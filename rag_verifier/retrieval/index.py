"""向量 + BM25 双索引，RRF 混合排序。千级块规模，numpy 暴力扫毫秒级，零向量库依赖。

向量检索的原罪是「只保证像不保证对」；关键词检索恰好补上「不像但对」的漏
（专有名词、编号、条款号向量embedding抓不住，BM25 一抓一个准）。
两路结果用 RRF（倒数排名融合）合并：只看名次不看分值，免调权重，结果确定。
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

# 模型下载走国内镜像，避免直连 huggingface.co 被墙（与 docling 同一约定）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from rag_verifier.retrieval.bm25 import BM25, tokenize  # noqa: E402
from rag_verifier.retrieval.corpus import DATA_DIR, ChunkRecord  # noqa: E402

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 旧「全库单文件」缓存：仅作一次性迁移来源（seed 完新分片缓存后即可退休）。
CACHE_VEC = DATA_DIR / "retrieval_index.npz"
CACHE_META = DATA_DIR / "retrieval_index_meta.json"
# 新「按块分片」缓存：key = sha1(doc|chunk_id|content)，向量按块存。
# 改一份文档只重嵌它变化的块，其余文档的向量原样复用，全库重建从此不再发生。
SHARD_VEC = DATA_DIR / "embed_shards.npz"
SHARD_META = DATA_DIR / "embed_shards_meta.json"
RRF_K = 60  # RRF 平滑常数，标准取值
# 多语言模型（paraphrase-multilingual-MiniLM-L12-v2）中文向量质量不如专属模型，
# 但 BM25 关键词匹配对中英都是主力。实测：W_VEC=1.0, W_BM25=2.0 时中文 recall@5
# 从 100% 降到 88%，把 BM25 权重拉到 5.0 让关键词主导，向量只做辅助。
W_VEC, W_BM25 = 1.0, 5.0

# ── 超长块分片嵌入（父子块）──────────────────────────────────────────────
# 起因：fastembed 的 paraphrase-multilingual-MiniLM-L12-v2 是「128 input tokens
# truncation」——超过 128 token 的部分直接被丢掉，不参与向量。实测本语料 1933 块
# 里有 650 块（33.6%）超限，即三分之一的块后半段在向量检索里等于不存在。
# 做法：块本身不动（块=引用单元，切开就没法逐字核验），只在建索引时按句子装箱
# 切成 ≤110 token 的窗口，每片一个向量并指回原块；命中任一片 → 原块参与排序。
# 检索单元变小，引用单元不变。
SHARD_MAX_TOKENS = 110        # 留在 128 截断线以内，留一点 tokenizer 误差余量
SHARD_OVERLAP_TOKENS = 25     # 相邻窗口重叠，避免命中句被切在边界上
SHARD_MAX_PER_CHUNK = 40      # 单块分片上限：防病态超长文档（如 11 万字样张）撑爆索引
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9]+")
_SENT = re.compile(r"(?<=[。！？；!?;\n])")


def _fingerprint(records: list[ChunkRecord]) -> str:
    """全库语料指纹：仅用于「旧全库缓存 ↔ 当前语料」一致性判定（迁移 seed 前的把门）。"""
    h = hashlib.sha1()
    for r in records:
        h.update(r.doc.encode())
        h.update(r.chunk_id.encode())
        h.update(r.content.encode())
    return h.hexdigest()


def _chunk_key(r: ChunkRecord) -> str:
    """块级嵌入缓存的 key：同一 (doc, chunk_id, 内容) 恒映射同一向量。

    内容不变则 key 不变、向量直接复用——这是「改一份文档只重嵌它的块」的依据。
    """
    h = hashlib.sha1()
    h.update(r.doc.encode())
    h.update(b"\x1f")
    h.update(r.chunk_id.encode())
    h.update(b"\x1f")
    h.update(r.content.encode())
    return h.hexdigest()  # 40 位十六进制，与 SHARD_VEC 的 "<U40" 定长一致


def _normalize(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


def _shard_key(r: ChunkRecord, i: int) -> str:
    """多窗块的分片缓存 key：在块 key 之外再加窗口序号。"""
    h = hashlib.sha1()
    h.update(r.doc.encode())
    h.update(b"\x1f")
    h.update(r.chunk_id.encode())
    h.update(b"\x1f")
    h.update(r.content.encode())
    h.update(b"\x1f")
    h.update(str(i).encode())
    return h.hexdigest()


def _est_tokens(s: str) -> int:
    """粗估 token：CJK 每字≈1，英文/数字按词≈1.3。只用来切窗，不追求精确计费。"""
    return len(_CJK.findall(s)) + int(len(_WORD.findall(s)) * 1.3) + 1


def _windows(text: str) -> list[str]:
    """把长块按句子装箱成 ≤SHARD_MAX_TOKENS 的窗口，相邻窗口重叠一小段。

    单句就超长时硬切；总片数封顶，避免病态超长文档把索引撑爆。
    返回长度 1 表示无需分片，调用方沿用整块嵌入。
    """
    if _est_tokens(text) <= SHARD_MAX_TOKENS:
        return [text]
    wins: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in _SENT.split(text):
        if not sent.strip():
            continue
        t = _est_tokens(sent)
        while t > SHARD_MAX_TOKENS and sent.strip():
            if cur:
                wins.append("".join(cur))
                cur, cur_tok = [], 0
            wins.append(sent[:SHARD_MAX_TOKENS])
            sent = sent[SHARD_MAX_TOKENS - SHARD_OVERLAP_TOKENS:]
            t = _est_tokens(sent)
            if len(wins) >= SHARD_MAX_PER_CHUNK:
                break
        if len(wins) >= SHARD_MAX_PER_CHUNK:
            break
        if cur and cur_tok + t > SHARD_MAX_TOKENS:
            wins.append("".join(cur))
            tail = "".join(cur)[-SHARD_OVERLAP_TOKENS:]
            cur, cur_tok = ([tail] if tail else []), _est_tokens(tail)
        cur.append(sent)
        cur_tok += t
    if cur and len(wins) < SHARD_MAX_PER_CHUNK:
        wins.append("".join(cur))
    return wins[:SHARD_MAX_PER_CHUNK] or [text]


def _shard_plan(records: list[ChunkRecord]) -> list[tuple[str, int, str]]:
    """(缓存 key, 原块下标, 待嵌入文本)。单窗块沿用旧 key，老缓存可直接复用。"""
    plan: list[tuple[str, int, str]] = []
    for i, r in enumerate(records):
        wins = _windows(r.content)
        if len(wins) == 1:
            plan.append((_chunk_key(r), i, r.content))
        else:
            for j, w in enumerate(wins):
                plan.append((_shard_key(r, j), i, w))
    return plan


def _rank_of(scores: list[float] | np.ndarray) -> np.ndarray:
    """分数 → 名次（0=最相关）。同分时按下标稳定排序，保证确定性。"""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = np.empty(len(scores), dtype=np.int64)
    for pos, i in enumerate(order):
        ranks[i] = pos
    return ranks


class HybridIndex:
    def __init__(self, records: list[ChunkRecord], matrix: np.ndarray, model,
                 owner: np.ndarray | None = None):
        self.records = records
        self.matrix = matrix              # L2 归一化的「分片」向量，余弦=点积
        self.model = model
        # owner[k] = 第 k 个分片属于哪个块；None 表示一行一块（未分片，兼容旧用法）
        self.owner = (np.arange(len(records), dtype=np.int64)
                      if owner is None else owner)
        # section_path 不拼进 BM25 文本（2026-10-18 实测）：六场景对照净收益≈0
        # （狠版 rewrite recall -2.6 与补全 +14.3 相抵，其余场景零漂移）——标题词
        # 扰动 BM25 全库排序，误伤不可接受，按「宁可漏不可误伤」回滚拼接。
        # 字段仍在语料中（emit 提供），hierarchical_english 实现后可复评。
        self.bm25 = BM25([tokenize(r.content) for r in records])

    def embed_query(self, query: str) -> np.ndarray:
        q = np.array(list(self.model.embed([query])), dtype=np.float32)
        return _normalize(q)[0]

    def search(self, query: str, top_k: int = 5, doc: str | None = None) -> list[dict]:
        """RRF 混合检索。doc 限定单文档（演示/调试用）。"""
        n = len(self.records)
        # 向量这一路：先算每个分片与问题的相似度，再按块取「最像的那一片」。
        # 这样命中块后半段的内容也能把整块救回来（截断问题由此解决），
        # 排序仍以块为单位，与 BM25 名次一一对应。
        shard_scores = (self.matrix @ self.embed_query(query)).astype(np.float32)
        rec_scores = np.full(n, -np.inf, dtype=np.float32)
        np.maximum.at(rec_scores, self.owner, shard_scores)
        vec_ranks = _rank_of(rec_scores)
        bm25_ranks = _rank_of(self.bm25.score(query))

        fused = []
        for i in range(n):
            if doc and self.records[i].doc != doc:
                continue
            score = W_VEC / (RRF_K + vec_ranks[i]) + W_BM25 / (RRF_K + bm25_ranks[i])
            fused.append((i, score, int(vec_ranks[i]), int(bm25_ranks[i])))
        fused.sort(key=lambda x: (-x[1], x[0]))
        # 同文档内同内容去重：PDF 逐页重复的页眉/大标题（release-notes 实测：10 个
        # 内容完全相同的「DJI Mini 4 Pro 发布记录」标题块霸榜前 10，把日期块挤到
        # #11，LLM 诚实拒答）。相同文本信息量相同，去重零信息损失，只留名次最好的
        # 一块；跨文档同文不去重（来源归属是产品信息，宁可漏不可误伤）。
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i, s, vr, br in fused:
            key = (self.records[i].doc, " ".join(self.records[i].content.split()))
            if key in seen:
                continue
            seen.add(key)
            out.append({"record": self.records[i], "score": s,
                        "vec_rank": vr, "bm25_rank": br})
            if len(out) == top_k:
                break
        return out


def _load_shards() -> dict[str, np.ndarray]:
    """读分片缓存 key→向量。换过嵌入模型则整体作废，避免新旧向量混用。"""
    if not (SHARD_VEC.exists() and SHARD_META.exists()):
        return {}
    try:
        meta = json.loads(SHARD_META.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if meta.get("model") != EMBED_MODEL:
        return {}
    data = np.load(SHARD_VEC, allow_pickle=False)
    return dict(zip([str(k) for k in data["keys"].tolist()], data["vectors"]))


def _save_shards(cache: dict[str, np.ndarray]) -> None:
    keys = sorted(cache)                       # 排序落盘，文件内容确定（便于校验/对比）
    vecs = (np.stack([cache[k] for k in keys]) if keys
            else np.zeros((0, 0), dtype=np.float32))
    np.savez(SHARD_VEC, keys=np.array(keys, dtype="<U40"), vectors=vecs)
    SHARD_META.write_text(json.dumps({"model": EMBED_MODEL, "n_keys": len(keys)},
                                     ensure_ascii=False, indent=1), encoding="utf-8")


def _seed_from_legacy(records: list[ChunkRecord]) -> dict[str, np.ndarray]:
    """一次性迁移：旧全库缓存与当前语料一致时，按记录顺序把向量 seed 进分片缓存。

    旧缓存的行序 = 构建时 records 的顺序；全库指纹一致即证明 records 逐条相同，
    故可直接按序配对，免去 ~19 分钟全量重建。指纹不一致就放弃（宁可重嵌，不冒险错配）。
    """
    if not (CACHE_VEC.exists() and CACHE_META.exists()):
        return {}
    meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    if meta.get("model") != EMBED_MODEL or meta.get("fingerprint") != _fingerprint(records):
        return {}
    old = np.load(CACHE_VEC)["matrix"]
    if old.shape[0] != len(records):
        return {}
    # 旧矩阵在构建时已 L2 归一化，直接搬入分片缓存。只对「单窗块」有效：
    # 多窗块的首片是截断后的窗口，与旧向量（整块截断）不完全等价，宁可重嵌。
    return {_chunk_key(r): old[i] for i, r in enumerate(records)
            if len(_windows(r.content)) == 1}


def build_index(records: list[ChunkRecord]) -> HybridIndex:
    """全量嵌入建索引（忽略缓存，重嵌所有分片），并刷新分片缓存。"""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=EMBED_MODEL)
    plan = _shard_plan(records)
    print(f"[build_index] embedding {len(plan)} shards from {len(records)} chunks "
          f"with {EMBED_MODEL} ...", flush=True)
    vecs = _normalize(np.array(list(model.embed([t for _, _, t in plan])), dtype=np.float32))
    print(f"[build_index] done, dim={vecs.shape[1]}, saving shards ...", flush=True)
    _save_shards({k: vecs[i] for i, (k, _, _) in enumerate(plan)})
    print(f"[build_index] shards saved.", flush=True)
    owner = np.array([i for _, i, _ in plan], dtype=np.int64)
    return HybridIndex(records, vecs, model, owner)


def load_or_build(records: list[ChunkRecord], rebuild: bool = False) -> HybridIndex:
    """按分片缓存组装索引：只重嵌缺失/变化的分片，其余复用。

    - 缓存命中（常规问答路径）：零嵌入，直接组装矩阵。
    - 首次运行：旧全库缓存指纹匹配则 seed 单窗块，其余照常补嵌。
    - rebuild=True：丢弃缓存，全部分片重嵌。
    """
    from fastembed import TextEmbedding

    plan = _shard_plan(records)
    keys = [k for k, _, _ in plan]
    cache = {} if rebuild else _load_shards()
    migrated = False
    if not cache and not rebuild and not SHARD_VEC.exists():
        cache = _seed_from_legacy(records)
        migrated = bool(cache)

    text_of = {k: t for k, _, t in plan}
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        model = TextEmbedding(model_name=EMBED_MODEL)
        vecs = _normalize(np.array(list(model.embed([text_of[k] for k in missing])),
                                   dtype=np.float32))
        for k, v in zip(missing, vecs):
            cache[k] = v

    pruned = {k: cache[k] for k in keys}       # 裁掉已删文档残留的分片，缓存与语料同步
    if missing or migrated or set(pruned) != set(cache):
        _save_shards(pruned)
        if migrated:                           # 迁移完成，旧全库缓存可退休
            CACHE_VEC.unlink(missing_ok=True)
            CACHE_META.unlink(missing_ok=True)

    matrix = np.stack([pruned[k] for k in keys])
    owner = np.array([i for _, i, _ in plan], dtype=np.int64)
    return HybridIndex(records, matrix, TextEmbedding(model_name=EMBED_MODEL), owner)
