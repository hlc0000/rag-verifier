"""向量 + BM25 双索引，RRF 混合排序。千级块规模，numpy 暴力扫毫秒级，零向量库依赖。

向量检索的原罪是「只保证像不保证对」；关键词检索恰好补上「不像但对」的漏
（专有名词、编号、条款号向量embedding抓不住，BM25 一抓一个准）。
两路结果用 RRF（倒数排名融合）合并：只看名次不看分值，免调权重，结果确定。
"""
import hashlib
import json
import os
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


def _rank_of(scores: list[float] | np.ndarray) -> np.ndarray:
    """分数 → 名次（0=最相关）。同分时按下标稳定排序，保证确定性。"""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = np.empty(len(scores), dtype=np.int64)
    for pos, i in enumerate(order):
        ranks[i] = pos
    return ranks


class HybridIndex:
    def __init__(self, records: list[ChunkRecord], matrix: np.ndarray, model):
        self.records = records
        self.matrix = matrix              # L2 归一化的块向量，余弦=点积
        self.model = model
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
        vec_ranks = _rank_of(self.matrix @ self.embed_query(query))
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
    # 旧矩阵在构建时已 L2 归一化，直接搬入分片缓存，数值逐位与旧缓存一致（检索零漂移）。
    return {_chunk_key(r): old[i] for i, r in enumerate(records)}


def build_index(records: list[ChunkRecord]) -> HybridIndex:
    """全量嵌入建索引（忽略缓存，重嵌所有块），并刷新分片缓存。"""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=EMBED_MODEL)
    print(f"[build_index] embedding {len(records)} chunks with {EMBED_MODEL} ...", flush=True)
    vecs = _normalize(np.array(list(model.embed([r.content for r in records])), dtype=np.float32))
    print(f"[build_index] done, dim={vecs.shape[1]}, saving shards ...", flush=True)
    _save_shards({_chunk_key(r): vecs[i] for i, r in enumerate(records)})
    print(f"[build_index] shards saved.", flush=True)
    return HybridIndex(records, vecs, model)


def load_or_build(records: list[ChunkRecord], rebuild: bool = False) -> HybridIndex:
    """按块分片缓存组装索引：只重嵌缺失/变化的块，其余复用。

    - 缓存命中（常规问答路径）：零嵌入，直接组装矩阵，数值与旧全库缓存逐位一致。
    - 首次运行：旧全库缓存指纹匹配则 seed 新分片缓存（免全量重建），成功后旧缓存退休。
    - rebuild=True：丢弃缓存，全部重嵌。
    """
    from fastembed import TextEmbedding

    keys = [_chunk_key(r) for r in records]
    cache = {} if rebuild else _load_shards()
    migrated = False
    if not cache and not rebuild and not SHARD_VEC.exists():
        cache = _seed_from_legacy(records)
        migrated = bool(cache)

    content_of = {k: r.content for k, r in zip(keys, records)}
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        model = TextEmbedding(model_name=EMBED_MODEL)
        vecs = _normalize(np.array(list(model.embed([content_of[k] for k in missing])),
                                   dtype=np.float32))
        for k, v in zip(missing, vecs):
            cache[k] = v

    pruned = {k: cache[k] for k in keys}       # 裁掉已删文档残留的块，缓存与语料同步
    if missing or migrated or set(pruned) != set(cache):
        _save_shards(pruned)
        if migrated:                           # 迁移完成，旧全库缓存可退休
            CACHE_VEC.unlink(missing_ok=True)
            CACHE_META.unlink(missing_ok=True)

    matrix = np.stack([pruned[k] for k in keys])
    return HybridIndex(records, matrix, TextEmbedding(model_name=EMBED_MODEL))
