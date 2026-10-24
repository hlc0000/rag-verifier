"""查询改写：口语化提问 → 检索友好变体，多路并行检索 RRF 融合。

真实用户不会按文档用词提问（"那个钱咋算" vs 文档里的"保管费按日计收"），
纯确定性检索对乱问法没有缓冲。改写层把口语问题翻译成 2 个检索友好变体。

安全性设计（宁可漏不可误伤）：
- 原问题永远参与检索且只增不替——多路名次 RRF 融合时原路分数始终在，
  改写变体即使被 LLM 写坏也只能增益，不能把原路的好结果挤出去；
- LLM 挂掉/返回畸形 → 退化为单路原问题检索（fail-open，绝不硬编造）。

这是全系统第 3 处 LLM 调用（口语问法才值得花这个钱），标准问法走
--no-rewrite 单路即可零额外成本。
"""
import json
import re
from pathlib import Path

from src.retrieval.generate import chat
from src.retrieval.corpus import DATA_DIR

# 改写结果磁盘缓存：同问题不重复烧 LLM；评估权重模拟全靠它
_CACHE_FILE = DATA_DIR / "rewrite_cache.json"
_HYDE_CACHE_FILE = DATA_DIR / "hyde_cache.json"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_hyde_cache() -> dict:
    if _HYDE_CACHE_FILE.exists():
        return json.loads(_HYDE_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_hyde_cache(cache: dict) -> None:
    _HYDE_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


HYDE_PROMPT = """写一段约 80-150 词、可能含答案的文档片段，用来检索技术文档。
要求：
1. 直接写文档段落，不要回答用户；用陈述句、术语规范
2. 保留问题原语言（中文问题输出中文，英文问题输出英文）
3. 可以包含问题里没明说但相关的术语，但不要编造数值/具体条款号
4. 只输出文档片段本身，不要任何说明

用户问题：{question}"""


def hyde_doc(question: str) -> str:
    """LLM 生成假设答案文档片段。失败时返回 question 本身（fallback 到原路检索）。"""
    cache = _load_hyde_cache()
    if question in cache:
        return cache[question]
    try:
        doc = chat(HYDE_PROMPT.format(question=question), max_tokens=400).strip()
        if len(doc) < 20:
            return question
        cache[question] = doc
        _save_hyde_cache(cache)
        return doc
    except Exception:
        return question

REWRITE_PROMPT = """把下面的用户问题改写成 {n} 个适合文档检索的查询变体。

要求：
1. 每个变体补上问题中省略的规范用词（口语→书面），但不要编造问题没提的概念
2. 保留问题原语言（中文问题输出中文，英文问题输出英文）
3. 变体之间用词尽量不同（同义替换、详略不同），提高至少一路命中的概率
4. 只输出 JSON 数组，形如 ["变体一", "变体二"]，不要输出别的

用户问题：{question}"""


def rewrite_query(question: str, n: int = 2) -> list[str]:
    """口语问题 → [原问题, 变体...]。任何故障都退化为 [原问题]。缓存命中不调 LLM。"""
    cache = _load_cache()
    if question in cache:
        return [question] + cache[question]
    try:
        raw = chat(REWRITE_PROMPT.format(n=n, question=question), max_tokens=300)
        variants = json.loads(re.search(r'\[.*\]', raw, re.S).group(0))
        variants = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
        # 去重并排除与原问题几乎相同的变体（改写没价值还费一路检索）
        variants = [v for v in variants
                    if v != question and re.sub(r'\W', '', v) != re.sub(r'\W', '', question)]
        variants = variants[:n]
        cache[question] = variants
        _save_cache(cache)
        return [question] + variants
    except Exception:
        return [question]


def search_multi(index, question: str, top_k: int = 5, doc: str | None = None,
                 variants: list[str] | None = None, orig_weight: float = 2.0) -> list[dict]:
    """多路检索 RRF 融合。variants 为空时等价于单路 search()。

    原问题路加权（×2，38 题模拟选定）：不加权时改写变体的一致意见会把
    原路强命中挤出 top5（标准题 92.1%→89.5% 实测翻车）；×2 后标准题
    零漂移、口语题 79.0%→86.8%。×1 口语虽到 89.5% 但伤标准题，×3/×5
    口语收益递减——按「宁可漏不可误伤」取 ×2。
    """
    if variants is None:
        return index.search(question, top_k=top_k, doc=doc)
    if variants == [question]:
        return index.search(question, top_k=top_k, doc=doc)

    k = 60
    weights = [orig_weight] + [1.0] * (len(variants) - 1)
    pools = [index.search(v, top_k=20, doc=doc) for v in variants]

    def key(h):
        return (h["record"].doc, h["record"].chunk_id)

    fused: dict[tuple, float] = {}
    for w, pool in zip(weights, pools):
        for rank, h in enumerate(pool):
            fused[key(h)] = fused.get(key(h), 0.0) + w / (k + rank)

    # 原问题那一路的命中带出 vec/bm25 名次做展示参考
    first_pool = {key(h): h for h in pools[0]}
    any_pool = {key(h): h for pool in pools for h in pool}
    items = []
    for cid, score in fused.items():
        base = first_pool.get(cid) or any_pool[cid]
        # 仅改写路命中的块没有原问题单路名次，补 -1 保持返回结构一致
        base.setdefault("vec_rank", -1)
        base.setdefault("bm25_rank", -1)
        items.append((score, cid, dict(base, rr=score)))
    items.sort(key=lambda x: (-x[0], x[1]))
    return [it[2] for it in items[:top_k]]
