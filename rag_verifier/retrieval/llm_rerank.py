"""LLM 精排：RRF top20 候选池 → LLM 一次调用挑 5 个最相关 → 重排。

背景（盲测 2 起答错实测）：RRF 只看词面/向量名次，把 gold 块「埋」在同主题
干扰块后面（B013-E3 gold 第 11 名、B025-E1 第 19 名——都在池内但进不了 top5，
LLM 拿着部分相关块硬答/过度推理）。交叉编码器精排在 8GB Intel Mac 不可用
（~1.7s/对、机器忙时 5min/题），改用已有 LLM 做集外精排：单次调用判语义
相关性，恰好补 RRF 的词面盲区。

fail-open 原则：LLM 异常/输出不可解析/未选满时，退回 RRF 原序——宁可退回
旧排序，不阻塞链路。
"""
import json

from src.retrieval.generate import chat

RERANK_PROMPT = """You are a relevance judge for a document retrieval system.

Question: {question}

Candidate passages (numbered):
{candidates}

Pick the {top_k} passages most relevant to answering the question.
- Judge by semantic relevance, not keyword overlap
- Prefer passages that directly contain the answer or define what the question asks about
- If fewer than {top_k} are truly relevant, pick only those that are
- Output ONLY a JSON array of passage numbers in relevance order, e.g. [3, 1, 7]"""


def rerank_hits(hits: list[dict], question: str, top_k: int = 5,
                pool: int = 20, preview_chars: int = 650,
                rrf_keep: int = 2) -> list[dict]:
    """从 RRF 候选中精排 top_k。任何失败退回 hits[:top_k]（fail-open）。

    融合规则「RRF 前 rrf_keep 名保底」：Q006 实测 LLM 会推翻 RRF 第 1 名
    （预览截掉答案句，LLM 判它不如别的块相关），强命中不能交给 LLM 裁决
    ——前 2 名无条件保留，LLM 只竞争其余席位。

    preview_chars=650：B013 实测「175 billion trainable parameters」在 gold
    块第 606 字符，150 字预览看不到 → LLM 漏选。650 覆盖多数块全文
    （中位 150 字符），20 块 ≈ 4k token/次调用，成本可接受。
    """
    pool_hits = hits[:pool]
    if len(pool_hits) <= top_k:
        return pool_hits
    keep = min(rrf_keep, top_k)
    lines = [f"{i+1}. {h['record'].content[:preview_chars].strip()}"
             for i, h in enumerate(pool_hits)]
    try:
        raw = chat(RERANK_PROMPT.format(question=question,
                                        candidates="\n".join(lines),
                                        top_k=top_k - keep), max_tokens=60)
        picks = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
        # 只接受池内序号、去重、排除保底席位
        allowed = {p for p in picks if isinstance(p, int) and keep < p <= len(pool_hits)}
        llm_picks = [pool_hits[p - 1] for p in picks if isinstance(p, int) and p in allowed]
        rest = [h for i, h in enumerate(pool_hits) if (i + 1) not in allowed
                and i >= keep]
        return (pool_hits[:keep] + llm_picks + rest)[:top_k]
    except Exception:
        return pool_hits[:top_k]
