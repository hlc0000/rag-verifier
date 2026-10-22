"""精排（rerank）：交叉编码器给粗排候选逐对精读重打分。

向量/BM25 都是「单向量概览」——只看得出大概像不像；交叉编码器把问题
和候选拼在一起逐字精读，能分清近重复块的细微差别（changelog 实测：多个
翻译块都在说「加了什么翻译」，粗排分不清是哪个版本的）。

只重排粗排前 N 名（默认 20），成本可控；ONNX 本地跑，同输入同输出。
"""
import os

# 模型下载走国内镜像（与嵌入模型、docling 同一约定）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# hf-mirror 不代理 xet CAS 存储（直连会 401），强制回退普通 HTTP 下载通道
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"  # 多语言 int8，Intel CPU 可跑
CANDIDATE_K = 20  # 粗排池子：多捞些给精排挑，20 是质量/成本平衡点

_reranker = None


def _get_reranker():
    """懒加载单例：模型只在首次使用时加载，避免不用精排时白付启动成本。"""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    """对粗排 hits 精排，返回前 top_k。结构不变，追加 rerank_score/rerank_rank。

    同分按原粗排顺序稳定（粗排名次当裁判），保证全链路确定性。
    """
    if len(hits) <= 1:
        return hits[:top_k]
    model = _get_reranker()
    scores = list(model.rerank(query, [h["record"].content for h in hits]))
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return [dict(hits[i], rerank_score=float(scores[i]), rerank_rank=rank)
            for rank, i in enumerate(order[:top_k])]
