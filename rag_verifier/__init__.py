"""rag-verifier 对外包入口。

    from rag_verifier import verify_chunks

只依赖验证层（不导入 docling/pymupdf/检索层），可作为独立组件嵌入任意
Python 流水线：任意解析器产出的块 → 五关质量验证。
"""
from rag_verifier.chunking.rules import CHECKS, RuleContext, detect_language, list_rules, register_rule
from rag_verifier.chunking.semantic_verifier import builtin_llm
from rag_verifier.citation.verifier import verify_citation
from rag_verifier.models import EvidenceCard, VerificationResult, VerificationStatus
from rag_verifier.verify_api import verify_chunk, verify_chunks

__all__ = [
    "verify_chunk",
    "verify_chunks",
    "verify_citation",
    "EvidenceCard",
    "VerificationResult",
    "VerificationStatus",
    "builtin_llm",
    "register_rule",
    "detect_language",
    "list_rules",
    "RuleContext",
    "CHECKS",
]
