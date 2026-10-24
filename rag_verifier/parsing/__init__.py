"""解析层：把任意格式的文档转换成统一的 StructuredDocument。

职责边界：只做「视觉/格式 → 逻辑/结构」的翻译，不做切分、不做验证、不做检索。
"""
from rag_verifier.parsing.base import (
    ParseError, Parser, available_engines, get_parser, parse_document, register,
)
from rag_verifier.parsing.models import Block, StructuredDocument, serialize_table
from rag_verifier.parsing.quality import (
    QualityReport, build_quality_report, check_against_raw_text, validate_parse_quality,
)

__all__ = [
    "Parser", "register", "get_parser", "parse_document", "available_engines",
    "ParseError", "QualityReport", "build_quality_report",
    "validate_parse_quality", "check_against_raw_text",
    "StructuredDocument", "Block", "serialize_table",
]
