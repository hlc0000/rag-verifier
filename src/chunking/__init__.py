"""Chunking layer: splitter + chunk-level verification.

Top-level exports are the ones users almost always want. Splitter classes and
the verifier module are still importable via ``src.chunking.verifier`` etc.
"""
from src.chunking.splitter import split_document, detect_structure, select_splitter
from src.chunking.verifier import ALL_CHECKS, verify_chunk as verify_chunk_obj
from src.chunking.template_verifier import verify_template_completeness

__all__ = [
    "split_document",
    "detect_structure",
    "select_splitter",
    "ALL_CHECKS",
    "verify_chunk_obj",
    "verify_template_completeness",
]
