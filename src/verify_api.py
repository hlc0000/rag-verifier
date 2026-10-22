"""对外 API：任何解析器产出的块 → 块级质量验证结果。

设计目标（通用可插拔）：
  - 不依赖本仓库的解析器/切分器，块只需要 content（source/locator 可选）；
  - 可追溯性支持 exact / normalized / custom 三种匹配策略——别人用自己的
    解析器做过文本规范化，不该因此被判「无法定位」；
  - LLM 由调用方注入（llm_callback=prompt -> str），不隐式绑定任何供应商；
    不传则语义层返回 uncertain，其余四关照常判定；
  - 检查项可裁剪（checks=...），只想要规则层就不花 LLM 钱。

用法：

    from rag_verifier import verify_chunks, builtin_llm

    chunks = [{"content": "...", "source": "a.pdf", "locator": "p3",
               "source_text": "该块所在章节原文"}]
    results = verify_chunks(chunks, traceability_mode="normalized")

    # 需要语义层时显式注入（内置客户端只是可选实现）：
    results = verify_chunks(chunks, llm_callback=builtin_llm)
"""
from src.chunking.semantic_verifier import builtin_llm
from src.chunking.verifier import ALL_CHECKS, verify_chunk as _verify_chunk_obj
from src.models import EvidenceCard

__all__ = ["verify_chunk", "verify_chunks", "ALL_CHECKS", "builtin_llm"]


def verify_chunk(content=None, source_text="", *, prev_chunk="", full_doc="",
                 text=None, checks=None, traceability_mode="exact",
                 traceability_normalizer=None, llm_callback=None) -> dict:
    """Verify a single chunk of text.

    The block can be passed either as the first positional argument or via the
    keyword ``text=`` (more descriptive for external callers). ``content=`` is
    also accepted for symmetry with the internal naming.

    Signature (all positional-arg variants accepted):
        verify_chunk("block text", source_text="...", prev_chunk="...", full_doc="...")
        verify_chunk(text="block text", prev_chunk="...")
        verify_chunk(content="block text", prev_chunk="...")

    Returns a dict:
        {"overall": VerificationStatus, "checks": {
             "independence":        {"status": "pass|fail|uncertain", "reason": "..."},
             "completeness":        {"status": "...", "reason": "..."},
             "traceability":        {"status": "...", "reason": "..."},
             "semantic_independence": {"status": "...", "reason": "..."},
             "template_completeness": {"status": "...", "reason": "..."},
         }}

    With ``llm_callback`` not provided the semantic-independence check reports
    ``uncertain`` (fail-open; never silently reclassified as pass).
    """
    # Resolve which kwarg actually carries the block text.
    if content is None and text is not None:
        content = text
    elif content is None and text is None:
        raise TypeError("verify_chunk() requires the chunk text, pass it as the first argument or via text=... / content=...")

    if isinstance(content, EvidenceCard):
        card = content
    elif isinstance(content, str):
        card = EvidenceCard(content=content)
    else:
        raise TypeError(f"content 必须是 str 或 EvidenceCard，得到 {type(content).__name__}")
    doc = full_doc or source_text
    return _verify_chunk_obj(
        card, source_text=source_text, prev_chunk=prev_chunk,
        full_doc=doc, checks=checks, traceability_mode=traceability_mode,
        traceability_normalizer=traceability_normalizer,
        llm_callback=llm_callback,
    )


def verify_chunks(chunks, *, checks=None, traceability_mode: str = "exact",
                  traceability_normalizer=None, llm_callback=None,
                  full_doc: str = "", source_text_key: str = "source_text") -> list[dict]:
    """批量验证。

    chunks: 可迭代的 dict，每项至少含 "content"；可选 "source"/"locator"/
            "prev_chunk"/对应 source_text_key（该块所在章节原文，缺省用 full_doc）。
            也接受已有 EvidenceCard 实例。
    checks: 要跑的检查项子集，默认五关全跑（语义层仍需 llm_callback 才实际调用）。
    traceability_mode: exact（逐字）/ normalized（去空白）/ custom（配 normalizer）。
    llm_callback: prompt -> str；不传则语义层返回 uncertain。
    full_doc: 整份文档全文，作为未提供 source_text 时的兜底上下文。

    返回与输入同序的 list[dict]：{"index", "overall", "checks"}，
    输入带 "id" 时原样带回。
    """
    items = list(chunks)
    rows = []
    prev = ""

    for i, item in enumerate(items):
        if isinstance(item, EvidenceCard):
            card = item
            source_text = full_doc
            row_id = None
        else:
            if not isinstance(item, dict):
                raise ValueError(f"第 {i} 项不是 dict 或 EvidenceCard: {type(item).__name__}")
            content = item.get("content")
            if not isinstance(content, str):
                raise ValueError(f"第 {i} 项缺少字符串字段 'content'")
            card = EvidenceCard(content=content,
                                source=item.get("source", ""),
                                locator=item.get("locator", ""))
            source_text = item.get(source_text_key) or full_doc
            row_id = item.get("id")

        chunk_prev = item.get("prev_chunk", prev) if isinstance(item, dict) else prev
        result = verify_chunk(
            card,
            source_text=source_text,
            prev_chunk=chunk_prev,
            full_doc=full_doc,
            checks=checks,
            traceability_mode=traceability_mode,
            traceability_normalizer=traceability_normalizer,
            llm_callback=llm_callback,
        )

        row = {"index": i, "overall": result["overall"].value, "checks": result["checks"]}
        if row_id is not None:
            row["id"] = row_id
        rows.append(row)
        prev = card.content

    return rows
