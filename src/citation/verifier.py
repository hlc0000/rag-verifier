import json
import re

from src.chunking.semantic_verifier import _post_chat
from src.chunking.rules import detect_language

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
INSUFFICIENT = "insufficient"
UNCERTAIN = "uncertain"

VALID_VERDICTS = (SUPPORTED, UNSUPPORTED, INSUFFICIENT)


def normalize(text: str) -> str:
    """去掉所有空白，用于容忍排版/空白差异的逐字比对。"""
    return re.sub(r"\s+", "", text or "").strip()


def _faithfulness_prompt(claim: str, citation: str, source_text: str) -> str:
    """忠实性判定 prompt。

    必须给 LLM 完整的 source_text，而不只是 citation —— 否则它只能看到"引用说了什么"，
    看不到"引用在原文中处于什么位置"，判不出 005（引用在原文里是条件句的结果）。

    按输入文本语种选中文/英文模板：英文文档配中文 prompt 会让 reason 输出中文，
    且中文规则里的措辞（"按照约定"等）对英文合同不适用。
    """
    if detect_language(claim + " " + citation + " " + source_text[:200]) == "en":
        return _faithfulness_prompt_en(claim, citation, source_text)
    return f"""你是一个引证忠实性审查员。下面给出「原文」「模型引用的片段」「模型给出的声明」，
请判断该引用片段能否支持该声明。

【原文】（引用的完整来源，请通读，并注意引用片段在其中处于什么位置）
{source_text}

【引用的片段】
{citation}

【声明】
{claim}

三分类判定：
- supported：在原文语境下，该引用片段足以推出声明，且片段自足（不因切分丢失必要前提）。
- unsupported：引用片段与声明相矛盾（数字、金额、期限、权利义务方向等不一致）。
- insufficient：引用片段与声明无关；或引用片段是被切出的残缺片段，其成立依赖原文中被省略的
  条件/主语/前提，单独引用会误导。

unsupported 与 insufficient 不可混用：unsupported 是"矛盾"，insufficient 是"无关或证据不足"。

关键规则：若引用片段在原文中处于条件句的结果位置（即"如果A，则B"里的 B），而触发条件 A 未被
包含在引用片段内，那么即使片段字面与声明完全一致，也判 insufficient —— 因为该引用不自足，
单独引用会让读者误以为 B 无条件成立。
注意：若触发条件 A 已经出现在引用片段内部，本规则不适用；也不得因为"A 自身在原文中还有更早的
上下文"而判 insufficient —— 只要条件与结果在片段内都已完整写明，片段即自足，应判 supported。


只输出 JSON，不要任何其他文字，格式：
{{"verdict": "supported|unsupported|insufficient", "reason": "..."}}"""


def _faithfulness_prompt_en(claim: str, citation: str, source_text: str) -> str:
    """英文版忠实性判定 prompt（与中文版规则等价）。"""
    return f"""You are a citation faithfulness auditor. Below are the SOURCE DOCUMENT, the EXCERPT the model
cited, and the CLAIM the model made. Decide whether the cited excerpt supports the claim.

[SOURCE DOCUMENT] (the full origin of the citation — read it through, and note where the excerpt sits in it)
{source_text}

[CITED EXCERPT]
{citation}

[CLAIM]
{claim}

Three-way verdict:
- supported: in the context of the source, the cited excerpt is sufficient to derive the claim, and the
  excerpt is self-contained (it does not lose a necessary premise because of chunking).
- unsupported: the excerpt contradicts the claim (different number, amount, term, or direction of rights
  and obligations).
- insufficient: the excerpt is unrelated to the claim; or the excerpt is a truncated fragment whose
  validity depends on a condition/subject/premise omitted from the source, so quoting it alone misleads.

unsupported and insufficient must not be confused: unsupported means "contradiction", insufficient means
"irrelevant or insufficient evidence".

Key rule: if the excerpt sits in the consequent position of a conditional sentence in the source (the B
in "if A, then B") and the triggering condition A is not included in the excerpt, judge insufficient even
if the excerpt matches the claim word for word — the citation is not self-contained, and quoting it alone
would make the reader believe B holds unconditionally.
Note: if the triggering condition A is already present inside the cited excerpt, this rule does not apply;
do not judge insufficient merely because A itself has still earlier context in the source — as long as
the condition and the consequent are both fully stated inside the excerpt, the excerpt is self-contained
and must be judged supported.


Output JSON only, no other text, in this format (write `reason` in English):
{{"verdict": "supported|unsupported|insufficient", "reason": "..."}}"""


def _llm_faithfulness(claim: str, citation: str, source_text: str, lang: str = "zh") -> dict:
    """返回 {"verdict": str, "reason": str} 或 {"llm_unavailable": True, "reason": str}。"""
    en = lang == "en"
    try:
        content = _post_chat(_faithfulness_prompt(claim, citation, source_text))
    except Exception as e:
        return {"llm_unavailable": True, "reason": f"{type(e).__name__}: {e}"}

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"llm_unavailable": True, "reason": (
            f"Cannot parse the LLM output: {content[:200]}" if en
            else f"无法解析LLM输出: {content[:200]}")}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"llm_unavailable": True, "reason": (
            f"JSON decode failed: {e}" if en else f"JSON解析失败: {e}")}

    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        return {"llm_unavailable": True, "reason": (
            f"Invalid verdict: {verdict!r}" if en else f"verdict 非法: {verdict!r}")}
    return {"verdict": verdict, "reason": parsed.get("reason", "")}


def verify_citation(claim: str, citation: str, source_text: str) -> dict:
    """
    验证 answer 中的一条声明是否被它声称的引用支持。

    返回:
    {
      "status": "supported" | "unsupported" | "insufficient" | "uncertain",
      "reason": "...",
      "checks": {"existence": {...}, "verbatim": {...}, "faithfulness": {...}}
    }

    status 语义（恢复策略不同，不可混用）:
      supported    引用自足且支持声明                      -> 通过
      unsupported  引用与声明矛盾（数字/条件相反）          -> 重新生成
      insufficient 引用与声明无关，或引用不自足/证据不足    -> 重新检索
      uncertain    LLM 不可用，技术性降级（非语义判断）      -> 重试或人工
    """
    checks = {}
    cite_stripped = citation.strip()
    en = detect_language(claim + " " + citation + " " + source_text[:200]) == "en"

    # 第一层：引用存在性（精确）
    exact = bool(cite_stripped) and cite_stripped in source_text
    checks["existence"] = {
        "status": "pass" if exact else "fail",
        "reason": (("Citation appears verbatim in the source" if exact
                    else "Citation does not appear verbatim in the source")
                   if en else
                   ("引用逐字存在于原文" if exact else "引用未逐字出现在原文中")),
    }

    # 第二层：引用逐字性（归一化，容忍空白/排版差异）
    # 说明：精确与归一化都命中才算"存在"；改数字这类篡改在精确层即失败。
    norm = bool(normalize(citation)) and normalize(citation) in normalize(source_text)
    checks["verbatim"] = {
        "status": "pass" if norm else "fail",
        "reason": (("Located in the source after whitespace normalization" if norm
                    else "Still not located in the source after normalization")
                   if en else
                   ("归一化后可在原文中定位" if norm else "归一化后仍无法在原文中定位")),
    }

    # 存在性闸门：精确与归一化都定位不到 → 伪造/篡改，直接 unsupported，不调 LLM
    if not exact and not norm:
        return {
            "status": UNSUPPORTED,
            "reason": (
                "Citation does not exist in the source, or differs from it (fabricated/altered citation)"
                if en else
                "引用不存在于原文，或与原文不一致（伪造/篡改的引用）"
            ),
            "checks": checks,
        }

    # 第三层：忠实性（NLI 三分类）
    result = _llm_faithfulness(claim, citation, source_text, lang="en" if en else "zh")
    if result.get("llm_unavailable"):
        checks["faithfulness"] = {"status": "uncertain", "reason": result.get("reason", "")}
        return {
            "status": UNCERTAIN,
            "reason": ((f"LLM unavailable; faithfulness cannot be determined: {result.get('reason', '')}")
                       if en else
                       f"LLM不可用，忠实性无法判定: {result.get('reason', '')}"),
            "checks": checks,
        }

    verdict = result["verdict"]
    checks["faithfulness"] = {
        "status": "pass" if verdict == SUPPORTED else "fail",
        "reason": result.get("reason", ""),
    }
    return {"status": verdict, "reason": result.get("reason", ""), "checks": checks}
