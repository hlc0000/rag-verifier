"""从 heading 序列的编号走势推断层级（不依赖文档类型、不依赖编号约定）。

为什么不用「按文档类型加载规则表」：同一类文档内部编号也不同——这份合同用
「一、+1.」，下一份可能用「第一章+第一条」，再下一份用「1.1.1」。硬编码规则表
永远追不上。但**编号序列的走势**是共通的：同模式同级的编号 = 同级；
新模式 = 子级；栈里出现过的模式 = 弹栈回上一层。

核心洞察：层级不是外部规定的，是从文档内部的编号序列推断出来的。

边界处理：
- 无编号标题（前言/说明/大标题）→ 归到「当前栈顶层级」的同级，不改栈。
- 跳号（1. → 3.）→ 同级，跳号不改变层级。
- 换套（一、二、后突然 (a)(b)）→ 新模式即当前栈顶的子级。
- 回跳（一、1. 二、1.）→ 第二个 1. 弹回 cn_upper 后重新变子级。
"""
from __future__ import annotations

import re

from src.parsing.adapter import _normalize
from src.parsing.models import StructuredDocument

_CN = "一二三四五六七八九十百零〇两"


def pattern_of(text: str) -> str:
    """识别编号模式（已归一化空格，兼容 docling 的「3 . 2」）。"""
    t = _normalize((text or "").strip())
    if re.match(rf"^第[{_CN}]+章", t):
        return "cn_chapter"
    if re.match(rf"^第[{_CN}]+[条节]", t):
        return "cn_article"
    if re.match(rf"^[{_CN}]+、", t):
        return "cn_upper"
    if re.match(rf"^[（(][{_CN}]+[）)]", t):
        return "cn_paren"
    if re.match(r"^附\s*录\s*[A-Z]", t):
        return "appendix"
    if re.match(r"^[A-Z]\.\d+\.\d+\.\d+", t):
        return "appendix_d4"
    if re.match(r"^[A-Z]\.\d+\.\d+", t):
        return "appendix_d3"
    if re.match(r"^[A-Z]\.\d+(?!\.\d)", t):
        return "appendix_d2"
    if re.match(r"^\d+\.\d+", t):
        return "multi"                       # 3.1 / 5.1.1
    if re.match(r"^\d+\s*[\.、]", t):
        return "arabic"                      # 1. / 1、
    if re.match(r"^[（(]\d+[）)]", t):
        return "ar_paren"
    if re.match(r"^\d+(?=\s|$)", t):
        return "arabic"                      # 裸数字 1 2 3（GB/T 章节号）
    return "none"


# 附录编号是跨文档通用约定（GB/T、ISO 等都用 A.1 / A.1.1），按深度给绝对层级。
# 从深到浅匹配：先 4 级，再 3 级，再 2 级——否则 A.1.1 会被 A.1 模式截断。
APPENDIX_RULES = [
    (r"^[A-Z]\.\d+\.\d+\.\d+", 4),
    (r"^[A-Z]\.\d+\.\d+", 3),
    (r"^[A-Z]\.\d+(?!\.\d)", 2),
    (r"^附\s*录\s*[A-Z]", 1),
]


def appendix_level(text: str) -> int | None:
    """附录标题/编号的绝对层级；非附录返回 None。"""
    t = _normalize((text or "").strip())
    for pattern, level in APPENDIX_RULES:
        if re.match(pattern, t):
            return level
    return None


def infer_levels(headings: list[str]) -> list[int]:
    """栈式推断：返回与 headings 等长的层级列表。"""
    stack: list[tuple[str, int]] = []   # [(pattern, level)]
    out: list[int] = []
    for text in headings:
        # 附录走绝对层级（跨文档通用约定），并重置栈基准——结果与输入顺序无关
        alvl = appendix_level(text)
        if alvl is not None:
            out.append(alvl)
            stack = [("appendix", alvl)]
            continue
        pat = pattern_of(text)
        if pat == "none":
            out.append(stack[-1][1] if stack else 1)
            continue
        # 弹到栈里已出现过的同模式（回跳），否则停住（新模式→子级）
        while stack and stack[-1][0] != pat and any(p == pat for p, _ in stack):
            stack.pop()
        if stack and stack[-1][0] == pat:
            out.append(stack[-1][1])          # 同级
        else:
            lvl = len(stack) + 1              # 新的子级
            stack.append((pat, lvl))
            out.append(lvl)
    return out


def infer_levels_from_sequence(doc: StructuredDocument) -> StructuredDocument:
    """用 heading 序列推断层级，覆盖 docling 给的 level。"""
    idx = [i for i, b in enumerate(doc.blocks) if b.type == "heading"]
    levels = infer_levels([doc.blocks[i].content for i in idx])

    trace = []
    for i, lvl in zip(idx, levels):
        b = doc.blocks[i]
        trace.append({"text": (b.content or "").strip(),
                      "before": b.level, "after": lvl,
                      "pattern": pattern_of(b.content)})
        b.level = lvl

    doc.metadata["level_method"] = "sequence_stack"
    doc.metadata["headings_total"] = len(idx)
    doc.metadata["headings_changed"] = sum(1 for t in trace if t["before"] != t["after"])
    doc.metadata["heading_patterns"] = sorted({t["pattern"] for t in trace})
    doc.metadata["level_trace"] = trace
    return doc
