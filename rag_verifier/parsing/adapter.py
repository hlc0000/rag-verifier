"""解析层 → 切分层的桥：StructuredDocument → 切分器的行输入。

切分层只面对 (行文本, 页码) 的列表 + 表格注册表；它不知道原始格式。
表格以哨兵行进入行流（保证阅读顺序），实体内容放在 tables 注册表里，
切分器遇到哨兵行时用 emit_table 整表成块——表格不参与段落切分。
"""
from __future__ import annotations

import re

from src.parsing.models import StructuredDocument

TABLE_SENTINEL = "\x00T{}\x00"

# docling 输出在标点两侧插空格（"3 . 1"、"C . 1 . 1"），编号切分锚点会全部失配。
# 仅归一化「数字/大写字母 . 数字/大写字母」（中文句号是「。」不受影响；宁可漏，不可误伤）。
_NUM_DOT = re.compile(r"(?<=[0-9A-Z]) \. (?=[0-9A-Z])")

# 标签-值配对：PDF 双列布局（如发布记录的「发布日期：| 2025.11.05」）被解析成
# 「标签段」「值段」两组独立段落——Docling 按列读，标签连排、值连排，不是
# 「标签-值」交替（release-notes 实测 C075：日期行单独成行还被 hierarchical_numeric
# 误当编号锚点，标签全进前置块、值全连成怪串）。合并条件从窄：标签以冒号结尾且
# 是普通段落（标题行豁免——标题与正文是两个语义单元），值行不含冒号（防吞双标签）、
# 短、同页；标签组与值组数量相等才配对（k==m，按序一一对应）。
_LABEL_LINE_MAX = 30   # 「DJI Fly App Android 版本：」≈20 字符
_VALUE_LINE_MAX = 40   # 值超过 40 字符更可能是正文段（「注意：」后的完整句）


def _is_label(text: str) -> bool:
    return text.endswith(("：", ":")) and len(text) <= _LABEL_LINE_MAX


def _is_value(text: str, page: int, page0: int) -> bool:
    if text.startswith("["):
        return False  # 方括号占位（[_____]、[NAME OF ...]）是填空位不是值——
        # 信贷协议附件签表单实测：占位行配进标签会改变表单块判定（宁可漏，不可误伤）
    if text.startswith("\x00"):
        return False  # 表格哨兵行绝不能被当值吞掉：信贷协议实测「Title:」+
        # 哨兵被并成一行，T19 等表格从行流消失，整表内容丢失（切分 141→140 漂移）
    return (page == page0 and len(text) <= _VALUE_LINE_MAX
            and "：" not in text and ":" not in text)


def _merge_label_value_pairs(
    lines: list[tuple[str, int]], heading_levels: dict[int, int]
) -> tuple[list[tuple[str, int]], dict[int, int]]:
    merged: list[tuple[str, int]] = []
    new_headings: dict[int, int] = {}
    i = 0
    while i < len(lines):
        text, page = lines[i]
        # 收集连续标签组（标题行豁免）
        if _is_label(text) and i not in heading_levels:
            labels: list[int] = []
            j = i
            while j < len(lines) and _is_label(lines[j][0]) and j not in heading_levels:
                labels.append(j)
                j += 1
            # 收集紧随的连续值组（同页）
            values: list[int] = []
            while j < len(lines) and _is_value(lines[j][0], lines[j][1], page):
                values.append(j)
                j += 1
            # 数量相等才按序配对；否则原样放行（宁可漏，不可误伤）
            if len(labels) == len(values) and labels:
                for li, vi in zip(labels, values):
                    merged.append((lines[li][0] + lines[vi][0], lines[li][1]))
                i = j
                continue
        if i in heading_levels:
            new_headings[len(merged)] = heading_levels[i]
        merged.append((text, page))
        i += 1
    return merged, new_headings


def _normalize(text: str) -> str:
    return _NUM_DOT.sub(".", text)


def document_to_lines(doc: StructuredDocument) -> tuple[list[tuple[str, int]], dict[str, dict], dict[int, int]]:
    """返回 ([(行文本, 页码)], {表格ID: {"page","content","data"}}, {行号: 标题层级})。

    heading_levels 记录行流中哪些行是解析层认定的标题及其层级。结构明写的格式
    （MD 的 #、HTML 的 h1-h6、DOCX 的内置标题样式）把结构写在标记里，这是切分层
    最可靠的第一类锚点——不能在过桥时和正文一锅端压平丢掉。
    """
    lines: list[tuple[str, int]] = []
    tables: dict[str, dict] = {}
    heading_levels: dict[int, int] = {}
    ti = 0
    for b in doc.blocks:
        page = b.page or 1
        if b.type in ("heading", "paragraph"):
            if b.meta.get("kind") == "code" and b.content.strip():
                # 代码块换行是语义不是折行：整块作为一行进流（前后垫 \n 作边界），
                # 内部换行原样保留；逐行拆分会让拼块时把代码压成一行
                lines.append(("\n" + b.content.strip() + "\n", page))
                continue
            for ln in b.content.split("\n"):
                if ln.strip():
                    if b.type == "heading" and b.level is not None:
                        heading_levels[len(lines)] = b.level
                    lines.append((_normalize(ln.strip()), page))
        elif b.type == "list" and b.items:
            for it in b.items:
                if str(it).strip():
                    lines.append((_normalize(str(it).strip()), page))
        elif b.type == "table":
            ti += 1
            tid = f"T{ti:03d}"
            tables[tid] = {"page": page,
                           "content": b.content or "", "data": b.data}
            lines.append((TABLE_SENTINEL.format(ti), page))
        # image 块：解析层只记录存在，不进切分行流
    lines, heading_levels = _merge_label_value_pairs(lines, heading_levels)
    return lines, tables, heading_levels
