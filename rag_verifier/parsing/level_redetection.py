"""按文档类型的编号约定，重定 heading 层级（解析层后处理）。

为什么需要：docling 是视觉/布局模型，它判标题层级只看字号与缩进两个信号。
中文公文/合同里「一、」和「1.」往往**同字号、同左边界**（实测：都 13.9pt / x0=101.8），
两个视觉信号全部失效 → 一律压成 L1。但层级本质是**文本约定**（编号格式），
不是视觉信息，因此这里用纯文本规则把层级贴回去。

设计边界：规则表按文档类型**可插拔**，解析层主逻辑不写死任何编号约定——
换文档类型 = 换规则表，解析层不动。规则表只做「编号 → 层级」的确定性映射。
"""
from __future__ import annotations

import re

from rag_verifier.parsing.adapter import _normalize
from rag_verifier.parsing.models import StructuredDocument

# 文档类型 → [(正则, 层级)]。列表内顺序即优先级：更具体的编号规则必须排在前面
# （如 "5.1" 要排在 "5." 前，否则会被 '\d+\.' 抢先匹配成二级）。
# 层级数字越小越靠上（1 = 最高级）。
LEVEL_RULES: dict[str, list[tuple[str, int]]] = {
    # 中文公文 / 合同：中文序号、第X章/条、阿拉伯数字、括号序号
    "cn_official": [
        (r"^第[一二三四五六七八九十百零〇]+章", 1),
        (r"^[一二三四五六七八九十]+、", 1),
        (r"^第[一二三四五六七八九十百零〇]+[条节]", 2),
        (r"^\d+\.\d+(?=\s|$)", 3),          # 3.1 / 3.1.1
        (r"^\d+\.\d+\.\d+(?=\s|$)", 3),
        (r"^\d+\s*[\.、]", 2),               # 1. / 1、
        (r"^[（(][一二三四五六七八九十\d]+[）)]", 3),  # （1）/（一）
    ],
    # GB/T 等数字编号标准：1 / 5.1 / 5.1.1 / A.1
    "gbt_numeric": [
        (r"^\d+\.\d+\.\d+(?=\s|$)", 3),
        (r"^\d+\.\d+(?=\s|$)", 2),
        (r"^[A-Z]\.\d+\.\d+(?=\s|$)", 3),
        (r"^[A-Z]\.\d+(?=\s|$)", 2),
        (r"^\d+(?=\s|$)", 1),
    ],
    # 英文合同：Article / Section / (a)
    "en_contract": [
        (r"^Article\s+\d+", 1),
        (r"^Section\s+\d+(\.\d+)?", 2),
        (r"^\([a-z]\)", 3),
    ],
}

DEFAULT_TYPE = "cn_official"


def detect_doc_type(doc: StructuredDocument) -> str:
    """按全文的编号模式分布推断文档类型（决定加载哪套规则表）。"""
    text = _normalize(doc.full_text())
    if len(re.findall(r"^\s*\d+\.\d+", text, re.M)) > 5:
        return "gbt_numeric"
    if len(re.findall(r"第[一二三四五六七八九十百零〇]+[条节]", text)) > 5:
        return "cn_official"
    if len(re.findall(r"\bArticle\s+\d+", text)) > 3:
        return "en_contract"
    return DEFAULT_TYPE


def redetect_levels(doc: StructuredDocument,
                    doc_type: str | None = None) -> StructuredDocument:
    """按文档类型规则表重贴 heading.level；记录未匹配项（供方法2 取舍判断）。"""
    if doc_type is None:
        doc_type = detect_doc_type(doc)
    rules = LEVEL_RULES.get(doc_type, LEVEL_RULES[DEFAULT_TYPE])

    changed = 0
    unmatched: list[str] = []
    for b in doc.blocks:
        if b.type != "heading":
            continue
        text = _normalize((b.content or "").strip())
        for pattern, level in rules:
            if re.match(pattern, text):
                if b.level != level:
                    changed += 1
                b.level = level
                break
        else:
            unmatched.append(text)

    doc.metadata["level_rules"] = doc_type
    doc.metadata["headings_releveled"] = changed
    doc.metadata["headings_unmatched"] = unmatched
    return doc
