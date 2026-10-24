"""解析质量检测：第三类失败（解析"成功"但输出是垃圾）在这里拦住。

两层防线：
1. validate_parse_quality——自检（空文本/乱码/无标题），不需要基准；
2. check_against_raw_text——原文回查（核心）。原文就是 ground truth：
   pymupdf 秒级提取的纯文本不完美（无结构、表格乱），但它忠实反映
   「原文里有哪些字」——不会漏字、不会编造。用它做基准问两个问题：
   原文有的，解析出来了吗（content_lost）？解析出来的，原文里有吗（hallucination）？

职责边界（库 vs 页面）：本模块只产出结构化信息（QualityReport），
message 为事实陈述（数量/占比），不写任何面向最终用户的操作建议——
文案、分级展示、通知渠道都是产品层的事。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rag_verifier.parsing.models import StructuredDocument

# 全文文本低于该字数视为"空"（扫描件、纯图片页、解析翻车都会命中）
EMPTY_TEXT_MIN = 50
# 不可打印字符 / U+FFFD 替换字符占比超过该阈值视为乱码（编码错乱、字体映射失败）
GARBLED_RATIO = 0.1
GARBLED_FFFD_RATIO = 0.03
# 块数超过该值却零标题才提示 no_headings（短文档没有标题很正常）
NO_HEADING_MIN_BLOCKS = 10
# 纯文本格式本来就没有标题概念，no_headings 对它们是误伤
_NO_HEADING_EXEMPT = {".txt"}

# 原文回查阈值：解析文本量低于原文 70% 判内容丢失
# （30% 冗余用于吸收合法丢弃：页眉页脚、页码、装饰性重复文本）
CONTENT_LOST_MIN_RATIO = 0.7
# 超过 20% 的块在原文中找不到才报警幻觉（容忍个别块的合法改写/标点差异）
HALLUCINATION_BLOCK_RATIO = 0.2
# 幻觉检测只取每块前 100 个归一化字符做子串比对（够定位、控成本）
_HALLUCINATION_PROBE_LEN = 100

# 各 issue code 的严重度（产品层据此决定展示级别；库只标级，不做文案）
ISSUE_SEVERITY = {
    "empty_text": "high",         # 全文为空/极少——产出不可用
    "content_lost": "high",       # 明显漏解析
    "hallucination": "high",      # 输出含原文没有的内容——最危险
    "all_empty_blocks": "high",   # 有块无内容——产出不可用
    "likely_scanned": "medium",   # 疑似扫描件——大概率不可用，但需用户确认
    "garbled_text": "medium",     # 乱码占比异常
    "no_headings": "low",         # 无标题结构——影响有限
}
_SEVERITY_ORDER = ("low", "medium", "high")


@dataclass
class QualityReport:
    """质量检测结构化报告（库的对外接口）：passed/severity/issues 三要素。

    passed=False 不代表阻断——只表示质量门未通过，需产品层把选择权交给用户。
    """
    passed: bool
    severity: str                  # high | medium | low（无问题时为 low）
    issues: list[dict] = field(default_factory=list)  # [{"code","severity","message"}]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(s: str) -> str:
    """NFKC 折叠 + 去空白、去标点：只留文字本身。

    消除三类合法差异，避免误报：
    - docling 的 CJK 字形间隙空格（"东 方" vs "东方"）；
    - 表格序列化的 " | "、换行等结构字符（原文里是空白/版式）；
    - 连字/兼容字形差异（pymupdf 抽出 "ﬁ" U+FB01、docling 抽出 "fi"），
      NFKC 会把兼容字符折叠成标准形式再比对。
    """
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", s))


def _issue(code: str, message: str) -> dict:
    return {"code": code, "severity": ISSUE_SEVERITY[code], "message": message}


def validate_parse_quality(doc: StructuredDocument) -> list[dict]:
    """自检问题清单（不需要基准）；空列表 = 未检出异常。"""
    issues: list[dict] = []
    text = doc.full_text().strip()

    # 1. 全文为空或极少——最典型的"解析成功但产出垃圾"
    if len(text) < EMPTY_TEXT_MIN:
        issues.append(_issue("empty_text", f"全文仅 {len(text)} 字符"))
        # 归因细化：PDF 有图无文 → 高度疑似无文本层的扫描件
        if doc.source.lower().endswith(".pdf") and doc.metadata.get("images"):
            issues.append(_issue("likely_scanned",
                                 f"PDF 含 {doc.metadata['images']} 张图片但几乎无文本"))

    # 2. 乱码：不可打印字符占比异常，或 U+FFFD 替换字符占比异常（字体映射失败的特征）
    if text:
        non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\t")
        fffd = text.count("�")
        if non_printable / len(text) > GARBLED_RATIO or fffd / len(text) > GARBLED_FFFD_RATIO:
            pct = round(max(non_printable, fffd) / len(text) * 100, 1)
            issues.append(_issue("garbled_text", f"异常字符占比约 {pct}%"))

    # 3. 有块但全部无内容（块结构在、文本全空）
    if doc.blocks and not any(b.content or b.items or b.data for b in doc.blocks):
        issues.append(_issue("all_empty_blocks", f"{len(doc.blocks)} 个块均无内容"))

    # 4. 结构化文档应有标题：块数不少却零标题（TXT 豁免——纯文本没有标题概念）
    if (len(doc.blocks) > NO_HEADING_MIN_BLOCKS
            and not any(b.type == "heading" for b in doc.blocks)
            and Path(doc.source).suffix.lower() not in _NO_HEADING_EXEMPT):
        issues.append(_issue("no_headings", f"{len(doc.blocks)} 个块中无标题块"))

    return issues


def check_against_raw_text(doc: StructuredDocument, path: str) -> list[dict]:
    """原文回查（仅 PDF）：pymupdf 纯文本做基准，检测内容丢失与幻觉。

    不需要预知"解析应该是什么样"，只问原文两个问题：
    - content_lost：原文有的，解析出来了吗？
    - hallucination：解析出来的，原文里有吗？

    返回明细列表（code/message/severity），供产品层展示；基准本身读不到时
    返回空（宁可漏，不因基准故障误伤正常结果）。
    """
    import pymupdf

    try:
        with pymupdf.open(path) as pdf:
            raw_text = "".join(page.get_text() for page in pdf)
    except Exception:
        return []

    raw_n = _normalize(raw_text)
    parsed_n = _normalize(doc.full_text())
    issues: list[dict] = []

    # 检测1：内容丢失——解析文本量显著少于原文
    if raw_n and len(parsed_n) < len(raw_n) * CONTENT_LOST_MIN_RATIO:
        lost_pct = round((1 - len(parsed_n) / len(raw_n)) * 100)
        issues.append(_issue(
            "content_lost",
            f"原文 {len(raw_n)} 字符，解析得到 {len(parsed_n)} 字符（少约 {lost_pct}%）"))

    # 检测2：内容幻觉——块内容应能在原文中找到
    # 表格块豁免：两个引擎的单元格抽取顺序本就不同，顺序差异是特性不是幻觉
    checked = 0
    unmatched = 0
    for b in doc.blocks:
        if b.type in ("image", "table"):
            continue
        text = b.content or "\n".join(str(i) for i in (b.items or []))
        probe = _normalize(text)[:_HALLUCINATION_PROBE_LEN]
        if not probe:
            continue
        checked += 1
        if probe not in raw_n:
            unmatched += 1
    if checked and unmatched / checked > HALLUCINATION_BLOCK_RATIO:
        issues.append(_issue(
            "hallucination",
            f"{unmatched}/{checked} 个块的内容在原文中找不到"))

    return issues


def build_quality_report(doc: StructuredDocument, path: str | None = None) -> QualityReport:
    """汇总自检 + 原文回查（PDF 且给了路径时），产出库对外的 QualityReport。"""
    issues = validate_parse_quality(doc)
    if path and Path(doc.source).suffix.lower() == ".pdf":
        issues.extend(check_against_raw_text(doc, path))
    severity = "low"
    for item in issues:
        if _SEVERITY_ORDER.index(item["severity"]) > _SEVERITY_ORDER.index(severity):
            severity = item["severity"]
    return QualityReport(passed=not issues, severity=severity, issues=issues)
