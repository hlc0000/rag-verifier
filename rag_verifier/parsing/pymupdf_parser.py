"""pymupdf 解析器：把 PDF 的「视觉信号」翻译成逻辑块。

做四件事（全部是解析层职责，不碰切分）：
  1. 表格  ：page.find_tables() 提取为 {type:"table", data}，不再散成文本行
  2. 标题  ：字体排版信号（字号 > 正文 / 加粗且短行）识别为 {type:"heading", level}
  3. 页脚/页码/目次页：重复短行、页首页尾纯数字、目次页 —— 抽取噪声，剔除并记入 metadata
  4. 拆号合并：被换行拆开的编号（"5." + "1"）合并回一行

诚实边界：扫描件无文本层时 get_text 为空 —— metadata 里标注
text_layer=false，本解析器不做 OCR（那是另一个可插拔解析器的事）。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from rag_verifier.parsing.base import Parser, register
from rag_verifier.parsing.models import Block, StructuredDocument, serialize_table
from rag_verifier.parsing.text_clean import DIGIT_LINE, TOC_HEADING, HEADING_FRAGMENT, merge_fragments

# 页码判定：纯数字行落在页顶 10% 或页底 20% 区域内
NUM_LINE = re.compile(r"^\d{1,3}$")
END_PUNCT = re.compile(r"[。；，,；]$")


def _center_in(bbox: tuple, outer: tuple) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _line_text(spans: list[dict]) -> str:
    """span 拼接成行文本，按字形间隙补空格。

    get_text() 纯文本模式会按字距自动插空格（"5.1 范围"），而 dict 模式的
    span 拼接不会——直接 "".join 会把 "5.1 范围" 变成 "5.1范围"，
    行首编号正则（要求编号后跟空白）全部失配。这里按 span 间的横向间隙
    恢复空格：间隙 > 字号的 15%（且至少 0.9pt）视为一个空格。
    """
    parts: list[str] = []
    prev: dict | None = None
    for s in spans:
        t = s["text"]
        if not t:
            continue
        if prev is not None:
            gap = s["bbox"][0] - prev["bbox"][2]
            if gap > max(0.15 * s["size"], 0.9):
                parts.append(" ")
        parts.append(t)
        prev = s
    return "".join(parts)


@register
class PyMuPDFParser(Parser):
    name = "pymupdf"

    def parse(self, path: str) -> StructuredDocument:
        import pymupdf

        pdf = pymupdf.open(path)
        npages = len(pdf)
        heights = {p + 1: pdf[p].rect.height for p in range(npages)}

        items: list[dict] = []          # 文本行（已排除表格区域）
        tables: list[dict] = []         # {"page","y","rows"}
        freq: Counter = Counter()
        toc_pages: set[int] = set()
        images = 0

        for pno, page in enumerate(pdf, 1):
            finder = page.find_tables()
            bboxes = [t.bbox for t in finder.tables]
            for t in finder.tables:
                tables.append({"page": pno, "y": t.bbox[1], "rows": t.extract()})
            d = page.get_text("dict")
            page_lines: list[tuple[float, str]] = []
            for blk in d["blocks"]:
                if blk.get("type") != 0:
                    images += 1
                    continue
                if any(_center_in(blk["bbox"], bb) for bb in bboxes):
                    continue                              # 表格区域的文本不重复算
                for ln in blk["lines"]:
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    txt = _line_text(spans).strip()
                    page_lines.append((ln["bbox"][1], txt))
                    items.append({
                        "page": pno, "y": ln["bbox"][1], "text": txt,
                        "size": max(s["size"] for s in spans),
                        "bold": any(s["flags"] & 16 for s in spans),
                    })
            if any(TOC_HEADING.match(t) for _, t in page_lines):
                toc_pages.add(pno)
            freq.update({t for _, t in page_lines})

        text_chars = sum(len(it["text"]) for it in items)
        footers = {t for t, n in freq.items()
                   if n >= max(3, npages // 2) and len(t) <= 40}

        # 剔除：目次页整页、页脚、页首页尾页码
        kept = []
        removed = {"footer": 0, "page_number": 0, "toc_page": 0}
        prev_kept: dict | None = None
        for it in items:
            if it["page"] in toc_pages:
                removed["toc_page"] += 1
                continue
            if it["text"] in footers:
                removed["footer"] += 1
                continue
            if NUM_LINE.match(it["text"]):
                # 裸数字行紧跟拆号片段（'3.' + '6'）时是编号的下半段，不是页码——
                # dict 模式会把 "3.6 xxx" 按字体变化拆成 '3.'/'6' 两个"行"（同一视觉行）。
                prev_is_fragment = prev_kept is not None and HEADING_FRAGMENT.match(prev_kept["text"])
                rel = it["y"] / heights[it["page"]]
                if not prev_is_fragment and (rel <= 0.10 or rel >= 0.80):
                    removed["page_number"] += 1
                    continue
            kept.append(it)
            prev_kept = it
        # 保持抽取顺序（dict 模式的块序与 plain 模式一致）——
        # 不要按 y 重排：图注/上下标行的 y 有微小抖动，重排会把 "3.6" 这类
        # 相邻行拆散、打乱图注区行的先后（旧 pages 路径也从未排序）。
        merged = merge_fragments(kept)

        # 正文基准字号（按字符数加权的主流字号）
        size_freq: Counter = Counter()
        for it in merged:
            size_freq[it["size"]] += len(it["text"])
        body_size = size_freq.most_common(1)[0][0] if size_freq else 0.0
        heading_sizes = sorted({s for s in size_freq if s > body_size + 0.8}, reverse=True)

        def is_heading(it: dict) -> bool:
            if it["size"] in heading_sizes:
                return True
            # 加粗兜底：同字号但加粗的短行（无句尾标点）视为标题（GB 章标题常为同字号加粗）
            return (it["bold"] and it["size"] >= body_size - 0.1
                    and len(it["text"]) <= 60 and not END_PUNCT.search(it["text"]))

        def level_of(it: dict) -> int:
            if it["size"] in heading_sizes:
                return heading_sizes.index(it["size"]) + 1
            return len(heading_sizes) + 1

        # 表格按「页内首次越过其顶边的文本行之前」插入行流——保持文本抽取顺序，不做 y 重排
        page_tables: dict[int, list[dict]] = {}
        for t in tables:
            page_tables.setdefault(t["page"], []).append(
                {"page": t["page"], "y": t["y"], "kind": "table", "rows": t["rows"]})
        for pt in page_tables.values():
            pt.sort(key=lambda x: x["y"])

        stream: list[dict] = []
        for it in merged:
            pt = page_tables.get(it["page"], [])
            while pt and pt[0]["y"] <= it["y"]:
                stream.append(pt.pop(0))
            stream.append(it)
        for pg in sorted(page_tables):
            stream.extend(page_tables[pg])

        blocks: list[Block] = []
        para: list[dict] = []

        def flush_para():
            if para:
                blocks.append(Block(
                    type="paragraph",
                    content="\n".join(p["text"] for p in para),
                    page=para[0]["page"],
                ))
                para.clear()

        for it in stream:
            if it.get("kind") == "table":
                flush_para()
                blocks.append(Block(type="table", content=serialize_table(it["rows"]),
                                    data=it["rows"], page=it["page"]))
            elif is_heading(it):
                flush_para()
                blocks.append(Block(type="heading", content=it["text"],
                                    level=level_of(it), page=it["page"],
                                    meta={"bold": it["bold"], "size": it["size"]}))
            else:
                para.append(it)
        flush_para()

        n_head = sum(1 for b in blocks if b.type == "heading")
        metadata = {
            "pages": npages,
            "text_layer": text_chars > 0,
            "body_font_size": round(body_size, 1),
            "heading_font_sizes": [round(s, 1) for s in heading_sizes],
            "headings_detected": n_head,
            "heading_detection": "typography" if n_head >= 3 else "weak",
            "tables": len(tables),
            "images": images,
            "footers_removed": sorted(footers),
            "toc_pages": sorted(toc_pages),
            "lines_removed": removed,
        }
        return StructuredDocument(source=Path(path).name, engine=self.name,
                                  blocks=blocks, metadata=metadata)
