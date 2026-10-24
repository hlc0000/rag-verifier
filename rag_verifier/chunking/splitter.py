"""切分统一入口：清洗 → 探测结构 → 选择切分器 → 执行切分。

为什么清洗必须保留换行（而不是把全文拼成一行）：
  1) 结构探测的编号模式全部是行首锚定（^...），先把换行拼掉的话，
     `^` 就只剩全文首行有效，所有计数失真、选择也跟着错；
  2) GB/T 类文档存在「5.」换行「1」的拆号，需要先把拆号合并回 5.1。
块内容在输出阶段才拼成单行（中文没有词间空格），与既有语料格式保持一致。

输入三选一：
  doc   : StructuredDocument（解析层产物，首选——页脚/页码/表格已由解析层处理）
  pages : extract_pdf.py 的原始页文本（旧路径，清洗在本模块内做）
  text  : 纯文本
"""
import re
from importlib import import_module

from rag_verifier.chunking.detector import detect_structure, select_splitter
from rag_verifier.parsing.text_clean import DIGIT_LINE, TOC_HEADING, HEADING_FRAGMENT
from rag_verifier.parsing.adapter import document_to_lines


def clean_pages(pages: list[dict]) -> list[tuple[str, int]]:
    """去页脚/页码/目次页，合并被换行拆开的编号；返回 [(行文本, 页码)]。

    页脚不用硬编码文档类型：凡是「在多数页面上重复出现的短行」就视作页脚
    （如 GB 的 "GB45438—2025"）。页码则是紧贴页脚、或落在页尾的纯数字行。
    """
    total = len(pages)
    freq: dict[str, int] = {}
    for p in pages:
        for ln in {l.strip() for l in p["text"].split("\n") if l.strip()}:
            freq[ln] = freq.get(ln, 0) + 1
    footers = {ln for ln, n in freq.items() if n >= max(3, total // 2) and len(ln) <= 40}

    kept: list[tuple[str, int]] = []
    for p in pages:
        raw = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if any(TOC_HEADING.match(l) for l in raw):
            continue                                     # 目次页不是正文，跳过
        n = len(raw)
        for i, ln in enumerate(raw):
            if ln in footers:
                continue
            if DIGIT_LINE.match(ln):
                # 页码：贴着页脚，或落在页首页尾（含被拆成多行的两位数页码，如 "1" + "0" = 10）
                near_footer = any(1 <= j - i <= 2 and raw[j] in footers for j in range(i + 1, n))
                if near_footer or i <= 1 or i >= n - 3:
                    continue
            kept.append((ln, p["page"]))

    # 合并拆号："5." + "1 文本…" → "5.1 文本…"；"C." + "1." + "1 文字…" → "C.1.1 文字…"
    merged: list[tuple[str, int]] = []
    i = 0
    while i < len(kept):
        line, page = kept[i]
        while i + 1 < len(kept) and HEADING_FRAGMENT.match(line) and re.match(r"^\d", kept[i + 1][0]):
            line += kept[i + 1][0]
            i += 1
        merged.append((line, page))
        i += 1
    return merged


# 结构明写的格式（原生 AST 解析）：结构写在标记里（# / h1-h6 / Word 标题样式），
# 标题层级是第一类锚点，优先级高于行首编号正则——正则会把网页引用编号、
# 年份误当标题，AST 读出来的 heading 块不会。
AST_ENGINES = ("markdown", "html", "docx")


def split_document(doc=None, pages: list[dict] | None = None, text: str | None = None,
                   source: str = "document.pdf", mode: str | None = None) -> list[dict]:
    """探测文档结构并切分。mode 非空时强制使用指定切分器（用于对比测试）。"""
    tables: dict[str, dict] | None = None
    heading_levels: dict[int, int] = {}
    if doc is not None:
        lines, tables, heading_levels = document_to_lines(doc)
        plain = "\n".join(l for l, _ in lines)
    elif pages is not None:
        lines = clean_pages(pages)
        plain = "\n".join(l for l, _ in lines)
    else:
        plain = text or ""
        lines = [(l.strip(), 1) for l in plain.split("\n") if l.strip()]

    structure = detect_structure(plain)
    chosen = mode
    if not chosen:
        if doc is not None and doc.engine in AST_ENGINES and heading_levels:
            chosen = "heading_ast"
        else:
            chosen = select_splitter(structure["active"])
    module = import_module(f"rag_verifier.chunking.splitters.{chosen}")
    kwargs = dict(source=source, structure=structure, chosen=chosen, tables=tables)
    if chosen == "heading_ast":
        kwargs["heading_levels"] = heading_levels
    return module.split(lines, **kwargs)
