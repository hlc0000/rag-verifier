"""层级数字编号文档（GB/T、行标、论文等）切分。

编号层级：
  顶级章    '1 范围' / '5 显式标识'（也被拆成裸数字 '5' 的情况）
  子条款    '5.1' / '5.1.1'
  附录标题  '附 录 A'
  附录条款  'A.1' / 'C.1.1' / 'D.2.1'

切分单元 = 一个编号行 + 其后的内容行，直到下一个编号行。
纯标题单元（编号行后紧跟另一个编号行、自身无正文，如 '5 显式标识'）不单独成块，
只作为 source_text 的分组边界——避免切出一堆只有标题的碎片。
"""
import re

from rag_verifier.chunking.splitters import (
    TABLE_LINE, collect_preamble, emit, emit_preamble, emit_table, is_table_line,
    join_span, join_unit,
)

CHAPTER = re.compile(r"^\d{1,2}(?:\s+\S.*)?$")              # 顶级章：5 / 5 显式标识 / 1 范围
TOP_DOT = re.compile(r"^\d{1,2}\.\s+\S")                    # RFC 顶级章：1.  Introduction
CLAUSE = re.compile(r"^\d+(?:\.\d+)+\.?(\s|$)")             # 子条款：5.1 / 5.1.1 / 5.6.（RFC 尾点）
APPENDIX_CLAUSE = re.compile(r"^[A-Z]\.\d+(?:\.\d+)*(\s|$)")  # 附录条款：A.1 / C.1.1
APPENDIX_HEAD = re.compile(r"^附\s*录\s*[A-Z]")             # 附录标题：附 录 A

CHOSEN = "hierarchical_numeric"


def _is_heading(line: str) -> bool:
    return bool(CHAPTER.match(line) or TOP_DOT.match(line) or CLAUSE.match(line)
                or APPENDIX_CLAUSE.match(line) or APPENDIX_HEAD.match(line))


def _is_group_head(line: str) -> bool:
    return bool(CHAPTER.match(line) or TOP_DOT.match(line) or APPENDIX_HEAD.match(line))


def split(lines: list, source: str = "document.pdf", structure: dict | None = None,
          chosen: str = CHOSEN, tables: dict | None = None) -> list[dict]:
    structure = structure or {"active": {}}
    tables = tables or {}
    headings = [i for i, (line, _) in enumerate(lines) if _is_heading(line)]
    groups = [i for i, (line, _) in enumerate(lines) if _is_group_head(line)]
    hset = set(headings)

    def group_text(idx: int) -> str:
        """取 idx 所属的整组文本（一级章/附录），作为该块的参考上下文。表格哨兵行不进组文本。

        锚点之前没有任何分组头时必须用全文兜底——不能取 groups[0]（那是锚点
        之后的分组，窗口与块内容完全无关，可追溯性检查必然冤判（RFC 实测 272 块））。
        标题行后插换行，与块内容同一变换（join_span/join_unit），保子串关系。
        """
        whole = join_span(lines, 0, len(lines), hset)
        before = [x for x in groups if x <= idx]
        if not before:
            return whole
        g = before[-1]
        ge = next((x for x in groups if x > g), len(lines))
        return join_span(lines, g, ge, hset)

    chunks: list[dict] = []
    # 首个锚点之前的内容（封面/前言）：不静默丢弃，显式标注
    if headings:
        pre_text, pre_page = collect_preamble(lines, headings[0])
        if pre_text:
            chunks.append(emit_preamble(pre_text, pre_page, source, structure, chosen))
    n = 0
    cur_section = ""          # 最近一个一级章/附录标题行文本（块的检索用章节路径）
    for k, head in enumerate(headings):
        # 裸数字行（合同里的孤立 "60" 等）会被 CHAPTER 误判为章标题——
        # 路径里只收含非数字字符的标题，纯数字垃圾路径宁可不写（宁可漏不可误伤）
        if _is_group_head(lines[head][0]) and re.search(r"[^\d\s]", lines[head][0]):
            cur_section = lines[head][0]
        end = headings[k + 1] if k + 1 < len(headings) else len(lines)
        # 三元组 (绝对行号, 行文本, 页码)：标题行插换行需要绝对行号对照 hset
        seg = [(i, l, p) for i, (l, p) in enumerate(lines[head:end], start=head)]

        # 段内按表格哨兵行切段：表格整表成块（不参与条款切分），文本段照旧处理
        units: list = []
        cur: list = []
        for tr in seg:
            if is_table_line(tr[1]):
                if cur:
                    units.append(cur)
                    cur = []
                units.append(tr)
            else:
                cur.append(tr)
        if cur:
            units.append(cur)

        for unit in units:
            if not isinstance(unit, list):
                # 哨兵行 → 整表成块，编号沿用解析层的表格序号（与文档顺序一致）
                tid = f"T{int(TABLE_LINE.match(unit[1]).group(1)):03d}"
                if tid in tables:
                    chunks.append(emit_table(tid, tables[tid], source, structure, chosen))
                continue
            # 纯标题单元：一级章（如 '5 显式标识' / '6 隐式标识'）后面直接跟下一个编号行，
            # 自身没有条款正文 —— 不单独成块，只作为 source_text 的分组边界。
            # 子条款（'5.1 xxx'）即便只有一行也保留，因为它的全部内容可能就在标题行上。
            if len(unit) == 1 and (CHAPTER.match(unit[0][1]) or TOP_DOT.match(unit[0][1])
                                   or APPENDIX_HEAD.match(unit[0][1])):
                continue
            n += 1
            content = join_unit(unit, hset)
            chunks.append(emit(f"R{n:03d}", content, group_text(head),
                               unit[0][2], source, structure, chosen,
                               section_path=cur_section))
    return chunks
