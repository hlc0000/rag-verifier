"""英文编号文档切分（Article X / Section X）。

编号层级：
  顶层  'Article 5' / 'ARTICLE III'
  子层  'Section 5.1' / 'Section 5(a)'（Section 行自身也是块锚点）

切分单元 = 一个锚点行 + 其后的内容行，直到下一个锚点行。
锚点识别以 detector 的 article_en / section_en 为基准，并放宽到合同惯例的全大写与
罗马数字写法（'ARTICLE III'）——路由仍由 detector 决定，这里只让「已被路由到本
切分器的文档」切得更准；放宽部分不参与探测，因此不影响任何既有文档的路由。
编号后的 '.1'、'(a)' 后缀属于标题文本的一部分，不再单独切一层——英文合同里
'Section 5.1' 是 'Section 5' 的下一层，但两者的切分粒度相同（都是「一个标题 + 其正文」）。

section_path 取块直属的锚点行文本（最近一个 Article 或 Section 行）；文档同时存在
Article 与 Section 时，Article 行另作分组边界，为块提供章节级参考上下文。

纯标题单元（锚点行后直接跟另一个锚点行、自身无正文）不单独成块，只作为 source_text
的分组边界——避免切出一堆只有标题的碎片（与 hierarchical_numeric 同策略）。
"""
import re

from src.chunking.splitters import (
    TABLE_LINE, collect_preamble, emit, emit_preamble, emit_table, is_table_line,
    join_span, join_unit,
)

# 锚点词只认「标题式」与「全大写」两种写法：'Article 5' / 'ARTICLE III' 是合同惯例，
# 不会误伤正文里行首的小写 'section 3 of the Act applies.'（宁可漏，不可误伤）。
# Article 支持罗马数字（英文合同极常见），Section 只认阿拉伯数字（与 detector 同口径）。
ARTICLE = re.compile(r"^(?:Article|ARTICLE)\s+(?:\d+|[IVXLC]+)\b")
SECTION = re.compile(r"^(?:Section|SECTION)\s+\d+\b")

CHOSEN = "hierarchical_english"


def _is_heading(line: str) -> bool:
    return bool(ARTICLE.match(line) or SECTION.match(line))


def split(lines: list, source: str = "document.pdf", structure: dict | None = None,
          chosen: str = CHOSEN, tables: dict | None = None) -> list[dict]:
    structure = structure or {"active": {}}
    tables = tables or {}
    headings = [i for i, (line, _) in enumerate(lines) if _is_heading(line)]
    # 分组边界优先用顶层 Article；文档只有 Section（无 Article）时，Section 即顶层
    group_re = ARTICLE if any(ARTICLE.match(l) for l, _ in lines) else SECTION
    groups = [i for i, (line, _) in enumerate(lines) if group_re.match(line)]
    hset = set(headings)

    # 整份文档全是「裸标题行」（如一张 Article 目录）时不做纯标题剔除，
    # 否则一个块都产不出来——宁可产出标题块，也不静默 0 块。
    has_body = any(
        (headings[k + 1] if k + 1 < len(headings) else len(lines)) - head > 1
        for k, head in enumerate(headings)
    )

    def group_text(idx: int) -> str:
        """取 idx 所属整组的文本（顶层 Article 分组），作为该块的参考上下文。

        锚点之前没有任何分组头时必须用全文兜底——不能取 groups[0]（那是锚点之后的
        分组，窗口与块内容完全无关，可追溯性检查必然冤判）。标题行后插换行，与块内容
        同一变换（join_span/join_unit），保子串关系。
        """
        whole = join_span(lines, 0, len(lines), hset)
        before = [x for x in groups if x <= idx]
        if not before:
            return whole
        g = before[-1]
        ge = next((x for x in groups if x > g), len(lines))
        return join_span(lines, g, ge, hset)

    chunks: list[dict] = []
    # 首个锚点之前的内容（封面/合同抬头/当事方信息）：不静默丢弃，显式标注
    if headings:
        pre_text, pre_page = collect_preamble(lines, headings[0])
        if pre_text:
            chunks.append(emit_preamble(pre_text, pre_page, source, structure, chosen))
    n = 0
    cur_section = ""          # 最近的锚点行文本（块的检索用章节路径）
    for k, head in enumerate(headings):
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
            # 纯标题单元：顶层锚点行（如 'ARTICLE II - PAYMENT'）后面直接跟下一个锚点行，
            # 自身没有正文 —— 不单独成块，只作为 source_text 的分组边界。
            # 子层 Section 行即便只有一行也保留，因为它的内容可能就在标题行上。
            if len(unit) == 1 and has_body and group_re.match(unit[0][1]):
                continue
            n += 1
            content = join_unit(unit, hset)
            chunks.append(emit(f"R{n:03d}", content, group_text(head),
                               unit[0][2], source, structure, chosen,
                               section_path=cur_section))
    return chunks
