"""结构明写文档（Markdown / HTML / DOCX）切分：按解析层给出的标题层级切块。

与编号切分器的根本区别：锚点不靠行首正则去猜，而是解析层从格式标记里
直接读出来的（# 的数量、h1-h6、Word 内置标题样式）——结构是明写的，不需要猜。
这正是它的价值：网页里的 "[4]" 引用编号、年份、表格序号在正则眼里都像标题，
但在 AST 眼里它们根本不是 heading 块，不会进入 heading_levels。

切分单元 = 一个标题行 + 其后的内容行，直到下一个标题行（任何层级）。
纯标题单元（标题后紧跟下一个标题、自身无正文）不单独成块，
只作为 source_text 的分组边界——避免切出一堆只有标题的碎片。
"""
from rag_verifier.chunking.splitters import (
    TABLE_LINE, collect_preamble, emit, emit_preamble, emit_table, is_table_line,
    join_span, join_unit,
)

CHOSEN = "heading_ast"


def split(lines: list, source: str = "document.pdf", structure: dict | None = None,
          chosen: str = CHOSEN, tables: dict | None = None,
          heading_levels: dict[int, int] | None = None) -> list[dict]:
    structure = structure or {"active": {}}
    tables = tables or {}
    heading_levels = heading_levels or {}
    heads = sorted(heading_levels)
    hset = set(heads)

    def ancestor_text(idx: int) -> str:
        """idx 所属最近上级的整节文本，作为该块的参考上下文。表格哨兵行不进节文本。

        标题行后插换行，与块内容同一变换（join_span），保子串关系。
        """
        lv = heading_levels[idx]
        anc = next((h for h in reversed(heads) if h < idx and heading_levels[h] < lv), None)
        if anc is None:
            return ""
        alv = heading_levels[anc]
        end = next((h for h in heads if h > anc and heading_levels[h] <= alv), len(lines))
        return join_span(lines, anc, end, hset)

    chunks: list[dict] = []
    # 首个标题之前的内容（文档大标题/导言）：不静默丢弃，显式标注
    if heads:
        pre_text, pre_page = collect_preamble(lines, heads[0])
        if pre_text:
            chunks.append(emit_preamble(pre_text, pre_page, source, structure, chosen))
    n = 0
    for k, head in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        # 三元组 (绝对行号, 行文本, 页码)：标题行插换行需要绝对行号对照 hset
        seg = [(i, l, p) for i, (l, p) in enumerate(lines[head:end], start=head)]

        # 段内按表格哨兵行切段：表格整表成块，文本段照旧处理
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
                tid = f"T{int(TABLE_LINE.match(unit[1]).group(1)):03d}"
                if tid in tables:
                    chunks.append(emit_table(tid, tables[tid], source, structure, chosen))
                continue
            # 纯标题单元：标题后紧跟下一个标题、自身无正文 —— 不单独成块，
            # 只作为 source_text 的分组边界（标题文字仍在下级块的 source_text 里）。
            # 注意只豁免段首标题本身；表格后的单行正文仍保留。
            if len(unit) == 1 and unit[0][0] == head:
                continue
            n += 1
            content = join_unit(unit, hset)
            chunks.append(emit(f"R{n:03d}", content, ancestor_text(head) or content,
                               unit[0][2], source, structure, chosen,
                               section_path=lines[head][0]))  # 直属标题行文本
    return chunks
