"""合同类文档切分：一级「第X条」，二级「（X）」。

逻辑原本写死在 scripts/split_contract.py，这里抽出为通用模块（接口与其他切分器一致）。
切分单元：
  - 有（X）子条款的条 → 每个（X）一个块，source_text 取整条（提供被省略的上下文）
  - 无子条款的条     → 整条一个块
"""
import re

from src.chunking.splitters import (
    TABLE_LINE, collect_preamble, emit, emit_preamble, emit_table, is_table_line,
    join_unit,
)

CLAUSE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")
ITEM = re.compile(r"^（[一二三四五六七八九十]+）")

CHOSEN = "hierarchical_chinese"


def split(lines: list, source: str = "document.pdf", structure: dict | None = None,
          chosen: str = CHOSEN, tables: dict | None = None) -> list[dict]:
    structure = structure or {"active": {}}
    tables = tables or {}
    starts = [i for i, (line, _) in enumerate(lines) if CLAUSE.match(line)]
    hset = set(starts)                           # 「第X条」行是标题，拼接时其后插换行

    chunks: list[dict] = []
    # 首个锚点之前的内容（合同抬头/甲乙方信息/鉴于条款）：不静默丢弃，显式标注
    if starts:
        pre_text, pre_page = collect_preamble(lines, starts[0])
        if pre_text:
            chunks.append(emit_preamble(pre_text, pre_page, source, structure, chosen))
    n = 0
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else len(lines)
        # 三元组 (绝对行号, 行文本, 页码)：标题行插换行需要绝对行号对照 hset
        seg = [(i, l, p) for i, (l, p) in enumerate(lines[start:end], start=start)]  # 整条

        # 条内按表格哨兵行切段：表格整表成块（不参与（X）子条款切分），文本段照旧
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

            # （X）子条款标记不是标题（合同正文流），不插换行——
            # 块内容与整条文本同走 join_unit、同一 hset，子串关系不破
            item_at = [j for j, (_, line, _) in enumerate(unit) if ITEM.match(line)]
            if not item_at:
                blocks = [unit]
            else:
                blocks = [
                    unit[j:(item_at[k + 1] if k + 1 < len(item_at) else len(unit))]
                    for k, j in enumerate(item_at)
                ]

            for block in blocks:
                n += 1
                content = join_unit(block, hset)
                chunks.append(emit(f"R{n:03d}", content, join_unit(unit, hset),
                                   block[0][2], source, structure, chosen,
                                   section_path=lines[start][0]))  # 直属「第X条」标题
    return chunks
