"""段落兜底切分：文档里探测不到任何编号体系时使用。

降级不是失败，是诚实——没有编号结构就没有结构可依赖，只能按物理边界（段落/空行）切。
输出块的 structure_detection.failed = True，让下游知道「这份文档没有识别到编号结构」。

注意：清洗阶段已经去掉了空行，因此这里按「行」这一物理边界切；
连续不含编号的行各自成块（行即原文的排版段落）。

例外：纯文本（TXT）的硬折行是排版产物，不是语义段落。若不还原，一段话会被拆成
十几个"半个句子"的块（Apache 2.0 许可证实测：33 段 → 30+ 碎片块，全是中间截断）。
这里只做「英文硬折行还原」：上一行没有句末标点、且以常见功能词收尾，下一行以小写开头，
就把两行拼成一行。中文行一般以句号收尾，不会被合并；含 CJK 的行完全不参与合并。
"""
import re

from rag_verifier.chunking.splitters import TABLE_LINE, emit, emit_table, is_table_line

CHOSEN = "paragraph_fallback"

# 句末标点：出现即认为是完整句子的结尾，不与下一行合并
_TERMINAL = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":", '"', "”", ")", "）")
# 以这些词收尾的英文行，几乎必然是硬折行的上半句
_JOIN_TAIL = re.compile(
    r'\b(?:of|the|a|an|and|or|to|in|for|with|by|as|that|which|shall|will|be|is|are|'
    r'not|nor|but|from|on|at|under|this|these|those|such|any|all|each|its|their)\s*$',
    re.IGNORECASE,
)
_CJK = re.compile(r'[\u4e00-\u9fff]')
_LETTER = re.compile(r'[A-Za-z]')
_TRAILING_CLOSERS = "\"'”’)]}»"


def _is_all_caps(text: str) -> bool:
    """全大写段落（法律免责声明常见）：字母数 ≥20 且大写占比 ≥80%。"""
    letters = _LETTER.findall(text)
    if len(letters) < 20:
        return False
    return sum(ch.isupper() for ch in letters) / len(letters) >= 0.8


def _ends_sentence(text: str) -> bool:
    """剥掉结尾引号/右括号后再看句末标点——"),"(续行) 不是句子结束，"." 才是。"""
    t = text.rstrip().rstrip(_TRAILING_CLOSERS).rstrip()
    return t.endswith((".", "!", "?", ";", ":"))


def _is_continuation(prev: str, cur: str) -> bool:
    """prev 是否是硬折行的上半句、cur 是它的下半句。"""
    if not prev or not cur:
        return False
    if _CJK.search(prev) or _CJK.search(cur):
        return False
    # 全大写段落没有大小写信号：GPL-3.0 免责声明整段大写，原先被拆成 5 个半句块。
    # 只要不是句末标点收尾，就是折行的上半句。
    if _is_all_caps(prev):
        return not _ends_sentence(prev)
    if prev.rstrip().endswith(_TERMINAL):
        return False
    head = cur.lstrip()
    return head[:1].islower() or _JOIN_TAIL.search(prev.rstrip()) is not None


def _merge_wrapped(lines: list) -> list:
    """把连续的硬折行拼回一行；返回 [(line, page)]，页号取上半句所在页。"""
    merged: list = []
    for line, page in lines:
        if merged and _is_continuation(merged[-1][0], line):
            prev_line, prev_page = merged[-1]
            merged[-1] = (prev_line.rstrip() + " " + line.lstrip(), prev_page)
        else:
            merged.append((line, page))
    return merged


def split(lines: list, source: str = "document.pdf", structure: dict | None = None,
          chosen: str = CHOSEN, tables: dict | None = None) -> list[dict]:
    structure = structure or {"active": {}}
    tables = tables or {}
    chunks: list[dict] = []
    n = 0
    for line, page in _merge_wrapped(lines):
        if not line.strip():
            continue
        if is_table_line(line):
            tid = f"T{int(TABLE_LINE.match(line).group(1)):03d}"
            if tid in tables:
                chunks.append(emit_table(tid, tables[tid], source, structure, chosen))
            continue
        n += 1
        chunks.append(emit(f"R{n:03d}", line, line, page, source, structure, chosen))
    return chunks
