"""结构探测：扫描全文，识别文档的编号体系，据此选择切分策略。

设计原则：切分器不预设文档类型。先统计「哪种编号模式在本文档中占主导」，
再据此选切分器；探测不到任何结构时诚实降级到段落切分，并标注探测失败。

注意：本模块的编号模式全部是行首锚定（配合 re.MULTILINE），
因此探测必须跑在「保留换行」的清洗文本上——若先把换行拼掉，
`^` 就只剩全文首行能匹配，所有计数都会失真。
"""
import re

# 编号模式表：name -> 行首锚定正则
PATTERNS = {
    "clause_cn":     r"^第[一二三四五六七八九十百零〇\d]+条",      # 合同：第九条
    "item_cn":       r"^（[一二三四五六七八九十]+）",              # 合同子条款：（一）
    "chapter_cn":    r"^第[一二三四五六七八九十]+章",              # 章节：第三章
    "clause_num":    r"^\d+(?:\.\d+)+\.?(\s|$)",                  # GB/T：5.1 / 5.1.1；RFC：5.6.（尾点）
    "chapter_num":   r"^\d{1,2}(\s+\S.*)?$",                      # GB/T 顶级章：1 范围 / 5 显式标识
    "appendix_num":  r"^[A-Z]\.\d+(?:\.\d+)*(\s|$)",             # 附录编号：A.1 / C.1.1 / D.2.1
    "appendix_head": r"^附\s*录\s*[A-Z]",                        # 附录标题：附 录 A
    "section_num":   r"^\d+\.\s",                                # 论文：1. 标题
    "section_alpha": r"^[a-z]\.\s",                              # 论文：a. 标题
    "article_en":    r"^Article\s+\d+",                          # 英文合同：Article 5
    "section_en":    r"^Section\s+\d+",                          # 英文文件：Section 5
}

MIN_HITS = 3  # 某模式命中 ≥3 次，才认为它是本文档的编号体系

# 切分器选择优先级（互斥判定，避免互相踩）：
#   1) 「数字点层级」或附录编号 → 层级数字编号文档（GB/T 这类）
#   2) 中文条款（第X条 / （一） / 第X章）→ 合同类
#   3) 英文条款（Article / Section）
#   4) 只有顶级章、没有子条款 → 仍是层级数字编号文档
#   5) 都没有 → 段落兜底
# clause_num 与 section_num 互斥：前者要求「数字点数字」(5.1)，后者要求「数字点 + 空格」(1. 标题)。
# chapter_num 单独不足以判定：页码、"ICS 35.030"、表格序号这类裸数字也会命中它，
# 所以它只在「已排除其他体系」之后作为兜底信号，不作为首选依据。
NUMERIC_STRONG = ("clause_num", "appendix_num", "appendix_head")
CHINESE_KEYS = ("clause_cn", "item_cn", "chapter_cn")
ENGLISH_KEYS = ("article_en", "section_en")


def is_active(active: dict, keys) -> bool:
    return any(k in active for k in keys)


def detect_structure(text: str) -> dict:
    """统计各编号模式的命中数，返回命中数以及「主导模式」（命中 ≥ MIN_HITS）。"""
    counts = {name: len(re.findall(pattern, text, re.MULTILINE)) for name, pattern in PATTERNS.items()}
    active = {name: n for name, n in counts.items() if n >= MIN_HITS}
    return {"counts": counts, "active": active}


def select_splitter(active: dict) -> str:
    """按优先级从「主导模式」里选切分器；一个都没有则段落兜底。

    选中 ≠ 可用：hierarchical_chinese 的切块锚点只有「第X条」——（X）只是条内
    二级切分、「第X章」不是它的锚点。主导信号若只有 item_cn/chapter_cn 而无
    clause_cn，选中后一个锚点都打不到 → 静默 0 块（盲测白皮书实测：「（一）」
    顿号体系文档 0 块）。锚点存在性校验：clause_cn 不在 active 就诚实降级。
    """
    if is_active(active, NUMERIC_STRONG):
        return "hierarchical_numeric"
    if is_active(active, CHINESE_KEYS):
        if "clause_cn" in active:
            return "hierarchical_chinese"
        return "paragraph_fallback"
    if is_active(active, ENGLISH_KEYS):
        return "hierarchical_english"
    if "chapter_num" in active:
        return "hierarchical_numeric"
    return "paragraph_fallback"
