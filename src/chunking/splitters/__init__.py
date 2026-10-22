"""各切分器共用的块构造与模板分类工具。

所有切分器接口一致：split(lines, source=..., structure=..., chosen=...) -> list[dict]
  lines     : [(行文本, 页码)]，已经过清洗（去页脚/页码、合并拆号）
  structure : detector.detect_structure 的结果
  chosen    : 本次使用的切分器名（写进 structure_detection 便于下游追溯）
"""
import re

# 模板空位信号（只标注、不剔除）：与文档类型无关，合同/标准都可能出现未填写空位。
#   blank_field        内容空位，如"保管物为：。"
#   placeholder_missing 数量/日期空位，如"每逾期___日""自变更之日起___日内"
BLANK_FIELD = re.compile(r"[：:]\s*[。；，]|为。")
PLACEHOLDER = re.compile(r"为限。|每逾期日|之日起日内|之日起日|标准支付|之日日")

# 表格哨兵行：解析层桥（document_to_lines）把表格插进行流的占位标记。
# 表格不参与段落切分——遇到哨兵行必须整表成块，而不是把序列化文本粘进某个条款。
TABLE_LINE = re.compile(r"^\x00T(\d+)\x00$")


def is_table_line(line: str) -> bool:
    return bool(TABLE_LINE.match(line))


def join_span(lines: list, start: int, end: int,
              heading_idx: set[int] | None = None) -> str:
    """把 lines[start:end] 拼成文本，标题行后插换行；表格哨兵行不进文本。

    标题与正文是两个语义单元，空串直连会把「安装docling」「rag-verifier面向」
    粘成怪词（readme 实测）。heading_idx 是标题行的绝对行号集合
    （各切分器用自己的锚点判定：AST 层级表 / 编号正则 / 「第X条」）。
    """
    hp = heading_idx or set()
    parts = []
    for i in range(start, min(end, len(lines))):
        l, _ = lines[i]
        if is_table_line(l):
            continue
        parts.append(l)
        if i in hp:
            parts.append("\n")
    return "".join(parts)


def join_unit(unit: list, heading_idx: set[int] | None = None) -> str:
    """拼一个切分单元（[(绝对行号, 行文本, 页码), ...] 三元组），标题行后插换行。

    与 join_span 是同一变换——单元拼出的块内容必须是组文本/整条文本的
    精确子串：可追溯性检查两边各拼一次，变换不一致会大面积冤判
    （RFC 272 块教训）。
    """
    hp = heading_idx or set()
    parts = []
    for abs_i, l, _ in unit:
        parts.append(l)
        if abs_i in hp:
            parts.append("\n")
    return "".join(parts)


def emit_preamble(content: str, page: int, source: str, structure: dict, chosen: str) -> dict:
    """首个编号锚点之前的内容（封面/前言/合同抬头）：不静默丢弃，显式标注成块。

    它不是证据块，不参与块级验证（验证层按 category 跳过）；保留原文是为了
    诚实与可审计——合同前置部分含甲乙方信息，是否启用交给下游决定。
    """
    return {
        "id": "P000",
        "card": {"content": content.strip(), "source": source, "locator": f"p{page}"},
        "source_text": content.strip(),
        "category": "preamble_dropped",
        "ground_truth": None,
        "failure_type": None,
        "structure_detection": {
            "mode": chosen,
            "active": structure.get("active", {}),
            "failed": chosen == "paragraph_fallback",
        },
    }


def collect_preamble(lines: list, first_anchor: int) -> tuple[str, int]:
    """取第一个锚点之前的非表格行，拼成前置内容文本；返回 (文本, 起始页码)。"""
    pre = [(l, p) for l, p in lines[:first_anchor] if not is_table_line(l)]
    if not pre:
        return "", 0
    return "".join(l for l, _ in pre).strip(), pre[0][1]


def emit_table(table_id: str, info: dict, source: str, structure: dict, chosen: str) -> dict:
    """表格整体成块：content = 序列化表格文本，source_text 取自身（自足、可追溯）。"""
    return {
        "id": table_id,
        "card": {"content": info["content"], "source": source, "locator": f"p{info['page']}"},
        "source_text": info["content"],
        "category": "table",
        "ground_truth": None,
        "failure_type": None,
        "structure_detection": {
            "mode": chosen,
            "active": structure.get("active", {}),
            "failed": chosen == "paragraph_fallback",
        },
    }


def classify_template(content: str) -> str:
    if BLANK_FIELD.search(content):
        return "blank_field"
    if PLACEHOLDER.search(content):
        return "placeholder_missing"
    return "normal"


def emit(chunk_id: str, content: str, source_text: str, page: int,
         source: str, structure: dict, chosen: str, section_path: str = "") -> dict:
    """构造统一块 schema（与既有语料一致，下游 prescreen/run_eval 无需改动）。

    section_path：块直属标题行文本（如 "5 显式标识" / "第七条 保密义务"），
    检索层 BM25 拼接用（关键词在邻块/章节标题的块靠它可检索）。
    可选参数默认空——不改变块 ID/content，verify 标签拼接与嵌入缓存均不受影响。
    """
    category = classify_template(content)
    entry = {
        "id": chunk_id,
        "card": {"content": content.strip(), "source": source, "locator": f"p{page}"},
        "source_text": source_text.strip(),
        "category": category,
        "ground_truth": None,
        "failure_type": None,
        "structure_detection": {
            "mode": chosen,
            "active": structure.get("active", {}),
            "failed": chosen == "paragraph_fallback",
        },
    }
    if section_path.strip():
        entry["section_path"] = section_path.strip()
    # 模板空位：规则预标注为 fail + template_blank（恢复策略与 missing_referent 不同，不可混）
    if category != "normal":
        entry["ground_truth"] = "fail"
        entry["failure_type"] = "template_blank"
    return entry
