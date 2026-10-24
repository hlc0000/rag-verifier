"""fail 标签消费 + 上下文拼接：检索层的核心差异化。

别人的 RAG「切完就信」，我们「切完先验，坏块带标签，检索按标签用药」：
- 碎片 fail 块：入库时已剔除（corpus.py），检索永远碰不到它；
- cross_block_ref fail 块：命中时按其引用锚点（表6/第8条…）在本文档内
  确定性取出定义块补全上下文——把「引用块→定义块」从概率搜索变成查字典；
- 其余 fail 块：仍可被检出，但在上下文中带「不完整证据」标注，交给生成端知情。
"""
import re

from rag_verifier.retrieval.bm25 import tokenize
from rag_verifier.retrieval.corpus import ChunkRecord, TABLE_ANCHOR_RE


def _norm(s: str) -> str:
    """锚点匹配归一化：去空白、英文小写（"第 8 条"↔"第8条"、"Table 6"↔"table6"）。"""
    return re.sub(r'\s+', '', s).lower()


def _find_bounded(body: str, a: str) -> int:
    """归一化文本中找锚点位置。数字结尾的锚点要求后随字符非数字：
    "表1" 不得匹配 "表10"/"表11"（实测子串误配把补全引到了错块）。"""
    start = 0
    while True:
        i = body.find(a, start)
        if i == -1:
            return -1
        nxt = body[i + len(a):i + len(a) + 1]
        if not (a[-1].isdigit() and nxt.isdigit()):
            return i
        start = i + 1


def _nearest_table(doc_records: list[ChunkRecord], src_idx: int,
                   src_content: str) -> ChunkRecord | None:
    """在源块邻近窗口内找其引用的表数据块。切分层把表序列化成无 caption 的
    独立 table 块，字符串匹配够不到；而抽取顺序随文档排版漂移（中文论文表题
    在表上、BERT 论文表体在 caption 前），方向规则不可能通用。

    排序主键是距离（正确表块恒为 caption 的紧邻表块，与排版方向无关）；
    词重叠只在同距离时当裁判（BERT 实测：远距离表靠 caption 引用句里的
    "leaderboard/2018" 撞词能赢过紧邻的正确表，重叠当主排序会被骗）。
    窗口内按 距离→重叠度→后向 依次裁决（全确定，无随机）。
    """
    src_tokens = set(tokenize(src_content))
    cands = []
    for i, r in enumerate(doc_records):
        if r.category != "table" or i == src_idx:
            continue
        d = i - src_idx
        if abs(d) > 8:  # 表与其 caption 在文档顺序中必然相邻，8 块窗口足够
            continue
        overlap = len(src_tokens & set(tokenize(r.content)))
        cands.append((abs(d), -overlap, 0 if d > 0 else 1, r))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    return cands[0][3]


def resolve_anchor(anchor: str, source_rec: ChunkRecord,
                   doc_records: list[ChunkRecord]) -> ChunkRecord | None:
    """在本文档内找锚点的定义块。排除引用块自身（引用点≠定义点）。

    规则1：源块自身以锚点开头（它是"表N./第N条"的标题块）→ 定义=紧邻 table 块。
    规则2：其余按内容匹配（带数字边界），以锚点开头者优先，再按出现位置取先。
    """
    a = _norm(anchor)
    src_idx = next((i for i, r in enumerate(doc_records)
                    if r.chunk_id == source_rec.chunk_id), -1)
    if _norm(source_rec.content).startswith(a) and src_idx != -1:
        return _nearest_table(doc_records, src_idx, source_rec.content)

    cands = []
    for r in doc_records:
        if r.chunk_id == source_rec.chunk_id:
            continue
        body = _norm(r.content)
        pos = _find_bounded(body, a)
        if pos == -1:
            continue
        tier = 0 if body.startswith(a) else 1
        cands.append((tier, pos, r))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[0][2]


def consume_labels(hits: list[dict], all_records: list[ChunkRecord]) -> tuple[list[dict], list[dict]]:
    """对命中块消费 fail 标签，返回（上下文块序列, 补全记录）。

    上下文块序列每项：{"rec": ChunkRecord, "role": "hit" / "completion", "for": 源块ID}
    """
    by_doc: dict[str, list[ChunkRecord]] = {}
    for r in all_records:
        by_doc.setdefault(r.doc, []).append(r)

    sequence: list[dict] = []
    completions: list[dict] = []
    seen_ids = {h["record"].chunk_id for h in hits}

    for h in hits:
        rec = h["record"]
        sequence.append({"rec": rec, "role": "hit", "for": ""})
        if rec.fail_kind != "cross_block_ref":
            continue
        for anchor in rec.anchors:
            target = resolve_anchor(anchor, rec, by_doc.get(rec.doc, []))
            if target is None:
                completions.append({"anchor": anchor, "for": rec.chunk_id, "found": False})
                continue
            completions.append({"anchor": anchor, "for": rec.chunk_id,
                                "found": True, "target": target.chunk_id})
            if target.chunk_id in seen_ids:
                continue  # 定义块已在命中列表里，不重复进上下文
            seen_ids.add(target.chunk_id)
            sequence.append({"rec": target, "role": "completion",
                             "for": rec.chunk_id, "anchor": anchor})
    return sequence, completions


def format_context(sequence: list[dict]) -> str:
    """拼接喂给 LLM 的上下文：标注块角色与验证标签，让生成端知情。"""
    blocks = []
    for item in sequence:
        rec = item["rec"]
        tag = ""
        if item["role"] == "completion":
            # 显式声明补全目标锚点：表数据块序列化后无 caption（"SWAG"等关键词
            # 只活在 caption 里），不说破身份 LLM 无法把数据表对上号（BERT 实测翻案失败）
            tag = f"（引用补全：本块是 {item['for']} 引用的「{item.get('anchor', '')}」的定义内容）"
        elif rec.overall == "fail":
            tag = "（不完整证据：验证未通过，内容可能缺上下文）"
        blocks.append(f"【{rec.doc} / {rec.chunk_id}】{tag}\n{rec.content}")
    return "\n\n".join(blocks)
