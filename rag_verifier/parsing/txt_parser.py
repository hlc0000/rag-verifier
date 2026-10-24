"""TXT 解析器：纯文本直读，不做任何视觉/启发式加工。

TXT 没有结构标记（标题、表格、列表都靠编号惯例），能做的就是无损读出段落流：
按空行分段、段内保留硬换行。实测教训：TXT 交给 docling 视觉管线会被当版面
猜——"1.  Introduction" 的 "1." 被当列表项标记剥掉，编号锚点全失配，结构探测
整条报废（RFC 9110 实测）。纯文本的规则是：不编造、不丢弃，逐字保真。
"""
from __future__ import annotations

from pathlib import Path

from rag_verifier.parsing.base import Parser, register
from rag_verifier.parsing.models import Block, StructuredDocument


def convert_txt(path: str) -> StructuredDocument:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")

    blocks: list[Block] = []
    cur: list[str] = []
    for line in raw.split("\n"):
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            blocks.append(Block(type="paragraph", content="\n".join(cur)))
            cur = []
    if cur:  # 文件以段落结尾：收尾 flush（列表累积器同款教训）
        blocks.append(Block(type="paragraph", content="\n".join(cur)))

    metadata = {"paragraphs": len(blocks), "parser": "plain-text"}
    return StructuredDocument(source=Path(path).name, engine="txt",
                              blocks=blocks, metadata=metadata)


@register
class TXTParser(Parser):
    name = "txt"

    @classmethod
    def is_available(cls) -> bool:
        return True  # 纯标准库，永远可用

    def parse(self, path: str) -> StructuredDocument:
        return convert_txt(path)
