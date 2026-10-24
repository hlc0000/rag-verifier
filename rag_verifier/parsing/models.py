"""解析层统一数据模型：StructuredDocument 是切分层唯一面对的输入格式。

设计原则：解析层只做「视觉/格式 → 逻辑/结构」的忠实翻译，不做切分、不做验证。
Block.type 取值：heading / paragraph / table / list / image。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


def serialize_table(data: list[list] | None) -> str:
    """表格二维数组 → 行文本（"单元格 | 单元格"），空行剔除。

    空单元格一并剔除：它们是版式伪影（未填写的表格框线），不是文档内容。
    把 "1 |  |  |  |" 这样的行喂给语义层会毒化上下文——LLM 会把
    「文档里有很多空位」的印象泛化到别的条款上，造成判定翻转。
    完整网格仍在 Block.data 里，供下游结构化使用。
    """
    if not data:
        return ""
    rows = []
    for row in data:
        cells = [("" if c is None else str(c)).strip().replace("\n", " ") for c in row]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


@dataclass
class Block:
    type: str                  # heading | paragraph | table | list | image
    content: str = ""          # heading/paragraph 文本；table 为序列化文本；image 为空
    level: int | None = None   # heading 层级（1 最大）
    data: list | None = None   # table：二维数组
    items: list | None = None  # list：列表项文本
    page: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredDocument:
    source: str
    engine: str
    blocks: list[Block] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "engine": self.engine,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredDocument":
        blocks = [Block(**b) for b in d.get("blocks", [])]
        return cls(source=d.get("source", ""), engine=d.get("engine", ""),
                   blocks=blocks, metadata=d.get("metadata", {}))

    def full_text(self) -> str:
        """全文文本（含表格序列化），供验证层做整份文档上下文。"""
        parts = []
        for b in self.blocks:
            if b.type == "table":
                parts.append(b.content or serialize_table(b.data))
            elif b.type == "list" and b.items:
                parts.append("\n".join(str(i) for i in b.items))
            elif b.type in ("heading", "paragraph") and b.content:
                parts.append(b.content)
        return "\n".join(parts)
