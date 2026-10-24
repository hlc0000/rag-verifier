"""Markdown 解析器：用 marko 按 CommonMark 规范直读，不走视觉模型。

为什么不用 docling：markdown 的结构是「写在字符里」的——
`#` 的个数就是层级、反引号就是行内代码、`|` 就是表格、`-`/`1.` 就是列表。
docling 把 md 当"文档"过布局管线，反而把明写结构弄丢：
三级标题丢文本、层级全压平、行内代码被拆成独立块（见 parse_test_report #8）。
marko 按规范还原语法树，与作者无关：按语法写的，一律读对。

映射：
  Heading(level=#个数) → heading；Paragraph → paragraph（行内代码保留不拆块）；
  List → list；GFM Table → table；FencedCode/HTMLBlock → paragraph（保留原文）。
"""
from __future__ import annotations

from pathlib import Path

from rag_verifier.parsing.base import Parser, register
from rag_verifier.parsing.models import Block, StructuredDocument, serialize_table


def _inline_text(node) -> str:
    """把段落的 inline 子节点拼成纯文本；行内代码/链接/强调保留其文字，不拆块。

    marko 的 children 有三种形态：str（单行情形）、list[节点]、tuple[str, ...]
    （多行段落/代码块按行拆分）。漏掉任何一种都会把节点 repr 漏进文本。
    """
    parts: list[str] = []

    def walk(n):
        if isinstance(n, str):          # tuple[str, ...] 里的纯文本行
            parts.append(n)
            return
        children = getattr(n, "children", None)
        if isinstance(children, str):   # 叶子节点：children 就是文本
            parts.append(children)
            return
        if children is None:
            return
        for c in children:
            walk(c)

    walk(node)
    return "".join(parts)


def _table_to_grid(table) -> list[list[str]]:
    """GFM Table → 二维数组。表头一行 + 各数据行。"""
    grid: list[list[str]] = []
    head = getattr(table, "head", None)
    if head is not None:
        grid.append([_inline_text(cell).strip() for cell in getattr(head, "children", [])])
    for row in getattr(table, "children", []) or []:
        grid.append([_inline_text(cell).strip() for cell in getattr(row, "children", [])])
    return grid


def convert_markdown(path: str) -> StructuredDocument:
    from marko.ext.gfm import gfm  # gfm 是预配置好的 Markdown 实例（含表格等扩展）

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    doc = gfm.parse(text)

    blocks: list[Block] = []
    tables = 0
    lists = 0
    headings = 0

    def emit(node):
        nonlocal tables, lists, headings
        name = type(node).__name__

        if name == "Heading":
            headings += 1
            blocks.append(Block(
                type="heading",
                content=_inline_text(node).strip(),
                level=getattr(node, "level", None),
            ))
        elif name == "Paragraph":
            t = _inline_text(node).strip()
            if t:
                blocks.append(Block(type="paragraph", content=t))
        elif name == "List":
            lists += 1
            items = []
            for li in getattr(node, "children", []) or []:
                txt = _inline_text(li).strip()
                if txt:
                    items.append(txt)
            if items:
                blocks.append(Block(type="list", items=items))
        elif name == "Table":  # GFM
            tables += 1
            grid = _table_to_grid(node)
            blocks.append(Block(type="table", content=serialize_table(grid), data=grid))
        elif name in ("FencedCode", "CodeBlock"):
            # 代码块同样走 _inline_text：其 children 是按行拆的 tuple[str, ...]，
            # 手动 str() 会把节点 repr 漏进文本
            body = _inline_text(node).rstrip("\n")
            if body.strip():
                blocks.append(Block(type="paragraph", content=body, meta={"kind": "code"}))
        elif name == "HTMLBlock":
            body = str(getattr(node, "children", "") or "").strip()
            if body:
                blocks.append(Block(type="paragraph", content=body, meta={"kind": "html"}))
        elif name in ("Quote", "BlankLine", "ThematicBreak"):
            return  # 引用块内容会在子节点；空行/分隔线无内容
        else:
            # 容器类（Document/Quote 等）或未知块级：递归子节点
            for c in getattr(node, "children", []) or []:
                if not isinstance(c, str):
                    emit(c)

    for child in getattr(doc, "children", []) or []:
        emit(child)

    metadata = {
        "headings_detected": headings,
        "tables": tables,
        "lists": lists,
        "parser": "marko+gfm",
    }
    return StructuredDocument(source=Path(path).name, engine="markdown",
                              blocks=blocks, metadata=metadata)


@register
class MarkdownParser(Parser):
    name = "markdown"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import marko  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: str) -> StructuredDocument:
        return convert_markdown(path)
