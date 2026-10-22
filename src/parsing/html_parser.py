"""HTML 解析器：用 BeautifulSoup 直读 DOM，不走视觉模型。

和 markdown 同理：HTML 的结构写在标签里——h1~h6 就是层级、<table> 就是表格、
<ul>/<ol> 就是列表、<p> 就是段落。docling 把 HTML 渲染后过布局管线，大页面
（如 #16 那份 2.2MB Wikipedia）直接超时，且丢层级。bs4 解析 DOM，秒出且层级
天然正确（h1→L1 … h6→L6）。

只提取正文语义块（heading/paragraph/list/table），跳过导航/脚本/样式等噪音。
"""
from __future__ import annotations

import re
from pathlib import Path

from src.parsing.base import Parser, register
from src.parsing.models import Block, StructuredDocument, serialize_table

# 这些标签及其子树不提取正文
_SKIP = {"script", "style", "noscript", "template", "head", "nav",
         "header", "footer", "form", "button", "select", "iframe"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# 块级元素清单：版式表判定与 div/font 段落提取共用
_BLOCKY = list(_HEADINGS) + ["p", "div", "ul", "ol"]


def _text(el) -> str:
    return " ".join(el.get_text(" ", strip=True).split())


def _table_grid(table) -> list[list[str]]:
    grid: list[list[str]] = []
    for tr in table.find_all("tr"):
        row = [_text(c) for c in tr.find_all(["th", "td"])]
        if row:
            grid.append(row)
    return grid


def _is_layout_table(table) -> bool:
    """单个单元格装 ≥3 个块级元素的表是版式容器（如 gov.cn 把整篇文章塞进
    一个单元格），不是数据表。数据表单元格最多包一两个短段落包装。"""
    for cell in table.find_all(["td", "th"]):
        if len(cell.find_all(_BLOCKY)) >= 3:
            return True
    return False


# 目录条目锚点：Section/Article + 数字、第 X 条/章、N.N 点号编号
_TOC_ANCHOR = re.compile(
    r"^(?:SECTION|Section|ARTICLE|Article)\s+\d+"
    r"|^第[一二三四五六七八九十百零〇\d]+[条章节]"
    r"|^\d+(?:\.\d+)+"
)


def _is_toc_table(table) -> bool:
    """目录（TOC）型表格判据：单元格以短文本为主 + 命中多个编号锚点。

    目录表被序列化成「表格块」的同时，其单元格还会被额外抽成一批「标题-only
    段落块」，后者把行首编号锚点带偏（英文切分器会把每条目录当成 Section 起块，
    真实合同实测产出上百个碎片块）。只对这类目录表去重其单元格段落——贷款方
    清单/附表/通知地址等真实表内容不满足判据，不受影响，避免误伤验证覆盖。
    """
    cells = [c for c in (_text(c) for c in table.find_all(["td", "th"])) if c]
    if len(cells) < 3:
        return False
    short = sum(1 for c in cells if len(c) < 50)
    anchors = sum(1 for c in cells if _TOC_ANCHOR.match(c))
    return short > 0.6 * len(cells) and anchors >= 3


def convert_html(path: str) -> StructuredDocument:
    from bs4 import BeautifulSoup

    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    # 解析后端优先 lxml：html.parser 不会自动闭合未结束的 <p>，SEC 等老旧 HTML 大量
    # 未关 <p> 会层层嵌套，get_text 把整篇正文吐成一条巨行（实测真实合并协议单块达
    # 1.7MB）。lxml 按 HTML 规范自动闭合 <p>，消除该伪影；缺 lxml 时回退 html.parser
    # （行为与旧版一致）。实测换 lxml 对现有 5 份 HTML 基线解析结果字节级零漂移。
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # 去噪：删掉脚本/样式/导航等（head 也在其中，故根节点直接用整个文档——
    # 部分站点（如 gov.cn）正文写在 </body> 之外，取 soup.body 会漏掉正文）
    for tag in soup.find_all(_SKIP):
        tag.decompose()

    root = soup
    blocks: list[Block] = []
    headings = tables = lists = 0

    # 目录型表格的单元格 <p>/<div> 不再单独抽出——否则同一份目录被表达两次：
    # 一次是序列化表块，一次是逐段「标题-only」小块，后者把行首编号锚点带偏（英文
    # 切分器会把每条目录当成 Section 起块）。仅限目录表：贷款方清单/附表/通知地址等
    # 真实表内容照常逐段抽出，不丢验证覆盖。版式表本就不序列化，也不在此列。
    toc_tables = {id(t) for t in soup.find_all("table")
                  if not _is_layout_table(t) and _is_toc_table(t)}

    # 按文档顺序找语义块
    for el in root.find_all(list(_HEADINGS) + ["p", "ul", "ol", "table", "div", "font"]):
        name = el.name
        if name in _HEADINGS:
            t = _text(el)
            if t:
                headings += 1
                blocks.append(Block(type="heading", content=t, level=_HEADINGS[name]))
        elif name == "p":
            # 跳过嵌在 li 里的 p（已在列表块提取）；版式表内的 p 是正文，必须提取；
            # 目录表内的 p 已由序列化表块覆盖，不重复出块
            if el.find_parent("li") is not None:
                continue
            anc = el.find_parent("table")
            if anc is not None and id(anc) in toc_tables:
                continue
            t = _text(el)
            if t:
                blocks.append(Block(type="paragraph", content=t))
        elif name in ("ul", "ol"):
            if el.find_parent("li") is not None:
                continue  # 嵌套列表不重复
            items = [_text(li) for li in el.find_all("li", recursive=False)]
            items = [i for i in items if i]
            if items:
                lists += 1
                blocks.append(Block(type="list", items=items))
        elif name == "table":
            if _is_layout_table(el):
                continue  # 版式表不当数据表序列化，内容交给块级子元素正常提取
            grid = _table_grid(el)
            if grid:
                tables += 1
                blocks.append(Block(type="table", content=serialize_table(grid), data=grid))
        else:
            # div/font：无块级子元素且有文本的按段落提取（SEC 老式 HTML 正文全在
            # div>font 里）。父链防重：p/li 内的行内片段已由父块覆盖；font 在
            # div 或单元格内时由外层容器路径覆盖。
            if el.find(_BLOCKY):
                continue
            if el.find_parent(["p", "li"]) is not None:
                continue
            if name == "font" and el.find_parent(["div", "td", "th"]) is not None:
                continue
            anc = el.find_parent("table")
            if anc is not None and id(anc) in toc_tables:
                continue
            t = _text(el)
            if t:
                blocks.append(Block(type="paragraph", content=t))

    metadata = {"headings_detected": headings, "tables": tables,
                "lists": lists, "parser": "bs4"}
    return StructuredDocument(source=Path(path).name, engine="html",
                              blocks=blocks, metadata=metadata)


@register
class HTMLParser(Parser):
    name = "html"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import bs4  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, path: str) -> StructuredDocument:
        return convert_html(path)
