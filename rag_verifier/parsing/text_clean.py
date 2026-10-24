"""PDF 抽取产物的通用清洗规则（去页脚/页码/目次页、合并拆号）。

这些脏数据是「PDF 文本抽取」这一步带来的（页脚重复出现、编号被换行拆开），
所以清洗规则放在解析层，而不是切分层——切分层不应该知道"页脚"这种视觉概念。
切分层旧入口 clean_pages 也从这里导入同一组正则，避免两处漂移。
"""
import re

DIGIT_LINE = re.compile(r"^\d{1,3}$")                  # 页码行
TOC_HEADING = re.compile(r"^目\s*次$")                 # 目次页标题（\s 含全角空格）
# 被换行拆开的编号片段："5." / "C." / "C.1." / 图题 "图A." —— 需与下一行合并回 "5.1" / "C.1.1" / "图A.1"
# 必须允许「多级片段」：'C.' 与 '1.' 合并成 'C.1.' 后，还要能继续与下一行的 '1 文字形式' 合并。
# 图题允许「图 A.」带空格：dict 模式按字距补空格后，"图A." 常变成 "图 A."。
HEADING_FRAGMENT = re.compile(r"^(?:[A-Z](?:\.\d+)*\.|\d+(?:\.\d+)*\.|图\s*[A-Z](?:\.\d+)*\.)$")


def merge_fragments(kept: list) -> list:
    """合并被换行拆开的编号："5." + "1 文本…" → "5.1 文本…"。

    kept 的元素需为 dict 且含 text 字段（可含 page/size/bold 等），
    合并时 text 拼接、size 取 max、bold 取或，其余字段沿用首行。
    允许跨页合并（与旧 clean_pages 行为一致：拆号恰在翻页处时同样要接回来）。
    """
    merged: list = []
    i = 0
    while i < len(kept):
        it = dict(kept[i])
        while (i + 1 < len(kept)
               and HEADING_FRAGMENT.match(it["text"]) and re.match(r"^\d", kept[i + 1]["text"])):
            nxt = kept[i + 1]
            it["text"] = it["text"] + nxt["text"]
            it["size"] = max(it.get("size", 0), nxt.get("size", 0))
            it["bold"] = bool(it.get("bold")) or bool(nxt.get("bold"))
            i += 1
        merged.append(it)
        i += 1
    return merged
