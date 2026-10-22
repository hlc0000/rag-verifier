"""语料加载：重跑确定性切分复现块全文，按 ID 拼接 verify 报告的五关标签。

verify_*.json 每行只存 status+reason+前50字预览（fail details 未持久化）。
但切分是纯正则确定性的——重跑 split_document 可原样复现每块全文与 ID，
与验证报告按 ID 连接即得「全文 + 标签」，无需重跑 LLM。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking.splitter import split_document
from src.parsing.models import StructuredDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# preamble_dropped（封面/前言等首个编号锚点前的内容）是噪声，不入库。
# table 块按设计不参与块级验证（无标签），但它们是跨块引用的定义目标，必须入库。
SKIP_INDEX_CATEGORIES = ("preamble_dropped",)


@dataclass
class ChunkRecord:
    doc: str                       # 文档名（与 verify 文件同名 stem）
    chunk_id: str                  # 块 ID（R001...，切分顺序号，跨次运行稳定）
    category: str                  # normal / table / blank_field / placeholder_missing
    content: str                   # 块全文
    locator: str
    overall: str = "unlabeled"     # pass / fail / unlabeled（table 块无验证标签）
    fail_kind: str = ""            # fragment / cross_block_ref / other（仅 fail 时有值）
    anchors: list = field(default_factory=list)   # 跨块引用锚点（如 "表6"、"第8条"）
    fail_reason: str = ""          # 首个 fail 关的 reason（展示用）
    indexable: bool = True         # 碎片 fail 不进索引
    section_path: str = ""         # 直属标题行文本（切分层 emit 提供；BM25 拼接用）


# 跨块引用锚点：编号类引用（中英）。只匹配带编号的硬引用，
# 不匹配"该费用""前述款项"这类无锚点软引用（那些本来就没法确定性定位）。
ANCHOR_RES = [
    re.compile(r'第\s*\d+\s*条'), re.compile(r'第[一二三四五六七八九十百]+\s*条'),
    re.compile(r'表\s*\d+'), re.compile(r'图\s*\d+'),
    re.compile(r'附[录件]\s*[A-Za-z\d]+'),
    re.compile(r'\bTable\s+\d+', re.I), re.compile(r'\bFig(?:ure)?\.?\s*\d+', re.I),
    re.compile(r'\bSection\s+\d+(?:\.\d+)*', re.I),
    re.compile(r'\bClause\s+\d+(?:\.\d+)*', re.I),
    re.compile(r'\bAppendix\s+[A-Z]\b', re.I),
]

TABLE_ANCHOR_RE = re.compile(r'^(表\s*\d+|Table\s+\d+|Fig(?:ure)?\.?\s*\d+)$', re.I)


def extract_anchors(text: str) -> list[str]:
    """从块内容中提取编号类引用锚点，去重保序。"""
    seen, out = set(), []
    for rx in ANCHOR_RES:
        for m in rx.finditer(text):
            a = re.sub(r'\s+', '', m.group()) if not m.group().isascii() else m.group()
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def _classify_fail(row: dict, content: str) -> tuple[str, list[str]]:
    """从持久化的 checks 反推失败类型（details 未持久化，按 reason 文本 + 内容锚点分类）。

    fragment：规则层拦的碎片块（"碎片"字样），入库时剔除。
    cross_block_ref：规则层"跨块表格引用"，或语义层 fail 且内容含编号锚点——
                     恢复策略=按锚点确定性补全定义块。
    other：其余 fail（自身残缺、缺主语等），仍入库但带标签。
    """
    reasons = [v["reason"] for v in row["checks"].values() if v["status"] == "fail"]
    joined = " ".join(reasons)
    if "碎片" in joined:
        return "fragment", []
    anchors = extract_anchors(content)
    # 中英文跨块引用规则的 reason 文本都要认（规则层可插拔，reason 语种随规则走）
    if "跨块表格引用" in joined or "Cross-chunk reference" in joined:
        return "cross_block_ref", anchors
    if row["checks"].get("semantic_independence", {}).get("status") == "fail" and anchors:
        return "cross_block_ref", anchors
    return "other", []


def find_pairs(data_dir: Path = DATA_DIR) -> list[tuple[Path, Path, str]]:
    """发现 parsed_<stem>.json ↔ verify_<stem>.json 配对。"""
    pairs = []
    for vf in sorted(data_dir.glob("verify_*.json")):
        stem = vf.stem[len("verify_"):]
        pf = data_dir / f"parsed_{stem}.json"
        if pf.exists():
            pairs.append((pf, vf, stem))
    return pairs


def load_corpus(data_dir: Path = DATA_DIR) -> list[ChunkRecord]:
    """加载全部配对文档，返回带标签的块记录（按文档序、块序）。"""
    records: list[ChunkRecord] = []
    for pf, vf, stem in find_pairs(data_dir):
        doc = StructuredDocument.from_dict(json.loads(pf.read_text(encoding="utf-8")))
        chunks = split_document(doc=doc, source=doc.source)
        labels = {r["id"]: r for r in json.loads(vf.read_text(encoding="utf-8"))["rows"]}
        for c in chunks:
            if c["category"] in SKIP_INDEX_CATEGORIES:
                continue
            rec = ChunkRecord(
                doc=stem,
                chunk_id=c["id"],
                category=c["category"],
                content=c["card"]["content"],
                locator=c["card"]["locator"],
                section_path=c.get("section_path", ""),
            )
            row = labels.get(rec.chunk_id)
            if row is not None:
                rec.overall = row["overall"]
                if rec.overall == "fail":
                    rec.fail_kind, rec.anchors = _classify_fail(row, rec.content)
                    rec.fail_reason = next(
                        (v["reason"] for v in row["checks"].values() if v["status"] == "fail"), "")
                    if rec.fail_kind == "fragment":
                        rec.indexable = False
            records.append(rec)
    return records
