"""文档入库（ingest）引擎：幂等入库 + 版本替换。

解决的问题：
  1) 同一份文档不该每次重复上传后重复解析/切分/验证/嵌入——内容 sha1 未变即整步跳过；
  2) 文档更新后要能替换旧版，且替换只波及这一份，不触发全库重建。

身份认定：文件名 stem（如 "en-readme-requests"）。同一 stem 的新文件视为「同一文档的
新版本」，走 replaced；首见为 added；内容 sha1 与注册表一致且产物齐备为 unchanged。

产物落盘约定：parsed_<stem>.json / verify_<stem>.json 一律写进 data/，与
corpus.find_pairs() 的配对口径一致，源文件放在哪里都能被检索层发现。

不负责：「最新版去哪儿拿」是上层应用的事。引擎只认「你这次给我的这份文件」。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag_verifier.chunking.splitter import split_document
from rag_verifier.chunking.verifier import verify_chunk
from rag_verifier.models import EvidenceCard
from rag_verifier.parsing import parse_document
from rag_verifier.parsing.models import StructuredDocument
from rag_verifier.retrieval.corpus import DATA_DIR, PROJECT_ROOT

REGISTRY_PATH = DATA_DIR / "doc_registry.json"
# 与 scripts/verify_chunks.py 同口径：前置内容与整表块不参与块级验证。
SKIP_CATEGORIES = ("preamble_dropped", "table")


def sha1_file(path: Path) -> str:
    """文件内容 sha1（分块读，大文件不吃内存）。"""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_registry() -> dict:
    """读文档注册表：stem → {source_path, source_sha1, parsed_file, verify_file,
    n_chunks, updated_at}。不存在则返回空表。"""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def save_registry(reg: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def _parsed_path(stem: str) -> Path:
    return DATA_DIR / f"parsed_{stem}.json"


def _verify_path(stem: str) -> Path:
    return DATA_DIR / f"verify_{stem}.json"


def _verify_parsed(stem: str) -> int:
    """对 parsed_<stem>.json 跑五关验证，写 verify_<stem>.json，返回证据块数。

    与 scripts/verify_chunks.py 的 verify_document 同口径：逐块调 verify_chunk，
    prev_chunk 传递前一个证据块的正文（判断条件从句是否被切到前文）。
    """
    pf = _parsed_path(stem)
    doc = StructuredDocument.from_dict(json.loads(pf.read_text(encoding="utf-8")))
    chunks = split_document(doc=doc, source=doc.source)
    full_doc = doc.full_text()
    evidence = [c for c in chunks if c["category"] not in SKIP_CATEGORIES]

    rows, prev = [], ""
    for c in evidence:
        card = EvidenceCard(content=c["card"]["content"],
                            source=c["card"]["source"],
                            locator=c["card"]["locator"])
        r = verify_chunk(card, source_text=c.get("source_text", ""),
                         prev_chunk=prev, full_doc=full_doc)
        rows.append({
            "id": c["id"],
            "category": c["category"],
            "locator": c["card"]["locator"],
            "overall": r["overall"].value,
            "checks": r["checks"],
            "preview": card.content[:50],
        })
        prev = card.content

    summary = Counter(r["overall"] for r in rows)
    _verify_path(stem).write_text(json.dumps(
        {"source": doc.source, "engine": doc.engine,
         "summary": dict(summary), "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def ingest(path: str | Path, force: bool = False) -> dict:
    """入库一份文档，返回 {doc, status: added|replaced|unchanged, source_path, n_chunks}。

    unchanged 时解析/切分/验证/嵌入全部跳过（零成本）；force=True 忽略 sha1 未变强制重跑。
    """
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {src}")

    stem = src.stem
    digest = sha1_file(src)
    reg = load_registry()
    prev = reg.get(stem)

    if (not force and prev and prev.get("source_sha1") == digest
            and _parsed_path(stem).exists() and _verify_path(stem).exists()):
        return {"doc": stem, "status": "unchanged",
                "source_path": prev.get("source_path", str(src)),
                "n_chunks": prev.get("n_chunks", 0)}

    doc = parse_document(str(src))
    _parsed_path(stem).write_text(
        json.dumps(doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    n_evidence = _verify_parsed(stem)

    reg[stem] = {
        "source_path": str(src),
        "source_sha1": digest,
        "parsed_file": str(_parsed_path(stem)),
        "verify_file": str(_verify_path(stem)),
        "n_chunks": n_evidence,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_registry(reg)
    return {"doc": stem, "status": "replaced" if prev else "added",
            "source_path": str(src), "n_chunks": n_evidence}


def remove(doc_id: str) -> dict:
    """移除一份文档：删产物 + 撤注册表条目。

    检索层的分片缓存下次加载时按当前语料自动裁掉该文档的块，无需在此手动清缓存。
    """
    reg = load_registry()
    entry = reg.pop(doc_id, None)
    deleted = []
    for p in (_parsed_path(doc_id), _verify_path(doc_id)):
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if entry is not None:
        save_registry(reg)
    return {"doc": doc_id, "status": "removed" if entry is not None else "not_found",
            "deleted_files": deleted}


def list_documents() -> list[dict]:
    """已入库文档清单（按 stem 排序），供上层展示/管理。"""
    reg = load_registry()
    out = []
    for stem in sorted(reg):
        entry = dict(reg[stem])
        entry["doc"] = stem
        out.append(entry)
    return out


def rebuild_registry() -> dict:
    """扫描既有 parsed/verify 产物重建注册表（迁移用：把 ingest 之前入库的老文档纳入管理）。

    源文件定位：data/ 下同 stem 且扩展名受支持的原始文件，找不到再到项目根找（README 类
    文件在根目录）。都找不到（原始文件未保留）则 source_path/source_sha1 留空——这类条目
    无法做 unchanged 快判，下次 ingest 同一 stem 会重跑，属预期行为（宁可重跑，不假装未变）。
    """
    from rag_verifier.parsing.base import SUPPORTED_FORMATS

    def locate(stem: str) -> Path | None:
        for base in (DATA_DIR, PROJECT_ROOT):
            for ext in SUPPORTED_FORMATS:
                cand = base / f"{stem}{ext}"
                if cand.exists():
                    return cand
        return None

    reg = {}
    for vf in sorted(DATA_DIR.glob("verify_*.json")):
        stem = vf.stem[len("verify_"):]
        pf = _parsed_path(stem)
        if not pf.exists():
            continue  # 孤儿 verify 文件（如 verify_gb45438_results.json），跳过
        rows = json.loads(vf.read_text(encoding="utf-8")).get("rows", [])
        src_file = locate(stem)
        # 只敢在「产物不早于源文件」时打上 sha1：否则源文件在验证之后又被改过，
        # 产物对应的是旧内容，打上当前 sha1 会让后续 unchanged 快判撒谎（宁可留空重跑）。
        fresh = (src_file is not None
                 and src_file.stat().st_mtime <= vf.stat().st_mtime)
        reg[stem] = {
            "source_path": str(src_file) if src_file else "",
            "source_sha1": sha1_file(src_file) if fresh else "",
            "parsed_file": str(pf),
            "verify_file": str(vf),
            "n_chunks": len(rows),
            "updated_at": datetime.fromtimestamp(
                vf.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        }
    save_registry(reg)
    return reg
