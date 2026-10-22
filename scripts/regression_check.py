"""一键回归：20 份基线全部重新切分，与存档验证报告逐块比对。

用法：
  python3 scripts/regression_check.py            # 快速：复用存档 parsed JSON（测切分层/验证层改动）
  python3 scripts/regression_check.py --full     # 完整：重新解析原始文件再切分（测解析层改动）

比对口径（不调 LLM——语义判定随 prompt 漂移，不作回归准绳）：
  1. 证据块 locator 序列完全一致（顺序敏感的最强身份比对）
  2. 类别分布一致（evidence 各类别 / preamble_dropped / table 计数）
  3. 解析层质量报告的 issue 集合一致（--full 模式比对重解析产物）

任何一项漂移即退出码 1——改代码后跑一遍，老基线被伤立刻现形。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.splitter import split_document  # noqa: E402
from src.parsing.models import StructuredDocument  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

BASELINES = [
    "gb45438", "long-doc", "readme", "sample-webpage", "chinese-doc",
    "chinese-corp-announcement-01", "chinese-paper-real-01", "chinese-paper-real-02",
    "chinese-product-release-notes-01", "en-readme-requests",
    "en-changelog-keepachangelog", "en-contract-employment-ice",
    "en-contract-credit-swk", "en-report-berkshire-2023", "en-report-nist-ai-rmf",
    "en-paper-arxiv-bert", "en-rfc-9110", "en-docs-python-tutorial",
    "zh-govcn-digital-gov", "real-contract-filled-gov",
]

SKIP = {"preamble_dropped", "table"}


def expected_profile(verify_path: Path) -> dict:
    d = json.loads(verify_path.read_text(encoding="utf-8"))
    rows = d["rows"]
    return {
        "locators": [r["locator"] for r in rows],
        "cats": Counter(r["category"] for r in rows),
        "total": len(rows),
    }


def current_profile(doc: StructuredDocument) -> dict:
    chunks = split_document(doc=doc, source=doc.source)
    evidence = [c for c in chunks if c["category"] not in SKIP]
    return {
        "locators": [c["card"]["locator"] for c in evidence],
        "cats": Counter(c["category"] for c in evidence),
        "total": len(evidence),
        "dropped": sum(1 for c in chunks if c["category"] == "preamble_dropped"),
        "tables": sum(1 for c in chunks if c["category"] == "table"),
        "quality_issues": sorted(
            i["code"] for i in doc.metadata.get("quality_report", {}).get("issues", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="20 份基线一键回归")
    ap.add_argument("--full", action="store_true",
                    help="重新解析原始文件（默认复用存档 parsed JSON）")
    args = ap.parse_args()

    if args.full:
        from src.parsing.base import parse_document

    drifted = []
    print(f"{'文档':34s} {'存档块数':>6s} {'现算块数':>6s}  结论")
    for name in BASELINES:
        verify_path = DATA / f"verify_{name}.json"
        parsed_path = DATA / f"parsed_{name}.json"
        if not verify_path.exists() or not parsed_path.exists():
            print(f"{name:34s}    --      --     缺存档，跳过")
            continue

        if args.full:
            raw = DATA / json.loads(parsed_path.read_text(encoding="utf-8"))["source"]
            if not raw.exists():
                print(f"{name:34s}    --      --     原始文件缺失（{raw.name}），跳过")
                continue
            try:
                doc = parse_document(str(raw))
            except Exception as e:
                print(f"{name:34s}    --      --     解析异常: {type(e).__name__}: {e}")
                drifted.append(name)
                continue
        else:
            doc = StructuredDocument.from_dict(
                json.loads(parsed_path.read_text(encoding="utf-8")))

        exp, cur = expected_profile(verify_path), current_profile(doc)
        ok = exp["locators"] == cur["locators"] and exp["cats"] == cur["cats"]
        mark = "一致" if ok else "漂移!"
        print(f"{name:34s} {exp['total']:6d} {cur['total']:6d}  {mark}")
        if not ok:
            drifted.append(name)
            print(f"      存档类别: {dict(sorted(exp['cats'].items()))}")
            print(f"      现算类别: {dict(sorted(cur['cats'].items()))}")
            extra = [l for l in cur["locators"] if l not in exp["locators"]]
            gone = [l for l in exp["locators"] if l not in cur["locators"]]
            if extra:
                print(f"      新增 locator: {extra}")
            if gone:
                print(f"      消失 locator: {gone}")

    print()
    if drifted:
        print(f"回归失败：{len(drifted)} 份漂移 -> {drifted}")
        sys.exit(1)
    print(f"回归通过：{len(BASELINES)} 份基线全部一致")


if __name__ == "__main__":
    main()
