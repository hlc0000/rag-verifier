"""规则层审计：逐条规则统计命中，用于中英对等性与误伤核查。

用法：
  python3 scripts/rule_audit.py               # 全部 20 份基线
  python3 scripts/rule_audit.py --only-en     # 只看英文基线
  python3 scripts/rule_audit.py --dump        # 打印每个命中块的全文（人工核实用）

与 verify_chunks 的区别：verify_chunks 每项检查命中第一条规则就返回，
无法知道「哪条规则命中」。本脚本绕过短路，直接逐条规则跑，统计每个规则 ID
在多少块上命中、命中样例是什么。输出 data/_rule_audit.json。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.rules import detect_language, rules_for, list_rules  # noqa: E402
from src.chunking.splitter import split_document  # noqa: E402

# 规则在模块导入时注册；不导入就等于规则表为空（首次运行踩过：全 0 命中）
import src.chunking.verifier  # noqa: E402,F401
import src.chunking.semantic_verifier  # noqa: E402,F401
import src.chunking.template_verifier  # noqa: E402,F401
from src.models import EvidenceCard  # noqa: E402
from src.parsing.models import StructuredDocument  # noqa: E402
from src.chunking.rules import RuleContext  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SKIP = {"preamble_dropped", "table"}

BASELINES = [
    "gb45438", "long-doc", "readme", "sample-webpage", "chinese-doc",
    "chinese-corp-announcement-01", "chinese-paper-real-01", "chinese-paper-real-02",
    "chinese-product-release-notes-01", "en-readme-requests",
    "en-changelog-keepachangelog", "en-contract-employment-ice",
    "en-contract-credit-swk", "en-report-berkshire-2023", "en-report-nist-ai-rmf",
    "en-paper-arxiv-bert", "en-rfc-9110", "en-docs-python-tutorial",
    "zh-govcn-digital-gov", "real-contract-filled-gov",
]

CHECKS = ("independence", "completeness", "traceability",
          "semantic_independence", "template_completeness")


def audit(dump: bool = False, only_en: bool = False) -> dict:
    hits: dict[str, list] = defaultdict(list)
    lang_of: dict[str, str] = {}
    totals = Counter()

    for name in BASELINES:
        if only_en and not name.startswith("en-"):
            continue
        parsed = DATA / f"parsed_{name}.json"
        if not parsed.exists():
            print(f"  ! 缺少 {parsed.name}，跳过")
            continue
        doc = StructuredDocument.from_dict(json.loads(parsed.read_text(encoding="utf-8")))
        chunks = [c for c in split_document(doc=doc, source=doc.source)
                  if c["category"] not in SKIP]
        prev = ""
        for c in chunks:
            content = c["card"]["content"]
            lang = detect_language(content)
            lang_of[name] = lang if name.startswith("en-") else lang_of.get(name, lang)
            totals[lang] += 1
            card = EvidenceCard(content=content, source=c["card"]["source"],
                                locator=c["card"]["locator"])
            ctx = RuleContext(lang=lang, prev_chunk=prev, source_text=c.get("source_text", ""),
                              full_doc=doc.full_text())
            for check in CHECKS:
                for rule in rules_for(check, lang):
                    try:
                        r = rule.fn(card, ctx)
                    except Exception as e:  # 规则抛错要显式记录，不能静默吞掉
                        hits[rule.id].append({"doc": name, "id": c["id"], "error": repr(e)})
                        continue
                    if r is not None:
                        hits[rule.id].append({
                            "doc": name, "chunk": c["id"], "locator": c["card"]["locator"],
                            "status": r.status.value, "reason": r.reason,
                            "content": content,
                        })
            prev = content

    return {"totals": dict(totals), "hits": {k: v for k, v in hits.items()},
            "rules": list_rules()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="打印命中块全文")
    ap.add_argument("--only-en", action="store_true", help="只跑英文基线")
    args = ap.parse_args()

    report = audit(args.dump, args.only_en)
    out = DATA / "_rule_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n审计块数: {report['totals']}")
    print("\n每条规则命中统计（命中块数 / 状态）:")
    by_check = defaultdict(list)
    for rid, items in report["hits"].items():
        rule = next((r for r in report["rules"] if r["id"] == rid), None)
        if rule is None:
            continue
        statuses = Counter(i.get("status", "error") for i in items)
        by_check[rule["check"]].append((rule["lang"], rid, len(items), dict(statuses)))

    for check in CHECKS:
        rows = sorted(by_check.get(check, []), key=lambda x: (x[0], x[1]))
        print(f"\n[{check}]")
        for lang, rid, n, st in rows:
            print(f"  {lang:<4} {rid:<28} {n:>4}  {st}")
            if args.dump:
                for item in report["hits"][rid][:3]:
                    print(f"        {item['doc']} {item['chunk']}: {item['content'][:160]}")

    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
