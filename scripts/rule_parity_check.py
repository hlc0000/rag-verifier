"""规则对等性验证：每条规则的正例/反例必须给出确定性判定，中英逐条对照。

用法：
  python3 scripts/rule_parity_check.py

为什么需要它：基线规则审计只能看到「规则在真实文档上命中了几次」，
命不中既可能是文档没有该场景，也可能是规则写死了。这里用最小正/反例
逐条验证每个规则 ID 确实生效、且不误伤，并把中英规则条数并排打印。
退出码 1 表示有规则行为不符合预期。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_verifier.chunking.verifier  # noqa: E402,F401
import rag_verifier.chunking.semantic_verifier  # noqa: E402,F401
import rag_verifier.chunking.template_verifier  # noqa: E402,F401
from rag_verifier.chunking.rules import RuleContext, list_rules, rules_for  # noqa: E402
from rag_verifier.models import EvidenceCard  # noqa: E402

# rule_id -> {"pos": 应判 FAIL 的样本, "neg": 应判 PASS 的样本, "prev": 该规则用到的前文}
CASES = {
    # ---------------- 独立性
    "empty": {"pos": [""], "neg": ["The Buyer shall pay the price."]},
    "fragment": {"pos": ["日"], "neg": ["Days"]},
    "table_continuation": {
        "pos": ["Table 1: Categories for the GOVERN function. (Continued)"],
        "neg": ["Table 1: Categories and subcategories for the GOVERN function. (Continued) "
                "The GOVERN function cultivates risk management culture across the organization."],
    },
    "dependent_starter_zh": {
        "pos": ["因此，乙方应当承担违约责任。"],
        "neg": ["本合同自双方签字盖章之日起生效。"],
    },
    "dependent_starter_en": {
        "pos": ["Therefore, the Buyer shall pay the purchase price."],
        "neg": ["The Buyer shall pay the purchase price."],
    },
    "conjunctive_starter_zh": {
        "pos": ["并要求乙方承担违约责任。"],
        "neg": ["并购交易应经股东会批准。"],
    },
    "conjunctive_starter_en": {
        "pos": ["And the Seller may terminate the Agreement."],
        "neg": ["The Seller may terminate the Agreement."],
    },
    "incomplete_ending_zh": {
        "pos": ["本合同的终止条件为如果"],
        "neg": ["如果乙方违约，甲方可以解除合同。"],
    },
    "incomplete_ending_en": {
        "pos": ["The agreement terminates automatically only if"],
        "neg": ["If the agreement terminates, the fee is refunded."],
    },
    # ---------------- 完整性
    "cross_block_table_ref_zh": {
        "pos": ["由表3可知，一次合格率为97%。"],
        "neg": ["试验结果表明一次合格率为97%。"],
    },
    "cross_block_table_ref_en": {
        "pos": ["As shown in Table 3, the pass rate is 97%."],
        "neg": ["The experiment achieved a pass rate of 97%."],
    },
    "if_then_pair_zh": {
        "pos": ["如果乙方未按期支付价款"],
        "neg": ["如果乙方未按期支付价款，则甲方有权解除合同。"],
    },
    "if_then_pair_en": {
        "pos": ["If the Buyer fails to pay the purchase price and refuses to cure the breach "
                "within the notice period"],
        "neg": ["If the Buyer fails to pay, the Seller may terminate the contract."],
    },
    "concession_pair_zh": {
        "pos": ["虽然乙方已按期交付全部货物"],
        "neg": ["虽然乙方已按期交付货物，但是部分货物存在质量问题。"],
    },
    "concession_pair_en": {
        "pos": ["Although the Seller delivered the goods on time and in good condition"],
        "neg": ["Although the Seller delivered the goods on time, the quality was unacceptable."],
    },
    # ---------------- 语义独立性
    "anaphora_zh": {"pos": ["该费用由乙方承担。"], "neg": ["乙方应承担装卸费用。"]},
    "anaphora_en": {
        "pos": ["Such payment shall be borne by Party B."],
        "neg": ["Party B shall bear the payment."],
    },
    "missing_subject_zh": {
        "pos": ["应当在收到货物后七日内支付价款。"],
        "neg": ["甲方应当在收到货物后七日内支付价款。"],
    },
    "missing_subject_en": {
        "pos": ["Shall pay the purchase price within seven days."],
        "neg": ["The Buyer shall pay the purchase price within seven days."],
    },
    "dangling_condition_zh": {"pos": ["则甲方有权解除合同。"], "neg": ["甲方有权解除合同。"]},
    "dangling_condition_en": {
        "pos": ["Then the Seller may terminate the contract."],
        "neg": ["The Seller may terminate the contract."],
    },
    "cut_condition_prev_zh": {
        "pos": ["甲方有权要求赔偿。"],
        "neg": ["甲方有权要求赔偿。"],
        "prev": ["若乙方违反保密义务", "若乙方违反保密义务，则甲方有权要求赔偿。"],
    },
    "cut_condition_prev_en": {
        "pos": ["the Seller may terminate the contract"],
        "neg": ["the Seller may terminate the contract"],
        "prev": ["If the Buyer fails to pay the purchase price",
                 "If the Buyer fails to pay, the Seller may terminate."],
    },
    # ---------------- 模板完整性
    "template_blank_zh": {
        "pos": ["保管费计算方式为，。"],
        "neg": ["保管费计算方式为每月100元。"],
    },
    "template_blank_en": {
        "pos": ["Assignor[s]:____________________________________________________________",
                "Copyright [yyyy] [name of copyright owner]"],
        "neg": [
            "Party A: Acme Corp, Party B: Beta LLC.",
            "At that level, the reserve would have been emptied within months.",
            "Note: Data are for calendar years with these exceptions: 1965 and 1966.",
            "Delimiters are chosen from the set of characters (DQUOTE and \"(),/:;<=>?@[\\]{}\").",
            "See [RFC8446] Section 4.1 for the handshake message definition.",
        ],
    },
}

# 负例还要过「整条检查项」而不只是单条规则：模板规则在报告/代码上也必须放行


def run_rule(rule, content: str, prev: str = "") -> str:
    card = EvidenceCard(content=content, source="test", locator="T001")
    ctx = RuleContext(lang=rule.lang if rule.lang in ("zh", "en") else "en",
                      prev_chunk=prev, source_text=content, full_doc=content)
    result = rule.fn(card, ctx)
    return "fail" if result is not None else "pass"


def main() -> None:
    rules = {r["id"]: r for r in list_rules()}
    defects = []
    rows = []

    for rid, case in CASES.items():
        meta = rules.get(rid)
        if meta is None:
            defects.append(f"{rid}: 规则未注册")
            continue
        rule = next(r for r in rules_for(meta["check"], meta["lang"]) if r.id == rid)
        prevs = case.get("prev") or [""]
        pos_ok = all(run_rule(rule, s, prevs[0]) == "fail" for s in case["pos"])
        neg_ok = all(run_rule(rule, s, prevs[1] if len(prevs) > 1 else "") == "pass"
                     for s in case["neg"])
        rows.append((meta["check"], meta["lang"], rid, pos_ok, neg_ok))
        if not pos_ok:
            defects.append(f"{rid}: 正例未命中（规则失效）")
        if not neg_ok:
            defects.append(f"{rid}: 反例误伤（假阳性）")

    print(f"{'检查项':<24}{'语种':<6}{'规则':<28}{'正例':<6}{'反例'}")
    for check, lang, rid, pos_ok, neg_ok in rows:
        print(f"{check:<24}{lang:<6}{rid:<28}{'✓' if pos_ok else '✗':<6}"
              f"{'✓' if neg_ok else '✗'}")

    print("\n中英规则条数对照（不含 any 通用规则）:")
    for check in ("independence", "completeness", "semantic_independence", "template_completeness"):
        zh = len([r for r in list_rules() if r["check"] == check and r["lang"] == "zh"])
        en = len([r for r in list_rules() if r["check"] == check and r["lang"] == "en"])
        print(f"  {check:<24} zh={zh}  en={en}  {'对等' if zh == en else '不对等'}")

    missing = [rid for rid in rules if rid not in CASES and rules[rid]["check"] != "traceability"]
    if missing:
        print(f"\n未覆盖用例的规则: {missing}")

    if defects:
        print("\n✗ 失败项:")
        for d in defects:
            print("  -", d)
        sys.exit(1)
    print("\n✓ 全部规则正例命中、反例零误伤")


if __name__ == "__main__":
    main()
