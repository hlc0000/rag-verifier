"""模板完整性：检出"空位未填写"的合同/表单块。

规则按语种分两套（zh / en），由 src/chunking/rules.py 调度，可注册扩展。
"""
import re

from rag_verifier.models import EvidenceCard, VerificationResult, VerificationStatus
from rag_verifier.chunking.rules import RuleContext, detect_language, register_rule, rules_for

# 标点空位
PUNCTUATION_PLACEHOLDER = [
    r'：\s*[。；，]',
    # "为"紧跟标点=空位（"计算方式为，"）；但"行为/认为/成为/作为"等双字词
    # 中的"为"是构词成分，用否定后顾排除，避免误伤正常文本
    r'(?<![行认成作以因视称评分甚颇尤化转定选列归誉])为\s*[。；，]',
    r'[：，]\s*[。；]',
]

# 数量/时间空位
QUANTITY_PLACEHOLDER = [
    r'每逾期\s*日',
    r'每提前\s*日',
    r'之日起\s*日',
    r'赔偿金额以\s*为限',
    r'按照\s*标准',
    # 数字空位（"第_种方式""一式_份""甲方_份""乙方_份"）
    r'第\s*种',
    r'一式\s*份',
    r'甲方\s*份',
    r'乙方\s*份',
    # 语义层曾判出的物理空缺（收进规则层，比语义层更稳、不摇摆）
    r'期限\s*日',        # "超过保管期限日不领取" → R016，"日"前缺数字
    r'即\s*[，。；]',     # "其他保密信息即，" → R017，"即"后缺内容
    r'应当\s*在\s*日',   # "应当在日内向对方提供" → R039，"日"前缺数字
]

# 日期空位
DATE_PLACEHOLDER = [
    r'自\s*年\s*月\s*日',
    r'至\s*年\s*月\s*日',
    r'年\s*月\s*日\s*时',
    # 裸"年月日"（无数字）：兜住签名/落款区的空日期，如 R050
    r'年\s*月\s*日',
]

# 签名/盖章空位（"（签名/盖章）" 与冒号之间常夹"）"，需容忍）
# 收紧：冒号后不得紧跟非空白字符 —— 避免对"签名/盖章）：张三"这类已签署合同误伤。
SIGNATURE_PLACEHOLDER = [
    r'签名/盖章\s*）?\s*[：:](?!\S)',
]

# 英文空位：与中文四类一一对应（标点 / 数量时间 / 日期 / 签名），
# 只是英文合同的空位形态是下划线、空方括号、只有标签没有值的字段行、TBD 占位。
# 下划线不在行首匹配，避免把 markdown 分隔线 ---/___ 当空位；方括号要求 ≥2 空格，
# 避免代码里的 arr[]。
EN_PUNCTUATION_PLACEHOLDER = [
    # 冒号前必须是单词字符：排除 ABNF 语法集 "(),/:;<=>?@"（RFC 9110 实测误伤）
    # 与代码片段 "a < 10 : ..."（Python 教程实测误伤）
    r'(?<=\w)[：:]\s*[,.;]',         # "Party A: , Party B:" / "Fee: ." —— 冒号后直接标点
    r'\(\s{2,}\)',                  # "(    )" 空括号，常是待填的勾选/金额位
]

EN_QUANTITY_PLACEHOLDER = [
    # "Amount: ." / "the fee shall be ;" —— 值的位置是标点
    r'\b(?:amount|fee|price|rate|penalty|interest)\b[^.\n]{0,20}?\b(?:of|is|shall\s+be)\s*[,.;]',
    r'\bNo\.\s*(?=[,.;]|$)',
]
# 说明：不再把 "within days"/"for calendar years" 当空位——这两句本身是合法英语
# （伯克希尔年报实测 2 条误伤）。真缺失数字的场景会留下下划线或标点空位，已被覆盖。

EN_DATE_PLACEHOLDER = [
    r'\b(?:dated?|date)\s*:\s*[,.;]',
    r'_{1,}\s*/\s*_{1,}\s*/\s*_{1,}',   # ____/____/____ 日期格
]

# 只有标签没有值的字段行（签名/日期/姓名/职务等），整行只剩标签和冒号
EN_SIGNATURE_PLACEHOLDER = [
    r'(?im)^\s*(?:By|Signature|Authorized\s+Signature|Name|Printed\s+Name|Title|Witness|'
    r'Company|Address|E-?mail|Phone|Date|Dated|Signed|Date\s+of\s+Birth)\s*:\s*$',
]

EN_BLANK_UNDERSCORE = re.compile(r'(?<=[^\n])_{3,}')
EN_BLANK_BRACKET = re.compile(r'\[[ \t]{2,}\]')
# 方括号占位符：英文合同/许可证常用 "[yyyy]" "[name of copyright owner]" "[insert date]"。
# 只认占位词开头并在 ] 前带可选说明，避免把 RFC 的 "[RFC8446]" 引用当空位（实测零误伤）。
EN_BLANK_BRACKET_WORD = re.compile(
    r'\[(?:(?:yyyy|yy|mm|dd|xxxx)\b|(?:insert|specify|name|date|amount|address|title|'
    r'company|signatory)\b)[^\]\n]{0,40}\]',
    re.IGNORECASE,
)
EN_BLANK_TBD = re.compile(r'\b(?:TBD|TBA)\b|\b(?:To\s+be\s+determined|To\s+be\s+advised)\b', re.IGNORECASE)


@register_rule("template_completeness", "zh", id="template_blank_zh")
def _rule_template_zh(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    all_patterns = (
        PUNCTUATION_PLACEHOLDER
        + QUANTITY_PLACEHOLDER
        + DATE_PLACEHOLDER
        + SIGNATURE_PLACEHOLDER
    )
    for pattern in all_patterns:
        match = re.search(pattern, text)
        if match:
            return VerificationResult(
                VerificationStatus.FAIL,
                f"模板空位未填写: 匹配'{match.group()}'",
                {"pattern": pattern, "matched": match.group()},
            )
    return None


@register_rule("template_completeness", "en", id="template_blank_en")
def _rule_template_en(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    compiled = [(EN_BLANK_UNDERSCORE, "underscore blank"),
                (EN_BLANK_BRACKET, "bracket blank"),
                (EN_BLANK_BRACKET_WORD, "bracketed placeholder"),
                (EN_BLANK_TBD, "TBD/TBA placeholder")]
    for pattern, kind in compiled:
        match = pattern.search(text)
        if match:
            return VerificationResult(
                VerificationStatus.FAIL,
                f"Unfilled template slot ({kind}): '{match.group()}'",
                {"pattern": pattern.pattern, "matched": match.group()},
            )
    # 字符串类模式：与中文一样按类别逐条扫描
    for patterns, kind in (
        (EN_PUNCTUATION_PLACEHOLDER, "punctuation blank"),
        (EN_QUANTITY_PLACEHOLDER, "missing quantity/number"),
        (EN_DATE_PLACEHOLDER, "date blank"),
        (EN_SIGNATURE_PLACEHOLDER, "empty labeled field"),
    ):
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return VerificationResult(
                    VerificationStatus.FAIL,
                    f"Unfilled template slot ({kind}): '{match.group()}'",
                    {"pattern": pattern, "matched": match.group()},
                )
    return None


def verify_template_completeness(card: EvidenceCard, ctx: RuleContext | None = None) -> VerificationResult:
    ctx = ctx or RuleContext(lang=detect_language(card.content))
    for rule in rules_for("template_completeness", ctx.lang):
        result = rule.fn(card, ctx)
        if result is not None:
            return result
    return VerificationResult(
        VerificationStatus.PASS,
        "Template completeness check passed" if ctx.lang == "en" else "模板完整性检查通过")
