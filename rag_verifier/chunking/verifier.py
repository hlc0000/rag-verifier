"""块级验证：独立性 / 完整性 / 可追溯性 / 语义独立性 / 模板完整性。

规则层按语种分两套（zh / en），由 src/chunking/rules.py 的注册表调度：
  - 通用规则（碎片、空块、跨页表标记）lang="any"
  - 中文规则（承接词、条件句配对、由表N可知…）lang="zh"
  - 英文规则（Therefore/If…、as shown in Table N…）lang="en"
第三方可通过 register_rule 追加自己的规则，不改本文件。
"""
import re

from src.models import EvidenceCard, VerificationResult, VerificationStatus
from src.chunking.rules import RuleContext, detect_language, register_rule, rules_for
from src.chunking.semantic_verifier import (
    SKIP_LLM, has_multiple_sentences_en, has_multiple_sentences_zh,
    verify_semantic_independence,
)
from src.chunking.template_verifier import verify_template_completeness

DEPENDENT_STARTERS = [
    "因此", "所以", "但是", "然而", "不过", "综上所述", "由此可见",
    "于是", "从而", "总之", "换言之", "也就是说", "此外", "另外",
]

# 英文承接词必须带词边界：否则 "So" 误伤 "Social"/"Software"、"But" 误伤 "Button"
EN_DEPENDENT_STARTER_RE = re.compile(
    r'^(?:Therefore|Thus|However|But|So|In conclusion)\b'
)

# 单字承接词必须连到主语/谓语/情态词才算残缺：裸匹配"并"会误伤"并购/并行/并网/并发/并列/并存"。
# 例："并要求乙方承担违约责任。"命中（缺主语与触发条件）；"并购交易应经股东会批准。"放行。
# `并(?![购行网发列存入举联非])` 排除实测存在的复合词首字，宁可漏判也不误伤。
ZH_CONJ_STARTER_RE = re.compile(
    r'^(?:并且|并(?![购行网发列存入举联非])|以及|同时)(?:且)?'
    r'(?:要求|应当|应|须|需|需要|有权|可以|可|不得|不予|同意|承诺|保证|承担|支付|按|向|对|在|将|于'
    r'|甲方|乙方|双方|任何一方|当事人)'
)

# 英文对应规则（保持中英对等）：以并列/补充连词开头的块同样缺主语或前提。
EN_CONJ_STARTER_RE = re.compile(
    r'^(?:And|Also|Additionally|Moreover|Furthermore|Plus)\b', re.IGNORECASE
)

INCOMPLETE_ENDINGS = [
    "如果", "假如", "倘若", "因为", "由于", "虽然", "尽管",
    "若", "除非",
]
# 英文残缺结尾必须是"独立的词"（前边界为空白/行首）：连字符复合词不算，
# 否则 HTTP 请求头 "If-Modified-Since" 结尾的 Since 被误判为残缺连词（RFC 实测）。
# 不区分大小写：句尾小写 "…only if""…because" 同样是截断残缺。
EN_INCOMPLETE_ENDING_RE = re.compile(
    r'(?:^|\s)(?:If|Because|Since|Although|Though|Unless)$', re.IGNORECASE
)

# 表格跨页标记（通用排版惯例，非特定文档）：长表跨页时每页重复的续页标记
TABLE_CONT_MARK = re.compile(r"Continued on next page|\(Continued\)", re.IGNORECASE)
# 表格/图/附件标题行（剥离时按行匹配行首）
TABLE_CAPTION = re.compile(r"^(?:Table|Figure|Fig\.?|Exhibit)\s+\d+", re.IGNORECASE)

# 只用双字/三字条件词：单字"因""如""当"在正常文本中高频出现，
# 与条件句无关（如"如附件A所示""因乙方原因""当事人双方"），保留会造成误判。
# "由于/因为"是因果词不是条件词——"由于X，Y"是完整句，无需"就/那么"配对，移出防误伤。
IF_MARKERS = ["如果", "假如", "倘若", "若", "除非"]
# "请"是祈使结果句的合法承接："若升级失败，请重启"——条件自带祈使结果即完整，
# 无需"就/那么"。真实截断块（"若乙方违反保密义务"↘戛然而止）不含"请"，不受影响。
THEN_MARKERS = ["那么", "则", "就", "便", "所以", "因此", "故", "从而", "于是", "请"]

# 裸"若"的词界守卫："若干"的"若"不是条件标记（最高法文书名称"…若干问题的意见"
# 实测误伤）。"若非/若是/若无"仍是条件用法，只排除"若干"。
_ZH_IF_RUO_RE = re.compile(r'若(?!干)')


def _find_if_marker_zh(text: str):
    """在 text 中找首个条件标记；裸"若"走词界守卫，排除"若干"。"""
    for m in IF_MARKERS:
        if m == "若":
            if _ZH_IF_RUO_RE.search(text):
                return m
            continue
        if m in text:
            return m
    return None

# 让步/转折配对：让步句以"虽然/尽管"开头，必须配上主句的转折词才算完整。
# 不能直接塞进 DEPENDENT_STARTERS —— 完整的让步句也以"虽然"开头。
CONCESSION_MARKERS = ["虽然", "尽管", "即使", "纵然", "哪怕"]
CONTRAST_MARKERS = ["但是", "但", "然而", "却", "仍", "仍然", "依然", "可是", "不过"]


# ---------------------------------------------------------------- 独立性

@register_rule("independence", "any", id="empty")
def _rule_empty(card: EvidenceCard, ctx: RuleContext):
    if not card.content.strip():
        return VerificationResult(
            VerificationStatus.FAIL,
            "Empty chunk" if ctx.lang == "en" else "空块")
    return None


@register_rule("independence", "any", id="fragment")
def _rule_fragment(card: EvidenceCard, ctx: RuleContext):
    # 碎片块：≤2 字不可能独立成句（实测来源：公告盖章页被切出的"年""月""日"单字块）。
    # 标题类两字段（如"附则"）不成块——切分器只把标题作分组边界，不单独成块，无误伤面。
    text = card.content.strip()
    if text and len(text) <= 2:
        reason = (f"Fragment ({len(text)} chars): too short to stand as a sentence"
                  if ctx.lang == "en"
                  else f"碎片块（{len(text)}字），无法独立成句")
        return VerificationResult(
            VerificationStatus.FAIL,
            reason,
            {"fragment": True},
        )
    return None


@register_rule("independence", "any", id="table_continuation")
def _rule_table_continuation(card: EvidenceCard, ctx: RuleContext):
    # 表格跨页标记碎片：英文长表跨页是通用排版惯例，每页重复表头 + "(Continued)"/
    # "Continued on next page" 标记。这些标记被切分成独立块后，剥掉标记与表格/图标题
    # 剩余实质内容趋近于零——纯标记碎片，规则层直接拦，不交 LLM（同质块 LLM 判定摇摆，
    # NIST 实测 11 块同质仅 1 fail）。有实际数据的续表块剥完仍剩大段正文，不会命中。
    text = card.content.strip()
    if TABLE_CONT_MARK.search(text):
        residue = TABLE_CONT_MARK.sub("", text)
        kept = []
        for line in residue.split("\n"):
            s = line.strip()
            if TABLE_CAPTION.match(s):
                # 整行就是表题（含续页标记）→ 丢掉；表题与正文挤在同一行时只去表题，
                # 否则会把正文一起当碎片（单行解析产物实测）
                if len(s) <= 120:
                    continue
                s = TABLE_CAPTION.sub("", s).lstrip(":： ")
            kept.append(s)
        residue = "\n".join(kept)
        if len(residue.strip()) < 25:
            reason = ("Table continuation fragment: the chunk only carries a caption / "
                      "continued marker, with no substantive content"
                      if ctx.lang == "en"
                      else "表格跨页标记碎片：块仅含表头/续页标记，无实质内容")
            return VerificationResult(
                VerificationStatus.FAIL,
                reason,
                {"fragment": True, "kind": "table_continuation"},
            )
    return None


@register_rule("independence", "zh", id="dependent_starter_zh")
def _rule_dependent_starter_zh(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    for starter in DEPENDENT_STARTERS:
        if text.startswith(starter):
            # 多句豁免（与英文侧 known_issues #7 对等）：两三句的完整段落以
            # 「此外/因此」开头是文体过渡，结构自足，放行；单句仍拦截。
            # （最高法指导性案例实测：4 个完整句的"此外，关于…"块被误伤）
            if has_multiple_sentences_zh(text):
                return None
            return VerificationResult(
                VerificationStatus.FAIL,
                f"以依赖连接词开头: '{starter}'",
                {"starter": starter},
            )
    return None


@register_rule("independence", "zh", id="conjunctive_starter_zh")
def _rule_conjunctive_starter_zh(card: EvidenceCard, ctx: RuleContext):
    """以单字连词开头的块：主语与触发条件都在前文，单独成块必不自足。"""
    text = card.content.strip()
    match = ZH_CONJ_STARTER_RE.match(text)
    if match:
        return VerificationResult(
            VerificationStatus.FAIL,
            f"以承接词开头，主语与前文条件被切断: '{match.group()}'",
            {"starter": match.group()},
        )
    return None


@register_rule("independence", "en", id="dependent_starter_en")
def _rule_dependent_starter_en(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    match = EN_DEPENDENT_STARTER_RE.match(text)
    # 多句豁免（known_issues #7）：两三句的完整段落以 Therefore/So/But 开头是文体过渡，
    # 结构自足，放行；单句仍拦截。已知代价：多句真依赖（"So sanctified…"类）漏给语义层。
    if match and not has_multiple_sentences_en(text):
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Starts with a dependent connective: '{match.group()}'",
            {"starter": match.group()},
        )
    return None


@register_rule("independence", "zh", id="incomplete_ending_zh")
def _rule_incomplete_ending_zh(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    for ending in INCOMPLETE_ENDINGS:
        if text.endswith(ending):
            return VerificationResult(
                VerificationStatus.FAIL,
                f"以不完整条件句结尾: '{ending}'",
                {"ending": ending},
            )
    return None


@register_rule("independence", "en", id="conjunctive_starter_en")
def _rule_conjunctive_starter_en(card: EvidenceCard, ctx: RuleContext):
    """英文并列/补充连词开头的块：主语或前提在前文，单独成块不自足。"""
    text = card.content.strip()
    match = EN_CONJ_STARTER_RE.match(text)
    # 多句豁免（known_issues #7）：多句完整段落以 Furthermore/And then/Additionally
    # 开头是文体过渡，结构自足，放行；单句仍拦截。
    if match and not has_multiple_sentences_en(text):
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Starts with a conjunctive connector; subject/premise is in the prior chunk: '{match.group()}'",
            {"starter": match.group()},
        )
    return None


@register_rule("independence", "en", id="incomplete_ending_en")
def _rule_incomplete_ending_en(card: EvidenceCard, ctx: RuleContext):
    match = EN_INCOMPLETE_ENDING_RE.search(card.content.strip())
    if match:
        word = match.group().strip()
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Ends with an incomplete conditional: '{word}'",
            {"ending": word},
        )
    return None


# 跨边界条件回看：块自身没有任何结构标记，但上一块里有条件标记（if/notwithstanding/若X 等），
# 说明当前块是条件句的结果分句，单独拿出来不自足。
#
# 覆盖场景：切分器在逗号处劈开单句 —— `If Party A is 60 days late,` 成上一块、
# `Party B may terminate...` 成当前块。当前块无任何 marker，旧规则层碰不到。
#
# 窗口优先级：
#   1. prev_chunk 参数（调用方直接传了上一块内容，最精确）
#   2. full_doc 反推：找块起点 → 往前扫到句末标点停（退化路径，UI 手动输入时）
# 两种窗口覆盖等价（实测），prev_chunk 更干净且不受 full_doc.find 歧义影响。
# 英文 "?" 必须在集合里：反问/疑问句是完整思路，窗口若在句中 "When completed, ...?"
# 这种反问的条件状语处起切，会把完整问句误判成未配对条件前导（伯克希尔书信实测误伤）。
_CROSS_BOUNDARY_TERM = set("。；;.!?？？！")
_CROSS_BOUNDARY_WINDOW_MAX = 200

_ZH_CONDITION_PREFIX_RE = re.compile(r"^(若|如果|假如|倘若|除非|虽|尽管|即使|虽然)")

@register_rule("independence", "any", id="cross_boundary_condition")
def _rule_cross_boundary_condition(card: EvidenceCard, ctx: RuleContext):
    content = card.content.strip()
    if not content:
        return None

    prev_chunk = (ctx.prev_chunk or "").strip()
    full_doc = ctx.full_doc or ""

    # 取边界前的上下文尾巴：prev_chunk 优先，full_doc 反推做 fallback。
    # prev_chunk 可能是整块（实测 2000+ 字符，尾部还常黏 PDF 页脚），
    # 不能整段当窗口——否则句末标点守卫失效、30% 阈值形同虚设（基线实测大量误伤）。
    tail = ""
    if prev_chunk:
        tail = prev_chunk[-_CROSS_BOUNDARY_WINDOW_MAX:]
    elif full_doc:
        idx = full_doc.find(content)
        if idx == -1:
            return None
        tail = full_doc[max(0, idx - _CROSS_BOUNDARY_WINDOW_MAX):idx]

    # 窗口 = 尾巴里最后一个句末标点之后的片段；这样窗口要么是条件前导分句，
    # 要么是完整句的尾部/页脚，后两者不含行首条件标记，自然不触发。
    cut = -1
    for i in range(len(tail) - 1, -1, -1):
        if tail[i] in _CROSS_BOUNDARY_TERM:
            cut = i
            break
    if cut == -1 and len(tail) >= _CROSS_BOUNDARY_WINDOW_MAX:
        # 整个 200 字符都找不到句末标点：长块/页脚粘连，不是可判定的条件前导，跳过
        return None
    window = tail[cut + 1:].strip()

    if not window:
        return None

    # 条件前导分句被切分器劈开时，边界一定落在逗号处（切分只发生在标点上）。
    # 窗口里没有逗号 → 是标题/短语（实测 "When to Evaluate" 这种章节标题会误命中），
    # 不是被劈开的条件分句，跳过。
    if "," not in window and "，" not in window:
        return None

    # 窗口如果以句末标点结尾 → 上一块是完整句（不是逗号分隔的条件前导分句），
    # 条件标记在句中大概率是普通短语（如 "subject to change"），不是条件前导。
    # 跳过这种窗口。
    if window[-1] in _CROSS_BOUNDARY_TERM:
        return None

    # 中英文分别判。命中条件标记必须在窗口前半段（≤ 前 30%）——
    # 过滤句中短语："The tool is subject to change without notice" 里的 "subject to"
    # 位置在窗口中间，不是分句前导条件，不应该触发。
    hit = None
    if ctx.lang == "en":
        m = EN_CONDITION_MARKER_RE.search(window)
        if m and m.start() <= len(window) * 0.3:
            hit = m.group()
    else:  # zh
        # 中文标记更短，用更严格的阈值：标记必须在窗口前 20 字符内
        for marker in ("若", "如果", "假如", "倘若", "除非", "虽", "尽管", "即使", "虽然"):
            pos = window.find(marker)
            if pos != -1 and pos <= 20:
                hit = marker
                break

    if hit:
        if ctx.lang == "en":
            reason = (
                f"Prior chunk opens with a conditional/concessive lead '{hit}' — "
                f"this chunk is likely the result clause and is not self-contained."
            )
        else:
            reason = f"前文存在条件/让步标记 '{hit}'，当前块可能是条件句的结果分句，不自足"
        return VerificationResult(
            VerificationStatus.FAIL,
            reason,
            {"lead": hit, "window": window[:80]},
        )
    return None


def verify_independence(card: EvidenceCard, ctx: RuleContext | None = None) -> VerificationResult:
    ctx = ctx or RuleContext(lang=detect_language(card.content))
    for rule in rules_for("independence", ctx.lang):
        result = rule.fn(card, ctx)
        if result is not None:
            return result
    return VerificationResult(
        VerificationStatus.PASS,
        "Independence check passed" if ctx.lang == "en" else "独立性检查通过")


# ---------------------------------------------------------------- 完整性

# 跨块表格引用（2026-10-17 用户拍板的新失败类型）：正文引用"由表N可知"，
# 但表格本体是独立的 table 块，依据不在块内——块假设孤立的设计缺陷。
# 恢复策略=绑定同块或跨块召回，与 missing_referent 补前文不同，failure_type 不可混用。
TABLE_CONCLUSION_REF_RE = re.compile(r'由?表\s*\d+\s*可知')


@register_rule("completeness", "zh", id="cross_block_table_ref_zh")
def _rule_cross_block_ref_zh(card: EvidenceCard, ctx: RuleContext):
    match = TABLE_CONCLUSION_REF_RE.search(card.content.strip())
    if match:
        return VerificationResult(
            VerificationStatus.FAIL,
            f"跨块表格引用: '{match.group()}'的结论依据在独立表格块内，本块不自足（需跨块召回）",
            {"failure_type": "cross_block_reference", "ref": match.group()},
        )
    return None


# 英文等价物："as shown in Table 3" / "Table 4 summarizes…" —— 依据同样在独立表格块内
EN_TABLE_CONCLUSION_REF_RE = re.compile(
    r'\b(?:as\s+shown\s+in|shown\s+in|see|per|according\s+to)\s+'
    r'(?:Table|Figure|Fig\.?)\s*\d+'
    r'|\b(?:Table|Figure)\s*\d+\s+(?:shows|summarizes|lists|presents|reports)\b',
    re.IGNORECASE,
)


@register_rule("completeness", "en", id="cross_block_table_ref_en")
def _rule_cross_block_ref_en(card: EvidenceCard, ctx: RuleContext):
    match = EN_TABLE_CONCLUSION_REF_RE.search(card.content.strip())
    if match:
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Cross-chunk reference: '{match.group()}' points to a separate table/figure "
            f"chunk; this chunk is not self-contained (needs cross-chunk retrieval)",
            {"failure_type": "cross_block_reference", "ref": match.group()},
        )
    return None


@register_rule("completeness", "zh", id="if_then_pair_zh")
def _rule_if_then_zh(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    hit_if = _find_if_marker_zh(text)
    has_then = any(m in text for m in THEN_MARKERS)
    if hit_if and not has_then:
        # "若X，Y"是中文最常见的完整条件句形式，不需要"则/就/那么"承接
        # （英文侧 _rule_if_then_en 早已用逗号判主句，此处补齐对等）。
        # 只有逗号后确实还有实质主句才算完整；"若X，"逗号后空、或条件后无逗号 → 仍是残缺。
        tail = ""
        for sep in ("，", ","):
            idx = text.rfind(sep)
            if idx != -1:
                tail = text[idx + 1:].strip()
                break
        if len(tail) >= 4:
            return None
        return VerificationResult(
            VerificationStatus.FAIL,
            f"条件句不完整: 有'{hit_if}'但无结果承接词(那么/则/就)",
            {"has_if": True, "has_then": False, "marker": hit_if},
        )
    return None


@register_rule("completeness", "zh", id="concession_pair_zh")
def _rule_concession_zh(card: EvidenceCard, ctx: RuleContext):
    # 让步句完整性：有"虽然/尽管"但无"但是/却/仍"，说明主句被切掉了
    text = card.content.strip()
    has_concession = any(m in text for m in CONCESSION_MARKERS)
    has_contrast = any(m in text for m in CONTRAST_MARKERS)
    if has_concession and not has_contrast:
        return VerificationResult(
            VerificationStatus.FAIL,
            "让步句不完整: 有'虽然/尽管'但无'但是/却/仍'",
            {"has_concession": True, "has_contrast": False},
        )
    return None


# 英文条件句截断：块以条件从句开头、无逗号分隔主句、也没有结果承接词。
# "otherwise" 收进承接词——"Unless otherwise agreed." / "Where the context requires
# otherwise." 本身就是完整惯用句，不能判截断。
EN_CONDITION_LEAD_RE = re.compile(
    r'^(?:If|When|Unless|Where|Provided\s+that|In\s+the\s+event\s+that)\b', re.IGNORECASE
)
EN_CONDITION_RESOLVE_RE = re.compile(
    r'\b(?:then|shall|must|may|will|can|is|are|was|were|has|have|otherwise)\b',
    re.IGNORECASE,
)


# 词边界必须同时排除连字符：HTTP 头 "If-Match"/"If-None-Match" 里的 If 不是条件词
# （RFC 9110 实测 5 条标题被误判），因此左右都不许紧邻 - 或单词字符。
# Subject to / Notwithstanding：法律英语高频条件前导，词边界安全（不会在 "subject to change" 这种普通短语里
# 被误判为条件分句标记——跨边界回看时，这类短语所在的完整句在句末标点后自然截断，不会进入窗口）。
EN_CONDITION_MARKER_RE = re.compile(
    r'(?<![-\w])(?:If|When|Unless|Provided\s+that|In\s+the\s+event\s+that|Where'
    r'|Subject\s+to|Notwithstanding)(?![-\w])',
    re.IGNORECASE,
)


# 句中路径专用标记集：不含 When/Where——叙事文里句中 when/where 多为时间/地点副词
# （"imagine the surprise ... when that same company paid ..."），不是条件前导，
# 伯克希尔书信实测误伤。句中路径只保留 If/Unless/Provided that 等真条件标记。
EN_CONDITION_MID_RE = re.compile(
    r'(?<![-\w])(?:If|Unless|Provided\s+that|In\s+the\s+event\s+that'
    r'|Subject\s+to|Notwithstanding)(?![-\w])',
    re.IGNORECASE,
)


@register_rule("completeness", "en", id="if_then_pair_en")
def _rule_if_then_en(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    lead = EN_CONDITION_LEAD_RE.match(text)
    if lead and "," not in text and not EN_CONDITION_RESOLVE_RE.search(text):
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Incomplete conditional: starts with '{lead.group()}' but has no main clause "
            f"(no comma / then / result verb)",
            {"has_if": True, "has_then": False, "marker": lead.group()},
        )
    # 与中文 if_then_pair_zh 对齐：条件标记出现在块中间、但整块既没有逗号分隔主句、
    # 也没有任何结果动词/承接词 —— 说明主句被切到了别的块。
    if not lead:
        # 短块不检查：标题（"13.1.1. If-Match" / "13.2.1. When to Evaluate"）天然没有
        # 主句，RFC 9110 实测 6 条标题被误判。真实的条件从句截断不可能只有十几个字符。
        if len(text) < 40:
            return None
        marker = EN_CONDITION_MID_RE.search(text)
        if marker and "," not in text and not EN_CONDITION_RESOLVE_RE.search(text):
            return VerificationResult(
                VerificationStatus.FAIL,
                f"Incomplete conditional: contains '{marker.group()}' but neither the main "
                f"clause separator (comma) nor any result verb",
                {"has_if": True, "has_then": False, "marker": marker.group()},
            )
    return None


EN_CONCESSION_LEAD_RE = re.compile(
    r'^(?:Although|Though|While|Even\s+if|Even\s+though)\b', re.IGNORECASE
)
EN_CONTRAST_RE = re.compile(r'\b(?:but|however|nevertheless|yet|still)\b', re.IGNORECASE)


@register_rule("completeness", "en", id="concession_pair_en")
def _rule_concession_en(card: EvidenceCard, ctx: RuleContext):
    text = card.content.strip()
    lead = EN_CONCESSION_LEAD_RE.match(text)
    if lead and "," not in text and not EN_CONTRAST_RE.search(text):
        return VerificationResult(
            VerificationStatus.FAIL,
            f"Incomplete concession: starts with '{lead.group()}' but the main clause "
            f"(but/however/yet) is missing",
            {"has_concession": True, "has_contrast": False},
        )
    return None


def verify_completeness(card: EvidenceCard, ctx: RuleContext | None = None) -> VerificationResult:
    ctx = ctx or RuleContext(lang=detect_language(card.content))
    for rule in rules_for("completeness", ctx.lang):
        result = rule.fn(card, ctx)
        if result is not None:
            return result
    return VerificationResult(
        VerificationStatus.PASS,
        "Completeness check passed" if ctx.lang == "en" else "完整性检查通过")


# ---------------------------------------------------------------- 可追溯性

def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


def verify_traceability(card: EvidenceCard, source_text: str,
                        mode: str = "exact", normalizer=None) -> VerificationResult:
    """内容能否回到源文本精确定位。

    mode:
      exact      —— 逐字子串（默认，最严格）
      normalized —— 去空白后子串（适用于解析器做过换行/空格规范化的场景）
      custom     —— 调用方传入 normalizer(text) -> str

    别人用自家解析器时，文本规范化差异不该被判 fail——用 normalized/custom。
    """
    lang = detect_language(card.content)
    if not source_text:
        return VerificationResult(
            VerificationStatus.UNCERTAIN,
            "No source text provided" if lang == "en" else "无源文本")

    text = card.content.strip()
    if mode == "exact":
        ok = text in source_text
    elif mode == "normalized":
        ok = _collapse_ws(text) in _collapse_ws(source_text)
    elif mode == "custom":
        if normalizer is None:
            raise ValueError("traceability mode='custom' 需要传入 normalizer 函数")
        ok = normalizer(text) in normalizer(source_text)
    else:
        raise ValueError(f"未知 traceability mode: {mode}（可选: exact / normalized / custom）")

    if not ok:
        return VerificationResult(
            VerificationStatus.FAIL,
            ("Content cannot be located verbatim in the source text" if lang == "en"
             else "内容无法在源文本中精确定位"),
            {"locator": card.locator, "mode": mode},
        )

    return VerificationResult(
        VerificationStatus.PASS,
        "Traceability check passed" if lang == "en" else "可追溯性检查通过")


# ---------------------------------------------------------------- 总入口

ALL_CHECKS = (
    "independence",
    "completeness",
    "traceability",
    "semantic_independence",
    "template_completeness",
)

# 默认用内置 LLM 客户端（仓库内脚本行为不变）；显式传 None 表示跳过语义层。
_USE_BUILTIN_LLM = object()


def verify_chunk(card: EvidenceCard, source_text: str = "", prev_chunk: str = "",
                 full_doc: str = "", *, checks=None,
                 traceability_mode: str = "exact", traceability_normalizer=None,
                 llm_callback=_USE_BUILTIN_LLM) -> dict:
    """对单个块跑五关检查。

    checks 可裁剪要跑的检查项（如只要规则层，不花 LLM 钱）。
    llm_callback：签名 prompt -> str；默认用内置 OpenAI 兼容客户端，
    显式传 None 则语义层返回 uncertain（不隐式绑定任何供应商）。
    """
    selected = list(ALL_CHECKS) if checks is None else list(checks)
    unknown = set(selected) - set(ALL_CHECKS)
    if unknown:
        raise ValueError(f"未知检查项: {sorted(unknown)}（可选: {', '.join(ALL_CHECKS)}）")

    ctx = RuleContext(lang=detect_language(card.content), prev_chunk=prev_chunk,
                      source_text=source_text, full_doc=full_doc)
    results = {}

    if "independence" in selected:
        results["independence"] = verify_independence(card, ctx)
    if "completeness" in selected:
        results["completeness"] = verify_completeness(card, ctx)
    if "traceability" in selected:
        results["traceability"] = verify_traceability(
            card, source_text, mode=traceability_mode, normalizer=traceability_normalizer
        )
    if "semantic_independence" in selected:
        # None = 调用方明确不注入 LLM → 规则层照跑，LLM 层跳过；
        # 默认哨兵 = 仓库内置客户端；其他 = 调用方注入的回调。
        if llm_callback is None:
            cb = SKIP_LLM
        elif llm_callback is _USE_BUILTIN_LLM:
            cb = None
        else:
            cb = llm_callback
        results["semantic_independence"] = verify_semantic_independence(
            card, prev_chunk, section_text=source_text, full_doc=full_doc,
            llm_callback=cb,
        )
    if "template_completeness" in selected:
        results["template_completeness"] = verify_template_completeness(card, ctx)

    overall = VerificationStatus.PASS
    for r in results.values():
        if r.status == VerificationStatus.FAIL:
            overall = VerificationStatus.FAIL
            break
        elif r.status == VerificationStatus.UNCERTAIN:
            overall = VerificationStatus.UNCERTAIN

    return {
        "overall": overall,
        "checks": {k: {"status": v.status.value, "reason": v.reason} for k, v in results.items()},
    }
