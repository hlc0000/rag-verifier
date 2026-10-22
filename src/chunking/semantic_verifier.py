import os
import re
import ssl
import json
import bisect
import urllib.request
import urllib.error
from pathlib import Path

from src.models import EvidenceCard, VerificationResult, VerificationStatus
from src.chunking.rules import RuleContext, detect_language, register_rule, rules_for


def _load_dotenv() -> None:
    """从项目根目录 .env 加载配置（不覆盖已有环境变量）。"""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# LLM 配置：默认走 DeepSeek 的 OpenAI 兼容端点。
# 换供应商只需改环境变量（LLM_BASE_URL / LLM_MODEL / LLM_API_KEY），不必改代码。
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

ANAPHORA_MARKERS = ["该", "这", "其", "此", "上述", "前述", "该等", "这些", "那些"]

# 构词豁免："此"在"因此/由此/据此/为此"等双字词中是构词成分，不是独立指代词。
# 只豁免实际观察到的构词模式，不为想象中的罕见用例扩展列表。
ANAPHORA_COMPOUND_PREV = {
    "此": "因由据为特至从彼就如以除",
}
CONDITION_MARKERS = ["如果", "假如", "倘若", "若", "如", "当", "除非", "因", "由于"]
CLAUSE_PATTERN = re.compile(r'^第[一二三四五六七八九十\d]+条')


def _ssl_context() -> ssl.SSLContext:
    """macOS 自带 Python 常缺 CA 链，逐个回退到可用的 CA bundle。"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    for cafile in ("/etc/ssl/cert.pem", "/usr/local/etc/openssl/cert.pem"):
        if os.path.exists(cafile):
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def has_unresolved_anaphora(text: str) -> tuple[bool, str]:
    for marker in ANAPHORA_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            # 构词成分（如"因此"中的"此"）不是指代词，跳过继续找下一个
            if idx > 0 and text[idx - 1] in ANAPHORA_COMPOUND_PREV.get(marker, ""):
                start = idx + 1
                continue
            after = text[idx + len(marker):idx + len(marker) + 5]
            before = text[:idx]
            if not after or after[0] in "。，、；：":
                return True, f"指代词'{marker}'后无明确对象"
            if not before.strip():
                return True, f"指代词'{marker}'无前文指代对象"
            break  # 首个独立出现的指代词有明确对象即通过，与原判定口径一致
    return False, ""


def has_missing_subject(text: str) -> tuple[bool, str]:
    first_char = text.strip()[0] if text.strip() else ""
    if first_char in "应须就便可将":
        if not any(m in text for m in ["甲方", "乙方", "双方", "任何一方", "当事人"]):
            return True, "缺主语且以动词/助动词开头"
    return False, ""


def has_dangling_condition(text: str) -> tuple[bool, str]:
    """检查悬空结果句：以承接词开头，但触发条件留在前文。

    '则' 与 '应/须/就/便/可/将' 语义不同：前者承接的是前文的条件，
    后者缺的是主语。归因不同，恢复策略也不同（补条件 vs 补主语）。
    """
    first_char = text.strip()[0] if text.strip() else ""
    if first_char == "则":
        return True, "以承接词'则'开头，触发条件在前文"
    return False, ""


# 前文是"未配对的条件从句" → 块是孤立主句（"若…主句" 被切成两块）。
# 该构造语义层会摇摆，故下沉到规则层做确定性字符串判定。
CONDITION_LEAD = ["如果", "假如", "倘若", "若", "除非"]
CONDITION_RESOLVE = ["那么", "则", "就", "便", "所以", "因此", "故", "从而", "于是"]

# 全文≤该长度时语义层直接用全文做上下文：小文档全文成本可忽略（<0.01元/块），
# 没必要走"章节先行+全文兜底"两段式——两段式是为大文档省 token 设计的。
FULL_DOC_DIRECT_MAX = 8000


def has_cut_condition_in_prev(prev_chunk: str) -> tuple[bool, str]:
    """块自身句义完整，但其触发条件从句留在前文、且未与主句配对。

    如块"甲方有权要求赔偿。"、前文"若乙方违反保密义务"——条件被切走，块是孤立主句。
    """
    prev = (prev_chunk or "").strip()
    if not prev:
        return False, ""
    if not any(prev.startswith(lead) for lead in CONDITION_LEAD):
        return False, ""
    if any(res in prev for res in CONDITION_RESOLVE):
        return False, ""
    return True, f"触发条件从句'{prev[:24]}'留在前文，块是孤立主句"


def _build_prompt_en(text: str, context: str = "") -> str:
    """英文版块独立性判定 prompt（与中文版规则等价，按英文合同的措辞习惯表述）。"""
    ctx = context.strip() if context and context.strip() else "(empty)"
    return f"""You are a quality auditor for RAG evidence chunks. Below are an EVIDENCE CHUNK cut out of a
long document, and the REFERENCE CONTEXT extracted from the same document (the whole document, or the
text surrounding the chunk).

[REFERENCE CONTEXT]
{ctx}

[EVIDENCE CHUNK TO AUDIT]
{text}

Decide: when this chunk is retrieved and quoted on its own, can it be understood **independently and
unambiguously without the REFERENCE CONTEXT**?

Rules:
1. Substantive references **internal to this document** must be resolvable. If the chunk refers to a
   specific value that is necessary to execute the provision — an amount, a percentage, a time period,
   a standard, a method, a triggering condition, a defined term — that concrete value must actually be
   locatable in the REFERENCE CONTEXT. Typical substantive references include: "as agreed", "as agreed
   herein", "as set forth in this Contract", "the amount agreed by the parties", "the aforementioned
   payment", "such fees". If those references resolve to a **blank / unfilled slot** (e.g. "The
   custody fee shall be calculated as: .") or if the concrete value **cannot be found anywhere in the
   document**, judge independent=false and state in `reason` where the reference resolved to.
   Note in particular: "as agreed" and "the agreed X" are **not** acceptable generic designations —
   they point to a concrete value and must **not** be waved through as "it points to another clause"
   or "it is an acceptable reference".
   The following are **not** substantive references and must **not** make the chunk fail:
   (a) references to **externally available carriers**: exhibit/appendix numbers (e.g. "Exhibit A",
       "see Appendix B", "as provided in Exhibit A"), external laws and regulations, national
       standards — their content is not expected to appear inside this document;
   (b) terms that **only limit scope and do not affect understanding of the provision**, e.g. "within
       the agreed custody period, Party B may not request early collection" — the reader does not need
       the concrete period to understand the rule; judge pass;
   (c) the **bare designation "this Contract / this Agreement"** (e.g. "This Contract takes effect upon
       signature by both parties") is self-referential and acceptable; only phrases such as "the X
       **agreed in** this Contract" that point to a concrete value count as substantive references.
2. Ordinary designations are not substantive references: "Party A", "Party B", "the parties", "either
   party", "this Contract/this Agreement", clause numbers and similar pure designations or
   self-references are acceptable; do **not** fail a chunk merely because they are not defined inside
   it. This whitelist does **not** include "as agreed", "the agreed X", "the aforementioned X",
   "such X", "the said X" — those point to concrete content.
3. Required elements that were cut off: only when a required element **actually exists** in the
   REFERENCE CONTEXT (e.g. a triggering condition clause introduced by "if / when / where / unless /
   provided that", or an elided subject) and the chunk cuts it away so that the meaning can only be
   determined by looking back, judge independent=false. If that element does **not** exist in the
   REFERENCE CONTEXT, you must not fail the chunk on that ground.
4. Do not invent missing elements when the context is not comparable: if the REFERENCE CONTEXT is
   empty, or is essentially the same as the chunk itself (i.e. there is no source text to compare
   against), you must **not** fail the chunk for "missing triggering condition", "missing scope",
   "missing concrete value" or "an element was cut off" — with no comparable source, there is no proof
   that anything was cut. In that case fail only if the chunk is **explicitly defective on its own**
   (e.g. starts with a connective such as "Therefore / However / But / Accordingly", has a blank after
   "i.e.,", contains only a condition without a result, or is not a complete sentence — a single
   word/fragment, or clearly truncated at its start or end); otherwise judge independent=true.
5. Never fabricate conditions or premises that are absent from the REFERENCE CONTEXT.
6. The REFERENCE CONTEXT is usually just the **section** containing the chunk, not the whole document.
   If your **only or main** reason for independent=false is "the concrete value referenced by the chunk
   cannot be located in the REFERENCE CONTEXT", set missing_in_context=true and put the unresolvable
   target in `ref_target`: for numbered items give the number (e.g. "Section 8", "Appendix A",
   "Table 3"); for unnumbered ones give the shortest distinctive phrase (at most 15 words, e.g.
   "custody fee calculation method") — the driver will locate that target in the full document and
   re-judge, so do not create false positives.
   When failing for other reasons (chunk itself defective, starts with a connective, contradicts the
   context, etc.), set missing_in_context=false and leave ref_target empty.

Output JSON only, no other text, in this format (write `reason` in English):
{{"independent": true or false, "reason": "if false, state which part of the REFERENCE CONTEXT must be looked at",
  "missing_in_context": true or false,
  "ref_target": "reference target when missing_in_context=true, otherwise empty string"}}"""


def _build_prompt(text: str, context: str = "") -> str:
    ctx = context.strip() if context and context.strip() else "（空）"
    if detect_language(text + " " + context[:200]) == "en":
        return _build_prompt_en(text, context)
    return f"""你是一个 RAG 证据块质量审查员。下面是从一份长文档中切出的【证据块】，
以及从原文中提取的【参考上下文】（可能是整篇原文，也可能是该块所在段落的前后文）。

【参考上下文】
{ctx}

【待审证据块】
{text}

请判断：该块被单独检索引用时，能否**不借助【参考上下文】独立、无歧义地理解**？

判定规则：
1. **本文件内**的实质引用必须可解析：若块内出现对**执行该条款所必需的具体值**的引用——金额、比例、
   期限数值、标准、方式、触发条件、被定义的名词等，必须能在【参考上下文】中**实际定位到该具体值**。
   典型的**具体值引用**包括："按照约定""按照本合同约定""本合同约定的X""双方约定的其他X"，
   以及"前述款项""该费用"。若这些引用在上下文中定位到的是**空位/未填写**
   （如"保管费计算方式为：。"）或**在文档中找不到该具体值**，判 independent=false，
   并在 reason 中指明该引用解析到了哪里。
   特别注意："按照约定""约定的X"**不属于**可接受的常规称谓——它们引用的是具体值，
   **不得**以"指向合同其他条款""属于可接受引用"为由放行。
   但下列情形**不算**实质引用，**不得**因此判 false：
   （a）**指向外部可查阅载体**的引用：附件/附录编号（如"附件A""见附件B""按照附件A执行"）、
        外部法律法规、国家标准等——它们的内容本来就不在本文件内、不期望出现在上下文中；
   （b）**仅限定范围、不影响条款理解的术语**：如"在约定保管期限内，乙方不得请求甲方提前领取
        保管物"——读者无须知道具体期限即可理解该规定，判 pass；
   （c）**裸称谓"本合同/本协议"**（如"本合同自双方签字盖章之日起生效"）属自指，可接受；
        只有"本合同**约定的**X"这类**指向具体值**的才按实质引用处理。
2. 常规称谓不算实质引用：甲方、乙方、双方、任何一方、本合同/本协议、条款编号等
   纯称谓/自指，属于合同文本可接受的引用，**不要**仅因其"未在块内定义"就判不自足。
   但本条白名单**不包含**"按照约定""约定的X""前述/上述X""该X"等指向具体内容的引用。
3. 被切掉的必需成分：只有当【参考上下文】中**确实存在**某必需成分（如触发条件从句——由
   "如果/若/假如/倘若/当/除非"引导的条件，或"被省略的主语"），而该块把它切掉、句义必须回看
   上下文才能确定时，判 independent=false。
   若【参考上下文】中并不存在该成分，不得据此判 false。
4. 上下文不可比对时不得凭空判缺：若【参考上下文】为空、或与块本身基本相同（即没有可供比对的原文），
   **不得**以"缺少触发条件""缺少赔偿范围""缺少具体值""某成分被切掉"等为由判 false——
   没有可比对的原文，就无从证明这些成分"被切掉"。此时只在块**自身**出现显式残缺
   （如以"因此/但是/然而"等承接词开头、"即，"后空白、只有条件没有结果、块自身不成
   完整句——单字/单词碎片、以明显被截断的词语开头或结尾，如"司债券…"）时才判 false；
   否则判 independent=true。
5. 不要臆造【参考上下文】中并不存在的条件或前提。
6. 【参考上下文】通常只是该块**所在章节**，不是全文。若你判 independent=false 的**唯一或主要理由**
   是"块引用的具体值在【参考上下文】中无法定位"，请设 missing_in_context=true，并在 ref_target
   中给出你定位不到的引用目标：编号类写编号（如"第8条""附录A""表3"），无编号的写最短特征短语
   （≤15字，如"保管费计算方式"）——驱动层会按该目标去全文**定向取出对应内容**重判，不能误伤。
   因其他理由（块自身残缺、以承接词开头、与上下文矛盾等）判 false 时设 missing_in_context=false、
   ref_target 留空。

只输出 JSON，不要任何其他文字，格式：
{{"independent": true 或 false, "reason": "若 false，指明需要回看【参考上下文】的哪一部分",
  "missing_in_context": true 或 false,
  "ref_target": "missing_in_context=true 时填引用目标，否则空字符串"}}"""


def _post_chat(prompt: str) -> str:
    """调用 OpenAI 兼容的 chat/completions 端点，返回首个 choice 的文本内容。

    配置每次调用时从环境变量读取，方便调用方在进程运行期（如 UI 里填写）切换 key/model。
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    base_url = os.environ.get("LLM_BASE_URL") or LLM_BASE_URL
    model = os.environ.get("LLM_MODEL") or LLM_MODEL
    if not api_key:
        raise RuntimeError("未配置 LLM_API_KEY / DEEPSEEK_API_KEY")
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


# 对外别名：外部接入方可显式传入 builtin_llm 启用语义层，不传则语义层跳过
builtin_llm = _post_chat

# 哨兵：规则层跑完后不调 LLM（对外 API 未注入回调时使用）
SKIP_LLM = object()


def llm_independence_check(text: str, context: str = "", llm_callback=None) -> dict:
    """
    用 LLM 判断块能否脱离上下文独立理解。

    llm_callback：签名 prompt -> str；不传则用内置 OpenAI 兼容客户端（_post_chat）。
    外部接入方传自己的模型函数即可，不绑定供应商。

    返回约定：
      {"independent": bool, "reason": str,
       "missing_in_context": bool}                LLM 判定成功（missing_in_context=true 表示
                                                  false 理由是"引用值在上下文中定位不到"）
      {"llm_unavailable": True, "reason": str}    调用/解析失败（绝不伪装成通过）
    """
    prompt = _build_prompt(text, context)
    call = llm_callback or _post_chat

    try:
        content = call(prompt)
    except urllib.error.HTTPError as e:
        return {"llm_unavailable": True, "reason": f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}"}
    except Exception as e:
        return {"llm_unavailable": True, "reason": f"LLM调用失败: {type(e).__name__}: {e}"}

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        return {"llm_unavailable": True, "reason": f"无法解析LLM输出: {content[:200]}"}

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"llm_unavailable": True, "reason": f"JSON解析失败: {e}"}

    if "independent" not in parsed:
        return {"llm_unavailable": True, "reason": f"LLM输出缺少 independent 字段: {parsed}"}

    return {"independent": bool(parsed["independent"]), "reason": parsed.get("reason", ""),
            "missing_in_context": bool(parsed.get("missing_in_context", False)),
            "ref_target": str(parsed.get("ref_target", "") or "")}


def _locate_ref_window(full_doc: str, ref_target: str, exclude_text: str = "", window: int = 1200) -> str:
    """按 LLM 给出的引用目标在全文里定向定位，截取目标附近的窗口文本。

    定位三级：直接子串 → 字符间容忍任意空白（PDF 抽取的"第 8 条"常带空格）→ 失败返回空，
    由调用方退化为全文兜底（罕见路径，不是常态）。

    必须排除"引用点"（块自己提到"第8条"的那个位置，内容与上下文重复），取上下文之外
    的"定义点"——否则窗口取到的是引用处而非被引内容，重判必然失败（实测踩过的坑）。
    """
    ref = (ref_target or "").strip().strip('"“”\'')
    if not ref or not full_doc:
        return ""
    positions = []
    start = 0
    while True:
        i = full_doc.find(ref, start)
        if i == -1:
            break
        positions.append((i, i + len(ref)))
        start = i + 1
    if not positions and len(ref) <= 20:
        loose = re.compile(r"\s*".join(re.escape(ch) for ch in ref))
        positions = [(m.start(), m.end()) for m in loose.finditer(full_doc)]

    # 定位章节在全文中的区间（去空白后找子串，容忍两处的空白差异），
    # 落在区间内的候选即引用点，跳过——与章节长短无关。
    sec_lo = sec_hi = -1
    if exclude_text:
        chars, idx_map = [], []
        for i, ch in enumerate(full_doc):
            if not ch.isspace():
                chars.append(ch)
                idx_map.append(i)
        sq_full = "".join(chars)
        sq_section = re.sub(r"\s+", "", exclude_text)
        k = sq_full.find(sq_section[:60])
        if k != -1:
            sec_lo, sec_hi = k, k + len(sq_section)
        for s, e in positions:
            sq_s = bisect.bisect_left(idx_map, s)
            if sec_lo != -1 and sec_lo <= sq_s < sec_hi:
                continue  # 引用点：落在块所在章节内，跳过
            w0 = max(0, s - window)
            return full_doc[w0:e + window].strip()
        return ""
    for s, e in positions:
        w0 = max(0, s - window)
        return full_doc[w0:e + window].strip()
    return ""


# ---------------------------------------------------------------- 规则层（语种分套 + 可注册）

# 英文指代：块以 Such / Said / The foregoing / The same / The aforementioned 等开头，指代对象在前文。
# 不把裸 "This" 收进来——"This Agreement shall…" 是自指的合法写法。
EN_ANAPHORA_RE = re.compile(
    r'^(?:Such|Said|The\s+foregoing|The\s+same|The\s+aforementioned|The\s+above-mentioned|'
    r'The\s+preceding|The\s+said)\b',
    re.IGNORECASE,
)
# 英文缺主语：条款被截掉主语后以情态动词/助动词开头（"shall pay…""will notify…"）。
# "May" 不收——避免把 "May 2024" 这类月份当情态动词；"Can" 只在句首大写时成立。
EN_MISSING_SUBJECT_RE = re.compile(r'^(?:Shall|Must|Will|Should|Hereby)\b')
EN_CONDITION_LEAD_RE = re.compile(
    r'^(?:If|When|Unless|Provided\s+that|In\s+the\s+event\s+that)\b', re.IGNORECASE
)
EN_CONDITION_RESOLVE_RE = re.compile(r'\b(?:then|otherwise|shall|must|may|will|can)\b', re.IGNORECASE)
# 英文悬空结果句：块以承接词开头，触发条件留在了前文。
# 中文靠单字「则」判定；英文没有这种单字标记，只能用词边界 + 首词匹配。
# Therefore/Thus/However/But 已由完整性检查的依赖连接词规则覆盖，这里只补它没覆盖的承接词，
# 避免同一个块报两条同义失败。
EN_DANGLING_RE = re.compile(
    r'^(?:Then|In\s+(?:that|this|such)\s+case|Accordingly|Consequently|Hence)\b',
    re.IGNORECASE,
)


@register_rule("semantic_independence", "zh", id="anaphora_zh")
def _rule_anaphora_zh(card: EvidenceCard, ctx: RuleContext):
    bad, reason = has_unresolved_anaphora(card.content.strip())
    if bad:
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "zh", id="missing_subject_zh")
def _rule_missing_subject_zh(card: EvidenceCard, ctx: RuleContext):
    bad, reason = has_missing_subject(card.content.strip())
    if bad:
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "zh", id="dangling_condition_zh")
def _rule_dangling_zh(card: EvidenceCard, ctx: RuleContext):
    bad, reason = has_dangling_condition(card.content.strip())
    if bad:
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "zh", id="cut_condition_prev_zh")
def _rule_cut_condition_zh(card: EvidenceCard, ctx: RuleContext):
    bad, reason = has_cut_condition_in_prev(ctx.prev_chunk)
    if bad:
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "en", id="anaphora_en")
def _rule_anaphora_en(card: EvidenceCard, ctx: RuleContext):
    match = EN_ANAPHORA_RE.match(card.content.strip())
    if match:
        reason = (f"Unresolved reference '{match.group()}' at chunk start: "
                  f"the referent lives in a previous chunk")
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "en", id="missing_subject_en")
def _rule_missing_subject_en(card: EvidenceCard, ctx: RuleContext):
    match = EN_MISSING_SUBJECT_RE.match(card.content.strip())
    if match:
        reason = f"Missing subject: chunk starts with bare modal '{match.group()}'"
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "en", id="dangling_condition_en")
def _rule_dangling_en(card: EvidenceCard, ctx: RuleContext):
    match = EN_DANGLING_RE.match(card.content.strip())
    if match:
        reason = (f"Chunk starts with the connective '{match.group()}', "
                  f"so its triggering condition lives in a previous chunk")
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


@register_rule("semantic_independence", "en", id="cut_condition_prev_en")
def _rule_cut_condition_en(card: EvidenceCard, ctx: RuleContext):
    prev = (ctx.prev_chunk or "").strip()
    if not prev:
        return None
    lead = EN_CONDITION_LEAD_RE.match(prev)
    if lead and "," not in prev and not EN_CONDITION_RESOLVE_RE.search(prev):
        reason = (f"Triggering condition '{prev[:40]}…' is left in the previous chunk; "
                  f"this chunk is an orphan main clause")
        return VerificationResult(VerificationStatus.FAIL, reason, {"issues": [reason]})
    return None


def verify_semantic_independence(card: EvidenceCard, prev_chunk: str = "",
                                 section_text: str = "", full_doc: str = "",
                                 llm_callback=None) -> VerificationResult:
    text = card.content.strip()
    ctx = RuleContext(lang=detect_language(text), prev_chunk=prev_chunk,
                      source_text=section_text, full_doc=full_doc)

    issues = []
    for rule in rules_for("semantic_independence", ctx.lang):
        result = rule.fn(card, ctx)
        if result is not None:
            issues.extend(result.details.get("issues", [result.reason]))

    # 如果规则已判定失败，跳过LLM
    if issues:
        return VerificationResult(VerificationStatus.FAIL, "; ".join(issues), {"issues": issues})

    # 未注入回调：规则层已跑完，LLM 层不做判定（绝不伪装成 pass）
    if llm_callback is SKIP_LLM:
        reason = (
            "No LLM callback provided; semantic layer skipped (rule layer passed)"
            if ctx.lang == "en"
            else "未提供 LLM 回调，语义层未执行（规则层已通过）"
        )
        return VerificationResult(
            VerificationStatus.UNCERTAIN,
            reason,
            {"llm_skipped": True},
        )

    # 两段式上下文：小文档直接用全文（成本可忽略，换最大判定上下文）；
    # 大文档先所在章节，仅当 fail 理由是"引用值在章节中定位不到"时才用全文重判——
    # 跨章节引用的值定义在别处，只传章节会误伤（宁可漏不可误伤）。
    if full_doc and len(full_doc) <= FULL_DOC_DIRECT_MAX:
        context = full_doc
    else:
        context = section_text or full_doc or prev_chunk
    result = llm_independence_check(text, context, llm_callback=llm_callback)

    # LLM 不可用时降级为 UNCERTAIN —— 绝不能伪装成 PASS
    if result.get("llm_unavailable"):
        return VerificationResult(
            VerificationStatus.UNCERTAIN,
            (f"LLM unavailable; the semantic layer cannot decide: {result.get('reason', '')}"
             if ctx.lang == "en" else
             f"LLM不可用，语义层无法判定: {result.get('reason', '')}"),
            {"llm_unavailable": True, "reason": result.get("reason", "")},
        )

    if not result.get("independent", True):
        # 定向兜底：fail 理由是"引用值定位不到"时，按 LLM 给出的 ref_target 去全文定向取
        # 目标内容重判（缺什么取什么，不灌全文）；定位失败才退化为全文（罕见路径）。
        # 其余 fail 理由（自身残缺/承接词开头/与章节矛盾）换上下文也不会翻转，直接判 fail。
        if result.get("missing_in_context") and full_doc and full_doc != context:
            ref = result.get("ref_target", "")
            target_text = _locate_ref_window(full_doc, ref, exclude_text=context)
            if target_text:
                retry_context = (
                    context + f"\n\n[Content elsewhere in the document for the reference target "
                              f"\"{ref}\"]\n" + target_text if ctx.lang == "en" else
                    context + f"\n\n【引用目标“{ref}”在文档其他位置的内容】\n" + target_text)
                fallback_kind = "targeted_section"
            else:
                retry_context = full_doc
                fallback_kind = "full_doc"
            retry = llm_independence_check(text, retry_context, llm_callback=llm_callback)
            if retry.get("llm_unavailable"):
                return VerificationResult(
                    VerificationStatus.UNCERTAIN,
                    (f"LLM unavailable; targeted fallback could not decide: {retry.get('reason', '')}"
                     if ctx.lang == "en" else
                     f"LLM不可用，定向兜底无法判定: {retry.get('reason', '')}"),
                    {"llm_unavailable": True, "reason": retry.get("reason", "")},
                )
            if retry.get("independent", True):
                # 第一轮对"良性引用"（如仅限定范围的术语）判得过严：值补上后翻案为 pass。
                # 保留这次重判就是为了防这种误伤（宁可漏不可误伤），不是为跨块值引用翻案。
                return VerificationResult(
                    VerificationStatus.PASS,
                    (f"Reference '{ref}' not found in the section; passed after re-judging "
                     f"with the targeted content" if ctx.lang == "en" else
                     f"章节内未定位引用'{ref}'，经定向取目标内容重判通过"),
                    {"fallback": fallback_kind, "ref_target": ref,
                     "section_reason": result.get("reason", "")},
                )
            # 用户拍板口径（2026-10-17）：跨块值引用=不自足=fail（"按第8条费率"答非所问）。
            # 归因二分：值在文档其他位置定位到 → cross_block_reference（恢复=跨块召回）；
            # 定位不到（文档里真没有/空位）→ missing_referent（恢复=重新检索）。两者恢复策略不同，不可混。
            return VerificationResult(
                VerificationStatus.FAIL,
                (retry.get("reason") or
                 ("LLM judged the chunk not independent" if ctx.lang == "en"
                  else "LLM判定语义不独立")),
                {"llm_reason": retry.get("reason"), "fallback": fallback_kind, "ref_target": ref,
                 "failure_type": "cross_block_reference" if target_text else "missing_referent"},
            )
        return VerificationResult(
            VerificationStatus.FAIL,
            (result.get("reason") or
             ("LLM judged the chunk not independent" if ctx.lang == "en"
              else "LLM判定语义不独立")),
            {"llm_reason": result.get("reason")},
        )

    return VerificationResult(
        VerificationStatus.PASS,
        "Semantic independence check passed" if ctx.lang == "en" else "语义独立性检查通过")
