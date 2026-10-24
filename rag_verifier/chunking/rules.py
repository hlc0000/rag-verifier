"""规则注册表：按检查项 + 语种分组，内置规则与第三方规则走同一接口。

语种由 detect_language 按 CJK 字符占比自动判定（zh / en）；
lang="any" 的规则对所有语种生效。规则按注册顺序执行，先命中先返回。

第三方扩展示例（不改本仓库任何文件）：

    from src.chunking.rules import register_rule

    @register_rule("completeness", lang="en", id="no_currency_unit")
    def my_rule(card, ctx):
        # 返回 VerificationResult 表示命中，返回 None 表示放行
        ...
        return None
"""
import re
from dataclasses import dataclass
from typing import Callable

from src.models import EvidenceCard

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

CHECKS = (
    "independence",
    "completeness",
    "traceability",
    "semantic_independence",
    "template_completeness",
)


def detect_language(text: str) -> str:
    """按 CJK 字符占比判定语种：非空白字符中 CJK 占比 ≥15% 判 zh，否则 en。

    纯符号/数字块（如 "3.1"）判为 en——中英规则对这类块都不命中，
    不影响判定结果。
    """
    if not text:
        return "en"
    cjk = len(CJK_RE.findall(text))
    nonspace = len(re.sub(r"\s", "", text))
    if nonspace and cjk / nonspace >= 0.15:
        return "zh"
    return "en"


@dataclass
class RuleContext:
    """规则执行上下文：语种 + 前一块 + 源文本 + 全文。"""

    lang: str
    prev_chunk: str = ""
    source_text: str = ""
    full_doc: str = ""


@dataclass
class Rule:
    id: str
    check: str
    lang: str  # zh / en / any
    fn: Callable[[EvidenceCard, RuleContext], object]


_RULES: list[Rule] = []


def register_rule(check: str, lang: str = "any", id: str | None = None):
    """注册规则。check 必须是 CHECKS 之一；lang 为 zh / en / any。"""
    if check not in CHECKS:
        raise ValueError(f"未知检查项: {check}（可选: {', '.join(CHECKS)}）")
    if lang not in ("zh", "en", "any"):
        raise ValueError(f"未知语种: {lang}（可选: zh / en / any）")

    def deco(fn):
        _RULES.append(Rule(id=id or fn.__name__, check=check, lang=lang, fn=fn))
        return fn

    return deco


def rules_for(check: str, lang: str) -> list[Rule]:
    """取出某检查项在当前语种下生效的规则（保持注册顺序）。"""
    return [r for r in _RULES if r.check == check and r.lang in ("any", lang)]


def list_rules() -> list[dict]:
    """规则清单（调试/文档用途）。"""
    return [{"id": r.id, "check": r.check, "lang": r.lang} for r in _RULES]
