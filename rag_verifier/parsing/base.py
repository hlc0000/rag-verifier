"""解析器接口与注册表：接口统一、解析器可插拔。

新增解析器 = 写一个 Parser 子类并加 @register，切分层零改动。
engine="auto" 时的选择顺序：docling（布局识别强）→ pymupdf（轻量、已验证）。

失败处理哲学：解析失败是常态，不是异常。
- 格式不支持 → 入口白名单直接拒绝（fail-fast，不静默降级读纯文本）
- 解析抛异常 → 归一化为 ParseError 抛出（结构化、可归因），不甩引擎 traceback
- 解析成功但质量差 → build_quality_report() 挂 metadata["quality_report"]，
  只产出结构化信息（passed/severity/issues），不阻断、不写用户文案——
  文案与交互是产品层的事，库的职责是把结构化信息传出去
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path

from src.parsing.models import StructuredDocument
from src.parsing.quality import QualityReport, build_quality_report

# auto 模式的优先级：布局识别强的在前，缺依赖时自然落到下一个
AUTO_ORDER = ("docling", "pymupdf")

# 支持格式白名单（2026-10 决策：聚焦主战场，做深而非做广）。
# 不在名单内的一律拒绝，并给出可操作的下一步——
# 宁可明确拒绝，也不让低质量解析结果蒙混进下游。
SUPPORTED_FORMATS = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
}

# 结构明写的格式走 AST 直读：markdown 的 `#`、HTML 的 h1-h6 把结构写在字符里，
# 视觉模型反而会把明写的结构弄丢（实测：docling 解析 md 丢三级标题文本、层级全塌）。
# 仅 engine="auto" 时生效；显式 --engine 始终优先。
FORMAT_ENGINE = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
}

# 曾经支持、因实测局限明确下线的格式（拒绝时说明原因，而不是当成"未知格式"）
DROPPED_FORMATS = {
    ".rtf": "RTF 为低频遗留格式，标题常只靠字号区分、无样式信息时层级必丢，已于 2026-10 停止支持",
    ".pptx": "PPTX 属谨慎支持档，本轮聚焦 PDF/DOCX/MD/TXT/HTML 后暂时下线",
    ".xlsx": "XLSX 属谨慎支持档（系统导出件公式缓存为空会丢列），本轮聚焦后暂时下线",
}


class ParseError(Exception):
    """结构化解析失败（库的对外接口）：抛出而非返回，调用方按 code 分流处理。

    code:    unsupported_format | encrypted | corrupted | timeout | unknown（机器可读）
    message: 人类可读的事实陈述（不面向最终用户，不含操作建议）
    detail:  技术细节（内部记录用）
    """

    def __init__(self, code: str, message: str, source: str = "", detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.source = source
        self.detail = detail

    def to_dict(self) -> dict:
        return {"ok": False, "code": self.code, "message": self.message,
                "source": self.source, "detail": self.detail}

_REGISTRY: dict[str, type["Parser"]] = {}


def register(cls: type["Parser"]) -> type["Parser"]:
    _REGISTRY[cls.name] = cls
    return cls


def available_engines() -> list[str]:
    import importlib.util

    # 触发内置解析器模块导入（注册走装饰器；模块内部对重依赖延迟导入）
    for mod in ("src.parsing.pymupdf_parser", "src.parsing.docling_parser",
                "src.parsing.markdown_parser", "src.parsing.html_parser",
                "src.parsing.txt_parser"):
        try:
            import_module(mod)
        except ImportError:
            pass
    # 依赖缺失的引擎不可用（如 docling 未安装）——注册了也不算数
    # is_available() 优先（如 docling 走子进程桥，主环境无依赖但 venv 在即可用）
    usable = []
    for name, cls in _REGISTRY.items():
        avail = getattr(cls, "is_available", None)
        if avail is not None:
            if not avail():
                continue
        else:
            req = getattr(cls, "requirement", None)
            if req and importlib.util.find_spec(req) is None:
                continue
        usable.append(name)
    return sorted(usable)


def get_parser(engine: str = "auto") -> Parser:
    engines = available_engines()
    if engine == "auto":
        for name in AUTO_ORDER:
            if name in engines:
                return _REGISTRY[name]()
        raise RuntimeError(f"无可用解析器，已尝试 {AUTO_ORDER}")
    if engine not in engines:
        raise RuntimeError(f"解析器 '{engine}' 不可用，可用: {engines}")
    return _REGISTRY[engine]()


def parse_document(path: str, engine: str = "auto") -> StructuredDocument:
    """解析层统一入口：白名单闸口 → 解析 → 质量检测。

    文件 ingestion 一律走这里，而不是直接 get_parser().parse()。
    成功返回 StructuredDocument（metadata["quality_report"] 为 QualityReport）；
    失败抛 ParseError（code 机器可读，调用方据此分流处理）。
    """
    ext = Path(path).suffix.lower()

    # ── 第一类失败：格式不支持（入口直接拒绝）──
    if ext not in SUPPORTED_FORMATS:
        if ext in DROPPED_FORMATS:
            raise ParseError(code="unsupported_format",
                             message=f"暂不支持 {ext} 格式",
                             source=Path(path).name, detail=DROPPED_FORMATS[ext])
        supported = "、".join(sorted(set(SUPPORTED_FORMATS.values())))
        raise ParseError(code="unsupported_format",
                         message=f"暂不支持 {ext or '无扩展名'} 格式",
                         source=Path(path).name, detail=f"当前支持：{supported}")

    # ── 第二类失败：解析抛异常（归一化为 ParseError 抛出）──
    # auto 模式下按格式路由：结构明写的格式（md/html）走 AST 直读解析器
    if engine == "auto":
        engine = FORMAT_ENGINE.get(ext, "auto")
    try:
        doc = get_parser(engine).parse(path)
    except ParseError:
        raise
    except Exception as exc:
        raise _map_exception(exc, path) from exc

    # ── 第三类失败：解析成功但质量差（结构化报告，不阻断）──
    doc.metadata["quality_report"] = build_quality_report(doc, path).to_dict()
    return doc


def _map_exception(exc: Exception, path: str) -> ParseError:
    """把各引擎五花八门的异常归一成可归因的 ParseError。"""
    name = Path(path).name
    if isinstance(exc, subprocess.TimeoutExpired):
        return ParseError(code="timeout", message="解析超时", source=name,
                          detail=str(exc))
    text = f"{type(exc).__name__} {exc}".lower()
    if any(k in text for k in ("encrypt", "password", "口令", "密码")):
        return ParseError(code="encrypted", message="文档已加密",
                          source=name, detail=str(exc))
    if any(k in text for k in ("corrupt", "damaged", "invalid", "broken",
                               "cannot read", "could not load", "conversionerror",
                               "conversion failed", "not a ", "无法读取", "损坏")):
        return ParseError(code="corrupted", message="文档损坏或格式异常",
                          source=name, detail=str(exc))
    return ParseError(code="unknown", message=f"解析失败：{type(exc).__name__}",
                      source=name, detail=str(exc))


class Parser(ABC):
    """所有解析器的统一接口：文件路径 → StructuredDocument。"""

    name: str = "base"

    @abstractmethod
    def parse(self, path: str) -> StructuredDocument: ...
