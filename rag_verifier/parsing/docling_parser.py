"""Docling 解析器（默认引擎，布局识别强、CPU 可跑）。

依赖较重（torch 等），主程序环境（py3.13）不安装 docling——解析经子进程桥完成：
.venv-docling 的 py3.11 运行 scripts/docling_runner.py，以 JSON 往返 StructuredDocument。
venv 内直跑（或未来主环境装了 docling）时自动走进程内路径，翻译逻辑只此一份。
Docling 自带布局模型，标题层级/表格/双栏阅读顺序由它给出，
本解析器只做「Docling 项 → Block」的翻译，不再做字号猜测。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

# docling 对 CJK 文本会在每个 token 间插空格（"中 华 人 民 共 和 国"），
# 数字与点号之间也会插空格（"3 . 2"、"0 . 01%"），数字与相邻 CJK 之间也有
# （"2025 年"、"第 1 章"）。这些空格会冲掉切分层的编号锚点、Section 锚点。
# 在解析层统一清理一次（仅处理 item.text 内部，不跨段落/单元格，天然安全）。
# 只清理「单行内的空格字符」，不碰换行（\n \r）——换行承载段落边界语义。
# 全角空格 \u3000 也会被 docling 作为 token 分隔符输出。
_SPACE = r'[ \t\u3000]+'

_CJK_RANGE = r'\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af'
_CJK_SPACE = re.compile(rf'([{_CJK_RANGE}]){_SPACE}([{_CJK_RANGE}])')
_NUM_DOT_SPACE = re.compile(rf'(\d+){_SPACE}\.{_SPACE}(\d+)')
_LETTER_DOT_NUM = re.compile(rf'([A-Za-z]){_SPACE}\.{_SPACE}(\d+)')
_NUM_CJK_SPACE = re.compile(rf'(\d+){_SPACE}([{_CJK_RANGE}])')
_CJK_NUM_SPACE = re.compile(rf'([{_CJK_RANGE}]){_SPACE}(\d+)')


def _clean_docling_text(text: str) -> str:
    """清理 docling 输出的 token 间空格，尽量只碰明显 artifact 的场景。

    多轮循环直到稳定——re.sub 是非重叠扫描，连续的 CJK 间空格
    （如 "中 华 人 民" 有 3 个连续空格）单轮只能消一半。
    """
    if not text:
        return text
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        # 1. CJK 相邻字符间空格（最严重的 artifact，几乎必清）
        cur = _CJK_SPACE.sub(r'\1\2', cur)
        # 2. 数字 . 数字 间空格（"3 . 2" → "3.2"，"Figure 3 . 1" → "Figure 3.1"）
        cur = _NUM_DOT_SPACE.sub(r'\1.\2', cur)
        # 3. 字母 . 数字 间空格（"A . 1" → "A.1"，国标附录编号）
        cur = _LETTER_DOT_NUM.sub(r'\1.\2', cur)
        # 4. 数字 ↔ CJK 之间空格（"2025 年" → "2025年"、"第 1 章" → "第1章"）
        cur = _NUM_CJK_SPACE.sub(r'\1\2', cur)
        cur = _CJK_NUM_SPACE.sub(r'\1\2', cur)
    return cur

from src.parsing.base import Parser, register
from src.parsing.level_inference import infer_levels_from_sequence
from src.parsing.models import Block, StructuredDocument, serialize_table

# HF 直连在国内网络下被重置，走镜像（已显式设置时不覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已缓存；强制离线，避免 runner 每次反复联网探测/下载导致进程挂死。
# 若需首次下载新模型，运行前设 HF_HUB_OFFLINE=0 即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VENV_PYTHON = Path(os.environ.get("DOCLING_PYTHON",
                                   str(_PROJECT_ROOT / ".venv-docling" / "bin" / "python")))
_RUNNER = _PROJECT_ROOT / "scripts" / "docling_runner.py"

# Docling 的 DocItemLabel → 本系统 Block.type
_HEADING_LABELS = {"title", "section_header", "subtitle", "heading"}
_LIST_LABELS = {"list_item", "enumerated_item"}


def convert_pdf(path: str) -> StructuredDocument:
    """进程内 docling 解析（要求当前解释器已安装 docling）。"""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat

    # 扫描件/图片页需要 OCR。优先用 RapidOCR（支持中英文、ONNX 推理轻量）；
    # 未安装时关闭 OCR（有文本层的页面仍正常解析，纯扫描页返回空文本）。
    pdf_opt = {"backend": PyPdfiumDocumentBackend}
    # ── CPU 推理慢（实测：非死锁，是真慢）──
    # Intel Mac 4核纯 CPU（torch 无 CUDA/MPS），布局模型 RT-DETR + OCR 单页约 10s，
    # 20 页文档 ≈ 200s 跑完（exit 0、结构完整）。表象 CPU 0% 是 ps 采样撞上 GIL/OMP
    # 等待，进程实际在慢速前进。勿并行起多个 docling（互相抢 CPU，loadavg 飙升）。
    #
    # ── 表格结构默认开（正确性优先）──
    # do_table_structure=False 的代价实测：表格骨架在，但单元格内容散落成独立段落
    # （键值对断裂、跨行/列合并丢失、表头结构丢失）→ verify_completeness 判缺。
    # 故默认开 TableFormer，接受它在 CPU 上慢；仅当赶时间/无表格的大文档时
    # 用 DOCLING_DISABLE_TABLE=1 手动关（明确知道会牺牲表格内容归位）。
    ocr = os.environ.get("DOCLING_DISABLE_OCR", "0") not in ("1", "true", "True")
    table = os.environ.get("DOCLING_DISABLE_TABLE", "0") not in ("1", "true", "True")

    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    kwargs: dict = {"do_ocr": ocr, "do_table_structure": table}
    if ocr:
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions
            import rapidocr  # noqa: F401  验证可用
            kwargs["ocr_options"] = RapidOcrOptions(lang=["ch", "en"])
        except Exception:
            kwargs["do_ocr"] = False

    # batch_size=1：来一页算一页；限线程避免 torch 互抢。
    kwargs["layout_batch_size"] = int(os.environ.get("DOCLING_LAYOUT_BATCH", "1"))
    kwargs["ocr_batch_size"] = int(os.environ.get("DOCLING_OCR_BATCH", "1"))
    kwargs["table_batch_size"] = int(os.environ.get("DOCLING_TABLE_BATCH", "1"))
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions
        kwargs["accelerator_options"] = AcceleratorOptions(
            num_threads=int(os.environ.get("DOCLING_NUM_THREADS", "4")))
    except Exception:
        pass

    pdf_opt["pipeline_options"] = ThreadedPdfPipelineOptions(**kwargs)

    result = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(**pdf_opt)
    }).convert(str(path))
    dl = result.document

    blocks: list[Block] = []
    images = 0
    skipped: list[str] = []
    cur_list: list[str] | None = None

    def flush_list():
        nonlocal cur_list
        if cur_list:
            blocks.append(Block(type="list", items=cur_list))
            cur_list = None

    for item, level in dl.iterate_items():
        label = str(getattr(item, "label", "") or "").lower()
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None

        if label in _HEADING_LABELS:
            flush_list()
            blocks.append(Block(type="heading",
                                content=_clean_docling_text((item.text or "").strip()),
                                level=level, page=page))
        elif label in _LIST_LABELS:
            text = _clean_docling_text((getattr(item, "text", "") or "").strip())
            if text:
                if cur_list is None:
                    cur_list = []
                cur_list.append(text)
        elif label == "table" or hasattr(item, "data"):
            flush_list()
            try:
                grid = [[_clean_docling_text(getattr(c, "text", ""))
                         for c in row] for row in item.data.grid]
            except Exception:
                grid = []
            blocks.append(Block(type="table", content=serialize_table(grid),
                                data=grid, page=page))
        elif label == "picture" or type(item).__name__ == "PictureItem":
            images += 1
        elif label in ("paragraph", "text", "caption", "footnote",
                       "reference", "formula", "code", "page_header", "page_footer", ""):
            flush_list()
            text = _clean_docling_text((getattr(item, "text", "") or "").strip())
            # 页眉页脚 Docling 已标注，直接丢弃，不进正文块
            if label in ("page_header", "page_footer"):
                continue
            if text:
                blocks.append(Block(type="paragraph", content=text, page=page))
        else:
            skipped.append(label)

    # 收尾必须补一次 flush：文档若以列表结尾（参考文献、结论清单、附录列表），
    # 循环内不会再有非列表项触发 flush，尾部列表会被静默丢弃（chinese-paper-real-01 实测丢 11 条参考文献）
    flush_list()

    npages = result.pages if isinstance(getattr(result, "pages", None), int) else None
    metadata = {
        "pages": npages,
        "headings_detected": sum(1 for b in blocks if b.type == "heading"),
        "tables": sum(1 for b in blocks if b.type == "table"),
        "images": images,
        "skipped_labels": sorted(set(skipped)),
    }
    doc = StructuredDocument(source=Path(path).name, engine="docling",
                             blocks=blocks, metadata=metadata)
    # 视觉模型读不出编号层级；层级是「编号序列走势」，从序列推断，不依赖文档类型
    return infer_levels_from_sequence(doc)


@register
class DoclingParser(Parser):
    name = "docling"

    @classmethod
    def is_available(cls) -> bool:
        """主环境装了 docling 直跑即可；否则看子进程桥的 venv 解释器在不在。"""
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            return _VENV_PYTHON.exists()

    def parse(self, path: str) -> StructuredDocument:
        try:
            import docling  # noqa: F401
        except ImportError:
            return self._parse_subprocess(path)
        return convert_pdf(path)

    @staticmethod
    def _parse_subprocess(path: str) -> StructuredDocument:
        proc = subprocess.run(
            [str(_VENV_PYTHON), str(_RUNNER), path],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"docling runner 失败(exit {proc.returncode}): {proc.stderr[-800:]}")
        return StructuredDocument.from_dict(json.loads(proc.stdout))
