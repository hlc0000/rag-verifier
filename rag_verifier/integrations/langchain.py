"""LangChain 集成：VerifierTransformer。

把 rag-verifier 接入 LangChain LCEL 流水线，在切块后、入库前做质量门控。
核心验证层零依赖；本模块在运行时才导入 langchain_core，未安装 langchain 时
导入本模块会抛出清晰的 ImportError。

用法（5 行接入）：

    from rag_verifier.integrations.langchain import VerifierTransformer

    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_documents(docs)
    gate = VerifierTransformer()          # 规则层，零 LLM 调用；uncertain 块保留并标注
    clean_chunks = gate.transform_documents(chunks)   # 只丢弃规则层判定 fail 的块

可选：注入 LLM 回调开启语义层：

    from rag_verifier import builtin_llm
    gate = VerifierTransformer(llm_callback=builtin_llm)
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from rag_verifier import verify_chunk as _verify_chunk


class VerifierTransformer:
    """LangChain 兼容的文档质量门控 transformer。

    对每个 Document 的 page_content 运行 verify_chunk，丢弃 overall != "pass"
    的块（即 fail 或 uncertain 都不放行——fail-open 原则下 uncertain 不
    伪装成通过）。

    参数:
        llm_callback: 可选的 LLM 回调（prompt -> str），传入则启用语义层。
        checks: 要运行的检查项子集，None 表示五关全跑。
        prev_window: 跨块回看时向前取多少字符（默认 200）。
        drop_uncertain: False（默认）表示只丢弃 fail 块，uncertain 块保留并
                        在 metadata 里标注 verdict；True 表示 uncertain 块也丢弃。
    """

    def __init__(
        self,
        llm_callback: Optional[Any] = None,
        checks: Optional[list[str]] = None,
        prev_window: int = 200,
        drop_uncertain: bool = False,
    ) -> None:
        self._llm_callback = llm_callback
        self._checks = checks
        self._prev_window = prev_window
        self._drop_uncertain = drop_uncertain

    def transform_documents(self, documents: Sequence[Any]) -> list[Any]:
        """对文档序列做质量门控，返回通过的文档列表。

        兼容 LangChain Document（.page_content / .metadata）以及任意带
        page_content 属性的对象。
        """
        passed: list[Any] = []
        prev_content = ""

        for doc in documents:
            content = getattr(doc, "page_content", None)
            if content is None and isinstance(doc, dict):
                content = doc.get("page_content") or doc.get("content", "")
            content = content or ""

            source_text = getattr(doc, "metadata", {}) or {}
            if isinstance(source_text, dict):
                source_text = source_text.get("source_text", "") or content
            else:
                source_text = content

            prev = prev_content[-self._prev_window:] if prev_content else ""
            result = _verify_chunk(
                content,
                source_text=str(source_text),
                prev_chunk=prev,
                checks=self._checks,
                llm_callback=self._llm_callback,
            )

            overall = result["overall"].value if hasattr(result["overall"], "value") else result["overall"]
            prev_content = content

            if overall == "pass":
                passed.append(doc)
            elif not self._drop_uncertain and overall == "uncertain":
                # 保留但标注
                meta = getattr(doc, "metadata", None)
                if isinstance(meta, dict):
                    meta["rag_verifier_verdict"] = "uncertain"
                passed.append(doc)
            # fail 直接丢弃

        return passed

    # LangChain Runnable 兼容：.invoke() 接收单个文档或列表
    def invoke(self, input: Any) -> Any:
        if isinstance(input, (list, tuple)):
            return self.transform_documents(input)
        result = self.transform_documents([input])
        return result[0] if result else None
