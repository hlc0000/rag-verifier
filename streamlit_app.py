"""rag-verifier UI demo · 双语（中/英）切换 + 左右两栏布局

📑 Tab 1: Chunk Completeness（切分块完整性验证）
📑 Tab 2: Citation Faithfulness（引用忠实性验证）
📑 Tab 3: Batch Verification（批量验证）

本地运行：
    pip install streamlit
    streamlit run streamlit_app.py
"""
import os
import sys
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.citation.verifier import verify_citation  # noqa: E402
from src.verify_api import verify_chunk, verify_chunks  # noqa: E402
from src.models import EvidenceCard  # noqa: E402


# ──────────────────────────────────────────────────────────────
# 颜色 / 状态文案
# ──────────────────────────────────────────────────────────────

STATUS_COLOR = {
    "supported": "#16a34a",
    "unsupported": "#dc2626",
    "insufficient": "#ea580c",
    "uncertain": "#6b7280",
    "pass": "#16a34a",
    "fail": "#dc2626",
}

STATUS_TEXT = {
    "supported": "SUPPORTED",
    "unsupported": "UNSUPPORTED",
    "insufficient": "INSUFFICIENT",
    "uncertain": "UNCERTAIN",
    "pass": "PASS",
    "fail": "FAIL",
}

FAIL_KIND_TEXT = {
    "fragment": "fragment",
    "cross_block_ref": "cross_block_ref",
    "other": "issue",
}

CHECK_NAME_TEXT = {
    "independence": "independence",
    "completeness": "completeness",
    "traceability": "traceability",
    "semantic_independence": "semantic_independence",
    "template_completeness": "template_completeness",
    "existence": "existence",
    "verbatim": "verbatim",
    "faithfulness": "faithfulness",
}


# ──────────────────────────────────────────────────────────────
# i18n：所有 UI 文案
# ──────────────────────────────────────────────────────────────

I18N = {
    "zh": {
        # 顶部
        "title": "🛡️ RAG Verification Layer",
        "subtitle": "验证切分块是否完整、AI 引用是否真的支持论点。",
        "lang_label": "🌐 语言",
        # Tab 名
        "tab1": "📑 Tab 1: 切分块完整性",
        "tab2": "📑 Tab 2: 引用忠实性",
        "tab3": "📑 Tab 3: 批量验证",
        # Tab 1
        "tab1_sub": "📑 切分块完整性",
        "tab1_caption": "验证切分块是否完整——是否依赖前文条件、是否引用了别处的表/条/章、是否有模板空位。",
        "label_source": "原文全文",
        "label_chunk": "要验证的切分块",
        "preset_label": "**预置测试场景** ⚡️",
        "verify_chunk_btn": "🔍 验证这块",
        "right_placeholder_t1": "👈 把『原文全文』和『切分块』填好，点『验证这块』看结果。",
        "result_label": "**验证结果**",
        "empty_result_t1": "验证结果为空——可能验证未跑完。重新点『🔍 验证这块』试试。",
        "chunk_pass_msg": "✅ 块自足，无需额外上下文。",
        "chunk_uncertain_msg": "规则层未发现明显问题，但语义层未执行（未配置 LLM），无法确认。",
        "anchors_warning": "⚠️ 引用锚点: {anchors} — 这些引用指向别处的内容",
        "checks_detail": "查看每条检查明细",
        # Tab 2
        "tab2_sub": "📑 引用忠实性",
        "tab2_caption": "验证 AI 引用是否真实存在、是否真的支持 AI 给出的论点——三层核查：存在性 → 归一化 → 忠实性。",
        "label_citation": "AI 引用的原文片段",
        "citation_caption": "AI 在答案里说『原文说 XX』里的 XX 那段话。",
        "label_claim": "AI 给的答案",
        "claim_caption": "AI 回答你时给出的结论。",
        "verify_cite_btn": "🔍 验证引用",
        "right_placeholder_t2": "👈 把『原文全文』『AI 引用的原文片段』『AI 给的答案』填好，点『验证引用』看结果。",
        "empty_result_t2": "验证结果为空——可能验证未跑完。重新点『🔍 验证引用』试试。",
        "checks_detail_cite": "查看三层核查明细",
        "advanced_llm": "高级 · 启用第三层忠实性判定（可选）",
        "advanced_caption": "第一层『原文里能不能找到』+ 第二层『容忍空格差异』不需要 LLM。"
                             "第三层『引用真的支持论点吗』需要 LLM。"
                             "不配则第三层显示 uncertain（不假装通过）。",
        # Tab 3
        "tab3_sub": "📑 批量验证",
        "tab3_caption": "把切分器产出的所有块一次性传进来批量验证——支持 JSON 文件上传或直接粘 JSON。",
        "label_full_doc": "整份原文全文",
        "full_doc_caption": "所有块共享这份原文。如果原文很大，建议用文件上传。",
        "upload_txt": "或上传原文文件（TXT/MD）",
        "upload_txt_help": "文件内容会覆盖上面的文本框。",
        "label_chunks_json": "切分块列表（JSON）",
        "chunks_json_caption": "JSON 数组，每项至少有 content 字段。示例：[{\"content\": \"块1\"}, {\"content\": \"块2\"}]",
        "upload_json": "或上传 chunks JSON 文件",
        "upload_json_help": "文件内容会覆盖上面的 JSON 文本框。",
        "verify_batch_btn": "🔍 批量验证",
        "right_placeholder_t3": "👈 上传 / 粘贴原文和 chunks JSON，点『🔍 批量验证』看结果。",
        "metric_total": "总块数",
        "metric_pass": "通过",
        "metric_fail": "失败",
        "metric_uncertain": "不确定",
        "per_chunk_label": "**逐块结果**",
        "block_label": "块",
        "all_pass_msg": "所有检查项通过 ✅",
        "all_uncertain_msg": "规则层未发现失败项，但语义层未执行，无法确认 ⬜",
        "block_full_text": "块全文：",
        "json_must_be_list": "JSON 必须是数组（list），不能是单个对象。",
        "json_parse_fail": "JSON 解析失败：{err}",
        "batch_running": "批量验证中（共 {n} 块）……",
        "batch_failed": "批量验证失败：{err}",
        "read_file_fail": "读取文件失败：{err}",
        "read_json_fail": "读取 JSON 失败：{err}",
        # 杂项
        "placeholder_source": "把完整的原文粘进来……",
        "placeholder_chunk": "把要验证的切分块粘进来……",
        "placeholder_cite": "把 AI 拿出来当证据的那段原文粘进来……",
        "placeholder_claim": "把 AI 给你的答案粘进来……",
        "placeholder_full_doc": "把完整的原文粘进来……",
        "placeholder_chunks_json": '[{"content": "第一块文本"}, {"content": "第二块文本"}]',
    },
    "en": {
        "title": "🛡️ RAG Verification Layer",
        "subtitle": "Check if your chunks are complete, and if AI citations actually support the claims.",
        "lang_label": "🌐 Language",
        "tab1": "📑 Tab 1: Chunk Completeness",
        "tab2": "📑 Tab 2: Citation Faithfulness",
        "tab3": "📑 Tab 3: Batch Verification",
        "tab1_sub": "📑 Chunk Completeness",
        "tab1_caption": "Verify chunk completeness — whether it depends on prior conditions, references external tables/sections, or has template blanks.",
        "label_source": "Full Document",
        "label_chunk": "Chunk to Verify",
        "preset_label": "**Preset Test Cases** ⚡️",
        "verify_chunk_btn": "🔍 Verify This Chunk",
        "right_placeholder_t1": "👈 Fill in the 『Full Document』 and 『Chunk to Verify』, then click 『Verify This Chunk』 to see the result.",
        "result_label": "**Verification Result**",
        "empty_result_t1": "Empty result — verification may not have completed. Click 『🔍 Verify This Chunk』 again.",
        "chunk_pass_msg": "✅ Chunk is self-contained; no extra context needed.",
        "chunk_uncertain_msg": "Rule layer found no obvious issue, but the semantic layer did not run (no LLM configured), so it cannot be confirmed.",
        "anchors_warning": "⚠️ Reference anchors: {anchors} — these point to content elsewhere",
        "checks_detail": "Show check details",
        "tab2_sub": "📑 Citation Faithfulness",
        "tab2_caption": "Verify AI citations are real and actually support the claim — three layers: existence → normalized → faithfulness.",
        "label_citation": "AI-Cited Excerpt",
        "citation_caption": "The excerpt that AI quoted from the source as evidence.",
        "label_claim": "AI Claim",
        "claim_caption": "The answer AI gave you.",
        "verify_cite_btn": "🔍 Verify Citation",
        "right_placeholder_t2": "👈 Fill in 『Full Document』, 『AI-Cited Excerpt』, and 『AI Claim』, then click 『Verify Citation』.",
        "empty_result_t2": "Empty result — verification may not have completed. Click 『🔍 Verify Citation』 again.",
        "checks_detail_cite": "Show three-layer check details",
        "advanced_llm": "Advanced · Enable Layer-3 Faithfulness (optional)",
        "advanced_caption": "Layer-1 『does it exist in the source』 + Layer-2 『tolerate whitespace』 need no LLM. "
                             "Layer-3 『does the citation actually support the claim』 requires an LLM. "
                             "Without a key, Layer-3 returns uncertain (we never fake a pass).",
        "tab3_sub": "📑 Batch Verification",
        "tab3_caption": "Verify all chunks from your splitter in one batch — JSON file upload or paste JSON.",
        "label_full_doc": "Full Document",
        "full_doc_caption": "All chunks share this source. For large docs, use file upload.",
        "upload_txt": "Or upload source file (TXT/MD)",
        "upload_txt_help": "File content will override the text box above.",
        "label_chunks_json": "Chunk List (JSON)",
        "chunks_json_caption": "JSON array, each item must have a 'content' field. Example: [{\"content\": \"chunk1\"}, {\"content\": \"chunk2\"}]",
        "upload_json": "Or upload chunks JSON file",
        "upload_json_help": "File content will override the JSON text box above.",
        "verify_batch_btn": "🔍 Batch Verify",
        "right_placeholder_t3": "👈 Upload / paste the source and chunks JSON, then click 『🔍 Batch Verify』.",
        "metric_total": "Total",
        "metric_pass": "Pass",
        "metric_fail": "Fail",
        "metric_uncertain": "Uncertain",
        "per_chunk_label": "**Per-Chunk Result**",
        "block_label": "Chunk",
        "all_pass_msg": "All checks passed ✅",
        "all_uncertain_msg": "Rule layer found no failures, but the semantic layer did not run — cannot confirm ⬜",
        "block_full_text": "Chunk full text: ",
        "json_must_be_list": "JSON must be a list (array), not a single object.",
        "json_parse_fail": "JSON parse failed: {err}",
        "batch_running": "Batch verifying ({n} chunks)…",
        "batch_failed": "Batch verification failed: {err}",
        "read_file_fail": "Failed to read file: {err}",
        "read_json_fail": "Failed to read JSON: {err}",
        "placeholder_source": "Paste the full source document here…",
        "placeholder_chunk": "Paste the chunk to verify here…",
        "placeholder_cite": "Paste the excerpt AI cited as evidence…",
        "placeholder_claim": "Paste the claim AI gave you…",
        "placeholder_full_doc": "Paste the full source document here…",
        "placeholder_chunks_json": '[{"content": "chunk 1 text"}, {"content": "chunk 2 text"}]',
    },
}


def t(key, **kwargs):
    """按当前 session_state['lang'] 取文案，支持 {placeholder} 插值。"""
    lang = st.session_state.get("lang", "en")
    text = I18N.get(lang, I18N["zh"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text


# ──────────────────────────────────────────────────────────────
# 预置测试场景（中/英两套）
# ──────────────────────────────────────────────────────────────

CHUNK_TEST_CASES = {
    "zh": {
        "Missing Context (缺失前文条件)": {
            "source": (
                "第五条 付款方式\n"
                "如果乙方逾期超过30天，甲方有权解除合同，并要求乙方承担违约责任。"
            ),
            "chunk": "甲方有权解除合同，并要求乙方承担违约责任。",
        },
        "Cross-Chunk Reference (引用别处的表/条)": {
            "source": (
                "第一条 标的\n本合同标的为某商品。\n\n"
                "第二条 数量\n商品数量详见本合同附件一。"
            ),
            "chunk": "商品数量详见本合同附件一。",
        },
        "Template Blank (模板空位)": {
            "source": "甲方：，乙方：\n本合同经双方签署后生效。",
            "chunk": "甲方：，乙方：",
        },
        "Unresolved Reference (无锚点指代)": {
            "source": "本合同自双方签字之日起生效。该款项由乙方负责支付。",
            "chunk": "该款项由乙方负责支付。",
        },
        "Clean Block (通过案例)": {
            "source": (
                "第一条 标的\n本合同标的为某商品。\n\n"
                "第二条 付款方式\n乙方应于合同生效后30日内支付全部款项。"
            ),
            "chunk": "乙方应于合同生效后30日内支付全部款项。",
        },
    },
    "en": {
        "Missing Context (missing prior condition)": {
            "source": (
                "Section 5 Payment\n"
                "If Party B is more than 30 days overdue, Party A has the right to terminate the contract "
                "and hold Party B liable for breach."
            ),
            "chunk": "Party A has the right to terminate the contract and hold Party B liable for breach.",
        },
        "Cross-Chunk Reference (refers to external table/section)": {
            "source": (
                "Section 1 Subject\nThe subject of this contract is the Goods.\n\n"
                "Section 2 Quantity\nThe quantity of the Goods is set forth in Exhibit A."
            ),
            "chunk": "The quantity of the Goods is set forth in Exhibit A.",
        },
        "Template Blank (template placeholder)": {
            "source": (
                "Party A: ____________, Party B: ____________\n"
                "This contract takes effect upon signature by both parties."
            ),
            "chunk": "Party A: ____________, Party B: ____________",
        },
        "Unresolved Reference (dangling anaphora)": {
            "source": "This contract takes effect upon signature by both parties. Such payment shall be borne by Party B.",
            "chunk": "Such payment shall be borne by Party B.",
        },
        "Clean Block (passing case)": {
            "source": (
                "Section 1 Subject\nThe subject of this contract is the Goods.\n\n"
                "Section 2 Payment\nParty B shall pay the full amount within 30 days of the contract taking effect."
            ),
            "chunk": "Party B shall pay the full amount within 30 days of the contract taking effect.",
        },
    },
}

CITATION_TEST_CASES = {
    "zh": {
        "Fake Number (篡改数字)": {
            "source": "本产品年化利率为0.5%，自购买之日起计算。",
            "citation": "本产品年化利率为0.5%，自购买之日起计算。",
            "claim": "本产品年化利率为1%。",
        },
        "Decontextualized (断章取义)": {
            "source": (
                "第五条 解除权\n"
                "如果乙方逾期超过30天，甲方有权解除合同。"
            ),
            "citation": "甲方有权解除合同。",
            "claim": "甲方可以无条件解除合同。",
        },
        "Faithful (引用真实且支持)": {
            "source": "本产品保修期24个月，自购买之日起计算。",
            "citation": "本产品保修期24个月",
            "claim": "保修期为2年。",
        },
        "Fabricated Citation (引用不存在)": {
            "source": "本产品保修期24个月，自购买之日起计算。",
            "citation": "本产品保修期36个月。",
            "claim": "保修期为3年。",
        },
    },
    "en": {
        "Fake Number (altered number)": {
            "source": "The annual interest rate of this product is 0.5%, calculated from the date of purchase.",
            "citation": "The annual interest rate of this product is 0.5%, calculated from the date of purchase.",
            "claim": "The annual interest rate of this product is 1%.",
        },
        "Decontextualized (out of context)": {
            "source": (
                "Section 5 Termination Right\n"
                "If Party B is more than 30 days overdue, Party A has the right to terminate the contract."
            ),
            "citation": "Party A has the right to terminate the contract.",
            "claim": "Party A may terminate the contract unconditionally.",
        },
        "Faithful (citation is real and supports)": {
            "source": "The warranty period of this product is 24 months, calculated from the date of purchase.",
            "citation": "The warranty period of this product is 24 months",
            "claim": "The warranty period is 2 years.",
        },
        "Fabricated Citation (citation does not exist)": {
            "source": "The warranty period of this product is 24 months, calculated from the date of purchase.",
            "citation": "The warranty period of this product is 36 months.",
            "claim": "The warranty period is 3 years.",
        },
    },
}

BATCH_TEST_CASES = {
    "zh": {
        "Contract Mix (混合：好块+条件句截断+模板空位+跨块引用)": {
            "full_doc": (
                "第一条 标的\n本合同标的为某商品。\n\n"
                "第二条 数量\n商品数量详见本合同附件一。\n\n"
                "第三条 付款\n如果乙方逾期超过30天，甲方有权解除合同。\n\n"
                "第四条 签署\n甲方：，乙方："
            ),
            "chunks": [
                {"content": "本合同标的为某商品。"},
                {"content": "商品数量详见本合同附件一。"},
                {"content": "甲方有权解除合同。"},
                {"content": "甲方：，乙方："},
            ],
        },
    },
    "en": {
        "Contract Mix (mix: good chunk + truncated conditional + template blank + cross-chunk ref)": {
            "full_doc": (
                "Section 1 Subject\nThe subject of this contract is the Goods.\n\n"
                "Section 2 Quantity\nThe quantity of the Goods is set forth in Exhibit A.\n\n"
                "Section 3 Payment\nIf Party B is more than 30 days overdue, Party A has the right to terminate the contract.\n\n"
                "Section 4 Signature\nParty A: ____________, Party B: ____________"
            ),
            "chunks": [
                {"content": "The subject of this contract is the Goods."},
                {"content": "The quantity of the Goods is set forth in Exhibit A."},
                {"content": "Party A has the right to terminate the contract."},
                {"content": "Party A: ____________, Party B: ____________"},
            ],
        },
    },
}


# ──────────────────────────────────────────────────────────────
# 辅助：单块验证
# ──────────────────────────────────────────────────────────────

def _llm_callback():
    """配了 LLM key 就返回内置回调，否则返回 None（语义层跳过，绝不伪装通过）。"""
    if os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"):
        from src.chunking.semantic_verifier import builtin_llm
        return builtin_llm
    return None


def _verify_single_chunk(content: str, full_doc: str) -> dict:
    """跑单块验证。返回 {overall, fail_kind, fail_reason, anchors, checks}。"""
    try:
        result = verify_chunk(content, source_text=full_doc, traceability_mode="normalized",
                              llm_callback=_llm_callback())
        overall = result["overall"].value if hasattr(result["overall"], "value") else str(result["overall"])
        checks = result["checks"]
    except Exception as e:
        overall = "fail"
        checks = {"error": {"status": "fail", "reason": f"{type(e).__name__}: {e}"}}

    fail_kind = ""
    if overall == "fail":
        import re
        reasons = " ".join(v.get("reason", "") for v in checks.values() if v.get("status") == "fail")
        if "碎片" in reasons or "fragment" in reasons.lower():
            fail_kind = "fragment"
        elif "跨块" in reasons or "Cross-chunk" in reasons or "cross-chunk" in reasons.lower():
            fail_kind = "cross_block_ref"
        else:
            fail_kind = "other"

    import re
    anchor_res = [
        re.compile(r'第\s*\d+\s*条'), re.compile(r'第[一二三四五六七八九十百]+\s*条'),
        re.compile(r'表\s*\d+'), re.compile(r'图\s*\d+'),
        re.compile(r'\bTable\s+\d+', re.I), re.compile(r'\bSection\s+\d+', re.I),
        re.compile(r'\bExhibit\s+[A-Z]', re.I),
    ]
    seen, anchors = set(), []
    for rx in anchor_res:
        for m in rx.finditer(content):
            a = re.sub(r'\s+', '', m.group())
            if a not in seen:
                seen.add(a)
                anchors.append(a)

    first_fail_reason = next(
        (v.get("reason", "") for v in checks.values() if v.get("status") == "fail"), ""
    )
    return {
        "overall": overall,
        "fail_kind": fail_kind,
        "fail_reason": first_fail_reason,
        "anchors": anchors,
        "checks": checks,
    }


def _render_chunk_result(r):
    """渲染单块验证结果到右栏。"""
    if not r or not isinstance(r, dict) or "overall" not in r:
        st.warning(t("empty_result_t1"))
        st.session_state.pop("chunk_result", None)
        return
    overall = r["overall"]
    color = STATUS_COLOR.get(overall, "#6b7280")
    text = STATUS_TEXT.get(overall, overall)

    if overall == "fail":
        fail_kind = r["fail_kind"] or "issue"
        reason = r["fail_reason"] or "—"
        line = f"❌ {text} - {fail_kind}: {reason}"
    elif overall == "pass":
        line = f"✅ {text} - {t('chunk_pass_msg')}"
    else:
        # uncertain：规则层没抓到问题，但语义层未执行，不能伪装成通过
        reason = next(
            (v.get("reason", "") for v in r["checks"].values() if v.get("status") == "uncertain"),
            t("chunk_uncertain_msg"),
        )
        line = f"⬜ {text} - {reason}"

    st.markdown(
        f"<div style='padding:16px;border:1px solid {color};border-radius:8px;"
        f"background:#fafafa;font-family:monospace;font-size:14px;color:{color}'>"
        f"{line}</div>",
        unsafe_allow_html=True,
    )

    if r["anchors"]:
        anchors_str = ", ".join(r["anchors"])
        st.markdown(
            f"<div style='padding:12px;border:1px solid #fbbf24;border-radius:6px;"
            f"background:#fffbeb;color:#92400e;font-size:13px;margin-top:8px'>"
            f"{t('anchors_warning', anchors=anchors_str)}</div>",
            unsafe_allow_html=True,
        )

    with st.expander(t("checks_detail"), expanded=False):
        for name, chk in r["checks"].items():
            chk_status = chk.get("status", "uncertain")
            chk_color = STATUS_COLOR.get(chk_status, "#6b7280")
            label = CHECK_NAME_TEXT.get(name, name)
            st.markdown(
                f"<span style='color:{chk_color}'>●</span> **{label}**: {chk_status} — "
                f"{chk.get('reason', '')}",
                unsafe_allow_html=True,
            )


def _render_citation_result(result):
    """渲染引用核查结果到右栏。"""
    if not result or not isinstance(result, dict) or "status" not in result:
        st.warning(t("empty_result_t2"))
        st.session_state.pop("citation_result", None)
        return
    status = result["status"]
    color = STATUS_COLOR.get(status, "#6b7280")
    text = STATUS_TEXT.get(status, status)
    reason = result.get("reason", "—")

    if status == "supported":
        prefix = "✅"
    elif status == "unsupported":
        prefix = "❌"
    elif status == "insufficient":
        prefix = "⚠️"
    else:
        prefix = "⬜"

    line = f"{prefix} {text} - {reason}"
    st.markdown(
        f"<div style='padding:16px;border:1px solid {color};border-radius:8px;"
        f"background:#fafafa;font-family:monospace;font-size:14px;color:{color}'>"
        f"{line}</div>",
        unsafe_allow_html=True,
    )

    with st.expander(t("checks_detail_cite"), expanded=False):
        for name, chk in result["checks"].items():
            chk_status = chk.get("status", "uncertain")
            chk_color = STATUS_COLOR.get(chk_status, "#6b7280")
            label = CHECK_NAME_TEXT.get(name, name)
            st.markdown(
                f"<span style='color:{chk_color}'>●</span> **{label}**: {chk_status} — "
                f"{chk.get('reason', '')}",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────
# Tab 1 · Chunk Completeness
# ──────────────────────────────────────────────────────────────

def _load_chunk_case(case_data: dict):
    """on_click callback：在 widget 渲染前写入 session_state。"""
    st.session_state["chunk_source"] = case_data["source"]
    st.session_state["chunk_input"] = case_data["chunk"]
    st.session_state.pop("chunk_result", None)


def tab_chunk_completeness():
    st.subheader(t("tab1_sub"))
    st.caption(t("tab1_caption"))

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"**{t('label_source')}**")
        source = st.text_area(
            t("label_source"),
            value=st.session_state.get("chunk_source", ""),
            height=150,
            placeholder=t("placeholder_source"),
            label_visibility="collapsed",
            key="chunk_source",
        )

        st.markdown(f"**{t('label_chunk')}**")
        chunk = st.text_area(
            t("label_chunk"),
            value=st.session_state.get("chunk_input", ""),
            height=100,
            placeholder=t("placeholder_chunk"),
            label_visibility="collapsed",
            key="chunk_input",
        )

        st.markdown(t("preset_label"))
        cases = CHUNK_TEST_CASES[st.session_state["lang"]]
        case_cols = st.columns(2)
        for i, (case_name, case_data) in enumerate(cases.items()):
            with case_cols[i % 2]:
                st.button(case_name, key=f"chunk_case_{i}", use_container_width=True,
                          on_click=_load_chunk_case, args=(case_data,))

        st.write("")
        if st.button(t("verify_chunk_btn"), type="primary", use_container_width=True,
                     disabled=not (source and chunk)):
            with st.spinner("..."):
                result = _verify_single_chunk(chunk, source)
            st.session_state["chunk_result"] = result

    with right:
        st.markdown(t("result_label"))
        if "chunk_result" in st.session_state:
            _render_chunk_result(st.session_state["chunk_result"])
        else:
            st.info(t("right_placeholder_t1"))


# ──────────────────────────────────────────────────────────────
# Tab 2 · Citation Faithfulness
# ──────────────────────────────────────────────────────────────

def _load_citation_case(case_data: dict):
    """on_click callback：把 source/citation/claim 写入 session_state。"""
    st.session_state["cite_source"] = case_data["source"]
    st.session_state["cite_input"] = case_data["citation"]
    st.session_state["claim_input"] = case_data["claim"]
    st.session_state.pop("citation_result", None)


def tab_citation_faithfulness():
    st.subheader(t("tab2_sub"))
    st.caption(t("tab2_caption"))

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"**{t('label_source')}**")
        source = st.text_area(
            t("label_source"),
            value=st.session_state.get("cite_source", ""),
            height=120,
            placeholder=t("placeholder_source"),
            label_visibility="collapsed",
            key="cite_source",
        )

        st.markdown(f"**{t('label_citation')}**")
        st.caption(t("citation_caption"))
        citation = st.text_area(
            t("label_citation"),
            value=st.session_state.get("cite_input", ""),
            height=80,
            placeholder=t("placeholder_cite"),
            label_visibility="collapsed",
            key="cite_input",
        )

        st.markdown(f"**{t('label_claim')}**")
        st.caption(t("claim_caption"))
        claim = st.text_area(
            t("label_claim"),
            value=st.session_state.get("claim_input", ""),
            height=80,
            placeholder=t("placeholder_claim"),
            label_visibility="collapsed",
            key="claim_input",
        )

        st.markdown(t("preset_label"))
        cases = CITATION_TEST_CASES[st.session_state["lang"]]
        case_cols = st.columns(2)
        for i, (case_name, case_data) in enumerate(cases.items()):
            with case_cols[i % 2]:
                st.button(case_name, key=f"cite_case_{i}", use_container_width=True,
                          on_click=_load_citation_case, args=(case_data,))

        with st.expander(t("advanced_llm")):
            st.caption(t("advanced_caption"))
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                llm_base = st.text_input("LLM Base URL",
                                          value="")
            with col_b:
                llm_key = st.text_input("LLM API Key", type="password",
                                         value="")
            with col_c:
                llm_model = st.text_input("LLM Model",
                                          value="")
            if llm_key:
                os.environ["LLM_BASE_URL"] = llm_base or "https://api.deepseek.com/v1"
                os.environ["LLM_API_KEY"] = llm_key
                os.environ["LLM_MODEL"] = llm_model or "deepseek-chat"

        st.write("")
        if st.button(t("verify_cite_btn"), type="primary", use_container_width=True,
                     disabled=not (source and citation and claim)):
            with st.spinner("..."):
                try:
                    result = verify_citation(claim, citation, source)
                except Exception as e:
                    result = {
                        "status": "uncertain",
                        "reason": f"{type(e).__name__}: {e}",
                        "checks": {"error": {"status": "uncertain", "reason": str(e)}},
                    }
            st.session_state["citation_result"] = result

    with right:
        st.markdown(t("result_label"))
        if "citation_result" in st.session_state:
            _render_citation_result(st.session_state["citation_result"])
        else:
            st.info(t("right_placeholder_t2"))


# ──────────────────────────────────────────────────────────────
# Tab 3 · Batch Verification
# ──────────────────────────────────────────────────────────────

def _load_batch_case(case_data: dict):
    """一键加载批量测试场景。"""
    st.session_state["batch_full_doc"] = case_data["full_doc"]
    st.session_state["batch_chunks_json"] = json.dumps(case_data["chunks"], ensure_ascii=False, indent=2)
    st.session_state.pop("batch_result", None)


def tab_batch_verification():
    st.subheader(t("tab3_sub"))
    st.caption(t("tab3_caption"))

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"**{t('label_full_doc')}**")
        st.caption(t("full_doc_caption"))
        full_doc_text = st.text_area(
            t("label_full_doc"),
            value=st.session_state.get("batch_full_doc", ""),
            height=150,
            placeholder=t("placeholder_full_doc"),
            label_visibility="collapsed",
            key="batch_full_doc",
        )
        full_doc_file = st.file_uploader(
            t("upload_txt"),
            type=["txt", "md"],
            help=t("upload_txt_help"),
        )
        if full_doc_file is not None:
            try:
                content = full_doc_file.read().decode("utf-8", errors="replace")
                st.session_state["batch_full_doc"] = content
                full_doc_text = content
            except Exception as e:
                st.error(t("read_file_fail", err=f"{type(e).__name__}: {e}"))

        st.markdown(f"**{t('label_chunks_json')}**")
        st.caption(t("chunks_json_caption"))
        chunks_json = st.text_area(
            t("label_chunks_json"),
            value=st.session_state.get("batch_chunks_json", ""),
            height=200,
            placeholder=t("placeholder_chunks_json"),
            label_visibility="collapsed",
            key="batch_chunks_json",
        )
        chunks_file = st.file_uploader(
            t("upload_json"),
            type=["json"],
            help=t("upload_json_help"),
        )
        if chunks_file is not None:
            try:
                content = chunks_file.read().decode("utf-8", errors="replace")
                st.session_state["batch_chunks_json"] = content
                chunks_json = content
            except Exception as e:
                st.error(t("read_json_fail", err=f"{type(e).__name__}: {e}"))

        st.markdown(t("preset_label"))
        cases = BATCH_TEST_CASES[st.session_state["lang"]]
        case_cols = st.columns(len(cases))
        for i, (case_name, case_data) in enumerate(cases.items()):
            with case_cols[i]:
                st.button(case_name, key=f"batch_case_{i}", use_container_width=True,
                          on_click=_load_batch_case, args=(case_data,))

        st.write("")
        parse_ok = False
        chunks_list = []
        if chunks_json.strip():
            try:
                chunks_list = json.loads(chunks_json)
                if not isinstance(chunks_list, list):
                    st.error(t("json_must_be_list"))
                else:
                    parse_ok = True
            except json.JSONDecodeError as e:
                st.error(t("json_parse_fail", err=str(e)))

        if st.button(t("verify_batch_btn"), type="primary", use_container_width=True,
                     disabled=not (parse_ok and chunks_list and full_doc_text)):
            with st.spinner(t("batch_running", n=len(chunks_list))):
                try:
                    results = verify_chunks(
                        chunks_list,
                        full_doc=full_doc_text,
                        traceability_mode="normalized",
                        llm_callback=_llm_callback(),
                    )
                except Exception as e:
                    st.error(t("batch_failed", err=f"{type(e).__name__}: {e}"))
                    results = None
                st.session_state["batch_result"] = results

    with right:
        st.markdown(t("result_label"))
        if st.session_state.get("batch_result"):
            results = st.session_state["batch_result"]
            total = len(results)
            # overall 可能是枚举或字符串，统一取小写字符串再比较，避免 "fail".endswith("FAIL") 漏计
            def _status(r):
                v = r.get("overall", "")
                v = v.value if hasattr(v, "value") else str(v)
                return v.lower()
            passes = sum(1 for r in results if _status(r) == "pass")
            fails = sum(1 for r in results if _status(r) == "fail")
            uncertains = total - passes - fails

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t("metric_total"), total)
            col2.metric(t("metric_pass"), passes)
            col3.metric(t("metric_fail"), fails)
            col4.metric(t("metric_uncertain"), uncertains)

            st.write("")
            st.markdown(t("per_chunk_label"))
            for i, r in enumerate(results):
                overall_raw = r.get("overall", "")
                overall = overall_raw.value if hasattr(overall_raw, "value") else str(overall_raw)
                if overall == "pass":
                    color, emoji = "#16a34a", "✅"
                elif overall == "fail":
                    color, emoji = "#dc2626", "❌"
                else:
                    color, emoji = "#6b7280", "⬜"

                content_preview = chunks_list[i].get("content", "")[:80].replace("\n", " ")
                fails_in_block = [(k, v.get("reason", "")) for k, v in r.get("checks", {}).items()
                                  if v.get("status") == "fail"]
                with st.expander(f"{emoji} {t('block_label')} #{i+1} · {overall} · {content_preview}{'...' if len(chunks_list[i].get('content', '')) > 80 else ''}"):
                    if fails_in_block:
                        for name, reason in fails_in_block:
                            st.markdown(f"**[{name}]** {reason}")
                    elif overall == "pass":
                        st.markdown(t("all_pass_msg"))
                    else:
                        uncertain_reasons = [v.get("reason", "") for v in r.get("checks", {}).values()
                                             if v.get("status") == "uncertain"]
                        st.markdown(t("all_uncertain_msg"))
                        for reason in uncertain_reasons:
                            st.caption(f"· {reason}")
                    st.caption(f"{t('block_full_text')}{chunks_list[i].get('content', '')}")
        else:
            st.info(t("right_placeholder_t3"))


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RAG Verification Layer",
        page_icon="🛡️",
        layout="wide",
    )

    # 初始化 session_state
    defaults = {
        "lang": "en",
        "chunk_source": "", "chunk_input": "", "chunk_result": None,
        "cite_source": "", "cite_input": "", "claim_input": "", "citation_result": None,
        "batch_full_doc": "", "batch_chunks_json": "", "batch_result": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # 顶部标题
    st.markdown(
        f"<h1 style='text-align:center'>{t('title')}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:#6b7280'>{t('subtitle')}</p>",
        unsafe_allow_html=True,
    )

    # 语言切换器（标题下方，普通 selectbox，不套 columns 以免宽度被压没）
    lang_options = ["中文", "English"]
    lang_index = 0 if st.session_state["lang"] == "zh" else 1
    selected = st.selectbox(
        t("lang_label"), lang_options, index=lang_index, key="lang_selector"
    )
    new_lang = "zh" if selected == "中文" else "en"
    if new_lang != st.session_state["lang"]:
        st.session_state["lang"] = new_lang
        # 切语言时清空已填入的场景，避免中文用例留在英文模式里
        for k in ("chunk_source", "chunk_input", "chunk_result",
                  "cite_source", "cite_input", "claim_input", "citation_result",
                  "batch_full_doc", "batch_chunks_json", "batch_result"):
            st.session_state.pop(k, None)
        st.rerun()

    st.write("")

    tab1, tab2, tab3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])
    with tab1:
        tab_chunk_completeness()
    with tab2:
        tab_citation_faithfulness()
    with tab3:
        tab_batch_verification()


if __name__ == "__main__":
    main()
