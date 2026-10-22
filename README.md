# rag-verifier

RAG 证据块质量验证库。可以集成到任何 RAG 应用的切分/入库环节，在块进入向量库之前做自动质量闸门。

## 做什么

对每个待入库的证据块，依次过五道验证：

| 检查项 | 规则层（快筛） | LLM 语义层（兜底） |
|---|---|---|
| independence  独立性 | 承接词开头（However/因此）、残缺结尾（If/若...）、碎片、空块、跨页表标记 | 触发条件被切走、缺主语 |
| completeness  完整性 | 条件句配对、让步句配对、依赖连接词前缀 + 词边界 | — |
| traceability  可追溯性 | 指代缺失（因此/该合同 无前文）、跨块实质引用（as shown in Figure 2） | — |
| template_completeness  模板完整性 | 方括号占位符、下划线空位、数量/日期/签名空位 | — |
| semantic_independence  语义独立性 | — | 独立判定，带完整全文上下文 |

规则层确定性正则（0.5% 以内误伤），LLM 层做 fail-open（不可用 → `uncertain`，绝不伪装成通过）。

## 安装

```bash
pip install -e .                  # 仅核心（规则层）
pip install -e ".[parse]"         # + 文本层解析（pymupdf + lxml）
pip install -e ".[semantic]"      # + LLM 语义层（requests）
pip install -e ".[parse,semantic]" # 常用组合
```

## 用法

### 仅规则层（最快，零 LLM 调用）

```python
from src.verify_api import verify_chunk

result = verify_chunk(
    text="因此甲方有权要求乙方赔偿。",
    prev_chunk="若乙方违反保密义务，",
)
# result = {"status": "fail", "reason": "依赖连接词开头...", "checks": {...}}
```

### 带 LLM 语义层（需配置 API key）

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
```

```python
from src.verify_api import verify_chunk

result = verify_chunk(
    text="因此甲方有权要求乙方赔偿。",
    prev_chunk="若乙方违反保密义务，",
    full_doc="若乙方违反保密义务，因此甲方有权要求乙方赔偿。...",  # 可选，语义层用
)
```

### 引用忠实性验证

```python
from src.citation.verifier import verify_citation

result = verify_citation(
    claim="Apache License 2.0 requires copyright headers.",
    citation="Copyright [yyyy] [name of copyright owner]",
    source_text="...完整 Apache 2.0 原文...",
)
# result = {"status": "insufficient", "reason": "模板空位未填写...", "checks": {...}}
```

### 文档解析 + 切分 + 验证

```python
from src.parsing.adapter import parse_document
from src.chunking.splitter import split_document

doc = parse_document("contract.pdf")           # 返回 StructuredDocument
chunks = split_document(doc)                  # 返回 list[EvidenceCard]

for chunk in chunks:
    result = verify_chunk(chunk.content, prev_chunk=...)
    if result["status"] == "fail":
        print(f"丢弃 {chunk.id}: {result['reason']}")
```

## 支持的文档格式

PDF（文本层）、DOCX、TXT、MD、HTML。RTF / 扫描件 PDF 不支持（RTF 降级丢标题层级，扫描件缺 OCR）。

## 规则层中英对等

所有规则按语种分两套（`zh` / `en` / `any`），由 `detect_language()` 自动分发。`scripts/rule_parity_check.py` 逐条验证中英规则正例命中、反例零误伤。

## 测试

```bash
python scripts/rule_parity_check.py      # 规则对等性（必过）
python scripts/rule_audit.py             # 逐条规则命中统计
python scripts/regression_check.py       # 20 份基线回归（必过）
```
