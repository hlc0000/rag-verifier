# rag-verifier

A RAG chunk quality gate library. Drop it into your ingestion pipeline, run it on every chunk **before** it enters the vector store — filter out garbage upfront instead of letting it poison later retrieval and citation stages.

## What it does

Five checks run sequentially on each chunk. The rule layer is fast (pure regex, ~0.5% false positive rate) and catches the obvious defects deterministically. The LLM semantic layer kicks in only when rule-layer says "this looks OK" but the chunk might still be semantically broken (missing trigger condition, elided subject).

| Check | Rule layer (fast filter) | LLM semantic layer (fallback) |
|---|---|---|
| independence | dependent connective at start (However/Therefore/Accordingly), incomplete conditional ending (If/When/Unless), fragments, empty chunks, table-continuation markers | trigger condition cut away, elided subject |
| completeness | conditional pairings, concessive pairings, dependent connective with word boundary | — |
| traceability | unresolved anaphora (this/that/it with no referent in-chunk), cross-chunk substantive references (as shown in Figure 2) | — |
| template_completeness | bracketed placeholders (`[yyyy]`), underscore blanks, quantity/date/signature slots | — |
| semantic_independence | — | full independent judgment with the complete document as context |

The LLM layer is fail-open: if the API is down, the block gets `uncertain`, **never** silently reclassified as `supported` or `insufficient`.

## Installation

```bash
pip install -e .                          # core only (rule layer, zero deps)
pip install -e ".[parse]"                 # + text-layer parsing (pymupdf + lxml)
pip install -e ".[semantic]"              # + LLM semantic layer (requests)
pip install -e ".[parse,semantic]"         # common combo
```

## Usage

### Rule layer only (fast, zero LLM calls)

```python
from src.verify_api import verify_chunk

result = verify_chunk(
    text="Therefore the Contractor has the right to claim damages from the Client.",
    prev_chunk="If the Client breaches any material obligation under this Agreement,",
)
# result["overall"].value == "fail"
# result["checks"]["independence"]["status"] == "fail"
# result["checks"]["independence"]["reason"] == "Starts with a dependent connective: 'Therefore'"
```

### With LLM semantic layer (API key required)

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
```

Any OpenAI-compatible endpoint works — DashScope, DeepSeek, AGNES, your local one — change only env vars, no code.

```python
from src.verify_api import verify_chunk

result = verify_chunk(
    text="Therefore Party A has the right to claim compensation from Party B.",
    prev_chunk="If Party B breaches the confidentiality obligation,",
    full_doc="If Party B breaches the confidentiality obligation, therefore Party A ...",  # optional, semantic layer uses it
    llm_callback=builtin_llm,  # inject the LLM client (otherwise semantic check returns uncertain)
)
```

### Citation faithfulness

Three-way verdict (`supported` / `unsupported` / `insufficient`), plus `uncertain` when LLM is unavailable.

```python
from src.citation.verifier import verify_citation

result = verify_citation(
    claim="Apache License 2.0 requires copyright headers.",
    citation="Copyright [yyyy] [name of copyright owner]",
    source_text="... full Apache 2.0 original text ...",
)
# Citation API uses the simpler shape: result["status"], result["reason"], result["checks"]
```

### Full pipeline: parse → split → verify

```python
from src.parsing.adapter import parse_document
from src.chunking.splitter import split_document
from src.verify_api import verify_chunk

doc = parse_document("contract.pdf")    # StructuredDocument
chunks = split_document(doc)            # list[EvidenceCard]

prev = ""
for chunk in chunks:
    result = verify_chunk(chunk.content, prev_chunk=prev)
    if result["overall"].value == "fail":
        # Each check has its own status + reason
        failed = [f"{k}: {v['reason']}" for k, v in result["checks"].items() if v["status"] == "fail"]
        print(f"DROP {chunk.id}: {'; '.join(failed)}")
    prev = chunk.content
```

## Supported formats

Text-layer PDF, DOCX, TXT, Markdown, HTML. Not supported: RTF (no heading styles → levels lost), scanned PDF (no OCR, empty text layer).

## Bilingual (zh / en)

Every rule has a Chinese and an English copy, dispatched automatically by `detect_language()` (CJK ratio ≥15% → zh). The prompt templates, the `reason` strings, and the three citation verdicts all come in matching pairs. Run your document in English → English output. Run in Chinese → Chinese output. Never mixed.

`scripts/rule_parity_check.py` runs the positive and negative examples for every rule in both languages and fails the build if any positive case is missed or any negative case is hit by accident.

## Tests

```bash
python scripts/rule_parity_check.py    # zh/en rule parity — must pass
python scripts/rule_audit.py           # per-rule hit statistics
python scripts/regression_check.py     # 20-document baseline — must pass
```
