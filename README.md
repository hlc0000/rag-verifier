# rag-verifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/hlc0000/rag-verifier/actions/workflows/ci.yml/badge.svg)](https://github.com/hlc0000/rag-verifier/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyPI version](https://img.shields.io/pypi/v/rag-verifier.svg)](https://pypi.org/project/rag-verifier/)

🎯 **[Online Demo](https://rag-verifier.streamlit.app)** · 📦 `pip install rag-verifier`

![demo](docs/demo.png)

Most RAG failures aren't model failures — they're evidence failures. A clause gets split mid-condition: `If the Client breaches any material obligation,` goes into one vector, `the Contractor may terminate this Agreement.` into another. Both chunks look fine to your retriever; neither is answerable on its own. And when the LLM finally answers, it quotes a citation that *sounds* right — with the trigger condition quietly cut away.

rag-verifier is a quality gate for that evidence layer. It checks every chunk **before** it enters your vector store, and every citation **before** it reaches your user.

**Niche:** `chunkvet` audits chunk geometry. `ContextDoctor` does static analysis. `rag-verifier` targets **semantic integrity before ingestion** + **citation faithfulness after generation**. Rule layer is zero-dependency; LLM fallback only.

## One-line CLI

```bash
rag-verifier check "Therefore the Contractor may terminate this Agreement." \
  --prev "If the Client breaches any material obligation,"
```

```
[FAIL] overall = fail
  - independence: fail  (Starts with a dependent connective: 'Therefore')
  - completeness: pass
  - traceability: uncertain  (No source text provided)
  - semantic_independence: uncertain  (No LLM callback; semantic layer skipped)
  - template_completeness: pass
```

Exit code `0` = pass, `1` = fail or uncertain. Add `--json` for machine-readable output, `--llm` to enable the semantic layer.

## What it checks

Five checks run sequentially on each chunk. The rule layer is **fast** (pure regex) and catches structural defects deterministically — measured **false-positive rate on our evaluation set (1,388 chunks / 20 real-world documents)**. The LLM semantic layer kicks in only when the rule layer says "looks OK" but the chunk might still be semantically broken.

| Check | Rule layer (fast, zero LLM cost) | LLM semantic layer (fallback) |
|---|---|---|
| `independence` | dependent connective at start (`However`/`Therefore`/`Accordingly`), incomplete conditional ending (`If`/`When`/`Unless`), fragments, empty chunks, table-continuation markers | trigger condition cut away, elided subject |
| `completeness` | conditional pairings, concessive pairings, dependent connective with word boundary | — |
| `traceability` | unresolved anaphora (`this`/`that`/`it` with no in-chunk referent), cross-chunk substantive references (`as shown in Figure 2`) | — |
| `template_completeness` | bracketed placeholders (`[yyyy]`), underscore blanks, quantity/date/signature slots | — |
| `semantic_independence` | — | full independent judgment with the complete document as context |

The LLM layer is **fail-open**: if the API is down, the chunk gets `uncertain`, **never** silently reclassified as `supported` or `insufficient`.

## Why this exists: pure-rules vs pure-LLM vs this

| | Pure regex rules | Pure LLM judge | rag-verifier (hybrid) |
|---|---|---|---|
| **Cost per chunk** | $0 | $0.001–$0.01 | $0 for rule-pass; LLM only on rule-ambiguous |
| **Latency** | <1 ms | 500–2000 ms | <1 ms for the 95% rule-pass cases |
| **Consistency** (same chunk, same verdict every run) | ✅ deterministic | ❌ sampling variance, prompt drift | ✅ deterministic for rule layer; LLM only where needed |
| **Fail-open on API failure** | N/A | ❌ often silently passes | ✅ returns `uncertain`, never fakes `pass` |
| **Catches subtle semantic breaks** | ❌ only structural | ✅ but expensive & inconsistent | ✅ LLM layer for the hard cases |

The rule layer does the cheap, deterministic work. The LLM layer exists only for the cases rules can't decide.

## How it fits in the RAG quality stack

This is one layer of a healthy stack, designed to compose with the others:

- **Structure-aware chunkers** (e.g. chunkvet, LangChain text splitters) decide *where to cut*. rag-verifier runs after any chunker — yours, theirs, LangChain's — and gates *what may pass*.
- **Citation-side checkers** (e.g. spancheck) verify that quoted spans exist. We go one step further into *faithfulness*: a verbatim quote can still be `insufficient` if its trigger condition was cut away.
- **Static context analyzers** (e.g. ContextDoctor) lint your context before retrieval. We add a fail-open LLM semantic layer on top of deterministic rules, in both Chinese and English.

Parse with anything, chunk with anything, gate with rag-verifier.

## Installation

```bash
pip install rag-verifier
```

The core verification layer is **zero-dependency** (pure stdlib — the LLM calls go through `urllib`). Optional extras:

```bash
pip install "rag-verifier[parse]"        # PDF/DOCX/Markdown/HTML parsing
pip install "rag-verifier[langchain]"    # VerifierTransformer for LangChain pipelines
```

<details><summary>Install from source (development)</summary>

```bash
git clone https://github.com/hlc0000/rag-verifier.git
cd rag-verifier
pip install -e ".[parse,langchain]"
```

</details>

## Usage

### 1. Command line

```bash
# Single chunk (rule layer only)
rag-verifier check "Therefore the Contractor may terminate." --prev "If the Client breaches,"

# Read from stdin
echo "However, the agreement is terminated." | rag-verifier check -

# Citation faithfulness
rag-verifier citation \
  --claim "Apache License requires copyright headers." \
  --citation "Copyright [yyyy] [name of copyright owner]" \
  --source-text "$(cat apache2.txt)"

# JSON output for piping
rag-verifier check "..." --json | jq '.overall'
```

### 2. Chunk quality gate (Python API)

```python
from rag_verifier import verify_chunk

result = verify_chunk(
    "Therefore the Contractor has the right to claim damages from the Client.",
    prev_chunk="If the Client breaches any material obligation under this Agreement,",
)
# result["overall"].value == "fail"
# result["checks"]["independence"]["reason"] == "Starts with a dependent connective: 'Therefore'"
```

### 3. Citation faithfulness — does the citation actually support the claim?

People search for "RAG citation verification" more than "chunk quality gate" — this is the check that catches AI *appearing* to cite evidence while quietly omitting the premise.

**Three-way verdict** (plus `uncertain` when the LLM is unavailable):

- `supported` — citation both entails the claim **and** stands alone without misleading
- `unsupported` — citation contradicts the claim → **regenerate the answer**
- `insufficient` — citation is incomplete or off-topic (e.g. only the result clause of a conditional, trigger condition cut away) → **re-retrieve**

```python
from rag_verifier import verify_citation

result = verify_citation(
    claim="Apache License 2.0 requires copyright headers.",
    citation="Copyright [yyyy] [name of copyright owner]",
    source_text="... full Apache 2.0 license text ...",
)
# result["status"]  -> "insufficient"
# result["reason"]  -> why the citation doesn't support the claim
```

A **deterministic gate runs before any LLM call**: if the citation is located verbatim in the source and its preceding context shows a truncated conditional (the citation kept only the result clause), the verdict is `insufficient` immediately — no LLM variance, no API cost. This gate alone lifted truncated-conditional detection from 55% to 95% on our evaluation set.

### 4. With the LLM semantic layer

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1   # any OpenAI-compatible endpoint
export LLM_MODEL=deepseek-chat
```

```python
from rag_verifier import verify_chunk, builtin_llm

result = verify_chunk(
    "Therefore Party A has the right to claim compensation from Party B.",
    prev_chunk="If Party B breaches the confidentiality obligation,",
    full_doc="If Party B breaches the confidentiality obligation, therefore Party A ...",
    llm_callback=builtin_llm,   # inject any LLM callable; without it the semantic check returns uncertain
)
```

### 5. LangChain integration (5 lines)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_verifier.integrations.langchain import VerifierTransformer

splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_documents(docs)

gate = VerifierTransformer()              # rule layer, zero LLM calls
clean_chunks = gate.transform_documents(chunks)   # drops FAIL chunks; marks UNCERTAIN in metadata
# clean_chunks -> vectorstore
```

Enable the semantic layer by injecting an LLM callback:

```python
from rag_verifier import builtin_llm
gate = VerifierTransformer(llm_callback=builtin_llm)
```

### 6. Full pipeline: parse → split → verify

```python
from rag_verifier.parsing import parse_document
from rag_verifier.chunking import split_document
from rag_verifier import verify_chunk

doc = parse_document("contract.pdf")      # StructuredDocument
chunks = split_document(doc)              # list[dict]: {"id", "card", "source_text", ...}

prev = ""
for item in chunks:
    content = item["card"]["content"]
    result = verify_chunk(content, prev_chunk=prev, source_text=item.get("source_text", ""))
    if result["overall"].value == "fail":
        failed = [f"{k}: {v['reason']}" for k, v in result["checks"].items() if v["status"] == "fail"]
        print(f"DROP {item['id']}: {'; '.join(failed)}")
    prev = content
```

## Bilingual parity (zh / en) — enforced in CI

Every rule has a Chinese and an English copy, dispatched automatically by `detect_language()` (CJK ratio ≥ 15% → zh). The prompt templates, the `reason` strings, and the three citation verdicts all come in matching pairs. Run an English document → English output. Run a Chinese document → Chinese output. Never mixed.

This is a hard guarantee, not a promise: `scripts/rule_parity_check.py` runs the **positive and negative examples for every rule in both languages** and **fails the build** if any positive case is missed or any negative case is hit by accident. English coverage is not "good enough" — it is byte-for-byte symmetric with Chinese.

## Limitations — not a silver bullet

rag-verifier is a quality gate, not a repair tool. It tells you which chunks are broken; it does not fix them.

- **Does not repair chunks.** A failed chunk needs to be re-split (different chunker, larger overlap) or merged with its neighbor — that's your pipeline's job.
- **Does not evaluate retrieval quality.** It doesn't measure recall@k or whether the right chunk was retrieved. It only checks whether a chunk, once retrieved, is self-sufficient.
- **Does not support scanned PDFs.** No OCR; a scanned PDF yields an empty text layer. Use an OCR pipeline first.
- **Does not support RTF.** RTF headings rarely carry style information, so heading levels are lost during parsing.
- **Chinese-question-on-English-document is not supported.** The verifier detects the document's language and applies matching rules; cross-language question answering is out of scope.
- **Rule layer catches structural defects, not factual errors.** A chunk can be perfectly self-contained and still contain wrong information — that's the LLM / citation check's job, not the chunk gate's.

## Supported formats

Text-layer PDF, DOCX, TXT, Markdown, HTML. Not supported: RTF, scanned PDF (see Limitations above).

## Tests

```bash
python scripts/rule_parity_check.py    # zh/en rule parity — must pass (runs in CI)
```

## Roadmap & Contributing

- **LlamaIndex integration** ( `VerifierTransformer` for LlamaIndex nodes)
- **CLI batch mode** (verify a JSONL of chunks in one command)
- **More document types** for the parsing extra

Found a chunk that *should* be caught but isn't? Or a false positive that broke your pipeline? [Open an issue](https://github.com/hlc0000/rag-verifier/issues) with the chunk text and the expected verdict. Rule-layer fixes are deterministic and permanent — once a pattern is covered, it stays covered.

## License

MIT — see [LICENSE](LICENSE).
