"""命令行入口：rag-verifier check / rag-verifier citation。

零依赖（仅用 argparse + json），pip 安装后即可使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from rag_verifier import builtin_llm, verify_chunk as _verify_chunk
from rag_verifier.citation.verifier import verify_citation as _verify_citation


def _print_result(result: dict, as_json: bool) -> None:
    if as_json:
        # overall 是 enum，转成字符串便于 JSON 序列化
        out = {
            "overall": result["overall"].value if hasattr(result["overall"], "value") else result["overall"],
            "checks": {
                k: {
                    "status": v["status"].value if hasattr(v["status"], "value") else v["status"],
                    "reason": v["reason"],
                }
                for k, v in result["checks"].items()
            },
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    overall = result["overall"].value if hasattr(result["overall"], "value") else result["overall"]
    symbol = "PASS" if overall == "pass" else ("FAIL" if overall == "fail" else "UNCERTAIN")
    print(f"[{symbol}] overall = {overall}")
    for name, info in result["checks"].items():
        st = info["status"].value if hasattr(info["status"], "value") else info["status"]
        print(f"  - {name}: {st}  ({info['reason']})")


def cmd_check(args: argparse.Namespace) -> int:
    content = args.content
    if content == "-":
        content = sys.stdin.read()

    llm_cb = builtin_llm if args.llm else None
    result = _verify_chunk(
        content,
        source_text=args.source_text or "",
        prev_chunk=args.prev or "",
        full_doc=args.full_doc or "",
        llm_callback=llm_cb,
    )
    _print_result(result, args.json)
    overall = result["overall"].value if hasattr(result["overall"], "value") else result["overall"]
    return 0 if overall == "pass" else 1


def cmd_citation(args: argparse.Namespace) -> int:
    result = _verify_citation(
        claim=args.claim,
        citation=args.citation,
        source_text=args.source_text,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = result.get("status", "?")
        print(f"[{status.upper()}] {result.get('reason', '')}")
    return 0 if result.get("status") == "supported" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-verifier",
        description="Verify RAG chunk quality and citation faithfulness from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    p_check = sub.add_parser("check", help="Verify a single chunk (rule layer by default).")
    p_check.add_argument("content", help="Chunk text. Use '-' to read from stdin.")
    p_check.add_argument("--prev", default="", help="Previous chunk content (for cross-boundary checks).")
    p_check.add_argument("--source-text", default="", help="Source text of the chunk's section.")
    p_check.add_argument("--full-doc", default="", help="Full document text (semantic layer context).")
    p_check.add_argument("--llm", action="store_true", help="Enable LLM semantic layer (needs LLM_API_KEY etc.).")
    p_check.add_argument("--json", action="store_true", help="Output JSON instead of human-readable.")
    p_check.set_defaults(func=cmd_check)

    # citation
    p_cit = sub.add_parser("citation", help="Verify whether a citation supports a claim.")
    p_cit.add_argument("--claim", required=True, help="The AI-generated claim.")
    p_cit.add_argument("--citation", required=True, help="The cited excerpt.")
    p_cit.add_argument("--source-text", required=True, help="Full source text containing the citation.")
    p_cit.add_argument("--json", action="store_true", help="Output JSON instead of human-readable.")
    p_cit.set_defaults(func=cmd_citation)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
