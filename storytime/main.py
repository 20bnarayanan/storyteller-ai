"""Command-line entry point.

    python main.py "a little fox who gets lost in an enchanted forest" --age 6
    python main.py            # interactive prompts
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from llm_client import LLMClient, StorytimeError
from models import LENGTH_SCALE, StoryRequest
from pipeline import MAX_REVISIONS, run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Storytime: age-aware bedtime stories with a fading refrain.")
    p.add_argument("premise", nargs="?", help="what the story is about (omit to be prompted)")
    p.add_argument("--age", type=int, default=None, help="child's age, 5-10 (default 7)")
    p.add_argument("--interests", help="favourite things or characters, e.g. 'trains, owls'")
    p.add_argument("--mood", help="desired mood or theme, e.g. 'cozy and gentle'")
    p.add_argument("--length", choices=list(LENGTH_SCALE), default="medium", help="short / medium / long")
    p.add_argument("--out", type=Path, help="also save the story to this markdown file")
    p.add_argument("--trace-dir", type=Path, help="save every candidate, check, and judge output (trace.json) here")
    p.add_argument("--max-revisions", type=int, choices=(0, 1), default=MAX_REVISIONS, help="allow zero or one revision (default 1)")
    p.add_argument("-v", "--verbose", action="store_true", help="log pipeline decisions")
    return p.parse_args(argv)


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{label}{suffix}: ").strip() or default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        premise, age = args.premise, args.age
        if not premise:  # interactive mode
            premise = _prompt("What should the story be about?")
            if age is None:
                age = int(_prompt("Child's age (5-10)", "7"))
        request = StoryRequest(premise=premise, age=7 if age is None else age, interests=args.interests,
                               mood=args.mood, length=args.length)
        llm = LLMClient()
        result = run_pipeline(llm, request, max_revisions=args.max_revisions, trace_dir=args.trace_dir)
    except (ValueError, StorytimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if result.final_markdown is None:
        print(f"Sorry, I couldn't make a good story this time. {result.message}", file=sys.stderr)
        return 1

    print(result.final_markdown)
    if result.status == "best_effort":
        print(f"\n[note] {result.message}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.final_markdown, encoding="utf-8")
        print(f"[saved to {args.out}]", file=sys.stderr)
    u = result.usage
    print(
        f"[{result.status} | {result.revisions_used} revision(s) | {u.get('calls', 0)} API calls | "
        f"{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out tokens | {datetime.now():%H:%M:%S}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
