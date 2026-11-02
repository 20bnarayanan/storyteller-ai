"""Judging + the quality gate. Python, not the LLM, decides what passes.

The judge answers eight pass/fail questions (models.CHECK_NAMES), each with quoted evidence and, when it fails, one concrete
fix. This module counts the passes, applies the gate, lets deterministic validators overrule the judge on facts they can
verify, ranks candidates, and turns failed checks into revision instructions.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from llm_client import LLM, LLMOutputError, complete_validated
from models import (
    CHECK_NAMES, CRITICAL_CHECKS, JUDGE_SCHEMA, MAX_FIXES_PER_REVISION, Blueprint, Candidate, JudgeCheck, JudgeResult,
    StoryRequest,
)
from presentation import build_judge_view
from prompts import JUDGE_SYSTEM, build_judge_user
from validators import remaining_word_room, run_validators

logger = logging.getLogger(__name__)

# A story passes when at least this many of the eight checks pass (and no critical check fails, and nothing is unsafe).
DEFAULT_PASS_MIN_CHECKS = 6
# A story this far over its word target is too long for bedtime: it fails the gate (so it gets a trimming revision) but is
# never discarded for it. Slightly long stories pass.
LENGTH_GATE_RATIO = 1.25


def pass_min_checks_from_env() -> int:
    try:
        value = int(os.getenv("STORYTIME_PASS_MIN_CHECKS", DEFAULT_PASS_MIN_CHECKS))
    except ValueError:
        return DEFAULT_PASS_MIN_CHECKS
    return max(1, min(value, len(CHECK_NAMES)))


PASS_MIN_CHECKS = pass_min_checks_from_env()


def check_score(judge: JudgeResult) -> float:
    """Fraction of checks passed (0-1). Computed here, never taken from the judge."""
    return round(judge.passed_count / len(CHECK_NAMES), 2)


def gate_failures(judge: JudgeResult, word_count: int = 0, word_cap: int = 0) -> list[str]:
    """Reasons the story fails the quality gate (empty list = passes)."""
    failures: list[str] = []
    if word_cap and word_count > word_cap * LENGTH_GATE_RATIO:
        failures.append(f"{word_count} words is far over the {word_cap}-word bedtime target; trim to under {word_cap}")
    if judge.unsafe_content:
        failures.append("judge flagged content unsuitable for the child's age")
    failures += [f"critical check failed: {name}" for name in CRITICAL_CHECKS if not judge.checks[name].passed]
    if judge.passed_count < PASS_MIN_CHECKS:
        failures.append(f"only {judge.passed_count}/{len(CHECK_NAMES)} checks passed (need {PASS_MIN_CHECKS})")
    return failures


def apply_validator_overrides(judge: JudgeResult, candidate_checks) -> None:
    """Facts the code can verify beat the judge's opinion: a word that is not in the text is not in context, and a lead-in
    that trails off is not a natural refrain pause."""
    by_name = {c.name: c for c in candidate_checks}
    vocab = by_name.get("vocabulary_present")
    if vocab and not vocab.passed and judge.checks["vocab_in_context"].passed:
        judge.checks["vocab_in_context"] = JudgeCheck(False, vocab.detail, f"Work the missing word(s) into the narration. {vocab.detail}")
    lead = by_name.get("lead_ins_complete")
    if lead and not lead.passed and judge.checks["refrain_natural"].passed:
        judge.checks["refrain_natural"] = JudgeCheck(False, lead.detail, lead.detail)


def judge_story(
    llm: LLM, request: StoryRequest, blueprint: Blueprint, story: str, soft_warnings: list[str],
    previous: Optional[JudgeResult] = None,
) -> JudgeResult:
    """One judge call. The prompt asks for evidence before each verdict; deterministic validators stay authoritative.

    ``previous`` is the judge's earlier review when this story is a revision. The API gives no way to lower the model's
    randomness, so re-judging a rewrite from scratch can flip verdicts for no reason; showing the earlier review makes the
    judge check "was each requested fix done?" instead of re-litigating everything.
    """
    user = build_judge_user(request, blueprint, build_judge_view(story, blueprint), soft_warnings, previous)
    return complete_validated(
        llm, "judge", JUDGE_SYSTEM, user, "submit_evaluation", JUDGE_SCHEMA, JudgeResult.from_dict, attempts=2,
    )


def evaluate_candidate(
    llm: LLM, request: StoryRequest, blueprint: Blueprint, story: str, label: str, previous: Optional[JudgeResult] = None,
) -> Candidate:
    """Validate first. A hard failure skips the (paid) judge call and goes straight to revision."""
    candidate = Candidate(label=label, story=story, checks=run_validators(story, blueprint, request))
    if not candidate.hard_ok:
        logger.info("%s failed hard checks; skipping judge", label)
        return candidate
    soft = [f"{c.name}: {c.detail}" for c in candidate.checks if not c.passed and not c.hard]
    try:
        candidate.judge = judge_story(llm, request, blueprint, story, soft, previous)
    except LLMOutputError as exc:
        candidate.judge_error = str(exc)
        logger.warning("%s: judge output unusable: %s", label, exc)
        return candidate
    apply_validator_overrides(candidate.judge, candidate.checks)
    candidate.score = check_score(candidate.judge)
    count, _lo, hi, _room = remaining_word_room(story, request)
    candidate.gate_failures = gate_failures(candidate.judge, count, hi)
    return candidate


def rank_key(c: Candidate) -> tuple[int, int, float]:
    """Best candidate = passed the gate, then valid, then most checks passed. Unscored candidates rank lowest in their tier."""
    return (int(c.passed), int(c.hard_ok), c.score if c.score is not None else -1.0)


def best_candidate(candidates: list[Candidate]) -> Optional[Candidate]:
    # max() keeps the earliest of equal keys, so ties favour the original over a later rewrite.
    return max(candidates, key=rank_key) if candidates else None


def build_revision_instructions(c: Candidate) -> str:
    """Turn validator failures and failed judge checks into one short, targeted instruction block.

    Only the highest-priority fixes are sent (MAX_FIXES_PER_REVISION): a rewrite that tries to do everything tends to
    break something that was working.
    """
    lines: list[str] = []
    hard_failures = [x for x in c.checks if x.hard and not x.passed]
    if hard_failures:
        lines.append("MUST FIX (automatic checks failed):")
        lines += [f"- {x.name}: {x.detail}" for x in hard_failures]
    if c.judge:
        j = c.judge
        too_long = [f for f in c.gate_failures if "far over" in f]
        if too_long:
            lines.append("LENGTH (do this first): " + too_long[0])
        # Critical checks first, then the rest in checklist order.
        order = [n for n in CHECK_NAMES if n in CRITICAL_CHECKS] + [n for n in CHECK_NAMES if n not in CRITICAL_CHECKS]
        failed = [n for n in order if n in j.failed][:MAX_FIXES_PER_REVISION]
        if failed:
            lines.append("EDITOR'S FIXES (most important first):")
            for name in failed:
                check = j.checks[name]
                evidence = f" (evidence: \"{check.evidence}\")" if check.evidence and check.evidence not in check.fix else ""
                lines.append(f"- [{name}] {check.fix}{evidence}")
        if j.unsafe_content:
            lines.append("SAFETY: remove or soften anything frightening or unsuitable for the child's age.")
    # Advisory validator findings the judge checks do not already cover (those became failed checks via the overrides).
    minor = [x for x in c.checks if not x.hard and not x.passed and x.name not in ("vocabulary_present", "lead_ins_complete", "length_target")]
    if lines and minor:
        lines.append("ALSO IMPROVE (minor):")
        lines += [f"- {x.name}: {x.detail}" for x in minor[:2]]
    if lines and c.judge and c.judge.strengths_to_preserve:
        lines.append("PRESERVE THESE STRENGTHS (do not change them):")
        lines += [f"- {s}" for s in c.judge.strengths_to_preserve]
    return "\n".join(lines)
