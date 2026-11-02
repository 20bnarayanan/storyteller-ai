"""The controller. Python owns the whole loop; the judge only advises.

    plan -> write 2 drafts in parallel -> [fix hard failures -> judge -> gate] each -> keep the better draft
         -> (one quality revision -> judge) -> format

* Two drafts are written and judged in parallel and the better one continues (best-of-2 costs no extra waiting time, and picking
  between two drafts is more reliable than rewriting a weak one).
* A draft that breaks a hard check (refrain markers, runaway length) gets one repair attempt that does NOT use up the quality
  revision. If the markers are still wrong, code places them itself (story_engine.place_markers).
* At most MAX_REVISIONS quality revisions, so cost and time are bounded. The final answer is whichever valid candidate passed the
  most judge checks (ties keep the earlier one, so a rewrite must actually be better to replace it).

WHAT I WOULD BUILD WITH TWO MORE HOURS
1. I would focus on building an extra safety check that works independently of the LLM judge. For example, I would add Python-based 
    checks to catch potentially unsafe content, language that may not be appropriate for the child's age, and inconsistencies in the story, 
    such as characters or details changing unexpectedly. This way, the application would not have to rely entirely on the LLM to decide 
    whether a story is suitable. I would also test it with a few deliberately challenging story requests to see how well it catches 
    problems before delivering the final story.
2. Add the comparison harness I used ad hoc (~20 varied premises, a frozen judge, blind pairwise comparisons in both orders)
    to the repo as a regression test, so a "better" prompt has to win by more than noise.
3. Parent preferences: names, favourite animals and topics to avoid, remembered between nights, plus a "read it again but
    gentler" button that reuses the revision step with the parent's note as the instruction.
4. A second judge from a different model family, to reduce the bias of Claude grading Claude's own writing.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from evaluation_engine import best_candidate, build_revision_instructions, evaluate_candidate
from llm_client import LLM, StorytimeError
from models import Candidate, PipelineResult, StoryRequest
from presentation import build_final_markdown
from story_engine import place_markers, plan_story, repair_markers, revise_story, write_story
from validators import missing_markers, remaining_word_room, run_validators

logger = logging.getLogger(__name__)

MAX_REVISIONS = 1
DRAFTS = 2  # drafts written in parallel; the better one continues
TRIM_RATIO = 1.05  # a draft this far over the word cap gets one trimming pass before it is judged
TRIM_TARGET = 0.92  # ...aimed at this share of the cap, because rewrites tend to land a little over the target


def run_pipeline(
    llm: LLM,
    request: StoryRequest,
    max_revisions: int = MAX_REVISIONS,
    trace_dir: Optional[Path] = None,
    drafts: Optional[int] = None,
) -> PipelineResult:
    # Keep the cost and control-flow bound even if a caller passes a larger value.
    max_revisions = max(0, min(int(max_revisions), MAX_REVISIONS))
    drafts = max(1, min(int(DRAFTS if drafts is None else drafts), 3))
    candidates: list[Candidate] = []
    blueprint = None
    message = ""
    revisions = 0

    try:
        blueprint = plan_story(llm, request)
        logger.info("Blueprint ready: %s", blueprint.title)

        candidates = _write_drafts(llm, request, blueprint, drafts)
        for c in candidates:
            _log_candidate(c)

        while revisions < max_revisions:
            best = best_candidate(candidates)
            if best.passed:
                break
            hard_failed = {c.name for c in best.checks if c.hard and not c.passed}
            # Only take the cheap, surgical path when the missing-marker check is the SOLE hard failure.
            gone = missing_markers(best.story) if hard_failed == {"refrain_placement"} else []
            if gone:
                revisions += 1
                logger.info("Revision %d/%d: targeted repair of %d missing refrain marker(s)", revisions, max_revisions, len(gone))
                story = repair_markers(llm, best.story, gone)
            else:
                instructions = build_revision_instructions(best)
                if not instructions.strip():
                    message = "No actionable feedback was available to guide a revision."
                    break
                revisions += 1
                logger.info("Revision %d/%d, starting from the best candidate so far (%s)", revisions, max_revisions, best.label)
                story = revise_story(llm, request, blueprint, best.story, instructions)
            candidates.append(evaluate_candidate(llm, request, blueprint, story, f"revision_{revisions}", previous=best.judge))
            _log_candidate(candidates[-1])
    except StorytimeError as exc:  # API failure, unusable model output, config problem
        message = f"Stopped early: {exc}"
        logger.error(message)

    result = _finalize(llm, request, blueprint, candidates, message, revisions)
    if trace_dir is not None:
        _write_trace(trace_dir, request, result)
    return result


def _write_drafts(llm: LLM, request: StoryRequest, blueprint, n: int) -> list[Candidate]:
    """Write, fix and judge n drafts (in parallel threads when n > 1). A draft that fails with an API or output error is dropped
    as long as at least one other draft survives."""
    labels = [f"draft_{chr(ord('a') + i)}" for i in range(n)]
    if n == 1:
        return [_make_draft(llm, request, blueprint, labels[0])]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_make_draft, llm, request, blueprint, label) for label in labels]
        results, error = [], None
        for f in futures:
            try:
                results.append(f.result())
            except StorytimeError as exc:
                error = exc
                logger.warning("a draft failed: %s", exc)
    if not results:
        raise error
    return results


def _make_draft(llm: LLM, request: StoryRequest, blueprint, label: str) -> Candidate:
    """One draft: write it, repair hard failures once (outside the revision budget), then validate and judge it."""
    story = write_story(llm, request, blueprint)
    hard = [c for c in run_validators(story, blueprint, request) if c.hard and not c.passed]
    if hard:
        logger.info("%s broke hard check(s) %s; repairing", label, [c.name for c in hard])
        gone = missing_markers(story) if {c.name for c in hard} == {"refrain_placement"} else []
        if gone:
            story = repair_markers(llm, story, gone)
        else:
            instructions = build_revision_instructions(Candidate(label=label, story=story, checks=hard))
            story = revise_story(llm, request, blueprint, story, instructions)
        if any(c.name == "refrain_placement" and not c.passed for c in run_validators(story, blueprint, request)):
            logger.info("%s: markers still wrong; placing them in code", label)
            story = place_markers(story)
    story = _trim_if_long(llm, request, blueprint, story, label)
    return evaluate_candidate(llm, request, blueprint, story, label)


def _trim_if_long(llm: LLM, request: StoryRequest, blueprint, story: str, label: str) -> str:
    """Bedtime stories must stay short. When a draft is well over the word cap, ask for one focused cut and keep the result
    only if it is shorter and still passes every hard check."""
    count, _lo, hi, _room = remaining_word_room(story, request)
    if count <= hi * TRIM_RATIO:
        return story
    logger.info("%s has %d words (cap %d); trimming", label, count, hi)
    instructions = (
        f"LENGTH (do only this): the story is {count} words and the cap is {hi}. Cut about {count - int(hi * TRIM_TARGET)} words by merging or "
        "dropping the weakest scene or description, not by trimming every sentence. Keep the climax, the ending, all three "
        "vocabulary words and the refrain markers."
    )
    trimmed = revise_story(llm, request, blueprint, story, instructions)
    shorter = remaining_word_room(trimmed, request)[0] < count
    valid = all(c.passed for c in run_validators(trimmed, blueprint, request) if c.hard)
    return trimmed if shorter and valid else story


def _finalize(llm: LLM, request: StoryRequest, blueprint, candidates: list[Candidate], message: str, revisions_used: int = 0) -> PipelineResult:
    usage = llm.stats.snapshot()
    best = best_candidate(candidates)

    if blueprint is None or best is None or not best.hard_ok:
        reasons = []
        if best is not None:
            reasons = [f"{c.name}: {c.detail}" for c in best.checks if c.hard and not c.passed]
        detail = message or "No version of the story passed the automatic checks."
        if reasons:
            detail += " Remaining problems: " + "; ".join(reasons)
        return PipelineResult("failed", detail, blueprint, candidates, best, None, revisions_used, usage)

    markdown = build_final_markdown(best.story, blueprint)
    if best.passed:
        return PipelineResult("passed", f"Passed: {best.checks_label}.", blueprint, candidates, best, markdown, revisions_used, usage)

    problems = best.gate_failures or ([best.judge_error] if best.judge_error else ["the story was never scored"])
    detail = (
        f"No version cleared the quality gate after {revisions_used} revision(s). "
        f"Returning the best valid candidate ({best.label}, {best.checks_label}). Open issues: {'; '.join(problems)}"
    )
    if message:
        detail = f"{message} {detail}"
    return PipelineResult("best_effort", detail, blueprint, candidates, best, markdown, revisions_used, usage)


def _log_candidate(c: Candidate) -> None:
    failed = [f"{x.name}" for x in c.checks if x.hard and not x.passed]
    logger.info(
        "%s: hard_ok=%s %s passed=%s%s%s",
        c.label, c.hard_ok, c.checks_label, c.passed,
        f" hard_failures={failed}" if failed else "",
        f" gate_failures={c.gate_failures}" if c.gate_failures else "",
    )


def _write_trace(trace_dir: Path, request: StoryRequest, result: PipelineResult) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": request.__dict__,
        "status": result.status,
        "message": result.message,
        "revisions_used": result.revisions_used,
        "usage": result.usage,
        "blueprint": result.blueprint.to_dict() if result.blueprint else None,
        "candidates": [c.to_dict() for c in result.candidates],
        "best": result.best.label if result.best else None,
    }
    (trace_dir / "trace.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if result.final_markdown:
        (trace_dir / "story.md").write_text(result.final_markdown, encoding="utf-8")
