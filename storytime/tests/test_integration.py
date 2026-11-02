"""Integration tests: the real pipeline, controller, validators, gate, and presentation layer,
with only the LLM replaced by a scripted fake. They test how the pieces *interact*.
"""
import json
import re

import main as cli
from conftest import ScriptedLLM
from fixtures import FOX_BLUEPRINT, GOOD_STORY, blueprint_dict, judge_dict, story_broken_markers, story_with_marker
from llm_client import LLMError
from models import StoryRequest
from pipeline import MAX_REVISIONS, run_pipeline

FOX_REQUEST = StoryRequest("a little fox who gets lost in an enchanted forest", age=6)

FAIL_THREE = {  # 5/8 checks pass and a critical check fails: below the gate
    "wind_down": "Slow the last paragraphs: cut the running and keep only the quiet den.",
    "age_fit": "Replace 'trembling with trepidation' with simpler words.",
    "natural_prose": "Fix the tense slip in paragraph two.",
}
FAIL_FIVE = {**FAIL_THREE, "cause_and_effect": "Explain how Pip finds the brook.", "climax_has_room": "Show Pip's choice at the oak."}


# ---------------------------------------------------------------------------------------------
# Test 1: end-to-end successful generation
# ---------------------------------------------------------------------------------------------
def test_1_end_to_end_successful_generation(tmp_path):
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST, trace_dir=tmp_path)

    # Planner produced a valid blueprint, and the storyteller was built from it
    assert result.blueprint.refrain.final_whisper == "Follow glimmer."
    story_prompt = llm.calls_for("storyteller")[0].user
    for needle in ("glimmer", "scamper", "enormous", "[[REFRAIN_1]]", "World rules"):
        assert needle in story_prompt
    # Validator accepted it, so the judge ran exactly once, on the rendered story
    initial = result.candidates[0]
    assert initial.hard_ok and initial.judge is not None
    judge_prompt = llm.calls_for("judge")[0].user
    assert "Follow the glimmer, soft paws." in judge_prompt and "[[REFRAIN" not in judge_prompt
    # Controller published the story with no revision
    assert result.status == "passed" and result.revisions_used == 0 and result.best is initial
    assert (len(llm.calls_for("planner")), len(llm.calls_for("storyteller")), len(llm.calls_for("judge"))) == (1, 1, 1)
    # Presentation substituted the refrain exactly, in italics, fading in order
    md = result.final_markdown
    assert "[[" not in md and md.startswith("# Pip and the Glimmer")
    lines = [f"*{l}*" for l in result.blueprint.refrain.lines]
    assert all(l in md for l in lines)
    assert [md.index(l) for l in lines] == sorted(md.index(l) for l in lines)
    assert md.index(lines[-1]) < md.index("Words we met tonight")
    assert "**glimmer**" in md and "**scamper**" in md and "**enormous**" in md   # first use of each word is bold
    # The trace records everything for debugging
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["status"] == "passed" and trace["candidates"][0]["score"] == 1.0
    assert (tmp_path / "story.md").exists()


def test_1b_cli_prints_and_saves_the_final_story(monkeypatch, capsys, tmp_path):
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY], judge=[judge_dict()])
    monkeypatch.setattr(cli, "LLMClient", lambda: llm)
    out = tmp_path / "story.md"

    code = cli.main(["a little fox who gets lost", "--age", "6", "--out", str(out)])

    captured = capsys.readouterr()
    assert code == 0
    assert "# Pip and the Glimmer" in captured.out and "*Follow glimmer.*" in captured.out
    assert "passed" in captured.err
    assert captured.out == out.read_text() + "\n"


def test_1c_cli_rejects_bad_age_without_calling_the_model(monkeypatch, capsys):
    llm = ScriptedLLM()
    monkeypatch.setattr(cli, "LLMClient", lambda: llm)
    assert cli.main(["a fox", "--age", "42"]) == 2
    assert "Age must be" in capsys.readouterr().err
    assert llm.calls == []


# ---------------------------------------------------------------------------------------------
# Test 2: storyteller forgets a vocabulary word -> validator catches it, judge skipped, revision
# ---------------------------------------------------------------------------------------------
def test_2_broken_markers_are_repaired_inside_the_draft_without_using_the_revision():
    llm = ScriptedLLM(
        planner=[FOX_BLUEPRINT],
        storyteller=[story_broken_markers(), GOOD_STORY],
        judge=[judge_dict()],  # only ONE judge response exists: the broken draft must not use it
    )

    result = run_pipeline(llm, FOX_REQUEST)

    # Only one marker is missing, so the draft gets the cheap, surgical repair: ask for just the missing marker.
    repair_prompt = llm.calls_for("storyteller")[1].user
    assert "REFRAIN_2" in repair_prompt and "missing refrain marker" in repair_prompt
    assert "Pip was a little red fox" in repair_prompt  # the broken story was sent along for editing
    assert "World rules" not in repair_prompt  # no blueprint dump: this is a smaller, cheaper call
    # The repaired story is the only candidate, judged once, and the quality revision was never spent
    (draft,) = result.candidates
    assert draft.hard_ok and draft.judge is not None
    assert len(llm.calls_for("judge")) == 1 and len(llm.calls_for("storyteller")) == 2
    assert result.status == "passed" and result.revisions_used == 0 and result.best is draft
    assert "enormous" in result.final_markdown


def test_2a_markers_the_model_ignores_are_placed_by_code():
    """If the repair attempt still leaves the markers wrong, code inserts them (no extra LLM call) and the draft is judged."""
    no_markers = re.sub(r"\[\[REFRAIN_\d\]\]\n\n?", "", GOOD_STORY).strip()
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[no_markers, no_markers], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    (draft,) = result.candidates
    assert draft.hard_ok and draft.judge is not None
    assert all(f"[[REFRAIN_{i}]]" in draft.story for i in range(1, 5))
    assert draft.story.strip().endswith("[[REFRAIN_4]]")
    assert len(llm.calls_for("storyteller")) == 2 and result.revisions_used == 0
    assert result.status == "passed"


def test_2b_multiple_broken_hard_checks_still_get_a_full_revision():
    """missing_markers() only fires for a clean 'some markers missing' case; a story that is ALSO too short (a
    second hard failure) should still get the full revision with blueprint and instructions."""
    too_short = story_broken_markers().split("\n\n")[0] + "\n\n[[REFRAIN_1]]\n\n[[REFRAIN_3]]\n\n[[REFRAIN_4]]"
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[too_short, GOOD_STORY], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    revision_prompt = llm.calls_for("storyteller")[1].user
    assert "MUST FIX" in revision_prompt and "World rules" in revision_prompt  # full revision, not the repair path
    assert result.status == "passed"


# ---------------------------------------------------------------------------------------------
# Test 3: passes structure, judge finds poor pacing, feedback flows into the revision call
# ---------------------------------------------------------------------------------------------
def test_3_judge_feedback_is_passed_to_the_revision_and_revision_is_reevaluated():
    weak = judge_dict(
        {"climax_has_room": "Expand Pip's decision at the oak: show him choosing to stay quiet before he moves.",
         "natural_prose": "Fix the tense slip where 'Pip creeps' appears."},
        strengths_to_preserve=["Hush's explanation of what a glimmer is"],
    )
    weak["checks"]["climax_has_room"]["evidence"] = "He crept around the enormous roots"
    fix = "Pip stopped, closed his eyes, and decided to stay quiet instead of running."
    revised_story = story_with_marker(fix)
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY, revised_story], judge=[weak, judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    original = result.candidates[0]
    assert original.hard_ok and original.score == 0.75 and original.passed   # 6/8, no critical failure: passes the gate
    assert result.status == "passed"
    assert len(llm.calls_for("storyteller")) == 1  # a passing story is not revised


def test_3b_a_failed_critical_check_triggers_a_targeted_revision_that_is_rejudged_against_the_old_review():
    weak = judge_dict(
        {"wind_down": "Slow the last paragraphs: cut the running and keep only the quiet den.",
         "climax_has_room": "Expand Pip's decision at the oak: show him choosing to stay quiet before he moves."},
        strengths_to_preserve=["Hush's explanation of what a glimmer is"],
    )
    weak["checks"]["climax_has_room"]["evidence"] = "He crept around the enormous roots"
    fix = "Pip stopped, closed his eyes, and decided to stay quiet instead of running."
    revised_story = story_with_marker(fix)
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY, revised_story], judge=[weak, judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    original, revised = result.candidates
    assert original.hard_ok and original.score == 0.75 and not original.passed
    assert "critical check failed: wind_down" in original.gate_failures
    revision_prompt = llm.calls_for("storyteller")[1].user
    assert "Slow the last paragraphs" in revision_prompt                       # judge's fix
    assert "Expand Pip's decision at the oak" in revision_prompt
    assert "He crept around the enormous roots" in revision_prompt             # judge's evidence
    assert "Hush's explanation of what a glimmer is" in revision_prompt        # strengths preserved
    assert "[[REFRAIN_1]], [[REFRAIN_2]], [[REFRAIN_3]], [[REFRAIN_4]]" in revision_prompt
    rejudge_prompt = llm.calls_for("judge")[1].user
    assert fix in rejudge_prompt and "PREVIOUS REVIEW" in rejudge_prompt and "Slow the last paragraphs" in rejudge_prompt
    assert result.status == "passed" and result.best is revised and fix in result.final_markdown


# ---------------------------------------------------------------------------------------------
# Test 4: a revision that makes the story worse is rejected
# ---------------------------------------------------------------------------------------------
def test_4_worse_revision_is_rejected_and_original_is_kept():
    original_judge = judge_dict(FAIL_THREE)                    # 5/8
    worse_judge = judge_dict(FAIL_FIVE)                        # 3/8
    llm = ScriptedLLM(
        planner=[FOX_BLUEPRINT],
        storyteller=[GOOD_STORY, story_with_marker("REVISION ONE added a loud noisy sentence.")],
        judge=[original_judge, worse_judge],
    )

    result = run_pipeline(llm, FOX_REQUEST)

    assert [c.score for c in result.candidates] == [0.62, 0.38]
    assert result.status == "best_effort" and result.best.label == "draft_a"
    assert "REVISION" not in result.final_markdown            # worse rewrites are never published
    assert "draft_a" in result.message and "5/8" in result.message


# ---------------------------------------------------------------------------------------------
# Test 5: maximum revision attempts, no infinite loop, controlled outcomes
# ---------------------------------------------------------------------------------------------
def test_5a_every_attempt_fails_validation_returns_controlled_failure():
    too_short = story_broken_markers().split("\n\n")[0]
    llm = ScriptedLLM(  # draft + its repair + the one quality revision; an unexpected extra call fails loudly
        planner=[FOX_BLUEPRINT],
        storyteller=[too_short, too_short, too_short],
    )

    result = run_pipeline(llm, FOX_REQUEST)

    assert MAX_REVISIONS == 1
    assert len(llm.calls_for("storyteller")) == 3 and result.revisions_used == 1
    assert llm.calls_for("judge") == []                     # hard failures never reach the judge
    assert result.status == "failed" and result.final_markdown is None
    assert "Remaining problems" in result.message and "length" in result.message


def test_5b_valid_original_survives_when_the_revision_breaks_validation():
    failing_judge = judge_dict(FAIL_THREE)
    llm = ScriptedLLM(
        planner=[FOX_BLUEPRINT],
        storyteller=[GOOD_STORY, story_broken_markers()],
        judge=[failing_judge],
    )

    result = run_pipeline(llm, FOX_REQUEST)

    assert len(llm.calls_for("storyteller")) == 2 and len(llm.calls_for("judge")) == 1
    assert result.status == "best_effort" and result.best.label == "draft_a"
    assert result.final_markdown is not None and "enormous" in result.final_markdown
    assert "critical check failed: wind_down" in result.message


def test_5c_max_revisions_zero_means_no_revision():
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY],
                      judge=[judge_dict(FAIL_THREE)])
    result = run_pipeline(llm, FOX_REQUEST, max_revisions=0)
    assert result.revisions_used == 0 and len(llm.calls_for("storyteller")) == 1
    assert result.status == "best_effort"


def test_5d_api_failure_during_revision_returns_best_candidate_so_far():
    class FailingSecondStory(ScriptedLLM):
        def complete_text(self, role, system, user):
            if role == "storyteller" and len(self.calls_for("storyteller")) >= 1:
                raise LLMError("storyteller call failed after 4 attempts: connection reset")
            return super().complete_text(role, system, user)

    llm = FailingSecondStory(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY],
                             judge=[judge_dict(FAIL_THREE)])
    result = run_pipeline(llm, FOX_REQUEST)
    assert result.status == "best_effort" and result.final_markdown is not None
    assert "Stopped early" in result.message and "connection reset" in result.message


# ---------------------------------------------------------------------------------------------
# Extra: planner validation happens BEFORE any story is written
# ---------------------------------------------------------------------------------------------
def test_6_invalid_refrain_in_blueprint_is_fixed_by_a_cheap_planner_retry():
    bad = blueprint_dict()
    bad["refrain"] = {**FOX_BLUEPRINT["refrain"],
                      "versions": ["Follow the glimmer, soft paws.", "Follow the glimmer, soft paws, slow steps.", "Follow the glimmer."]}
    llm = ScriptedLLM(planner=[bad, FOX_BLUEPRINT], storyteller=[GOOD_STORY], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    assert len(llm.calls_for("planner")) == 2 and len(llm.calls_for("storyteller")) == 1
    assert "strictly shorter" in llm.calls_for("planner")[1].user
    assert result.status == "passed"


def test_6b_planner_that_never_produces_a_valid_blueprint_fails_before_writing_anything():
    bad = blueprint_dict(vocabulary=[])
    llm = ScriptedLLM(planner=[bad] * 5)
    result = run_pipeline(llm, FOX_REQUEST)
    assert result.status == "failed" and result.final_markdown is None
    assert "Stopped early" in result.message
    assert llm.calls_for("storyteller") == [] and llm.calls_for("judge") == []


# ---------------------------------------------------------------------------------------------
# Extra: safety flag from the judge blocks publication of that candidate
# ---------------------------------------------------------------------------------------------
def test_7_judge_unsafe_flag_fails_the_gate_and_forces_a_revision():
    unsafe = judge_dict({"age_fit": "Remove the frightening image of the fog creature."}, unsafe_content=True)
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY, story_with_marker("Everything stayed cozy.")],
                      judge=[unsafe, judge_dict()])
    result = run_pipeline(llm, FOX_REQUEST)
    assert "unsuitable" in " ".join(result.candidates[0].gate_failures)
    assert "Remove the frightening image" in llm.calls_for("storyteller")[1].user
    assert result.status == "passed" and result.best.label == "revision_1"


def test_8_usage_is_reported_for_cost_tracking():
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY], judge=[judge_dict()])
    result = run_pipeline(llm, FOX_REQUEST)
    assert result.usage["calls"] == 3 and result.usage["by_role"] == {"planner": 1, "storyteller": 1, "judge": 1}




def test_revision_limit_is_capped_at_one_even_for_direct_callers():
    too_short = story_broken_markers().split("\n\n")[0]
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[too_short] * 3)
    result = run_pipeline(llm, FOX_REQUEST, max_revisions=5)
    assert result.revisions_used == 1
    assert len(llm.calls_for("storyteller")) == 3  # draft, its repair, and exactly one revision


def test_unsafe_judged_story_is_never_returned_as_final_output():
    llm = ScriptedLLM(
        planner=[FOX_BLUEPRINT],
        storyteller=[GOOD_STORY, GOOD_STORY],
        judge=[judge_dict({"age_fit": "Remove unsafe details."}, unsafe_content=True),
               judge_dict(unsafe_content=True)],
    )
    result = run_pipeline(llm, FOX_REQUEST)
    assert result.status == "failed"
    assert result.final_markdown is None
    assert result.best is not None and not result.best.hard_ok


# ---------------------------------------------------------------------------------------------
# Best-of-2 drafts (threads, so the scripted answers are interchangeable between drafts)
# ---------------------------------------------------------------------------------------------
def test_two_drafts_are_judged_and_the_better_one_continues_without_a_revision():
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY, GOOD_STORY],
                      judge=[judge_dict(FAIL_THREE), judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST, drafts=2)

    assert sorted(c.label for c in result.candidates) == ["draft_a", "draft_b"]
    assert sorted(c.score for c in result.candidates) == [0.62, 1.0]
    assert result.best.score == 1.0 and result.status == "passed"
    assert result.revisions_used == 0 and len(llm.calls_for("storyteller")) == 2
    assert result.usage["calls"] == 5   # planner + 2 writes + 2 judges, counted safely across threads


def test_a_failed_draft_is_dropped_when_the_other_draft_survives():
    class FirstWriteFails(ScriptedLLM):
        failed = False

        def complete_text(self, role, system, user):
            if role == "storyteller" and not self.failed:
                self.failed = True
                raise LLMError("storyteller call failed: boom")
            return super().complete_text(role, system, user)

    llm = FirstWriteFails(planner=[FOX_BLUEPRINT], storyteller=[GOOD_STORY], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST, drafts=2)

    assert len(result.candidates) == 1 and result.status == "passed"


def test_all_drafts_failing_is_a_controlled_failure():
    class AlwaysFails(ScriptedLLM):
        def complete_text(self, role, system, user):
            raise LLMError("storyteller call failed: down")

    result = run_pipeline(AlwaysFails(planner=[FOX_BLUEPRINT]), FOX_REQUEST, drafts=2)

    assert result.status == "failed" and "Stopped early" in result.message


# ---------------------------------------------------------------------------------------------
# Trimming a draft that is well over the word cap
# ---------------------------------------------------------------------------------------------
def _too_long_story() -> str:
    from models import word_count
    first, rest = GOOD_STORY.split("\n\n", 1)
    lo, hi = FOX_REQUEST.word_range()
    need = int(hi * 1.2) - word_count(GOOD_STORY)
    pad = " ".join(["Pip walked on through the soft quiet woods."] * (need // 8 + 1))
    return f"{first}\n\n{pad}\n\n{rest}"


def test_a_draft_far_over_the_cap_is_trimmed_before_it_is_judged():
    long_story = _too_long_story()
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[long_story, GOOD_STORY], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    assert "LENGTH (do only this)" in llm.calls_for("storyteller")[1].user
    (draft,) = result.candidates
    assert draft.story == GOOD_STORY and result.revisions_used == 0 and result.status == "passed"


def test_a_trim_that_breaks_the_markers_is_thrown_away():
    long_story = _too_long_story()
    llm = ScriptedLLM(planner=[FOX_BLUEPRINT], storyteller=[long_story, story_broken_markers()], judge=[judge_dict()])

    result = run_pipeline(llm, FOX_REQUEST)

    assert result.candidates[0].story == long_story          # the valid long draft was kept, not the broken trim
    assert len(llm.calls_for("storyteller")) == 2
