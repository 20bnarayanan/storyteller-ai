import evaluation_engine
from evaluation_engine import (
    LENGTH_GATE_RATIO, PASS_MIN_CHECKS, best_candidate, build_revision_instructions, check_score, evaluate_candidate,
    gate_failures, rank_key,
)
from fixtures import GOOD_STORY, judge_dict, story_broken_markers
from conftest import ScriptedLLM
from models import CHECK_NAMES, CRITICAL_CHECKS, Candidate, CheckResult, JudgeResult

NON_CRITICAL = [n for n in CHECK_NAMES if n not in CRITICAL_CHECKS]


def J(failing=None, **kw) -> JudgeResult:
    return JudgeResult.from_dict(judge_dict(failing, **kw))


def fails(names, fix="Change the sentence about X to Y and cut Z."):
    return {n: fix for n in names}


class TestScoring:
    def test_score_is_fraction_of_checks_passed_computed_in_python(self):
        assert check_score(J()) == 1.0
        assert check_score(J(fails(NON_CRITICAL[:2]))) == 0.75


class TestGate:
    def test_all_pass(self):
        assert gate_failures(J()) == []

    def test_minimum_is_inclusive(self):
        allowed = len(CHECK_NAMES) - PASS_MIN_CHECKS
        assert gate_failures(J(fails(NON_CRITICAL[:allowed]))) == []
        assert any("checks passed" in f for f in gate_failures(J(fails(NON_CRITICAL[:allowed + 1]))))

    def test_a_critical_failure_blocks_even_with_many_passes(self):
        out = gate_failures(J(fails(["wind_down"])))
        assert out == ["critical check failed: wind_down"]

    def test_unsafe_fails(self):
        assert any("unsuitable" in f for f in gate_failures(J(unsafe_content=True)))

    def test_slightly_long_passes_but_far_over_fails(self):
        assert gate_failures(J(), word_count=int(500 * 1.2), word_cap=500) == []
        assert any("far over" in f for f in gate_failures(J(), word_count=int(500 * LENGTH_GATE_RATIO) + 5, word_cap=500))


class TestRanking:
    def mk(self, label, hard_ok=True, score=None, failures=()):
        c = Candidate(label, "s", checks=[CheckResult("x", hard_ok, True)], gate_failures=list(failures), score=score)
        if score is not None:
            c.judge = J()
        return c

    def test_passed_beats_everything(self):
        assert best_candidate([self.mk("f", score=0.9, failures=["bad"]), self.mk("p", score=0.75)]).label == "p"

    def test_valid_scored_beats_invalid_unscored(self):
        assert best_candidate([self.mk("invalid", hard_ok=False), self.mk("scored", score=0.4, failures=["low"])]).label == "scored"

    def test_valid_but_unscored_ranks_below_scored(self):
        assert best_candidate([self.mk("unscored"), self.mk("scored", score=0.3, failures=["low"])]).label == "scored"

    def test_higher_score_wins_among_failures(self):
        assert best_candidate([self.mk("a", score=0.4, failures=["x"]), self.mk("b", score=0.6, failures=["x"])]).label == "b"

    def test_tie_keeps_the_earlier_candidate(self):
        a, b = self.mk("original", score=0.5, failures=["x"]), self.mk("rewrite", score=0.5, failures=["x"])
        assert best_candidate([a, b]).label == "original"

    def test_rank_key_orders_tiers(self):
        assert rank_key(self.mk("p", score=0.9)) > rank_key(self.mk("f", score=0.4, failures=["x"])) > rank_key(self.mk("i", hard_ok=False))

    def test_empty(self):
        assert best_candidate([]) is None


class TestEvaluateCandidate:
    def test_validator_failure_skips_judge(self, blueprint, request_age6):
        llm = ScriptedLLM()
        c = evaluate_candidate(llm, request_age6, blueprint, story_broken_markers(), "initial")
        assert not c.hard_ok and c.judge is None and c.score is None and not c.passed
        assert llm.calls_for("judge") == []

    def test_judged_candidate_gets_score_and_gate(self, blueprint, request_age6):
        c = evaluate_candidate(ScriptedLLM(judge=[judge_dict()]), request_age6, blueprint, GOOD_STORY, "initial")
        assert c.passed and c.score == 1.0 and c.gate_failures == []

    def test_judge_sees_rendered_story(self, blueprint, request_age6):
        llm = ScriptedLLM(judge=[judge_dict()])
        evaluate_candidate(llm, request_age6, blueprint, GOOD_STORY, "initial")
        prompt = llm.calls_for("judge")[0].user
        assert "Follow the glimmer, soft paws, slow steps." in prompt and "[[REFRAIN" not in prompt

    def test_unusable_judge_output_is_recorded_not_raised(self, blueprint, request_age6):
        bad = judge_dict()
        bad["checks"]["wind_down"]["passed"] = "maybe"
        c = evaluate_candidate(ScriptedLLM(judge=[bad, bad]), request_age6, blueprint, GOOD_STORY, "initial")
        assert c.hard_ok and c.judge is None and c.judge_error and not c.passed

    def test_repair_retry_feeds_the_error_back(self, blueprint, request_age6):
        bad = judge_dict(fails(["wind_down"], fix=""))  # failed check without a fix is invalid
        llm = ScriptedLLM(judge=[bad, judge_dict()])
        assert evaluate_candidate(llm, request_age6, blueprint, GOOD_STORY, "initial").passed
        assert "wind_down" in llm.calls_for("judge")[1].user

    def test_rejudging_a_revision_shows_the_previous_review(self, blueprint, request_age6):
        previous = J(fails(["natural_prose"], "Fix the tense slip in paragraph two."))
        llm = ScriptedLLM(judge=[judge_dict()])
        evaluate_candidate(llm, request_age6, blueprint, GOOD_STORY, "revision_1", previous=previous)
        prompt = llm.calls_for("judge")[0].user
        assert "PREVIOUS REVIEW" in prompt and "Fix the tense slip in paragraph two." in prompt

    def test_validator_overrides_a_judge_that_missed_a_missing_word(self, blueprint, request_age6):
        # vocabulary_present is a hard validator here, so the judge never runs; check the soft override on the judge result directly.
        from evaluation_engine import apply_validator_overrides
        judge = J()
        apply_validator_overrides(judge, [CheckResult("vocabulary_present", False, False, "missing 'glimmer' in narration")])
        assert not judge.checks["vocab_in_context"].passed and "glimmer" in judge.checks["vocab_in_context"].fix

    def test_far_too_long_story_fails_the_gate_but_is_kept(self, blueprint, request_age6, monkeypatch):
        monkeypatch.setattr(evaluation_engine, "remaining_word_room", lambda story, req: (2000, 300, 400, -1600))
        c = evaluate_candidate(ScriptedLLM(judge=[judge_dict()]), request_age6, blueprint, GOOD_STORY, "initial")
        assert c.hard_ok and not c.passed and any("far over" in f for f in c.gate_failures)


class TestRevisionInstructions:
    def test_includes_validator_failures(self, blueprint, request_age6):
        c = evaluate_candidate(ScriptedLLM(), request_age6, blueprint, story_broken_markers(), "initial")
        text = build_revision_instructions(c)
        assert "MUST FIX" in text and "REFRAIN_2" in text

    def test_critical_fixes_first_capped_and_strengths_preserved(self, blueprint, request_age6):
        payload = judge_dict(fails(["age_fit", "natural_prose", "wind_down", "cause_and_effect", "climax_has_room"], "Fix it here."))
        payload["checks"]["wind_down"]["fix"] = "Slow the last paragraphs: cut the fox's running."
        c = evaluate_candidate(ScriptedLLM(judge=[payload]), request_age6, blueprint, GOOD_STORY, "initial")
        text = build_revision_instructions(c)
        assert text.index("[wind_down]") < text.index("[cause_and_effect]")
        assert text.count("\n- [") == evaluation_engine.MAX_FIXES_PER_REVISION
        assert "PRESERVE THESE STRENGTHS" in text and "owl's explanation" in text

    def test_passing_candidate_has_no_instructions(self, blueprint, request_age6):
        c = evaluate_candidate(ScriptedLLM(judge=[judge_dict()]), request_age6, blueprint, GOOD_STORY, "initial")
        assert build_revision_instructions(c).strip() == ""
