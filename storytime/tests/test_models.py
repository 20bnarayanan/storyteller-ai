import pytest

from fixtures import FOX_BLUEPRINT, blueprint_dict, judge_dict
from models import PLACEHOLDERS, Blueprint, JudgeResult, Refrain, StoryRequest


class TestStoryRequest:
    def test_valid_request_and_word_range(self):
        r = StoryRequest("a rabbit", age=7)
        assert r.word_range() == (500, 750)
        assert StoryRequest("a rabbit", age=5, length="short").word_range() == (320, 480)
        assert StoryRequest("a rabbit", age=10, length="long").word_range() == (780, 1080)

    @pytest.mark.parametrize("age", [4, 11, 0, -1, "7", 7.5, True])
    def test_age_out_of_range_or_wrong_type(self, age):
        with pytest.raises(ValueError, match="Age"):
            StoryRequest("a rabbit", age=age)

    def test_empty_and_overlong_premise(self):
        with pytest.raises(ValueError):
            StoryRequest("   ")
        with pytest.raises(ValueError, match="too long"):
            StoryRequest("x" * 501)

    def test_control_characters_and_whitespace_are_cleaned(self):
        r = StoryRequest("a  fox\n\n\twho   hops\x00", age=6, mood="  calm\n ")
        assert r.premise == "a fox who hops"
        assert r.mood == "calm"

    def test_bad_length(self):
        with pytest.raises(ValueError, match="Length"):
            StoryRequest("a fox", length="epic")


class TestRefrain:
    def test_lines_are_versions_plus_whisper(self):
        r = Refrain.from_dict(FOX_BLUEPRINT["refrain"])
        assert len(r.lines) == 4 == len(PLACEHOLDERS)
        assert r.lines[-1] == "Follow glimmer."

    def test_valid_refrain_passes(self):
        Refrain.from_dict(FOX_BLUEPRINT["refrain"]).validate()

    def test_lines_must_strictly_shrink(self):
        bad = {**FOX_BLUEPRINT["refrain"], "versions": ["Follow the glimmer, soft paws.", "Follow the glimmer, slow steps.", "Follow the glimmer."]}
        with pytest.raises(ValueError, match=r"strictly shorter.*\[5, 5, 3, 2\]"):
            Refrain.from_dict(bad).validate()

    def test_fade_wording_is_left_to_the_prompt_and_judge_not_hard_rules(self):
        # Words added, a one-word whisper and a line ending on "the" are all accepted by code; the judge's refrain check owns them.
        loose = {"versions": ["Follow the glimmer, soft paws, slow steps.", "Follow your light, soft paws.", "Gentle glimmer, the."], "final_whisper": "Glimmer."}
        Refrain.from_dict(loose).validate()

    def test_wrong_version_count(self):
        with pytest.raises(ValueError, match="exactly 3"):
            Refrain.from_dict({"versions": ["Follow the glimmer."], "final_whisper": "glimmer"})


class TestBlueprint:
    def test_valid_blueprint(self):
        bp = Blueprint.from_dict(FOX_BLUEPRINT)
        assert [v.word for v in bp.vocabulary] == ["glimmer", "scamper", "enormous"]

    def test_needs_three_distinct_words(self):
        d = blueprint_dict()
        d["vocabulary"] = d["vocabulary"][:2]
        with pytest.raises(ValueError, match="exactly 3 distinct"):
            Blueprint.from_dict(d)

    def test_vocab_word_must_be_a_single_lemma(self):
        d = blueprint_dict()
        d["vocabulary"][1]["word"] = "scampered quickly"
        with pytest.raises(ValueError, match="base-form"):
            Blueprint.from_dict(d)

    def test_missing_beat(self):
        d = blueprint_dict()
        d["beats"]["climax"] = ""
        with pytest.raises(ValueError, match="beats missing"):
            Blueprint.from_dict(d)

    def test_world_rules_required(self):
        with pytest.raises(ValueError, match="world_rules"):
            Blueprint.from_dict(blueprint_dict(world_rules=[]))

    def test_malformed_refrain_field_is_a_clean_validation_error_not_a_crash(self):
        """A model occasionally returns a field shaped wrong (e.g. a string where an object was required). That must
        become the usual ValueError the planner retry loop expects, never an unhandled AttributeError."""
        with pytest.raises(ValueError, match="refrain"):
            Blueprint.from_dict(blueprint_dict(refrain="Follow the glimmer."))

    def test_malformed_beats_field_is_a_clean_validation_error_not_a_crash(self):
        with pytest.raises(ValueError, match="beats"):
            Blueprint.from_dict(blueprint_dict(beats="the setup, then the climax"))


class TestJudgeResult:
    def test_parses_a_complete_payload(self):
        j = JudgeResult.from_dict(judge_dict())
        assert j.passed_count == 8 and j.failed == [] and len(j.strengths_to_preserve) == 2

    def test_failed_checks_are_listed_in_priority_order(self):
        j = JudgeResult.from_dict(judge_dict({"age_fit": "x y z", "hero_solves_problem": "a b c"}))
        assert j.failed == ["hero_solves_problem", "age_fit"] and j.passed_count == 6

    def test_a_failed_check_needs_a_concrete_fix(self):
        with pytest.raises(ValueError, match="needs a concrete 'fix'"):
            JudgeResult.from_dict(judge_dict({"wind_down": ""}))

    def test_missing_or_non_boolean_check_is_rejected(self):
        d = judge_dict()
        d["checks"]["age_fit"]["passed"] = "yes"
        with pytest.raises(ValueError, match="age_fit"):
            JudgeResult.from_dict(d)
        d = judge_dict()
        del d["checks"]["wind_down"]
        with pytest.raises(ValueError, match="wind_down"):
            JudgeResult.from_dict(d)

    def test_flags_must_be_booleans(self):
        with pytest.raises(ValueError, match="unsafe_content"):
            JudgeResult.from_dict(judge_dict(unsafe_content="no"))


class TestTitleCheck:
    def test_title_from_the_hero_and_premise_is_fine(self, blueprint):
        assert blueprint.title_problem("a little fox who gets lost in an enchanted forest") == ""

    def test_title_with_an_invented_name_is_flagged(self, blueprint):
        blueprint.title = "Brisk the Fox"
        assert "Brisk" in blueprint.title_problem("a little fox who gets lost in an enchanted forest")

    def test_vocabulary_word_must_be_one_word_and_the_schema_says_so(self):
        from models import BLUEPRINT_SCHEMA, VocabWord
        import pytest

        with pytest.raises(ValueError, match="single base-form"):
            VocabWord.from_dict({"word": "magnifying glass", "definition": "a lens"})
        word_schema = BLUEPRINT_SCHEMA["properties"]["vocabulary"]["items"]["properties"]["word"]
        assert "single word" in word_schema["description"].lower()


def test_refrain_error_names_the_pair_that_did_not_shrink():
    import pytest
    from models import Refrain

    refrain = Refrain(versions=["one two three four five", "one two three four", "one two three"], final_whisper="one two three")
    with pytest.raises(ValueError, match="version 3 has 3 words and final_whisper has 3"):
        refrain.validate()
