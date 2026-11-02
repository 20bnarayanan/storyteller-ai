import pytest

from conftest import ScriptedLLM
from fixtures import FOX_BLUEPRINT, GOOD_STORY, blueprint_dict, story_broken_markers
from llm_client import LLMOutputError
from prompts import build_marker_repair_user, build_planner_user, build_storyteller_user
from story_engine import clean_story, plan_story, repair_markers, revise_story, write_story


class TestCleanStory:
    def test_strips_title_heading_and_fences(self):
        assert clean_story("# Pip and the Glimmer\n\nPip ran.\n\n[[REFRAIN_4]]") == "Pip ran.\n\n[[REFRAIN_4]]"
        assert clean_story("```\nPip ran.\n```") == "Pip ran."

    def test_strips_title_label(self):
        assert clean_story("Title: Pip\n\nPip ran.") == "Pip ran."

    def test_leaves_normal_story_alone(self):
        assert clean_story(GOOD_STORY) == GOOD_STORY.strip()


class TestPlanner:
    def test_valid_blueprint_first_try(self, request_age6):
        llm = ScriptedLLM(planner=[FOX_BLUEPRINT])
        bp = plan_story(llm, request_age6)
        assert len(bp.vocabulary) == 3 and len(bp.refrain.lines) == 4

    def test_invalid_refrain_triggers_retry_with_specific_error(self, request_age6):
        bad = blueprint_dict()
        bad["refrain"] = {**FOX_BLUEPRINT["refrain"], "versions": ["Follow the glimmer, soft paws.", "Follow the glimmer, slow steps now.", "Follow the glimmer."]}
        llm = ScriptedLLM(planner=[bad, FOX_BLUEPRINT])
        bp = plan_story(llm, request_age6)
        assert bp.title == "Pip and the Glimmer"
        retry_prompt = llm.calls_for("planner")[1].user
        assert "rejected" in retry_prompt and "strictly shorter" in retry_prompt
        assert "Your previous answer was" in retry_prompt  # the model edits its rejected answer, not starts over

    def test_gives_up_after_five_invalid_attempts(self, request_age6):
        bad = blueprint_dict(vocabulary=[])
        llm = ScriptedLLM(planner=[bad] * 5)
        with pytest.raises(LLMOutputError, match="5 attempts"):
            plan_story(llm, request_age6)

    def test_premise_is_wrapped_as_untrusted_data(self, request_age6):
        prompt = build_planner_user(request_age6)
        assert "<premise>a little fox who gets lost in an enchanted forest</premise>" in prompt


class TestStoryteller:
    def test_write_story_uses_blueprint_and_cleans_output(self, request_age6, blueprint):
        llm = ScriptedLLM(storyteller=["# A Title\n\n" + GOOD_STORY])
        story = write_story(llm, request_age6, blueprint)
        assert story == GOOD_STORY.strip()
        prompt = llm.calls_for("storyteller")[0].user
        for needle in ("glimmer", "scamper", "enormous", "[[REFRAIN_1]]", "[[REFRAIN_4]]", "400-600", "World rules"):
            assert needle in prompt

    def test_storyteller_prompt_never_asks_model_to_write_refrain_text(self, request_age6, blueprint):
        prompt = build_storyteller_user(request_age6, blueprint)
        assert "do not write these lines yourself" in prompt

    def test_revision_prompt_contains_story_instructions_and_marker_rules(self, request_age6, blueprint):
        llm = ScriptedLLM(storyteller=[GOOD_STORY])
        revise_story(llm, request_age6, blueprint, "ORIGINAL TEXT [[REFRAIN_4]]", "- Expand the climax")
        prompt = llm.calls_for("storyteller")[0].user
        assert "ORIGINAL TEXT" in prompt and "- Expand the climax" in prompt
        assert "keep everything else" in prompt
        assert "[[REFRAIN_1]], [[REFRAIN_2]], [[REFRAIN_3]], [[REFRAIN_4]]" in prompt

    def test_age_changes_the_prompt(self, blueprint):
        from models import StoryRequest
        young = build_storyteller_user(StoryRequest("a fox", age=5), blueprint)
        old = build_storyteller_user(StoryRequest("a fox", age=10), blueprint)
        assert "sing-song" in build_planner_user(StoryRequest("a fox", age=5))
        assert "babyish" in build_planner_user(StoryRequest("a fox", age=10))
        assert "400-600" in young and "650-900" in old


class TestMarkerRepair:
    """The cheap, targeted retry used when a draft is missing some (not all) refrain markers."""

    def test_repair_prompt_asks_only_for_the_missing_marker(self):
        prompt = build_marker_repair_user(story_broken_markers(), ["[[REFRAIN_2]]"])
        assert "Insert [[REFRAIN_2]]" in prompt
        assert "Insert [[REFRAIN_1]]" not in prompt  # only the missing marker gets an insert instruction
        assert "Pip was a little red fox" in prompt  # the existing (correct) story text is included
        assert "Do not change, move, or remove any other" in prompt

    def test_repair_prompt_pluralizes_correctly(self):
        one = build_marker_repair_user(GOOD_STORY, ["[[REFRAIN_1]]"])
        two = build_marker_repair_user(GOOD_STORY, ["[[REFRAIN_1]]", "[[REFRAIN_3]]"])
        assert "1 missing refrain marker)" in one and "marker(s)" not in one
        assert "2 missing refrain markers)" in two

    def test_repair_markers_calls_storyteller_and_cleans_output(self):
        llm = ScriptedLLM(storyteller=["# Title\n\n" + GOOD_STORY])
        story = repair_markers(llm, story_broken_markers(), ["[[REFRAIN_2]]"])
        assert story == GOOD_STORY.strip()
        prompt = llm.calls_for("storyteller")[0].user
        assert "REFRAIN_2" in prompt and "World rules" not in prompt  # no blueprint: smaller, cheaper call


class TestPlaceMarkers:
    def test_inserts_four_markers_in_order_with_the_whisper_last(self):
        from models import PLACEHOLDERS
        from story_engine import place_markers

        story = "\n\n".join(f"Paragraph number {i} ends here." for i in range(12))
        out = place_markers(story).split("\n\n")
        positions = [out.index(ph) for ph in PLACEHOLDERS]
        assert positions == sorted(positions) and out[-1] == PLACEHOLDERS[3]
        assert all(out[i - 1].endswith(".") for i in positions)   # each marker follows a finished sentence

    def test_removes_broken_markers_before_placing(self):
        from story_engine import place_markers

        story = "\n\n".join(f"Paragraph {i} ends here." for i in range(10)) + "\n\n[[REFRAIN_9]]\n\n[[REFRAIN_1]] [[REFRAIN_1]]"
        out = place_markers(story)
        assert "[[REFRAIN_9]]" not in out and out.count("[[REFRAIN_1]]") == 1

    def test_too_short_to_place_markers_is_left_unplaced(self):
        from story_engine import place_markers

        assert "[[" not in place_markers("One short paragraph.\n\nAnother one.")


class TestPlannerTitleRetry:
    def test_planner_is_asked_again_when_the_title_invents_a_name(self):
        import copy
        from conftest import ScriptedLLM
        from fixtures import FOX_BLUEPRINT
        from models import StoryRequest
        from story_engine import plan_story

        bad = copy.deepcopy(FOX_BLUEPRINT)
        bad["title"] = "Brisk the Fox"
        llm = ScriptedLLM(planner=[bad, FOX_BLUEPRINT])
        bp = plan_story(llm, StoryRequest("a little fox who gets lost in an enchanted forest", age=6))
        assert bp.title == FOX_BLUEPRINT["title"]
        retry_prompt = llm.calls_for("planner")[1].user
        assert "Brisk" in retry_prompt and "not a character" in retry_prompt
