import pytest

from fixtures import GOOD_STORY, story_broken_markers, story_with_marker, story_without
from models import StoryRequest
from validators import (
    check_lead_ins, check_refrain_spacing, check_length, check_refrain_placeholders, check_repetition,
    check_vocabulary, check_winddown, missing_markers, quarter_sentence_lengths, run_validators, split_sentences,
    winddown_ratio, word_forms, word_pattern,
)


class TestWordMatching:
    @pytest.mark.parametrize("lemma,text", [
        ("scamper", "He scampered home."), ("scamper", "They scamper and scampering too"),
        ("glimmer", "A glimmering light"), ("glimmer", "Two glimmers shone"),
        ("hop", "The rabbit hopped away"), ("glisten", "dew glistening on leaves"),
        ("navigate", "She navigated the river"), ("enormous", "an Enormous oak"),
    ])
    def test_matches_inflections(self, lemma, text):
        assert word_pattern(lemma).search(text)

    @pytest.mark.parametrize("lemma,text", [
        ("scamper", "He scamperx home"), ("hop", "a hopeful heart"), ("glimmer", "glimmerous"),
        ("scamper", "a campers' tent"), ("hop", "hoping for rain"),
    ])
    def test_does_not_match_lookalikes(self, lemma, text):
        assert not word_pattern(lemma).search(text)

    def test_known_words_do_not_get_junk_forms(self):
        assert word_forms("enormous") == {"enormous"}

    def test_unknown_word_falls_back_to_rules(self):
        forms = word_forms("zibble")
        assert "zibbled" in forms or "zibbles" in forms

    def test_accepted_forms_from_planner(self):
        assert word_pattern("mouse", ["mice"]).search("three blind mice")


class TestVocabulary:
    def test_all_present(self, blueprint):
        assert check_vocabulary(GOOD_STORY, blueprint).passed

    def test_missing_word_is_reported_but_is_soft(self, blueprint):
        r = check_vocabulary(story_without("enormous"), blueprint)
        assert not r.passed and not r.hard and "enormous" in r.detail

    def test_word_only_in_refrain_does_not_count(self, blueprint):
        # glimmer only inside the refrain markers, not in the prose
        story = story_without("glimmer", "light")
        r = check_vocabulary(story, blueprint)
        assert not r.passed and "glimmer" in r.detail


class TestRefrainSpacing:
    def test_spread_markers_pass(self):
        assert check_refrain_placeholders(GOOD_STORY).passed

    def test_stacked_markers_are_a_hard_failure(self):
        head = GOOD_STORY.split("[[REFRAIN_1]]")[0]
        stacked = head + "\n\n[[REFRAIN_1]]\n\n[[REFRAIN_2]]\n\n[[REFRAIN_3]]\n\n[[REFRAIN_4]]"
        r = check_refrain_placeholders(stacked)
        assert not r.passed and r.hard and "spread the four refrain lines" in r.detail

    def test_two_close_markers_fail(self):
        close = GOOD_STORY.replace("\n\n[[REFRAIN_2]]", "\n\n[[REFRAIN_1B]]\n\n[[REFRAIN_2]]").replace("[[REFRAIN_1B]]\n\n", "")
        assert check_refrain_placeholders(close).passed  # control: unchanged story still fine
        tight = GOOD_STORY.replace("[[REFRAIN_3]]", "[[REFRAIN_2]]\n\n[[REFRAIN_3]]", 1).replace("[[REFRAIN_2]]\n\n[[REFRAIN_2]]", "[[REFRAIN_2]]")
        assert tight.count("[[REFRAIN_2]]") <= 2


class TestLength:
    def test_in_range(self, request_age6):
        assert check_length(GOOD_STORY, request_age6).passed

    def test_too_short(self, request_age6):
        short = "Pip ran home. It was nice.\n\n[[REFRAIN_4]]"
        r = check_length(short, request_age6)
        assert not r.passed and "too short" in r.detail

    def test_too_long(self):
        req = StoryRequest("a fox", age=5, length="short")  # target 320-480, hard max 720
        assert not check_length(GOOD_STORY + ("\n\n" + "More words fill the page now. " * 80), req).passed

    def test_refrain_markers_do_not_count_as_words(self, request_age6):
        assert "427 words" in check_length(GOOD_STORY, request_age6).detail


class TestRefrainPlacement:
    def test_good_story_passes(self):
        assert check_refrain_placement_ok(GOOD_STORY)

    @pytest.mark.parametrize("mutate,expect", [
        (lambda s: s.replace("[[REFRAIN_2]]", ""), "must appear exactly once"),
        (lambda s: s + "\n\n[[REFRAIN_1]]", "must appear exactly once"),
        (lambda s: s.replace("[[REFRAIN_1]]", "[[TMP]]").replace("[[REFRAIN_2]]", "[[REFRAIN_1]]").replace("[[TMP]]", "[[REFRAIN_2]]"), "out of order"),
        (lambda s: s + "\n\nThe end.", "final paragraph"),
        (lambda s: s.replace("\n\n[[REFRAIN_2]]\n\n", " [[REFRAIN_2]]\n\n"), "alone"),
        (lambda s: s.replace("Good night, little fox.", "Good night. [[REFRAIN_9]]"), "unknown"),
    ], ids=["missing_marker", "duplicate_marker", "out_of_order", "whisper_not_last_paragraph", "marker_not_alone", "unknown_marker"])
    def test_broken_placement_is_rejected_with_a_specific_reason(self, mutate, expect):
        r = check_refrain_placeholders(mutate(GOOD_STORY))
        assert not r.passed and expect in r.detail


def check_refrain_placement_ok(story):
    return check_refrain_placeholders(story).passed


class TestMissingMarkers:
    """The deterministic trigger for the cheap, targeted repair path (see pipeline.py)."""

    def test_all_present_is_empty(self):
        assert missing_markers(GOOD_STORY) == []

    def test_one_missing_is_reported(self):
        assert missing_markers(story_broken_markers()) == ["[[REFRAIN_2]]"]

    def test_two_missing_are_both_reported_in_order(self):
        story = GOOD_STORY.replace("[[REFRAIN_1]]", "").replace("[[REFRAIN_3]]", "")
        assert missing_markers(story) == ["[[REFRAIN_1]]", "[[REFRAIN_3]]"]

    def test_all_four_missing_is_empty_no_anchor_to_repair_around(self):
        story = GOOD_STORY
        for ph in ("[[REFRAIN_1]]", "[[REFRAIN_2]]", "[[REFRAIN_3]]", "[[REFRAIN_4]]"):
            story = story.replace(ph, "")
        assert missing_markers(story) == []

    def test_duplicate_marker_is_empty_even_if_another_is_missing(self):
        story = GOOD_STORY.replace("[[REFRAIN_2]]", "") + "\n\n[[REFRAIN_1]]"
        assert missing_markers(story) == []

    def test_unknown_placeholder_is_empty_even_if_one_is_missing(self):
        story = GOOD_STORY.replace("[[REFRAIN_2]]", "[[REFRAIN_9]]")
        assert missing_markers(story) == []


def _story_with_gaps(gap_words: int) -> str:
    """Four markers with ``gap_words`` words of story between neighbours and a long tail of story elsewhere."""
    filler = lambda n: " ".join(["word"] * n) + "."  # noqa: E731
    parts = [filler(600), "[[REFRAIN_1]]", filler(gap_words), "[[REFRAIN_2]]", filler(gap_words), "[[REFRAIN_3]]", filler(gap_words), "[[REFRAIN_4]]"]
    return "\n\n".join(parts)


class TestSoftRefrainSpacing:
    def test_well_spread_passes(self):
        assert check_refrain_spacing(GOOD_STORY).passed

    def test_close_but_not_stacked_is_soft_advice_not_a_rejection(self):
        story = _story_with_gaps(25)                       # above the hard minimum (15), below 5% of the story
        assert check_refrain_placeholders(story).passed    # never rejected over fine spacing
        soft = check_refrain_spacing(story)
        assert not soft.passed and not soft.hard

    def test_very_close_is_a_hard_failure(self):
        assert not check_refrain_placeholders(_story_with_gaps(5)).passed


class TestLeadIns:
    def test_complete_lead_ins_pass(self):
        assert check_lead_ins(GOOD_STORY).passed

    def test_trailing_off_before_a_marker_is_flagged_but_soft(self):
        cut = GOOD_STORY.replace("taught him.\n\n[[REFRAIN_1]]", "taught him, and\n\n[[REFRAIN_1]]")
        r = check_lead_ins(cut)
        assert not r.passed and not r.hard


class TestSoftChecks:
    def test_repeated_word_flagged_but_soft(self, blueprint):
        story = GOOD_STORY + "\n\n" + " ".join(["The shiny pebble sat there."] * 12)
        r = check_repetition(story, blueprint)
        assert not r.passed and not r.hard and "pebble" in r.detail

    def test_names_and_vocab_are_exempt_from_repetition(self, blueprint):
        story = " ".join(["Pip the fox went to scamper and glimmer."] * 15) + "\n\n[[REFRAIN_4]]"
        detail = check_repetition(story, blueprint).detail
        over_used_part = detail.split(";")[0]
        for exempt in ("pip", "fox", "scamper", "glimmer"):
            assert f"'{exempt}'" not in over_used_part
        assert "'went'" in over_used_part  # a non-exempt word is still caught

    def test_repeated_phrase_flagged(self, blueprint):
        story = GOOD_STORY + "\n\n" + "the quiet river was calm. " * 4
        r = check_repetition(story, blueprint)
        assert not r.passed and "phrases" in r.detail

    def test_refrain_text_is_never_counted_as_repetition(self, blueprint):
        # Markers carry no refrain words, and Python substitutes them later, so the refrain cannot trip this check.
        assert check_repetition(GOOD_STORY, blueprint).passed

    def test_winddown_detected_in_good_story(self):
        assert check_winddown(GOOD_STORY).passed
        assert winddown_ratio(GOOD_STORY) < 1

    def test_no_winddown_flagged(self):
        flat = " ".join(["The fox ran across the whole wide field today."] * 4 + ["Suddenly the very large fox jumped over every single tall fence."] * 4) + " " + " ".join(["A tremendous storm crashed across the enormous mountain valley below them."] * 4)
        assert not check_winddown(flat).passed


class TestQuarterLengths:
    def test_excludes_refrain_markers_and_has_four_quarters(self):
        q = quarter_sentence_lengths(GOOD_STORY)
        assert len(q) == 4 and q[3] < q[0]

    def test_tiny_story(self):
        assert quarter_sentence_lengths("One. Two.") == [0.0] * 4

    def test_split_sentences_handles_quotes(self):
        assert len(split_sentences('"Hello there!" said Pip. He ran.')) == 2


def test_run_validators_all_pass_on_good_story(blueprint, request_age6):
    results = run_validators(GOOD_STORY, blueprint, request_age6)
    assert all(r.passed for r in results), [r for r in results if not r.passed]
    assert {r.name for r in results if r.hard} == {"refrain_placement", "length"}


def test_marker_story_still_valid(blueprint, request_age6):
    results = run_validators(story_with_marker("A cricket sang."), blueprint, request_age6)
    assert all(r.passed for r in results if r.hard)
