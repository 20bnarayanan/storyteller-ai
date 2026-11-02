from fixtures import GOOD_STORY
from presentation import bold_first_occurrences, build_final_markdown, build_judge_view, substitute_refrain, vocabulary_footer


def test_refrain_is_substituted_verbatim_in_italics(blueprint):
    out = substitute_refrain(GOOD_STORY, blueprint)
    assert "[[REFRAIN" not in out
    for line in blueprint.refrain.lines:
        assert f"*{line}*" in out
    assert out.strip().endswith("*Follow glimmer.*")


def test_refrain_order_is_longest_first(blueprint):
    out = substitute_refrain(GOOD_STORY, blueprint)
    positions = [out.index(line) for line in blueprint.refrain.lines]
    assert positions == sorted(positions)


def test_judge_view_has_plain_refrain_without_markup(blueprint):
    view = build_judge_view(GOOD_STORY, blueprint)
    assert "Follow the glimmer, soft paws, slow steps." in view and "[[" not in view


def test_only_first_occurrence_of_each_word_is_bold(blueprint):
    out = bold_first_occurrences(GOOD_STORY, blueprint)
    assert out.count("**scamper**") + out.count("**Scamper**") == 1
    assert out.count("**glimmer**") == 1
    assert out.count("**enormous**") == 1
    # later uses stay plain
    assert "scampered farther" in out


def test_bolding_never_touches_refrain_markers(blueprint):
    out = bold_first_occurrences(GOOD_STORY, blueprint)
    for i in range(1, 5):
        assert f"[[REFRAIN_{i}]]" in out


def test_inflected_word_is_bolded(blueprint):
    story = "Pip scampered home.\n\n[[REFRAIN_4]]"
    assert "**scampered**" in bold_first_occurrences(story, blueprint)


def test_footer_lists_definitions_and_question(blueprint):
    footer = vocabulary_footer(blueprint)
    assert "Words we met tonight" in footer
    assert "**Glimmer:** A faint or gentle light." in footer
    assert "**Enormous:** Much bigger than usual." in footer
    assert 'Ask your little storyteller:** "What would you have done when the glimmer went out?"' in footer


def test_footer_leaves_out_words_the_story_never_used(blueprint):
    story = "Pip scampered home in the glimmer.\n\n[[REFRAIN_4]]"      # 'enormous' never appears
    footer = vocabulary_footer(blueprint, story)
    assert "**Glimmer:**" in footer and "**Scamper:**" in footer
    assert "Enormous" not in footer
    assert "Ask your little storyteller" in footer


def test_footer_with_no_used_words_keeps_only_the_question(blueprint):
    footer = vocabulary_footer(blueprint, "Pip slept.\n\n[[REFRAIN_4]]")
    assert "Words we met tonight" not in footer and "Ask your little storyteller" in footer


def test_final_markdown_structure(blueprint):
    md = build_final_markdown(GOOD_STORY, blueprint)
    assert md.startswith("# Pip and the Glimmer")
    assert md.index("*Follow glimmer.*") < md.index("Words we met tonight")
    assert "[[" not in md
