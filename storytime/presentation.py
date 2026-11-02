"""Final presentation layer: no LLM calls.

Substitutes the exact refrain text for the placeholders (so the wording cannot
drift), italicises it, bolds the first use of each vocabulary word, and appends
a vocabulary footer.
"""
from __future__ import annotations

from models import PLACEHOLDERS, Blueprint
from validators import paragraphs, prose_text, word_pattern


def substitute_refrain(story: str, blueprint: Blueprint, italics: bool = True) -> str:
    out = story
    for placeholder, line in zip(PLACEHOLDERS, blueprint.refrain.lines):
        out = out.replace(placeholder, f"*{line}*" if italics else line)
    return out


def bold_first_occurrences(story: str, blueprint: Blueprint) -> str:
    """Bold the first use of each vocabulary word in the prose (refrain placeholders are untouched)."""
    paras = paragraphs(story)
    for vocab in blueprint.vocabulary:
        pattern = word_pattern(vocab.word, vocab.accepted_forms)
        for i, para in enumerate(paras):
            if para in PLACEHOLDERS:
                continue
            new_para, n = pattern.subn(lambda m: f"**{m.group(0)}**", para, count=1)
            if n:
                paras[i] = new_para
                break
    return "\n\n".join(paras)


def vocabulary_footer(blueprint: Blueprint, story: str | None = None) -> str:
    """Definitions for the words the story really uses (a word the story never used would teach nothing), then the question."""
    prose = prose_text(story) if story is not None else None
    used = [v for v in blueprint.vocabulary if prose is None or word_pattern(v.word, v.accepted_forms).search(prose)]
    lines = ["---", ""]
    if used:
        lines += ["**Words we met tonight ✨**", ""]
        lines += [f"- **{v.word.capitalize()}:** {v.definition}" for v in used]
        lines.append("")
    lines.append(f'**Ask your little storyteller:** "{blueprint.discussion_question}"')
    return "\n".join(lines)


def build_final_markdown(story: str, blueprint: Blueprint) -> str:
    body = substitute_refrain(bold_first_occurrences(story, blueprint), blueprint)
    return f"# {blueprint.title}\n\n{body}\n\n{vocabulary_footer(blueprint, story)}\n"


def build_judge_view(story: str, blueprint: Blueprint) -> str:
    """What the judge reads: the story as a child would hear it, refrain in italics, no footer or bolding."""
    return substitute_refrain(story, blueprint)
