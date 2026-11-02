"""Planning, story generation, and revision. Each function is one LLM call."""
from __future__ import annotations

import re

from llm_client import LLM, complete_validated
from models import BLUEPRINT_SCHEMA, PLACEHOLDERS, Blueprint, StoryRequest
from prompts import (
    PLANNER_SYSTEM,
    STORYTELLER_SYSTEM,
    build_marker_repair_user,
    build_planner_user,
    build_revision_user,
    build_storyteller_user,
)

PLANNER_ATTEMPTS = 5  # first try + four validation-feedback retries (cheap: no story written yet)


def plan_story(llm: LLM, request: StoryRequest) -> Blueprint:
    """Planner call. Blueprint rules (refrain shape, vocab count, title, ...) are checked before any story is written."""
    def parse(data: dict) -> Blueprint:
        blueprint = Blueprint.from_dict(data)
        problem = blueprint.title_problem(request.premise)
        if problem:
            raise ValueError(problem)
        return blueprint

    return complete_validated(
        llm, "planner", PLANNER_SYSTEM, build_planner_user(request),
        "submit_blueprint", BLUEPRINT_SCHEMA, parse, attempts=PLANNER_ATTEMPTS,
    )


def clean_story(text: str) -> str:
    """Strip things models add despite instructions: code fences, a leading title/heading, stray 'Story:' labels."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    def looks_like_title(p: str) -> bool:
        return p.lstrip().startswith("#") or (len(p) < 80 and bool(re.fullmatch(r"(?i)\s*(story|title)\s*:.*", p)))

    while paras and looks_like_title(paras[0]):
        paras.pop(0)
    return "\n\n".join(p.strip() for p in paras)


def _drop_title(text: str, title: str) -> str:
    """The presentation adds the title, so drop a first paragraph that merely repeats it (models often add one anyway)."""
    first, _, rest = text.partition("\n\n")
    return rest.strip() if first.strip().strip("*#\"' ").lower() == title.strip().lower() else text


def write_story(llm: LLM, request: StoryRequest, blueprint: Blueprint) -> str:
    raw = llm.complete_text("storyteller", STORYTELLER_SYSTEM, build_storyteller_user(request, blueprint))
    return _drop_title(clean_story(raw), blueprint.title)


def revise_story(llm: LLM, request: StoryRequest, blueprint: Blueprint, story: str, instructions: str) -> str:
    raw = llm.complete_text("storyteller", STORYTELLER_SYSTEM, build_revision_user(request, blueprint, story, instructions))
    return _drop_title(clean_story(raw), blueprint.title)


def place_markers(story: str) -> str:
    """Last resort when the model ignored the refrain markers: code puts them in itself, at about 25%, 50% and 75% of the
    story and as the last paragraph, each after a paragraph that ends a sentence. Any broken markers are removed first. The
    judge can then ask for better lead-in sentences in the one quality revision."""
    cleaned = re.sub(r"\[\[[^\]]*\]\]", "", story)
    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(paras) < 6:
        return "\n\n".join(paras)
    spots: list[int] = []
    for frac in (0.25, 0.5, 0.75):
        i = round(len(paras) * frac) - 1
        while i < len(paras) - 2 and not paras[i].rstrip("\"'”’*").endswith((".", "!", "?")):
            i += 1
        spots.append(i)
    out: list[str] = []
    for i, para in enumerate(paras):
        out.append(para)
        if i in spots:
            out.append(PLACEHOLDERS[spots.index(i)])
    out.append(PLACEHOLDERS[3])
    return "\n\n".join(out)


def repair_markers(llm: LLM, story: str, missing: list[str]) -> str:
    """Cheap targeted retry for the one hard failure with an obvious surgical fix: markers simply missing. No
    blueprint needed, so the call is smaller and faster than a full revision."""
    raw = llm.complete_text("storyteller", STORYTELLER_SYSTEM, build_marker_repair_user(story, missing))
    return clean_story(raw)
