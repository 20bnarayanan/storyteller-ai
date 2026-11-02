"""Typed data models + the JSON schemas the LLM must satisfy.

``from_dict`` constructors raise ValueError with a *specific* message, and
``llm_client.complete_validated`` feeds that message back to the model for a retry.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants shared across modules
# ---------------------------------------------------------------------------
BEAT_NAMES = ["setup", "inciting_incident", "rising_action", "climax", "resolution"]
PLACEHOLDERS = [f"[[REFRAIN_{i}]]" for i in range(1, 5)]  # 3 fade versions + final whisper

# Target word ranges for the story body, by age band.
AGE_BANDS: dict[tuple[int, int], tuple[int, int]] = {
    (5, 6): (400, 600),
    (7, 8): (500, 750),
    (9, 10): (650, 900),
}
LENGTH_SCALE = {"short": 0.8, "medium": 1.0, "long": 1.2}

# The judge answers eight pass/fail questions (pass/fail is easier to apply consistently than a 1-5 scale and each
# failure carries a concrete fix). Listed in priority order: revisions fix the earliest failures first.
CHECK_NAMES = [
    "hero_solves_problem",
    "cause_and_effect",
    "climax_has_room",
    "natural_prose",
    "age_fit",
    "vocab_in_context",
    "wind_down",
    "refrain_natural",
]
# A story cannot pass the quality gate if one of these fails, however many other checks pass.
CRITICAL_CHECKS = ("hero_solves_problem", "wind_down")
MAX_FIXES_PER_REVISION = 3


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
@dataclass
class StoryRequest:
    premise: str
    age: int = 7
    interests: Optional[str] = None
    mood: Optional[str] = None
    length: str = "medium"

    def __post_init__(self) -> None:
        self.premise = re.sub(r"[\x00-\x1f\x7f]+", " ", str(self.premise or "")).strip()
        self.premise = re.sub(r"\s+", " ", self.premise)
        if not self.premise:
            raise ValueError("Please give a story premise.")
        if len(self.premise) > 500:
            raise ValueError("The premise is too long (500 characters max).")
        if not isinstance(self.age, int) or isinstance(self.age, bool) or not 5 <= self.age <= 10:
            raise ValueError("Age must be a whole number from 5 to 10.")
        if self.length not in LENGTH_SCALE:
            raise ValueError(f"Length must be one of: {', '.join(LENGTH_SCALE)}.")
        for name in ("interests", "mood"):
            value = getattr(self, name)
            if value is not None:
                value = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]+", " ", str(value))).strip()[:200]
                setattr(self, name, value or None)

    def word_range(self) -> tuple[int, int]:
        lo, hi = next(r for (a, b), r in AGE_BANDS.items() if a <= self.age <= b)
        scale = LENGTH_SCALE[self.length]
        return int(lo * scale), int(hi * scale)


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
@dataclass
class VocabWord:
    word: str
    definition: str
    context_clue: str
    usage_guidance: str
    accepted_forms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "VocabWord":
        word = str(d.get("word", "")).strip().lower()
        if not re.fullmatch(r"[a-z]+(?:-[a-z]+)?", word):
            raise ValueError(f"vocabulary word {word!r} must be a single base-form (lemma) word, letters only")
        definition = str(d.get("definition", "")).strip()
        if not definition:
            raise ValueError(f"vocabulary word '{word}' needs a child-friendly definition")
        forms = [str(f).strip().lower() for f in d.get("accepted_forms", []) or [] if str(f).strip()]
        return cls(
            word=word,
            definition=definition,
            context_clue=str(d.get("context_clue", "")).strip(),
            usage_guidance=str(d.get("usage_guidance", "")).strip(),
            accepted_forms=forms,
        )


@dataclass
class Refrain:
    """The story's lullaby line, returned four times, each shorter than the last.

    Code guarantees only two things: the exact text is inserted by Python (the model never retypes it), and each
    appearance is strictly shorter than the one before. Whether the fade sounds natural (each line built from the previous
    line's words, ending on a complete word, tied to the story) is left to the planner prompt and the judge's check.
    """
    versions: list[str]
    final_whisper: str

    @property
    def lines(self) -> list[str]:
        """The four appearances, longest first. Index i fills PLACEHOLDERS[i]."""
        return [*self.versions, self.final_whisper]

    @classmethod
    def from_dict(cls, d: dict) -> "Refrain":
        versions = [str(v).strip() for v in d.get("versions", [])]
        if len(versions) != 3 or not all(versions):
            raise ValueError("refrain must have exactly 3 non-empty versions")
        whisper = str(d.get("final_whisper", "")).strip()
        if not whisper:
            raise ValueError("refrain needs a non-empty final_whisper")
        return cls(versions=versions, final_whisper=whisper)

    def validate(self) -> None:
        counts = [word_count(line) for line in self.lines]
        names = ["version 1", "version 2", "version 3", "final_whisper"]
        for i, (a, b) in enumerate(zip(counts, counts[1:])):
            if a <= b:
                raise ValueError(
                    f"refrain lines must get strictly shorter each time, but {names[i]} has {a} words and {names[i + 1]} has {b}. "
                    f"Make {names[i + 1]} shorter than {names[i]} by dropping trailing words, or lengthen {names[i]}; "
                    f"all four word counts must shrink (now {counts})."
                )


@dataclass
class Blueprint:
    title: str
    main_character: str
    supporting_characters: list[str]
    setting: str
    goal: str
    conflict: str
    beats: dict[str, str]
    character_growth: str
    world_rules: list[str]
    vocabulary: list[VocabWord]
    refrain: Refrain
    discussion_question: str
    safety_note: str = ""  # not branched on; surfaces in trace.json so a human can audit any premise reframing

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        try:
            vocab = [VocabWord.from_dict(v) for v in d.get("vocabulary", [])]
        except AttributeError as exc:
            raise ValueError("vocabulary must be a list of objects") from exc
        words = [v.word for v in vocab]
        if len(vocab) != 3 or len(set(words)) != 3:
            raise ValueError(f"need exactly 3 distinct vocabulary words, got {words}")

        beats = d.get("beats") or {}
        if not isinstance(beats, dict):
            raise ValueError("beats must be an object mapping each beat name to its text")
        missing = [b for b in BEAT_NAMES if not str(beats.get(b, "")).strip()]
        if missing:
            raise ValueError(f"beats missing: {missing}")

        try:
            refrain = Refrain.from_dict(d.get("refrain") or {})
        except AttributeError as exc:
            raise ValueError("refrain must be an object with 'versions' and 'final_whisper'") from exc
        refrain.validate()

        for key in ("title", "main_character", "setting", "goal", "conflict", "discussion_question"):
            if not str(d.get(key, "")).strip():
                raise ValueError(f"blueprint field '{key}' is required")
        rules = [str(r).strip() for r in d.get("world_rules", []) if str(r).strip()]
        if not rules:
            raise ValueError("world_rules must list at least one rule")

        return cls(
            title=str(d["title"]).strip(),
            main_character=str(d["main_character"]).strip(),
            supporting_characters=[str(c).strip() for c in d.get("supporting_characters", []) if str(c).strip()],
            setting=str(d["setting"]).strip(),
            goal=str(d["goal"]).strip(),
            conflict=str(d["conflict"]).strip(),
            beats={b: str(beats[b]).strip() for b in BEAT_NAMES},
            character_growth=str(d.get("character_growth", "")).strip(),
            world_rules=rules,
            vocabulary=vocab,
            refrain=refrain,
            discussion_question=str(d["discussion_question"]).strip(),
            safety_note=str(d.get("safety_note", "")).strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def title_problem(self, premise: str) -> str:
        """Titles like "Brisk the Dragon" go wrong when the name was invented for the title alone and never appears in the
        story. Check only the name slots (the first word, and the word after "and"): they must come from the premise or the
        story's own plan. Ordinary title words are left alone. Returns "" when the title is fine."""
        plan_text = " ".join([
            premise, self.main_character, *self.supporting_characters, self.setting, self.goal, self.conflict,
            self.character_growth, *self.beats.values(), *self.world_rules,
        ])
        known = set(re.findall(r"[a-z']+", plan_text.lower()))
        words = re.findall(r"[A-Za-z']+", self.title)
        slots = [w for i, w in enumerate(words) if i == 0 or words[i - 1].lower() == "and"]
        unknown = [w for w in slots if w.lower() not in known and w.lower() not in _TITLE_WORDS]
        if not unknown:
            return ""
        return (f"the title uses {unknown} which is not a character or part of the story; build the title from the hero's name "
                "or plain words from the story")


_TITLE_WORDS = set("the a an and of to in on at for with who that is was goes saves finds night story tale adventure".split())


# ---------------------------------------------------------------------------
# Judge result
# ---------------------------------------------------------------------------
@dataclass
class JudgeCheck:
    """One pass/fail verdict, with the quoted evidence behind it and (when it failed) one concrete edit."""
    passed: bool
    evidence: str
    fix: str = ""


@dataclass
class JudgeResult:
    checks: dict[str, JudgeCheck]
    strengths_to_preserve: list[str]
    overall_assessment: str  # not branched on; surfaces in trace.json as a quick human-readable summary of the verdict
    unsafe_content: bool

    @property
    def passed_count(self) -> int:
        return sum(c.passed for c in self.checks.values())

    @property
    def failed(self) -> list[str]:
        """Names of failed checks, in priority order."""
        return [name for name in CHECK_NAMES if not self.checks[name].passed]

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeResult":
        raw = d.get("checks") or {}
        checks: dict[str, JudgeCheck] = {}
        for name in CHECK_NAMES:
            item = raw.get(name)
            if not isinstance(item, dict) or not isinstance(item.get("passed"), bool):
                raise ValueError(f"check '{name}' must be an object with a true/false 'passed'")
            check = JudgeCheck(
                passed=item["passed"], evidence=str(item.get("evidence", "")).strip(), fix=str(item.get("fix", "")).strip()
            )
            if not check.passed and not check.fix:
                raise ValueError(f"failed check '{name}' needs a concrete 'fix' (where to edit and what to change)")
            checks[name] = check
        if not isinstance(d.get("unsafe_content"), bool):
            raise ValueError("'unsafe_content' must be true or false")
        return cls(
            checks=checks,
            strengths_to_preserve=[str(s).strip() for s in d.get("strengths_to_preserve", []) if str(s).strip()],
            overall_assessment=str(d.get("overall_assessment", "")).strip(),
            unsafe_content=d["unsafe_content"],
        )


# ---------------------------------------------------------------------------
# Pipeline objects
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    hard: bool
    detail: str = ""


@dataclass
class Candidate:
    label: str  # "initial", "revision_1", ...
    story: str  # raw story with [[REFRAIN_n]] placeholders
    checks: list[CheckResult] = field(default_factory=list)
    judge: Optional[JudgeResult] = None
    score: Optional[float] = None  # fraction of judge checks passed (0-1); None if never judged
    gate_failures: list[str] = field(default_factory=list)
    judge_error: str = ""

    @property
    def hard_ok(self) -> bool:
        # A judge safety flag is a publication blocker, not merely a score penalty.
        safe = self.judge is None or not self.judge.unsafe_content
        return safe and all(c.passed for c in self.checks if c.hard)

    @property
    def passed(self) -> bool:
        return self.hard_ok and self.judge is not None and not self.gate_failures

    @property
    def checks_label(self) -> str:
        """Human-readable result, e.g. '7/8 checks passed'."""
        if self.judge is None:
            return "not judged"
        return f"{self.judge.passed_count}/{len(CHECK_NAMES)} checks passed"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "story": self.story,
            "hard_ok": self.hard_ok,
            "passed": self.passed,
            "score": self.score,
            "gate_failures": self.gate_failures,
            "judge_error": self.judge_error,
            "checks": [asdict(c) for c in self.checks],
            "judge": asdict(self.judge) if self.judge else None,
        }


@dataclass
class PipelineResult:
    status: str  # "passed" | "best_effort" | "failed"
    message: str
    blueprint: Optional[Blueprint] = None
    candidates: list[Candidate] = field(default_factory=list)
    best: Optional[Candidate] = None
    final_markdown: Optional[str] = None
    revisions_used: int = 0
    usage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON schemas for structured output
# ---------------------------------------------------------------------------
_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}

BLUEPRINT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": _STR,
        "main_character": _STR,
        "supporting_characters": _STR_LIST,
        "setting": _STR,
        "goal": _STR,
        "conflict": _STR,
        "beats": {
            "type": "object",
            "properties": {b: _STR for b in BEAT_NAMES},
            "required": BEAT_NAMES,
        },
        "character_growth": _STR,
        "world_rules": _STR_LIST,
        "vocabulary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "ONE single word, letters only, base form. Never a phrase."},
                    "definition": _STR,
                    "usage_guidance": _STR,
                    "context_clue": _STR,
                    "accepted_forms": _STR_LIST,
                },
                "required": ["word", "definition", "usage_guidance", "context_clue"],
            },
        },
        "refrain": {
            "type": "object",
            "properties": {
                # Scratch space, filled first: lay out the four lines and the words struck out of each. Ignored by the code.
                "workings": _STR,
                "versions": {"type": "array", "items": _STR, "minItems": 3, "maxItems": 3},
                "final_whisper": _STR,
            },
            "required": ["workings", "versions", "final_whisper"],
        },
        "discussion_question": _STR,
        "safety_note": _STR,
    },
    "required": [
        "title", "main_character", "supporting_characters", "setting", "goal", "conflict",
        "beats", "character_growth", "world_rules", "vocabulary", "refrain",
        "discussion_question",
    ],
}

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"evidence": _STR, "passed": {"type": "boolean"}, "fix": _STR},
    "required": ["evidence", "passed", "fix"],
}

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # Strengths come first so no fix can ask to cut something the judge has already protected; evidence comes
        # before the verdict inside each check, so the judge reasons from the text first.
        "strengths_to_preserve": _STR_LIST,
        "checks": {"type": "object", "properties": {n: _CHECK_SCHEMA for n in CHECK_NAMES}, "required": CHECK_NAMES},
        "overall_assessment": _STR,
        "unsafe_content": {"type": "boolean"},
    },
    "required": ["strengths_to_preserve", "checks", "overall_assessment", "unsafe_content"],
}
