"""Deterministic story checks. No LLM calls in this file.

Each check returns a CheckResult. Only two checks are ``hard`` (they block a story and skip the judge): the refrain
markers, because the final text cannot be built without them, and runaway/truncated length. Everything else is ``soft``:
it is reported to the judge and folded into the revision instructions, but never discards a story on its own.

The word matcher here is also used by presentation.py so "what counts as the
word appearing" is defined exactly once.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

from models import PLACEHOLDERS, Blueprint, CheckResult, StoryRequest, word_count

# ---------------------------------------------------------------------------
# Word matching with inflections
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _word_forms(lemma: str, extra: tuple[str, ...]) -> frozenset[str]:
    """Recognize a small set of common English inflections without a morphology dependency."""
    forms = {lemma, *extra}
    # Most adjectives (e.g. enormous, curious) should not receive verb/noun suffixes.
    if lemma.endswith(("ous", "ful", "less", "ive", "able", "ible", "al", "ic", "ish")):
        return frozenset(forms)
    if lemma.endswith("e"):
        forms.update({lemma + "d", lemma[:-1] + "ing", lemma + "s"})
    elif lemma.endswith("y") and len(lemma) > 1:
        forms.update({lemma[:-1] + "ies", lemma[:-1] + "ied", lemma[:-1] + "ying"})
    else:
        forms.add(lemma + "s")
        # Double a final consonant after a short vowel (hop -> hopped/hopping).
        cvc = (len(lemma) == 3 and lemma[-1] not in "aeiouwxy" and lemma[-2] in "aeiou"
               and lemma[-3] not in "aeiou")
        stem = lemma + lemma[-1] if cvc else lemma
        forms.update({stem + "ed", stem + "ing"})
    return frozenset(f.lower() for f in forms)

def word_forms(lemma: str, extra: Iterable[str] = ()) -> set[str]:
    """Base form, common inflections and explicitly accepted forms."""
    return set(_word_forms(lemma.lower().strip(), tuple(sorted(e.lower() for e in extra))))

def word_pattern(lemma: str, extra: Iterable[str] = ()) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(f) for f in sorted(word_forms(lemma, extra), key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_PLACEHOLDER_ANY = re.compile(r"\[\[[^\]]*\]\]")
# Split after . ! ? (optionally followed by a closing quote) only when the next sentence starts with a capital or
# an opening quote, so `"Hello!" said Pip.` stays one sentence.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?]["”’])\s+(?=[A-Z"“‘\'])|(?<=[.!?])\s+(?=[A-Z"“‘\'])')
TERMINAL_CHARS = tuple('.!?…"”’\'*')


def paragraphs(story: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", story.strip()) if p.strip()]


def prose_paragraphs(story: str) -> list[str]:
    """Paragraphs that are real story text (not refrain placeholders)."""
    return [p for p in paragraphs(story) if p not in PLACEHOLDERS]


def prose_text(story: str) -> str:
    return "\n\n".join(prose_paragraphs(story))


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENTENCE_SPLIT.split(flat) if s.strip()]


def story_sentences(story: str) -> list[str]:
    return [s for p in prose_paragraphs(story) for s in split_sentences(p)]


def quarter_sentence_lengths(story: str) -> list[float]:
    """Average sentence length (words) for each quarter of the prose, refrain lines excluded."""
    sents = story_sentences(story)
    if len(sents) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    size = len(sents) // 4
    chunks = [sents[0:size], sents[size : 2 * size], sents[2 * size : 3 * size], sents[3 * size :]]
    return [sum(word_count(s) for s in c) / max(1, len(c)) for c in chunks]


def winddown_ratio(story: str) -> float:
    """Final-quarter average sentence length divided by the mean of quarters 1-3. Below 1.0 = winds down."""
    q = quarter_sentence_lengths(story)
    first_three = sum(q[:3]) / 3
    return q[3] / first_three if first_three else 1.0


# Hard checks
# ---------------------------------------------------------------------------
def length_bounds(request: StoryRequest) -> tuple[int, int]:
    """Runaway / truncated bounds. Slightly over the bedtime target is still judged."""
    lo, hi = request.word_range()
    return max(1, int(lo * 0.5)), int(hi * 1.5)


def remaining_word_room(story: str, request: StoryRequest) -> tuple[int, int, int, int]:
    """current count, target lo, target hi, signed room until hi (negative = over cap)."""
    count = word_count(prose_text(story))
    lo, hi = request.word_range()
    return count, lo, hi, hi - count


def check_length(story: str, request: StoryRequest) -> CheckResult:
    count, lo, hi, _room = remaining_word_room(story, request)
    hard_lo, hard_hi = length_bounds(request)
    ok = hard_lo <= count <= hard_hi
    detail = f"{count} words (target {lo}-{hi}, hard limits {hard_lo}-{hard_hi})"
    if not ok:
        mid = (lo + hi) // 2
        if count < hard_lo:
            detail += f" - too short: add about {mid - count} words to the middle scenes"
        else:
            detail += f" - too long: cut about {count - mid} words (aim for about {mid}); merge or drop scenes rather than trimming sentences"
    return CheckResult("length", ok, True, detail)


MIN_GAP_WORDS = 15       # HARD: refrain lines closer than this are stacked like a list (a story with that is rejected)
SOFT_GAP_FRACTION = 0.05  # SOFT: ideally at least this share of the story between lines (advice to the judge only)


def _gap_problems(paras: list[str], need: int) -> list[str]:
    """Spots where two refrain lines have fewer than ``need`` words of story between them."""
    problems: list[str] = []
    positions = [paras.index(ph) for ph in PLACEHOLDERS]
    for i, (a, b) in enumerate(zip(positions, positions[1:]), start=1):
        gap = sum(word_count(p) for p in paras[a + 1:b])
        if gap < need:
            problems.append(
                f"{PLACEHOLDERS[i - 1]} and {PLACEHOLDERS[i]} have only {gap} words of story between them (need at least {need}); "
                "spread the four refrain lines through the story, each at its own calm moment, instead of stacking them"
            )
    return problems


def missing_markers(story: str) -> list[str]:
    """Markers that are simply absent (count 0), with nothing else wrong: no stray placeholder, no marker duplicated.

    Used to decide whether a hard refrain failure can get a cheap, surgical repair ("add just these markers") instead
    of a full revision. Anything messier than plain absence (a duplicate, an unknown placeholder, or all four missing
    at once, which leaves no anchor points to insert around) returns an empty list so the caller falls back to a full
    revision.
    """
    stray = [t for t in _PLACEHOLDER_ANY.findall(story) if t not in PLACEHOLDERS]
    counts = {ph: story.count(ph) for ph in PLACEHOLDERS}
    if stray or any(c > 1 for c in counts.values()):
        return []
    gone = [ph for ph in PLACEHOLDERS if counts[ph] == 0]
    return gone if 0 < len(gone) < len(PLACEHOLDERS) else []


def check_refrain_placeholders(story: str) -> CheckResult:
    """Each [[REFRAIN_n]] exactly once, alone in its paragraph, in order; the whisper is the last paragraph."""
    problems: list[str] = []
    stray = [t for t in _PLACEHOLDER_ANY.findall(story) if t not in PLACEHOLDERS]
    if stray:
        problems.append(f"unknown placeholder(s) {stray}")
    paras = paragraphs(story)
    last_index = -1
    for ph in PLACEHOLDERS:
        total = story.count(ph)
        if total != 1:
            problems.append(f"{ph} must appear exactly once (found {total})")
            continue
        if ph not in paras:
            problems.append(f"{ph} must sit alone in its own paragraph")
            continue
        index = paras.index(ph)
        if index < last_index:
            problems.append("refrain placeholders are out of order (longest first)")
        last_index = max(last_index, index)
    if paras and paras[-1] != PLACEHOLDERS[-1]:
        problems.append(f"the final paragraph must be exactly {PLACEHOLDERS[-1]}")
    if not problems:
        problems += _gap_problems(paras, MIN_GAP_WORDS)
    return CheckResult(
        "refrain_placement", not problems, True, "; ".join(problems) if problems else "4 placeholders in order, whisper last"
    )


# ---------------------------------------------------------------------------
# Soft checks (advice for the judge and the reviser; never discard a story)
# ---------------------------------------------------------------------------
def check_vocabulary(story: str, blueprint: Blueprint) -> CheckResult:
    """The three words should appear in the prose itself (the refrain doesn't count); common inflections allowed."""
    prose = prose_text(story)
    missing = [v.word for v in blueprint.vocabulary if not word_pattern(v.word, v.accepted_forms).search(prose)]
    return CheckResult(
        "vocabulary_present",
        not missing,
        False,
        (
            f"missing from the story text: {', '.join(missing)}. Use each word itself (or a plain -s/-ed/-ing form); "
            "a related word such as 'curiosity' for 'curious' does not count"
        )
        if missing
        else "all three words present",
    )


def check_refrain_spacing(story: str) -> CheckResult:
    """Soft: ideally the four lines are spread evenly enough that the story, not the refrain, fills the pages."""
    paras = paragraphs(story)
    if any(ph not in paras for ph in PLACEHOLDERS):
        return CheckResult("refrain_spacing", True, False, "markers missing (reported by refrain_placement)")
    total = sum(word_count(p) for p in paras if p not in PLACEHOLDERS)
    problems = _gap_problems(paras, int(total * SOFT_GAP_FRACTION))
    return CheckResult("refrain_spacing", not problems, False, "; ".join(problems) if problems else "refrain lines are well spread")


def check_lead_ins(story: str) -> CheckResult:
    """Sentences must be finished before a refrain line and at the end. A paragraph that trails off ('...very, very')
    leaves the refrain to complete the thought, which reads as a cut-off."""
    paras = paragraphs(story)
    cut_off: list[str] = []
    for i, para in enumerate(paras):
        before_marker = i + 1 < len(paras) and paras[i + 1] in PLACEHOLDERS
        is_last_prose = para not in PLACEHOLDERS and all(p in PLACEHOLDERS for p in paras[i + 1 :])
        if para in PLACEHOLDERS or not (before_marker or is_last_prose):
            continue
        if not para.rstrip().endswith(TERMINAL_CHARS):
            cut_off.append(f"...{para.rstrip()[-40:]!r}")
    detail = "every refrain lead-in is a finished sentence" if not cut_off else (
        "this paragraph trails off before a refrain line; end it with a complete sentence: " + "; ".join(cut_off)
    )
    return CheckResult("lead_ins_complete", not cut_off, False, detail)


def check_length_target(story: str, request: StoryRequest) -> CheckResult:
    """Soft bedtime cap: over-target drafts are still judged, then trimmed or compared."""
    count, lo, hi, room = remaining_word_room(story, request)
    ok = lo <= count <= hi
    if count < lo:
        detail = f"{count} words is under the {lo}-{hi} bedtime target by {lo - count}"
    elif count > hi:
        detail = f"{count} words is over the {hi}-word bedtime target by {count - hi}; trim while revising"
    else:
        detail = f"{count} words, {room} words of room under the {hi}-word cap"
    return CheckResult("length_target", ok, False, detail)


_STOPWORDS = set(
    "a about above after again all also am an and any are as at be because been before being below between both but by "
    "can could did do does down each few for from further had has have he her here hers herself him himself his how i if "
    "in into is it its itself just me more most my myself no nor not now of off on once only or other our out over own "
    "same she should so some such than that the their them then there these they this those through to too under until "
    "up very was we were what when where which while who whom why will with would you your said".split()
)


def _name_tokens(blueprint: Blueprint) -> set[str]:
    names = " ".join([blueprint.main_character, *blueprint.supporting_characters])
    return {t.lower() for t in re.findall(r"[A-Za-z']+", names)}


def check_repetition(story: str, blueprint: Blueprint) -> CheckResult:
    """Flags over-used content words and repeated 4-word phrases. Refrain lines are not in the prose, so never counted."""
    prose = prose_text(story)
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", prose)]
    exempt = _name_tokens(blueprint)
    for v in blueprint.vocabulary:
        exempt |= word_forms(v.word, v.accepted_forms)
    counts = Counter(t for t in tokens if len(t) > 3 and t not in _STOPWORDS and t not in exempt)
    limit = max(8, int(0.025 * len(tokens)))
    over = [f"'{w}' x{c}" for w, c in counts.most_common(5) if c > limit]
    grams = Counter(tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3))
    phrases = [" ".join(g) for g, c in grams.items() if c >= 3 and not all(t in _STOPWORDS for t in g)]
    problems = []
    if over:
        problems.append("over-used words: " + ", ".join(over))
    if phrases:
        problems.append("phrases repeated 3+ times: " + "; ".join(phrases[:3]))
    return CheckResult("repetition", not problems, False, "; ".join(problems) if problems else "no excessive repetition")


def check_winddown(story: str) -> CheckResult:
    q = quarter_sentence_lengths(story)
    too_short_to_judge = q == [0.0, 0.0, 0.0, 0.0]  # quarter_sentence_lengths' own guard for <4 sentences
    ok = too_short_to_judge or winddown_ratio(story) < 1.0
    return CheckResult(
        "winddown_sentence_length", ok, False,
        f"avg sentence length per quarter: {', '.join(f'{x:.1f}' for x in q)}"
        + ("" if ok else " - the final quarter should use shorter, softer sentences"),
    )


# ---------------------------------------------------------------------------
def run_validators(story: str, blueprint: Blueprint, request: StoryRequest) -> list[CheckResult]:
    return [
        check_refrain_placeholders(story),
        check_length(story, request),
        check_vocabulary(story, blueprint),
        check_lead_ins(story),
        check_refrain_spacing(story),
        check_length_target(story, request),
        check_repetition(story, blueprint),
        check_winddown(story),
    ]
