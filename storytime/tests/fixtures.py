"""Shared test data: a realistic fox blueprint, a story that passes every check, and variants of it."""
from __future__ import annotations

import copy
import re

# --- Blueprint (what a good planner call returns) --------------------------------------------
FOX_BLUEPRINT = {
    "title": "Pip and the Glimmer",
    "main_character": "Pip, a little red fox",
    "supporting_characters": ["Hush the owl", "Mama Fox"],
    "setting": "the Whispering Wood at dusk, thick with soft fog",
    "goal": "find the way home to the den",
    "conflict": "Pip chases a firefly too far and the fog hides the path home",
    "beats": {
        "setup": "Pip loves to scamper through the Whispering Wood at dusk.",
        "inciting_incident": "Chasing a firefly, Pip loses the path in the fog.",
        "rising_action": "Hush the owl explains the glimmer that guides lost travelers; Pip must stay quiet to follow it.",
        "climax": "An enormous oak blocks the glimmer, and Pip chooses to stay calm instead of running.",
        "resolution": "Pip follows the glimmer around the oak, finds the den, and falls asleep.",
    },
    "character_growth": "Pip learns that staying calm and quiet helps more than running.",
    "world_rules": [
        "A glimmer is a faint gentle light that always points toward home.",
        "A glimmer fades if anyone shouts, but returns when it is quiet.",
    ],
    "vocabulary": [
        {
            "word": "glimmer",
            "definition": "A faint or gentle light.",
            "usage_guidance": "Hush explains it, then Pip follows it.",
            "context_clue": "a tiny star that fell into the trees",
            "accepted_forms": [],
        },
        {
            "word": "scamper",
            "definition": "To run quickly with small, light steps.",
            "usage_guidance": "Show Pip's quick small feet.",
            "context_clue": "small, quick feet going swish, swish",
            "accepted_forms": [],
        },
        {
            "word": "enormous",
            "definition": "Much bigger than usual.",
            "usage_guidance": "Compare the oak to ten foxes.",
            "context_clue": "ten foxes holding paws could not reach around it",
            "accepted_forms": [],
        },
    ],
    "discussion_question": "What would you have done when the glimmer went out?",
    "refrain": {
        "versions": [
            "Follow the glimmer, soft paws, slow steps.",
            "Follow the glimmer, soft paws.",
            "Follow the glimmer.",
        ],
        "final_whisper": "Follow glimmer.",
    },
    "safety_note": "",
}


def blueprint_dict(**overrides) -> dict:
    d = copy.deepcopy(FOX_BLUEPRINT)
    d.update(overrides)
    return d


# --- A story (age 5-6) that passes every deterministic check -------------------------------------
GOOD_STORY = """Pip was a little red fox with big ears and a bushy tail. Every evening, when the sky turned the color of plums, Pip liked to scamper through the Whispering Wood. Scamper, scamper, scamper went his small, quick feet, and the leaves went swish, swish, swish. He chased crickets and tickled the ferns. It was his favorite time of the whole day.

Tonight Pip went farther than ever before. He chased a firefly past the mossy stones and over the singing brook. Then he looked up, and the path home was gone. Tall trees leaned close. Cold fog curled around his paws. Pip's heart went thump, thump, thump. He did not know which way to go.

A wise old owl named Hush blinked down from a branch. "Do not worry, little fox," said Hush in a voice as soft as a feather. "The wood keeps a glimmer for every lost traveler. A glimmer is a faint, gentle light, like a tiny star that fell into the trees. If you follow it quietly, it will lead you home."

Pip took a slow breath. He listened to the crickets and tried to be brave. Then he whispered the words Hush had taught him.

[[REFRAIN_1]]

Far ahead, a faint silver light winked between the ferns. Pip tiptoed toward it. But the fog grew thick, and the light slipped behind an enormous oak tree. The oak was so wide that ten foxes holding paws could not reach around it. It was much, much bigger than any tree Pip had ever seen.

Pip felt a wobble in his tummy. He wanted to run and shout. But he remembered what Hush had said. A glimmer fades if you shout. So Pip stayed quiet. He listened. The crickets chirped. The brook bubbled. He crept around the enormous roots, one careful step at a time, and he sang the words again, very softly.

[[REFRAIN_2]]

Around the roots, the glimmer glowed warm and gold. It floated over a tiny hill, and at the bottom of the hill was Pip's cozy den. Mama Fox was waiting at the door. "You found your way," she said, and she tucked Pip close. He had scampered farther than ever, and he had stayed calm when it mattered most.

The fog melted away. The stars blinked. Pip's eyes grew heavy. He curled up in his warm bed of leaves. The den was quiet. Mama hummed low.

[[REFRAIN_3]]

The night was soft and still. Pip's paws were tired. His breath was slow. The glimmer in the window grew dim and dreamy. Good night, little fox.

[[REFRAIN_4]]"""


def story_without(word: str, replacement: str = "huge") -> str:
    """GOOD_STORY with every form of a word swapped out (to simulate a forgotten vocabulary word)."""
    return re.sub(word, replacement, GOOD_STORY, flags=re.IGNORECASE)


def story_broken_markers() -> str:
    """GOOD_STORY with one refrain marker deleted: a hard failure (nothing to substitute), but a clean enough one
    (nothing else wrong with the refrain) that the controller repairs it with a cheap, targeted retry."""
    return GOOD_STORY.replace("[[REFRAIN_2]]", "")


def story_with_marker(marker: str) -> str:
    """GOOD_STORY plus one extra sentence, so a 'revised' story is distinguishable from the original."""
    first, rest = GOOD_STORY.split("\n\n", 1)
    return f"{first} {marker}\n\n{rest}"


def judge_dict(failing: dict | None = None, **overrides) -> dict:
    """A complete judge JSON payload. Default = every check passes. ``failing`` maps check name -> the fix the judge asks for."""
    from models import CHECK_NAMES

    failing = failing or {}
    checks = {
        name: ({"evidence": f"quote about {name}", "passed": False, "fix": failing[name]} if name in failing
               else {"evidence": f"quote about {name}", "passed": True, "fix": ""})
        for name in CHECK_NAMES
    }
    d = {
        "strengths_to_preserve": ["The owl's explanation of a glimmer", "The refrain lead-ins"],
        "checks": checks,
        "overall_assessment": "A calm, well-built bedtime story.",
        "unsafe_content": False,
    }
    d.update(overrides)
    return d
