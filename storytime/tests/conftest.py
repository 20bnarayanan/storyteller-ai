import sys
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fixtures import FOX_BLUEPRINT  # noqa: E402
from llm_client import UsageStats  # noqa: E402
from models import Blueprint, StoryRequest  # noqa: E402


@dataclass
class Call:
    role: str
    system: str
    user: str


class ScriptedLLM:
    """Fake LLM: answers each role from a queue and records every call.

    Asking for more responses than were scripted raises AssertionError, so a runaway
    revision loop makes a test fail loudly instead of hanging.
    """

    def __init__(self, planner=(), storyteller=(), judge=()):
        self.queues = {
            "planner": deque(planner),
            "storyteller": deque(storyteller),
            "judge": deque(judge),
        }
        self.calls: list[Call] = []
        self.stats = UsageStats()

    def _next(self, role: str):
        if not self.queues[role]:
            raise AssertionError(f"Unexpected extra '{role}' call (nothing scripted)")
        return self.queues[role].popleft()

    def complete_text(self, role, system, user):
        self.calls.append(Call(role, system, user))
        self.stats.record(role)
        return self._next(role)

    def complete_json(self, role, system, user, schema_name, schema):
        self.calls.append(Call(role, system, user))
        self.stats.record(role)
        return self._next(role)

    def calls_for(self, role: str) -> list[Call]:
        return [c for c in self.calls if c.role == role]


@pytest.fixture(autouse=True)
def one_draft_by_default(monkeypatch):
    """The scripted tests assume a single draft; best-of-2 has its own tests that pass drafts=2."""
    import pipeline
    monkeypatch.setattr(pipeline, "DRAFTS", 1)


@pytest.fixture
def blueprint() -> Blueprint:
    return Blueprint.from_dict(FOX_BLUEPRINT)


@pytest.fixture
def request_age6() -> StoryRequest:
    return StoryRequest("a little fox who gets lost in an enchanted forest", age=6)
