"""All Anthropic API access lives here: model config, retries, structured output.

Everything else in the project talks to an object with two methods,
``complete_text`` and ``complete_json``. Tests swap in a scripted fake.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (single place to change model / provider settings)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-haiku-5-5"

# Per-role max_tokens.
ROLE_SETTINGS: dict[str, int] = {
    "planner": 3000,
    "storyteller": 4000,
    "judge": 3000,
}

REQUEST_TIMEOUT_S = 90.0


def model_for(role: str) -> str:
    """STORYTIME_MODEL_<ROLE> overrides STORYTIME_MODEL, which overrides Haiku 5.5."""
    return os.getenv(f"STORYTIME_MODEL_{role.upper()}") or os.getenv("STORYTIME_MODEL") or DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class StorytimeError(Exception):
    """Base class for expected, reportable failures."""


class ConfigError(StorytimeError):
    """Missing key or bad configuration."""


class LLMError(StorytimeError):
    """The API call itself failed (after retries)."""


class LLMOutputError(StorytimeError):
    """The model answered, but the answer was unusable (bad JSON, truncated, ...)."""


# ---------------------------------------------------------------------------
# Usage tracking (for the cost section of the evals)
# ---------------------------------------------------------------------------
_STATS_LOCK = threading.Lock()


@dataclass
class UsageStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_role: dict[str, int] = field(default_factory=dict)

    def record(self, role: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with _STATS_LOCK:  # drafts are written in parallel threads
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.by_role[role] = self.by_role.get(role, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_role": dict(self.by_role),
        }


class LLM(Protocol):
    """The only interface the rest of the code depends on."""

    stats: UsageStats

    def complete_text(self, role: str, system: str, user: str) -> str: ...

    def complete_json(
        self, role: str, system: str, user: str, schema_name: str, schema: dict
    ) -> dict: ...


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, api_key: Optional[str] = None, client: Any = None):
        self.stats = UsageStats()
        if client is not None:  # injected (tests)
            self._client = client
            return
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # python-dotenv is optional at runtime
            pass
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        import anthropic

        # The SDK retries transient connection, rate-limit and server errors automatically.
        self._client = anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_S)

    # -- public API ---------------------------------------------------------
    def complete_text(self, role: str, system: str, user: str) -> str:
        resp = self._create(role, system, user)
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise LLMOutputError(f"{role} output was truncated (max_tokens reached).")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMOutputError(f"{role} returned empty text.")
        return text

    def complete_json(self, role: str, system: str, user: str, schema_name: str, schema: dict) -> dict:
        """Structured output via forced tool use, so the SDK hands back parsed JSON."""
        tool = {
            "name": schema_name,
            "description": "Submit the structured result.",
            "input_schema": schema,
        }
        resp = self._create(
            role, system, user, tools=[tool], tool_choice={"type": "tool", "name": schema_name}
        )
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise LLMOutputError(f"{role} JSON output was truncated (max_tokens reached).")
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and isinstance(block.input, dict):
                return block.input
        raise LLMOutputError(f"{role} did not return the expected structured result.")

    # -- internals ----------------------------------------------------------
    def _create(self, role: str, system: str, user: str, **extra: Any) -> Any:
        kwargs: dict[str, Any] = {
            "model": model_for(role),
            "max_tokens": ROLE_SETTINGS.get(role, 2000),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "disabled"},
            **extra,
        }
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK has already applied its built-in retry policy.
            raise LLMError(f"{role} call failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        self.stats.record(
            role,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )
        return response


# ---------------------------------------------------------------------------
# Validate-and-retry helper (JSON repair retry lives here, not in story checks)
# ---------------------------------------------------------------------------
T = TypeVar("T")


def complete_validated(
    llm: LLM,
    role: str,
    system: str,
    user: str,
    schema_name: str,
    schema: dict,
    parse: Callable[[dict], T],
    attempts: int = 2,
) -> T:
    """Call the model for JSON, parse/validate it, and retry with the error message."""
    last_error: Optional[str] = None
    last_data: Optional[dict] = None
    for _ in range(attempts):
        prompt = user
        if last_error:
            # Show the rejected answer so the model edits it instead of starting over (starting over tends to fix one rule
            # and break another).
            previous = json.dumps(last_data, ensure_ascii=False)[:4000]
            prompt = (
                f"{user}\n\nYour previous answer was rejected for this reason:\n- {last_error}\n\n"
                f"Your previous answer was:\n{previous}\n\n"
                "Change only what is needed to fix that problem, keep everything else, and submit the corrected answer."
            )
        data = llm.complete_json(role, system, prompt, schema_name, schema)
        last_data = data
        try:
            return parse(data)
        except ValueError as exc:
            last_error = str(exc)
            logger.info("%s output failed validation: %s", role, last_error)
    raise LLMOutputError(f"{role} could not produce valid output after {attempts} attempts: {last_error}")
