from types import SimpleNamespace

import pytest

from llm_client import ConfigError, LLMClient, LLMError, LLMOutputError, complete_validated, model_for, DEFAULT_MODEL


def response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(data):
    return SimpleNamespace(type="tool_use", input=data)


class StubMessages:
    def __init__(self, item=None, error=None):
        self.item, self.error, self.calls = item, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.item


def make_client(item=None, error=None):
    messages = StubMessages(item, error)
    return LLMClient(client=SimpleNamespace(messages=messages)), messages


def test_text_call_tracks_usage_and_disables_thinking():
    client, messages = make_client(response([text_block("Once upon a time")]))
    assert client.complete_text("storyteller", "sys", "usr") == "Once upon a time"
    assert client.stats.calls == 1 and client.stats.input_tokens == 10
    assert messages.calls[0]["model"] == DEFAULT_MODEL
    assert messages.calls[0]["thinking"] == {"type": "disabled"}
    assert "temperature" not in messages.calls[0]


def test_json_call_forces_tool_use():
    client, messages = make_client(response([tool_block({"a": 1})]))
    assert client.complete_json("planner", "sys", "usr", "submit_blueprint", {"type": "object"}) == {"a": 1}
    assert messages.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_blueprint"}


def test_json_call_without_tool_block_raises_output_error():
    client, _ = make_client(response([text_block("sorry")]))
    with pytest.raises(LLMOutputError):
        client.complete_json("judge", "s", "u", "x", {})


def test_truncated_text_is_rejected():
    client, _ = make_client(response([text_block("half a story")], stop_reason="max_tokens"))
    with pytest.raises(LLMOutputError, match="truncated"):
        client.complete_text("storyteller", "s", "u")


def test_api_errors_are_wrapped_with_role_context():
    client, messages = make_client(error=RuntimeError("service unavailable"))
    with pytest.raises(LLMError, match="storyteller call failed"):
        client.complete_text("storyteller", "s", "u")
    assert len(messages.calls) == 1


def test_missing_api_key_gives_a_helpful_error(monkeypatch):
    import dotenv

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # a developer's local .env must not leak into the test
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        LLMClient()


def test_model_is_configurable_per_role(monkeypatch):
    monkeypatch.delenv("STORYTIME_MODEL", raising=False)
    monkeypatch.delenv("STORYTIME_MODEL_JUDGE", raising=False)
    assert model_for("judge") == DEFAULT_MODEL
    monkeypatch.setenv("STORYTIME_MODEL", "model-a")
    monkeypatch.setenv("STORYTIME_MODEL_JUDGE", "model-b")
    assert model_for("planner") == "model-a" and model_for("judge") == "model-b"


def test_complete_validated_retries_invalid_structured_output():
    class Fake:
        def __init__(self): self.prompts = []
        def complete_json(self, role, system, user, name, schema):
            self.prompts.append(user)
            return {"n": len(self.prompts)}

    def parse(data):
        if data["n"] < 2:
            raise ValueError("n must be at least 2")
        return data["n"]

    fake = Fake()
    assert complete_validated(fake, "judge", "s", "base prompt", "x", {}, parse) == 2
    assert "n must be at least 2" in fake.prompts[1]
