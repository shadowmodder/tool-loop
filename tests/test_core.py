"""Tests for ToolLoop and fn_to_schema. No API key required — client is mocked."""
from typing import Optional
from unittest.mock import MagicMock
import pytest

from toolloop import ToolLoop, fn_to_schema


# ── Mock helpers ──────────────────────────────────────────────────────────

class _Text:
    type = "text"
    def __init__(self, text): self.text = text

class _ToolUse:
    type = "tool_use"
    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input

class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


def _resp(stop_reason, *content):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = list(content)
    r.usage = _Usage()
    return r


def _loop(responses, tools=None):
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return ToolLoop(
        client=client,
        model="claude-opus-4-8",
        tools=tools or [_add],
    )


# ── Sample tools ──────────────────────────────────────────────────────────

def _add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)

def _explode(msg: str) -> str:
    """Always raises ValueError."""
    raise ValueError(msg)

def _returns_dict(x: str) -> str:
    """Returns a dict (auto-serialised to JSON)."""
    return {"echo": x}  # type: ignore[return-value]


# ── Core loop tests ───────────────────────────────────────────────────────

def test_end_turn_no_tools():
    loop = _loop([_resp("end_turn", _Text("hello"))])
    result = loop.run([{"role": "user", "content": "hi"}])
    assert result.stop_reason == "end_turn"
    assert result.final_text() == "hello"
    assert len(result.trace) == 1
    assert result.trace[0]["tool_calls"] == []


def test_single_tool_call_and_followup():
    tu = _ToolUse("id1", "_add", {"a": 3, "b": 4})
    loop = _loop([
        _resp("tool_use", tu),
        _resp("end_turn", _Text("The answer is 7")),
    ])
    result = loop.run([{"role": "user", "content": "3+4?"}])
    assert result.stop_reason == "end_turn"
    assert result.final_text() == "The answer is 7"
    assert result.trace[0]["tool_results"][0]["result"] == "7"
    assert not result.trace[0]["tool_results"][0]["error"]


def test_tool_error_is_captured_not_raised():
    tu = _ToolUse("id1", "_explode", {"msg": "boom"})
    loop = _loop(
        [_resp("tool_use", tu), _resp("end_turn", _Text("sorry"))],
        tools=[_explode],
    )
    result = loop.run([{"role": "user", "content": "fail"}])
    tr = result.trace[0]["tool_results"][0]
    assert tr["error"] is True
    assert "ValueError" in tr["result"]
    assert result.stop_reason == "end_turn"


def test_unknown_tool_returns_error_result():
    tu = _ToolUse("id1", "does_not_exist", {})
    loop = _loop([_resp("tool_use", tu), _resp("end_turn", _Text("ok"))])
    result = loop.run([{"role": "user", "content": "x"}])
    tr = result.trace[0]["tool_results"][0]
    assert tr["error"] is True
    assert "Unknown tool" in tr["result"]


def test_max_iterations_hard_stop():
    tu = _ToolUse("id1", "_add", {"a": 1, "b": 2})
    # Provide more responses than max_iterations to confirm the guard fires
    loop = _loop([_resp("tool_use", tu)] * 10, tools=[_add])
    loop.max_iterations = 3
    result = loop.run([{"role": "user", "content": "loop"}])
    assert len(result.trace) == 3


def test_parallel_dispatch_both_tools_execute():
    b1 = _ToolUse("id1", "_add", {"a": 1, "b": 2})
    b2 = _ToolUse("id2", "_add", {"a": 10, "b": 20})
    loop = _loop([
        _resp("tool_use", b1, b2),
        _resp("end_turn", _Text("done")),
    ])
    result = loop.run([{"role": "user", "content": "two sums"}])
    results = {r["result"] for r in result.trace[0]["tool_results"]}
    assert results == {"3", "30"}


def test_serial_dispatch_option():
    b1 = _ToolUse("id1", "_add", {"a": 1, "b": 1})
    b2 = _ToolUse("id2", "_add", {"a": 2, "b": 2})
    loop = _loop([_resp("tool_use", b1, b2), _resp("end_turn", _Text("ok"))])
    loop.parallel = False
    result = loop.run([{"role": "user", "content": "x"}])
    results = {r["result"] for r in result.trace[0]["tool_results"]}
    assert results == {"2", "4"}


def test_token_accumulation():
    tu = _ToolUse("id1", "_add", {"a": 1, "b": 1})
    loop = _loop([_resp("tool_use", tu), _resp("end_turn", _Text("done"))])
    result = loop.run([{"role": "user", "content": "x"}])
    assert result.tokens["input"] == 20   # 10 per call × 2
    assert result.tokens["output"] == 10  # 5 per call × 2


def test_message_history_structure():
    """After one tool call the history must be: user | assistant | user(tool_result)."""
    tu = _ToolUse("id1", "_add", {"a": 5, "b": 5})
    loop = _loop([_resp("tool_use", tu), _resp("end_turn", _Text("10"))])
    result = loop.run([{"role": "user", "content": "5+5?"}])
    assert len(result.messages) == 3
    assert result.messages[0]["role"] == "user"
    assert result.messages[1]["role"] == "assistant"
    assert result.messages[2]["role"] == "user"
    tr = result.messages[2]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "id1"


def test_dict_return_serialised_to_json():
    tu = _ToolUse("id1", "_returns_dict", {"x": "hello"})
    loop = _loop(
        [_resp("tool_use", tu), _resp("end_turn", _Text("ok"))],
        tools=[_returns_dict],
    )
    result = loop.run([{"role": "user", "content": "echo"}])
    import json
    payload = json.loads(result.trace[0]["tool_results"][0]["result"])
    assert payload == {"echo": "hello"}


def test_system_prompt_forwarded():
    loop = _loop([_resp("end_turn", _Text("hi"))])
    loop.run([{"role": "user", "content": "x"}], system="You are helpful.")
    call_kwargs = loop.client.messages.create.call_args[1]
    assert call_kwargs.get("system") == "You are helpful."


def test_no_system_prompt_not_forwarded():
    loop = _loop([_resp("end_turn", _Text("hi"))])
    loop.run([{"role": "user", "content": "x"}])
    call_kwargs = loop.client.messages.create.call_args[1]
    assert "system" not in call_kwargs


def test_to_markdown_contains_tool_names():
    tu = _ToolUse("id1", "_add", {"a": 2, "b": 3})
    loop = _loop([_resp("tool_use", tu), _resp("end_turn", _Text("5"))])
    result = loop.run([{"role": "user", "content": "sum"}])
    md = result.to_markdown()
    assert "_add" in md
    assert "Total tokens" in md


# ── Schema tests ──────────────────────────────────────────────────────────

def test_schema_name_and_description():
    schema = fn_to_schema(_add)
    assert schema["name"] == "_add"
    assert "Add two integers" in schema["description"]


def test_schema_required_params():
    schema = fn_to_schema(_add)
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


def test_schema_types():
    schema = fn_to_schema(_add)
    props = schema["input_schema"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"


def test_schema_optional_not_required():
    def greet(name: str, greeting: Optional[str] = None) -> str:
        """Greet someone."""
        return f"{greeting or 'Hello'}, {name}"

    schema = fn_to_schema(greet)
    assert "name" in schema["input_schema"]["required"]
    assert "greeting" not in schema["input_schema"]["required"]


def test_schema_default_not_required():
    def connect(host: str, port: int = 8080) -> str:
        """Open a connection."""
        return f"{host}:{port}"

    schema = fn_to_schema(connect)
    assert "host" in schema["input_schema"]["required"]
    assert "port" not in schema["input_schema"]["required"]
    assert schema["input_schema"]["properties"]["port"]["type"] == "integer"


def test_schema_unannotated_defaults_to_string():
    def mystery(x) -> str:
        """Unknown type."""
        return str(x)

    schema = fn_to_schema(mystery)
    assert schema["input_schema"]["properties"]["x"]["type"] == "string"


def test_schema_float_type():
    def scale(x: float, factor: float = 1.0) -> str:
        """Scale a value."""
        return str(x * factor)

    schema = fn_to_schema(scale)
    assert schema["input_schema"]["properties"]["x"]["type"] == "number"
