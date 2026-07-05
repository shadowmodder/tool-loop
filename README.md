[![CI](https://github.com/shadowmodder/tool-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/shadowmodder/tool-loop/actions/workflows/ci.yml)

# tool-loop

A minimal, correct implementation of the Anthropic agentic tool-use loop.

Most implementations get at least one of these wrong. This one doesn't.

```python
import anthropic
from toolloop import ToolLoop

def search(query: str) -> str:
    """Search the knowledge base and return matching passages."""
    ...  # your real implementation

def lookup(ticker: str) -> str:
    """Return the current price for a stock ticker."""
    ...

loop = ToolLoop(client=anthropic.Anthropic(), model="claude-opus-4-8", tools=[search, lookup])
result = loop.run([{"role": "user", "content": "Compare AAPL and MSFT based on recent news."}])
print(result.final_text())
```

Tools are plain Python functions. Schemas are auto-generated from type annotations and docstrings — no JSON by hand.

---

## What the naive loop gets wrong

### 1. Serial tool dispatch

When the model returns two `tool_use` blocks in a single response, a naive loop calls them one at a time:

```python
# common — unnecessarily slow
for block in tool_blocks:
    result = call_tool(block)
```

`ToolLoop` dispatches them concurrently with `ThreadPoolExecutor`:

```python
# correct — runs in parallel
with ThreadPoolExecutor(max_workers=len(tool_blocks)) as ex:
    futures = {ex.submit(call_tool, b): b.id for b in tool_blocks}
```

If the model calls a slow external API and a fast local function in the same turn, both finish in the time of the slowest one.

### 2. Exceptions crashing the loop

A tool that raises should return an error *to the model* so it can recover — not propagate an exception to the caller:

```python
# broken — one bad tool call kills the entire agent
result = my_tool(**block.input)

# correct — model sees the error and can retry or recover
try:
    out = my_tool(**block.input)
    return str(out), False
except Exception as exc:
    return f"{type(exc).__name__}: {exc}", True  # is_error=True
```

### 3. Malformed message history

The assistant message must include **all** content blocks — both text and tool_use — not just the text:

```python
# broken — omits the tool_use blocks, API rejects it
messages.append({"role": "assistant", "content": response_text})

# correct — full content list
messages.append({"role": "assistant", "content": response.content})
```

The tool_result user turn must use the exact `tool_use_id` from the corresponding block. Mismatches produce silent hallucinations.

### 4. Ignoring stop_reason

A response can contain `tool_use` blocks but have `stop_reason = "max_tokens"` (output was truncated mid-tool-call). Treating this as a normal tool invocation sends a malformed request:

```python
# broken — ignores stop_reason
if any(b.type == "tool_use" for b in response.content):
    dispatch_tools(...)

# correct — only dispatch on explicit tool_use stop
if response.stop_reason == "tool_use":
    dispatch_tools(...)
```

---

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and the `anthropic` SDK.

---

## Usage

### Basic

```python
import anthropic
from toolloop import ToolLoop

def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 72°F in {city}"

loop = ToolLoop(
    client=anthropic.Anthropic(),
    model="claude-opus-4-8",
    tools=[get_weather],
)
result = loop.run([{"role": "user", "content": "What's the weather in Chicago?"}])
print(result.final_text())
```

### With a system prompt

```python
result = loop.run(
    messages=[{"role": "user", "content": "..."}],
    system="You are a precise assistant. Always use tools for external data.",
)
```

### Inspecting the trace

```python
print(result.to_markdown())
# Step 0  stop_reason=tool_use
# - `get_weather({"city": "Chicago"})` → [ok] Sunny, 72°F in Chicago
#   *tokens: 312 in / 48 out*
#
# Step 1  stop_reason=end_turn
# > It's sunny and 72°F in Chicago right now.
#   *tokens: 374 in / 21 out*
#
# Total tokens — input: 686  output: 69  cache_read: 0  cache_write: 0

for step in result.trace:
    print(step["iteration"], step["stop_reason"], step["tool_calls"])
```

### Auto-schema generation

```python
from toolloop import fn_to_schema
import json

def calculate(expression: str, precision: int = 6) -> str:
    """Evaluate a mathematical expression to the given decimal precision."""
    return str(round(eval(expression), precision))

print(json.dumps(fn_to_schema(calculate), indent=2))
```

```json
{
  "name": "calculate",
  "description": "Evaluate a mathematical expression to the given decimal precision.",
  "input_schema": {
    "type": "object",
    "properties": {
      "expression": {"type": "string"},
      "precision": {"type": "integer"}
    },
    "required": ["expression"]
  }
}
```

Parameters with defaults are not required. `Optional[X]` annotations map correctly. Unannotated parameters default to `"string"`.

---

## ToolLoop options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `client` | — | `anthropic.Anthropic()` instance |
| `model` | — | Model string, e.g. `"claude-opus-4-8"` |
| `tools` | — | List of Python callables |
| `max_iterations` | `10` | Hard stop — prevents infinite loops |
| `parallel` | `True` | Dispatch multiple tool calls concurrently |
| `max_tokens` | `4096` | Per-call output token budget |

---

## LoopResult

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list` | Full conversation history including tool turns |
| `trace` | `list` | Per-iteration dict with tool calls, results, tokens |
| `tokens` | `dict` | Cumulative `input`, `output`, `cache_read`, `cache_write` |
| `stop_reason` | `str` | Final stop reason from the last model call |
| `final_text()` | method | Last text block the model emitted |
| `to_markdown()` | method | Human-readable trace for debugging |

---

## Running examples

```bash
export ANTHROPIC_API_KEY=sk-...
python examples/calculator.py
python examples/filesystem_agent.py
```

`calculator.py` sends a multi-part math question, triggering parallel tool dispatch across three independent computations. `filesystem_agent.py` demonstrates error recovery: if a file doesn't exist, the error is returned as a `tool_result` with `is_error=True` so the model can adjust rather than crashing.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

All tests mock the Anthropic client — no API key needed.
