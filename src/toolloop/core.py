"""Correct, parallel Claude tool-use loop."""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .schema import fn_to_schema


@dataclass
class LoopResult:
    """Returned by ToolLoop.run()."""
    messages: list
    trace: list
    tokens: dict
    stop_reason: str

    def final_text(self) -> str:
        """Last text block the model emitted."""
        for step in reversed(self.trace):
            if step["text"]:
                return step["text"]
        return ""

    def to_markdown(self) -> str:
        """Human-readable trace of every iteration."""
        lines = ["# Tool loop trace", ""]
        for step in self.trace:
            lines.append(
                f"## Step {step['iteration']}  stop_reason={step['stop_reason']}"
            )
            if step["text"]:
                lines.append(f"> {step['text'][:300]}")
            for call, res in zip(step["tool_calls"], step["tool_results"]):
                tag = "ERR" if res["error"] else "ok"
                lines.append(
                    f"- `{call['name']}({json.dumps(call['input'])})` "
                    f"→ [{tag}] {str(res['result'])[:120]}"
                )
            t = step["tokens"]
            lines.append(
                f"  *tokens: {t['input']} in / {t['output']} out*"
            )
            lines.append("")
        t = self.tokens
        lines.append(
            f"**Total tokens** — input: {t['input']}  output: {t['output']}  "
            f"cache_read: {t['cache_read']}  cache_write: {t['cache_write']}"
        )
        return "\n".join(lines)


class ToolLoop:
    """Run a Claude tool-use loop until end_turn or max_iterations.

    Args:
        client:         ``anthropic.Anthropic()`` instance.
        model:          Claude model string, e.g. ``"claude-opus-4-8"``.
        tools:          Python callables — schemas auto-generated from type
                        annotations and docstrings. No JSON schema by hand.
        max_iterations: Hard ceiling on model calls. Prevents runaway loops.
        parallel:       Dispatch multiple tool calls from one response
                        concurrently (default True). Most implementations
                        skip this and leave latency on the table.
        max_tokens:     Per-call output token budget (default 4096).

    Common mistakes this class avoids:

    * **Serial tool dispatch** — when a response contains two tool_use blocks,
      most code dispatches them one at a time. We use ThreadPoolExecutor so
      independent tools run in parallel.

    * **Exceptions crashing the loop** — a tool that raises an exception
      should return an is_error result to the model so it can recover, not
      blow up the caller. _call_one() never raises.

    * **Malformed message history** — the assistant message must include the
      full content list (text + tool_use blocks together), not just the text.
      The tool_result user turn must reference the correct tool_use_id for each
      call. Both are handled correctly here.

    * **Ignoring stop_reason** — we check stop_reason, not just "is there a
      tool block." A response can contain tool blocks with stop_reason
      "max_tokens"; treating that as tool_use causes silent truncation bugs.
    """

    def __init__(
        self,
        client,
        model: str,
        tools: list,
        max_iterations: int = 10,
        parallel: bool = True,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.model = model
        self.tool_fns: dict[str, Callable] = {fn.__name__: fn for fn in tools}
        self.schemas: list[dict] = [fn_to_schema(fn) for fn in tools]
        self.max_iterations = max_iterations
        self.parallel = parallel
        self.max_tokens = max_tokens

    def run(self, messages: list, system: str | None = None) -> LoopResult:
        """Run the tool-use loop and return a LoopResult.

        Args:
            messages: Initial conversation as a list of role/content dicts.
                      Not mutated — a copy is made internally.
            system:   Optional system prompt string.

        Returns:
            LoopResult with the full message history, per-iteration trace,
            cumulative token counts, and the final stop_reason.
        """
        messages = list(messages)
        trace: list = []
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        response = None

        for iteration in range(self.max_iterations):
            kwargs: dict = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=self.schemas,
                messages=messages,
            )
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)

            usage = response.usage
            tokens["input"] += getattr(usage, "input_tokens", 0)
            tokens["output"] += getattr(usage, "output_tokens", 0)
            tokens["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
            tokens["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0

            tool_blocks = [
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            ]
            text_blocks = [
                b for b in response.content if getattr(b, "type", None) == "text"
            ]

            step: dict = {
                "iteration": iteration,
                "stop_reason": response.stop_reason,
                "text": " ".join(b.text for b in text_blocks),
                "tool_calls": [{"name": b.name, "input": b.input} for b in tool_blocks],
                "tool_results": [],
                "tokens": {
                    "input": getattr(usage, "input_tokens", 0),
                    "output": getattr(usage, "output_tokens", 0),
                },
            }

            # Stop if the model is done or didn't actually use a tool
            if response.stop_reason != "tool_use" or not tool_blocks:
                trace.append(step)
                break

            results = self._dispatch(tool_blocks)

            tool_result_content: list = []
            for block in tool_blocks:
                value, is_error = results[block.id]
                step["tool_results"].append(
                    {"name": block.name, "result": value, "error": is_error}
                )
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": value,
                    "is_error": is_error,
                })

            trace.append(step)
            # Full content list (text + tool_use blocks) must go into the
            # assistant message — omitting either breaks the conversation.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_result_content})

        return LoopResult(
            messages=messages,
            trace=trace,
            tokens=tokens,
            stop_reason=response.stop_reason if response else "max_iterations",
        )

    def _dispatch(self, tool_blocks) -> dict:
        """Call all tools, returning {tool_use_id: (result_str, is_error)}."""
        if self.parallel and len(tool_blocks) > 1:
            results: dict = {}
            with ThreadPoolExecutor(max_workers=len(tool_blocks)) as ex:
                future_to_id = {
                    ex.submit(self._call_one, b): b.id for b in tool_blocks
                }
                for future in as_completed(future_to_id):
                    results[future_to_id[future]] = future.result()
            return results
        return {b.id: self._call_one(b) for b in tool_blocks}

    def _call_one(self, block) -> tuple:
        """Call one tool. Never raises — exceptions become is_error results."""
        fn = self.tool_fns.get(block.name)
        if fn is None:
            return f"Unknown tool: {block.name!r}", True
        try:
            out = fn(**block.input)
            if isinstance(out, str):
                return out, False
            return json.dumps(out, default=str), False
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", True
