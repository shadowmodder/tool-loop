"""
Calculator agent — demonstrates parallel tool dispatch.

Claude receives a multi-part math question and must call several tools in one
response. ToolLoop dispatches them concurrently, then feeds the results back
so Claude can produce a final answer.

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/calculator.py
"""
import anthropic
from toolloop import ToolLoop


def add(a: float, b: float) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


def multiply(a: float, b: float) -> str:
    """Multiply two numbers and return the result."""
    return str(a * b)


def power(base: float, exp: float) -> str:
    """Raise base to the power of exp."""
    return str(base ** exp)


def main() -> None:
    client = anthropic.Anthropic()
    loop = ToolLoop(
        client=client,
        model="claude-opus-4-8",
        tools=[add, multiply, power],
        parallel=True,
    )

    result = loop.run(
        messages=[{
            "role": "user",
            "content": (
                "Compute each of the following independently, then multiply "
                "all three results together to give a grand total:\n"
                "  A = 47 + 83\n"
                "  B = 12 × 15\n"
                "  C = 2^10\n"
            ),
        }],
        system="You are a precise calculator assistant. Use the provided tools for every arithmetic step.",
    )

    print(result.to_markdown())
    print("\n=== Final answer ===")
    print(result.final_text())
    print(f"\nTotal tokens used: {result.tokens}")


if __name__ == "__main__":
    main()
