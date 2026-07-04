"""
Filesystem agent — tools that read/write files, with error recovery demo.

The agent is asked to read a file, transform its contents, and write the
result. If any tool call fails (file not found, etc.) the error is returned
to Claude as a tool_result with is_error=True so it can recover gracefully
rather than crashing the loop.

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/filesystem_agent.py
"""
import os
import anthropic
from toolloop import ToolLoop


def read_file(path: str) -> str:
    """Read a file and return its contents as a string."""
    with open(path) as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Write content to a file. Returns 'ok' on success."""
    with open(path, "w") as f:
        f.write(content)
    return "ok"


def list_directory(path: str) -> str:
    """List files in a directory."""
    entries = os.listdir(path)
    return "\n".join(sorted(entries))


def main() -> None:
    import tempfile, pathlib

    # Seed a temp file for the agent to work with
    tmp = tempfile.mkdtemp()
    src = pathlib.Path(tmp) / "notes.txt"
    src.write_text("apple\nbanana\ncherry\ndate\nelder berry\n")

    client = anthropic.Anthropic()
    loop = ToolLoop(
        client=client,
        model="claude-opus-4-8",
        tools=[read_file, write_file, list_directory],
    )

    result = loop.run(
        messages=[{
            "role": "user",
            "content": (
                f"In the directory {tmp!r} there is a file called notes.txt "
                "containing one fruit per line. Read it, capitalise every "
                "fruit name, and write the result to a new file called "
                "notes_upper.txt in the same directory. "
                "Then list the directory to confirm both files exist."
            ),
        }]
    )

    print(result.to_markdown())
    out = pathlib.Path(tmp) / "notes_upper.txt"
    if out.exists():
        print("\n=== notes_upper.txt ===")
        print(out.read_text())


if __name__ == "__main__":
    main()
