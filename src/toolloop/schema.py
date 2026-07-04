"""Auto-generate Claude tool schemas from Python function signatures."""
from __future__ import annotations
import inspect
import typing


_PY_TO_JSON: dict = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        non_none = [a for a in annotation.__args__ if a is not type(None)]
        if non_none:
            return _PY_TO_JSON.get(non_none[0], "string")
    return _PY_TO_JSON.get(annotation, "string")


def fn_to_schema(fn) -> dict:
    """Convert a Python callable to a Claude tool schema.

    Uses the function name as tool name, docstring as description, and
    parameter type annotations to build input_schema. Parameters without
    defaults are marked required. Optional[X] → not required, type = X.
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or fn.__name__
    properties: dict = {}
    required: list = []

    for name, param in sig.parameters.items():
        annotation = param.annotation

        # Unwrap Optional[X] to get X for the type, but mark as not required
        is_optional = False
        origin = getattr(annotation, "__origin__", None)
        if origin is typing.Union:
            non_none = [a for a in annotation.__args__ if a is not type(None)]
            if type(None) in annotation.__args__:
                is_optional = True
            annotation = non_none[0] if non_none else inspect.Parameter.empty

        json_type = _PY_TO_JSON.get(annotation, "string")
        properties[name] = {"type": json_type}

        has_default = param.default is not inspect.Parameter.empty
        if not has_default and not is_optional:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": doc,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
