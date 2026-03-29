from __future__ import annotations

import inspect
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


SPECIAL_ACTION_PARAMS = {"action_context", "action_agent"}


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """
    Convert a Python type annotation into a basic JSON schema fragment.

    Supported mappings:
    - str -> string
    - int -> integer
    - float -> number
    - bool -> boolean
    - list[T] -> array
    - dict[...] -> object
    - Optional[T] / Union[T, None] -> schema for T
    - fallback -> string
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    if annotation is list:
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}

    if origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }

    if origin is dict:
        return {"type": "object"}

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])

    return {"type": "string"}


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """
    Parse a docstring and extract:
    - the top-level description
    - parameter descriptions from an 'Args:' section

    Expected format:

    '''
    Short description.

    Longer description if needed.

    Args:
        param_one: Description for param one.
        param_two: Description for param two.

    Returns:
        Description of return value.
    '''
    """
    if not docstring:
        return "No description provided.", {}

    lines = docstring.strip().splitlines()

    description_lines: list[str] = []
    param_descriptions: dict[str, str] = {}

    in_args_section = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            if not in_args_section:
                description_lines.append("")
            continue

        if line == "Args:":
            in_args_section = True
            continue

        if line in {"Returns:", "Raises:"}:
            in_args_section = False
            continue

        if in_args_section:
            if ":" in line:
                param_name, param_description = line.split(":", 1)
                param_descriptions[param_name.strip()] = param_description.strip()
            continue

        description_lines.append(line)

    description = " ".join(part for part in description_lines if part).strip()
    if not description:
        description = "No description provided."

    return description, param_descriptions


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Build a JSON schema object from a function signature, type hints,
    and docstring parameter descriptions.
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    docstring = inspect.getdoc(func)
    _, param_docs = _parse_docstring(docstring)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    for param_name, param in signature.parameters.items():
        if param_name in SPECIAL_ACTION_PARAMS:
            continue

        annotation = type_hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(annotation)

        if param_name in param_docs:
            param_schema["description"] = param_docs[param_name]

        schema["properties"][param_name] = param_schema

        if param.default is inspect.Parameter.empty:
            schema["required"].append(param_name)

    return schema


def action(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    terminal: bool = False,
    tags: list[str] | None = None,
):
    """
    Decorator that attaches agent action metadata to a function.

    Rules:
    - name defaults to function.__name__
    - description defaults to parsed function docstring description
    - parameters default to schema inferred from function signature + type hints + Args:
    - tags are optional
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        docstring = inspect.getdoc(func)
        parsed_description, _ = _parse_docstring(docstring)

        resolved_name = name or func.__name__
        resolved_description = description or parsed_description
        resolved_parameters = parameters or _build_parameters_schema(func)

        setattr(
            func,
            "_action_meta",
            {
                "name": resolved_name,
                "description": resolved_description,
                "parameters": resolved_parameters,
                "terminal": terminal,
                "tags": tags or [],
            },
        )

        return func

    return decorator
