"""Export recorded tool parser seeds with the Python parse results.

Set PC_SEEDS_OUT=<file> to write one JSON line per recorded tool-call
input. Another implementation, such as vLLM's Rust frontend, can then parse
the same text and compare its result with the "python" field.
"""

from __future__ import annotations

from parser_conformance.checks import parser_factory


def _registered_names() -> dict[str, list[str]]:
    from vllm.tool_parsers import ToolParserManager

    names: dict[str, list[str]] = {}
    registry = {**ToolParserManager.lazy_parsers, **ToolParserManager.tool_parsers}
    for name in sorted(registry):
        try:
            cls = ToolParserManager.get_tool_parser(name)
        except ImportError:  # noqa: S112  vLLM lazy parsers whose optional dependency is missing
            continue
        names.setdefault(cls.__name__, []).append(name)
    return names


def _tools(parser, request) -> list[dict] | None:
    tools = getattr(request, "tools", None)
    if not isinstance(tools, list):  # engine parsers take the tools in __init__
        args, kwargs = parser._pc_init
        tools = args[1] if len(args) > 1 else kwargs.get("tools")
    if not isinstance(tools, list):  # a Mock request, or no tools
        return None
    return [tool.model_dump(mode="json", exclude_none=True) for tool in tools]


def _python_result(parser, text: str, request) -> dict:
    # A fresh instance: the checks have already called this one, and a
    # stateful parser can answer differently after that.
    try:
        info = parser_factory(parser)().extract_tool_calls(text, request)
    except Exception as e:  # noqa: BLE001  parser under test; exported as an error
        return {"error": repr(e)[:300]}
    return {
        "tools_called": info.tools_called,
        "calls": [
            {"name": c.function.name, "arguments": c.function.arguments}
            for c in info.tool_calls or []
        ],
        "content": info.content,
    }


def export_rows(seeds: list[tuple]) -> list[dict]:
    names = _registered_names()
    rows = []
    for family, parser, text, request, nodeid in seeds:
        if family != "tool":
            continue
        tokenizer = getattr(parser.model_tokenizer, "name_or_path", None)
        rows.append(
            {
                "parser_class": type(parser).__name__,
                "parser_names": names.get(type(parser).__name__, []),
                "nodeid": nodeid,
                # A Mock tokenizer answers every attribute with another Mock.
                "tokenizer": tokenizer if isinstance(tokenizer, str) else None,
                "text": text,
                "tools": _tools(parser, request),
                "python": _python_result(parser, text, request),
            }
        )
    return rows
