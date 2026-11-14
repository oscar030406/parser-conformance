"""Export recorded tool parser seeds with the Python parse results.

Set PC_SEEDS_OUT=<file> to write one JSON line per recorded tool-call
input. Another implementation, such as vLLM's Rust frontend, can then parse
the same text and compare its result with the "python" field.
"""

from __future__ import annotations

import json


def _registered_names() -> dict[str, list[str]]:
    from vllm.tool_parsers import ToolParserManager

    names: dict[str, list[str]] = {}
    registry = {**ToolParserManager.lazy_parsers, **ToolParserManager.tool_parsers}
    for name in sorted(registry):
        cls = ToolParserManager.get_tool_parser(name)
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


def export_seeds(seeds: list[tuple], path: str) -> int:
    names = _registered_names()
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for family, parser, text, request, nodeid in seeds:
            if family != "tool":
                continue
            # A fresh instance: the checks have already called this one, and
            # a stateful parser can answer differently after that.
            args, kwargs = parser._pc_init
            info = type(parser)(*args, **kwargs).extract_tool_calls(text, request)
            row = {
                "parser_class": type(parser).__name__,
                "parser_names": names.get(type(parser).__name__, []),
                "nodeid": nodeid,
                "tokenizer": getattr(parser.model_tokenizer, "name_or_path", None),
                "text": text,
                "tools": _tools(parser, request),
                "python": {
                    "tools_called": info.tools_called,
                    "calls": [
                        {"name": c.function.name, "arguments": c.function.arguments}
                        for c in info.tool_calls or []
                    ],
                    "content": info.content,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
