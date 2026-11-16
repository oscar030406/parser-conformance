from __future__ import annotations

from functools import cached_property


class ToolParser:
    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer
        self.tools = tools or []

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()


class ToolParserManager:
    tool_parsers: dict[str, type] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}

    @classmethod
    def register_module(cls, name: str):
        def register(parser: type) -> type:
            cls.tool_parsers[name] = parser
            return parser

        return register

    @classmethod
    def get_tool_parser(cls, name: str) -> type:
        return cls.tool_parsers[name]
