"""The result types of vLLM's parsers, with the fields that the checks read."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    function: FunctionCall
    id: str = "call_0"


@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


@dataclass
class DeltaFunctionCall:
    name: str | None = None
    arguments: str | None = None


@dataclass
class DeltaToolCall:
    index: int
    function: DeltaFunctionCall
    id: str | None = None
    type: str | None = None


@dataclass
class DeltaMessage:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = field(default_factory=list)


class ChatCompletionRequest:
    """Like vLLM's request: every instance gets a random request_id."""

    def __init__(self, **fields):
        self.request_id = uuid.uuid4().hex
        self.fields = fields

    def model_dump_json(self, exclude=None) -> str:
        data = {"request_id": self.request_id, **self.fields}
        return json.dumps({k: v for k, v in data.items() if k not in (exclude or set())})
