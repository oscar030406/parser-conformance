"""Demo tool parsers for the format  text<tool_call>{"name": ..., "arguments": {...}}</tool_call>.

DemoToolParser is correct: streaming gives the same calls and content as
non-streaming, and a call that does not close stays in the content. Each
subclass adds one bug that a check must find, or one behavior that a check
must report apart from bugs.
"""

from __future__ import annotations

import json
import re

from vllm.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

START, END = "<tool_call>", "</tool_call>"
_BLOCK = re.compile(re.escape(START) + "(.*?)" + re.escape(END), re.DOTALL)


def _call(body: str) -> ToolCall | None:
    try:
        obj = json.loads(body)
        return ToolCall(FunctionCall(obj["name"], json.dumps(obj["arguments"], ensure_ascii=False)))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def parse(text: str) -> tuple[list[ToolCall], str, str]:
    """The calls, the text outside them, and the text from a start marker that does not close.

    A block with invalid JSON stays in the text.
    """
    calls, outside, pos, last_end = [], [], 0, 0
    for match in _BLOCK.finditer(text):
        last_end = match.end()
        call = _call(match.group(1))
        if call is not None:
            outside.append(text[pos : match.start()])
            calls.append(call)
            pos = match.end()
    open_at = text.find(START, last_end)
    if open_at < 0:
        return calls, "".join(outside) + text[pos:], ""
    return calls, "".join(outside) + text[pos:open_at], text[open_at:]


def _info(calls: list[ToolCall], content: str) -> ExtractedToolCallInformation:
    return ExtractedToolCallInformation(bool(calls), calls, content or None)


@ToolParserManager.register_module("demo")
class DemoToolParser(ToolParser):
    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self.tool_call_start_token = START
        self.tool_call_end_token = END
        self._sent_content = ""
        self._sent_calls = 0

    def extract_tool_calls(self, model_output, request):
        calls, outside, tail = parse(model_output)
        return _info(calls, outside + tail)

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        # Markers are single tokens, so a delta never ends inside one.
        calls, outside, _tail = parse(current_text)
        content = outside[len(self._sent_content) :]
        new = [
            DeltaToolCall(
                index=i,
                id=f"call_{i}",
                function=DeltaFunctionCall(call.function.name, call.function.arguments),
            )
            for i, call in enumerate(calls)
            if i >= self._sent_calls
        ]
        self._sent_content, self._sent_calls = outside, len(calls)
        if not content and not new:
            return None
        return DeltaMessage(content=content or None, tool_calls=new)


class StickyToolParser(DemoToolParser):
    """After an output with a call that does not close, every later output is content."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._inside_call = False

    def extract_tool_calls(self, model_output, request):
        if self._inside_call:
            return _info([], model_output)
        self._inside_call = bool(parse(model_output)[2])
        return super().extract_tool_calls(model_output, request)


class DropsCutCallToolParser(DemoToolParser):
    """A call that does not close is dropped from the content."""

    def extract_tool_calls(self, model_output, request):
        calls, outside, _tail = parse(model_output)
        return _info(calls, outside)


class NoContentToolParser(DemoToolParser):
    """Returns no content when it finds no call, so plain text is lost."""

    def extract_tool_calls(self, model_output, request):
        info = super().extract_tool_calls(model_output, request)
        return info if info.tool_calls else _info([], "")


class AnchoredToolParser(DemoToolParser):
    """Finds calls only when the output starts with one, like the pythonic format."""

    def extract_tool_calls(self, model_output, request):
        if not model_output.startswith(START):
            return _info([], model_output)
        return super().extract_tool_calls(model_output, request)


class MissesCallToolParser(DemoToolParser):
    """Non-streaming finds calls only when the output ends with a call."""

    def extract_tool_calls(self, model_output, request):
        if not model_output.endswith(END):
            return _info([], model_output)
        return super().extract_tool_calls(model_output, request)


class LeakyToolParser(DemoToolParser):
    """Streaming puts the start marker of each call into the content."""

    def extract_tool_calls_streaming(self, *args):
        message = super().extract_tool_calls_streaming(*args)
        if message is not None and message.tool_calls:
            message.content = (message.content or "") + START
        return message


class EmptyArgsToolParser(DemoToolParser):
    """Streaming sends "" as the arguments of a call without arguments, not "{}"."""

    def extract_tool_calls_streaming(self, *args):
        message = super().extract_tool_calls_streaming(*args)
        for delta in message.tool_calls if message is not None else []:
            if delta.function.arguments == "{}":
                delta.function.arguments = ""
        return message


class PartialArgsToolParser(DemoToolParser):
    """Streams the arguments of a first call as they arrive, before the call closes."""

    _HEAD = re.compile(re.escape(START) + r'\{"name": "([^"]+)", "arguments": ')

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._args_sent: str | None = None

    def extract_tool_calls_streaming(self, previous_text, current_text, *args):
        head = self._HEAD.match(current_text)
        if head is None:
            return None
        arguments = current_text[head.end() :]
        if self._args_sent is None:
            self._args_sent = arguments
            call = DeltaFunctionCall(head.group(1), arguments)
            return DeltaMessage(tool_calls=[DeltaToolCall(index=0, id="call_0", function=call)])
        new, self._args_sent = arguments[len(self._args_sent) :], arguments
        if not new:
            return None
        return DeltaMessage(tool_calls=[DeltaToolCall(index=0, function=DeltaFunctionCall(None, new))])


class PerTokenToolParser(DemoToolParser):
    """Streaming finds a call only when its start marker is a delta of its own."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._buffer: str | None = None

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, *args):
        if self._buffer is None:
            if delta_text == START:
                self._buffer = ""
                return None
            return DeltaMessage(content=delta_text)
        self._buffer += delta_text
        if END not in self._buffer:
            return None
        body, after = self._buffer.split(END, 1)
        call, index = _call(body), self._sent_calls
        self._buffer, self._sent_calls = None, index + 1
        function = DeltaFunctionCall(call.function.name, call.function.arguments)
        delta = DeltaToolCall(index=index, id=f"call_{index}", function=function)
        return DeltaMessage(content=after or None, tool_calls=[delta])


class RaisingToolParser(DemoToolParser):
    """Raises on every output."""

    def extract_tool_calls(self, model_output, request):
        raise ValueError("demo parser error")
