# SPDX-License-Identifier: Apache-2.0
# Adapted from tests/tool_parsers/utils.py of vLLM at 191e5d5.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's streaming test helpers, with the logic that the checks depend on."""

from __future__ import annotations

from vllm.protocol import FunctionCall, ToolCall


class StreamingToolReconstructor:
    def __init__(self, assert_one_tool_per_delta: bool = True):
        self.tool_calls: list[ToolCall] = []
        self.other_content = ""
        self._assert_one_tool_per_delta = assert_one_tool_per_delta

    def append_delta(self, delta) -> None:
        if delta.content is not None:
            self.other_content += delta.content
        else:
            assert delta.tool_calls, "Streaming results should have either content or tool calls"
        if self._assert_one_tool_per_delta:
            assert len(delta.tool_calls) < 2, "Streaming should include only one tool call"
        for call_delta in delta.tool_calls:
            current = (
                self.tool_calls[call_delta.index]
                if call_delta.index < len(self.tool_calls)
                else None
            )
            if current:
                assert not call_delta.function.name
                assert not call_delta.id
                assert call_delta.index == len(self.tool_calls) - 1
                assert call_delta.function.arguments is not None
                current.function.arguments += call_delta.function.arguments
            else:
                assert call_delta.id is not None
                assert call_delta.function.name is not None
                assert call_delta.index == len(self.tool_calls)
                self.tool_calls.append(
                    ToolCall(
                        id=call_delta.id,
                        function=FunctionCall(
                            name=call_delta.function.name,
                            arguments=call_delta.function.arguments or "",
                        ),
                    )
                )


def split_string_into_token_deltas(tokenizer, text: str) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    previously_decoded_text = ""
    deltas = []
    for i in range(1, len(token_ids) + 1):
        current_text = tokenizer.decode(token_ids[:i])
        deltas.append(current_text[len(previously_decoded_text) :])
        previously_decoded_text = current_text
    return deltas


def run_tool_extraction_streaming(
    tool_parser, model_deltas, request=None, assert_one_tool_per_delta: bool = True
) -> StreamingToolReconstructor:
    if isinstance(model_deltas, str):
        model_deltas = split_string_into_token_deltas(tool_parser.model_tokenizer, model_deltas)
    reconstructor = StreamingToolReconstructor(assert_one_tool_per_delta)
    previous_text = ""
    previous_tokens: list[int] = []
    tokenizer = tool_parser.model_tokenizer
    for delta in model_deltas:
        tokens = tokenizer.tokenize(delta)
        token_delta = [tool_parser.vocab.get(t) for t in tokens if t in tool_parser.vocab]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
            request,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor
