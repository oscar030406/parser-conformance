"""Demo reasoning parsers for the format  <think>reasoning</think>content.

DemoReasoningParser is correct: streaming gives the same reasoning and
content as non-streaming. The start marker is optional. Each subclass adds
one bug that a check must find.
"""

from __future__ import annotations

from vllm.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

START, END = "<think>", "</think>"


def split(text: str) -> tuple[str | None, str | None]:
    body = text[len(START) :] if text.startswith(START) else text
    if END not in body:
        return body or None, None
    reasoning, content = body.split(END, 1)
    return reasoning or None, content or None


class DemoReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._sent = (0, 0)

    def extract_reasoning(self, model_output, request):
        return split(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        # Markers are single tokens, so a delta never ends inside one.
        reasoning, content = (part or "" for part in split(current_text))
        new = reasoning[self._sent[0] :], content[self._sent[1] :]
        self._sent = len(reasoning), len(content)
        if not any(new):
            return None
        return DeltaMessage(reasoning=new[0] or None, content=new[1] or None)


class LeakyReasoningParser(DemoReasoningParser):
    """Streaming puts the end marker into the content."""

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text, *args):
        message = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text, *args
        )
        return DeltaMessage(content=END) if delta_text == END else message


class OnceReasoningParser(DemoReasoningParser):
    """Parses only its first output. Every later output comes back as content."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._used = False

    def extract_reasoning(self, model_output, request):
        if self._used:
            return None, model_output
        self._used = True
        return super().extract_reasoning(model_output, request)
