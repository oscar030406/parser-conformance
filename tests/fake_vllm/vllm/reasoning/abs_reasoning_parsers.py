from __future__ import annotations

from functools import cached_property


class ReasoningParser:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()
