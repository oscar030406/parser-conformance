"""A byte-level tokenizer: one token per UTF-8 byte, and one token per marker.

Like a byte-level BPE, it splits a character of several bytes over several
tokens, so a prefix of the token ids can end in the middle of a character.
"""

from __future__ import annotations

import re

MARKERS = ("<tool_call>", "</tool_call>", "<think>", "</think>")


class ByteTokenizer:
    name_or_path = "fake/bytes"

    def __init__(self) -> None:
        self.all_special_tokens = list(MARKERS)
        self._marker_ids = {m: 256 + i for i, m in enumerate(MARKERS)}
        self._markers = {i: m for m, i in self._marker_ids.items()}
        self._split = re.compile("(" + "|".join(map(re.escape, MARKERS)) + ")")
        # vLLM's tests use real vocabularies, and the checks skip any parser
        # whose vocabulary has fewer than 1000 entries.
        self._vocab = {f"<0x{b:02X}>": b for b in range(256)}
        self._vocab.update(self._marker_ids)
        self._vocab.update({f"<unused{i}>": 300 + i for i in range(1000)})

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._marker_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for part in self._split.split(text):
            if part in self._marker_ids:
                ids.append(self._marker_ids[part])
            else:
                ids.extend(part.encode("utf-8"))
        return ids

    def tokenize(self, text: str) -> list[str]:
        return [self._markers.get(i, f"<0x{i:02X}>") for i in self.encode(text)]

    def decode(self, ids: list[int]) -> str:
        out, pending = [], bytearray()
        for i in ids:
            if i in self._markers:
                out.append(pending.decode("utf-8", errors="replace"))
                out.append(self._markers[i])
                pending.clear()
            else:
                pending.append(i)
        out.append(pending.decode("utf-8", errors="replace"))
        return "".join(out)
