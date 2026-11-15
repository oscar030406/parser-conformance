"""Streaming cost of tool parsers as the output grows.

Set PC_PERF=1 to run it at the end of the session. For each tool parser
with a real tokenizer, the longest recorded output gets n tokens of plain
text inside its first argument value, or in front of it when it has none.
A fresh parser streams the result token by token. If the time grows about
as fast as n, the parser does constant work per delta. If it grows about
as fast as n squared, each delta rescans the whole text.
"""

from __future__ import annotations

import re
import time

from parser_conformance.checks import parser_factory, token_deltas, token_pieces

PAD = "The weather report says it will be sunny with a light breeze today. "
SIZES = (256, 1024)
RUNS = 3
_STRING_VALUE = re.compile(r'"([^"\\]*)"\s*:\s*"')


def pad_position(text: str) -> int:
    """Where a long argument would be: inside the first string argument value."""
    start = max(text.find('"arguments"'), text.find('"parameters"'))
    if start >= 0:
        match = _STRING_VALUE.search(text, start)
        if match:
            return match.end()
    for match in _STRING_VALUE.finditer(text):
        if match.group(1) not in ("name", "type", "id"):
            return match.end()
    return 0


def _stream_time(make, deltas: list[str], request) -> float:
    from tests.tool_parsers.utils import run_tool_extraction_streaming

    fresh = make()
    _ = fresh.vocab  # built on first use; the tokenizer can take seconds for it
    start = time.perf_counter()
    run_tool_extraction_streaming(fresh, deltas, request, assert_one_tool_per_delta=False)
    return time.perf_counter() - start


def perf_seed(parser, text: str, request) -> dict | None:
    make = parser_factory(parser)
    if make is None or token_deltas(parser, text, []) is None:
        return None
    tokenizer = parser.model_tokenizer
    pad = token_pieces(tokenizer, PAD * 200)
    cut = pad_position(text)
    row: dict = {
        "parser": type(parser).__name__,
        "chars": len(text),
        "where": "argument" if cut else "front",
    }
    times = {}
    for n in SIZES:
        deltas = token_pieces(tokenizer, text[:cut] + "".join(pad[:n]) + text[cut:])
        try:
            times[n] = min(_stream_time(make, deltas, request) for _ in range(RUNS))
        except Exception as e:  # noqa: BLE001  parser under test; recorded in the row
            return {**row, "error": repr(e)[:200]}
    small, large = SIZES
    row.update(
        {
            f"ms_{small}": round(times[small] * 1000, 1),
            f"ms_{large}": round(times[large] * 1000, 1),
            "ratio": round(times[large] / times[small], 1),
        }
    )
    return row


def run_perf(seeds: list[tuple]) -> list[dict]:
    """Time the longest recorded output of each tool parser that has a real tokenizer."""
    longest: dict[str, tuple] = {}
    for family, parser, text, request, _nodeid in seeds:
        if family != "tool" or token_deltas(parser, text, []) is None:
            continue
        name = type(parser).__name__
        if name not in longest or len(text) > len(longest[name][1]):
            longest[name] = (parser, text, request)
    rows = [perf_seed(*seed) for seed in longest.values()]
    return [row for row in rows if row is not None]
