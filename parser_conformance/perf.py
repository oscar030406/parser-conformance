"""Streaming cost of tool parsers as the output grows.

Set PC_PERF=1 to run it at the end of the session. For each tool parser
with a real tokenizer, the recorded output gets n tokens of plain text in
front of it, and a fresh parser streams the result token by token. If the
time grows about as fast as n, the parser does constant work per delta. If
it grows about as fast as n squared, each delta rescans the whole text.
"""

from __future__ import annotations

import time

PAD = "The weather report says it will be sunny with a light breeze today. "
SIZES = (256, 1024)


def _stream_time(parser, deltas: list[str], request) -> float:
    from tests.tool_parsers.utils import run_tool_extraction_streaming

    args, kwargs = parser._pc_init
    fresh = type(parser)(*args, **kwargs)
    start = time.perf_counter()
    run_tool_extraction_streaming(fresh, deltas, request, assert_one_tool_per_delta=False)
    return time.perf_counter() - start


def perf_seed(parser, text: str, request) -> dict | None:
    from tests.tool_parsers.utils import split_string_into_token_deltas

    from parser_conformance.checks import token_deltas

    if token_deltas(parser, text, []) is None:
        return None
    pad = split_string_into_token_deltas(parser.model_tokenizer, PAD * 200)
    # Padding goes into the first JSON string value, where a long argument
    # would be, or in front of the output when there is none.
    cut = text.find('": "')
    cut = 0 if cut < 0 else cut + 4
    row: dict = {"parser": type(parser).__name__, "where": "argument" if cut else "front"}
    times = {}
    for n in SIZES:
        padded = text[:cut] + "".join(pad[:n]) + text[cut:]
        deltas = split_string_into_token_deltas(parser.model_tokenizer, padded)
        try:
            times[n] = _stream_time(parser, deltas, request)
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
    """Time the longest recorded output of each tool parser."""
    longest: dict[str, tuple] = {}
    for family, parser, text, request, _nodeid in seeds:
        if family != "tool":
            continue
        name = type(parser).__name__
        if name not in longest or len(text) > len(longest[name][1]):
            longest[name] = (parser, text, request)
    rows = []
    for parser, text, request in longest.values():
        row = perf_seed(parser, text, request)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda r: -r.get("ratio", 0))
