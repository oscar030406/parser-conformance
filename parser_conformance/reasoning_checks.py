"""Conformance properties checked for one recorded reasoning parser input.

R1  Plain text added in front of the output ("R1_prefix") or after it
    ("R1_suffix") is kept, in reasoning or in content.
R2  Streaming the output token by token gives the same reasoning and
    content as non-streaming extraction ("R2_stream"). "R2_trunc" compares
    the two modes on token-boundary prefixes, which is what a max_tokens
    cut produces. "_markers" means the two differ only in special tokens
    or whitespace.

Both properties compare the parser with itself, so they do not depend on
knowing each model's marker words. Every parse uses a fresh parser.
"""

from __future__ import annotations

import random

from parser_conformance.checks import (
    MAX_CUTS,
    PREFIX,
    _markers,
    _norm,
    _strip,
    parser_factory,
    token_deltas,
)

SUFFIX = " Thanks for asking."


def _kept(words: str, reasoning, content) -> bool:
    return words in (reasoning or "") or words in (content or "")


def _stream_states(parser, deltas: list[str]) -> list[tuple]:
    """Stream the deltas once and return (reasoning, content) after each."""
    tokenizer = parser.model_tokenizer
    previous_text, previous_tokens = "", []
    reasoning = content = None
    states = []
    for delta in deltas:
        token_delta = [
            parser.vocab[token] for token in tokenizer.tokenize(delta) if token in parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        message = parser.extract_reasoning_streaming(
            previous_text, current_text, delta, previous_tokens, current_tokens, token_delta
        )
        if message is not None:
            if message.content is not None:
                content = (content or "") + message.content
            if message.reasoning is not None:
                reasoning = (reasoning or "") + message.reasoning
        states.append((reasoning, content))
        previous_text, previous_tokens = current_text, current_tokens
    return states


def check_reasoning_seed(parser, text: str, request) -> list[dict]:
    make = parser_factory(parser)
    if make is None:
        return [{"check": "skip_fresh", "input": text}]
    findings: list[dict] = []
    markers = _markers(parser)

    def extract(source: str, check: str):
        try:
            return make().extract_reasoning(source, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": check, "input": source, "error": repr(e)[:300]})
            return None

    def compare(check: str, source: str, stream: tuple, nonstream: tuple) -> None:
        if all(_norm(a) == _norm(b) for a, b in zip(stream, nonstream, strict=True)):
            return
        same = all(
            _strip(a, markers) == _strip(b, markers) for a, b in zip(stream, nonstream, strict=True)
        )
        findings.append(
            {
                "check": f"{check}_markers" if same else check,
                "input": source,
                "stream": [_norm(s)[:200] for s in stream],
                "nonstream": [_norm(s)[:200] for s in nonstream],
            }
        )

    base = extract(text, "raises")
    if base is None:
        return findings

    for check, variant, words in (
        ("R1_prefix", PREFIX + text, PREFIX.strip()),
        ("R1_suffix", text + SUFFIX, SUFFIX.strip()),
    ):
        result = extract(variant, f"raises_{check}")
        if result is not None and not _kept(words, *result):
            reasoning, content = result
            findings.append(
                {"check": check, "input": variant, "reasoning": reasoning, "content": content}
            )

    deltas = token_deltas(parser, text, findings)
    if not deltas:
        return findings
    try:
        states = _stream_states(make(), deltas)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        findings.append({"check": "raises_stream", "input": text, "error": repr(e)[:300]})
        return findings

    joined = "".join(deltas)
    target = base if joined == text else extract(joined, "raises")
    if target is not None:
        compare("R2_stream", text, states[-1], target)
    cuts = list(range(1, len(deltas)))
    if len(cuts) > MAX_CUTS:
        cuts = sorted(random.Random(0).sample(cuts, MAX_CUTS))
    for k in cuts:
        prefix = "".join(deltas[:k])
        cut = extract(prefix, "raises_trunc")
        if cut is not None:
            compare("R2_trunc", prefix, states[k - 1], cut)
    return findings
