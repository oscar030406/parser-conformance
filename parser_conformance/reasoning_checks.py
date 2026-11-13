"""Conformance properties checked for one recorded reasoning parser input.

R1  Plain text added in front of the output ("R1_prefix") or after it
    ("R1_suffix") is kept, in reasoning or in content.
R2  Streaming the output token by token gives the same reasoning and
    content as non-streaming extraction ("R2_stream"). "R2_trunc" compares
    the two modes on token-boundary prefixes, which is what a max_tokens
    cut produces.

Both properties compare the parser with itself, so they do not depend on
knowing each model's marker words.
"""

from __future__ import annotations

import random

from parser_conformance.checks import MAX_CUTS, PREFIX, _markers, _norm, _strip, token_deltas

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
    findings: list[dict] = []
    markers = _markers(parser)

    def same(a, b) -> bool:
        return all(_strip(x, markers) == _strip(y, markers) for x, y in zip(a, b, strict=True))

    try:
        base = parser.extract_reasoning(text, request)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        return [{"check": "raises", "input": text, "error": repr(e)[:300]}]

    for check, variant, words in (
        ("R1_prefix", PREFIX + text, PREFIX.strip()),
        ("R1_suffix", text + SUFFIX, SUFFIX.strip()),
    ):
        try:
            reasoning, content = parser.extract_reasoning(variant, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": f"raises_{check}", "input": variant, "error": repr(e)[:200]})
            continue
        if not _kept(words, reasoning, content):
            findings.append(
                {"check": check, "input": variant, "reasoning": reasoning, "content": content}
            )

    deltas = token_deltas(parser, text, findings)
    if deltas is None:
        return findings
    if not deltas or "_pc_init" not in vars(parser):
        findings.append({"check": "skip_fresh", "input": text})
        return findings
    args, kwargs = parser._pc_init
    try:
        states = _stream_states(type(parser)(*args, **kwargs), deltas)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        findings.append({"check": "raises_stream", "input": text, "error": repr(e)[:300]})
        return findings

    if not same(states[-1], base):
        findings.append(
            {
                "check": "R2_stream",
                "input": text,
                "stream": [_norm(s)[:200] for s in states[-1]],
                "nonstream": [_norm(s)[:200] for s in base],
            }
        )
    cuts = list(range(1, len(deltas)))
    if len(cuts) > MAX_CUTS:
        cuts = sorted(random.Random(0).sample(cuts, MAX_CUTS))
    for k in cuts:
        prefix = "".join(deltas[:k])
        try:
            cut = parser.extract_reasoning(prefix, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": "raises_trunc", "input": prefix, "error": repr(e)[:200]})
            continue
        if not same(states[k - 1], cut):
            findings.append(
                {
                    "check": "R2_trunc",
                    "input": prefix,
                    "stream": [_norm(s)[:200] for s in states[k - 1]],
                    "nonstream": [_norm(s)[:200] for s in cut],
                }
            )
    return findings
