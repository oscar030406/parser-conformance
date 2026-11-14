"""Conformance properties checked for one recorded tool parser input.

I1  tools_called is true exactly when tool_calls is non-empty.
I2  When no call is parsed, content keeps the model output.
    "I2_text" means non-marker text was lost. "I2_markers" means only
    special tokens or whitespace differ.
I3  Streaming the output token by token gives the same calls and content
    as non-streaming extraction. "I3_lenient" marks inputs that only the
    streaming path turns into a call, such as JSON that never closes.

P   Putting plain text in front of the output keeps the same calls
    ("prefix_calls") and the text reaches content ("prefix_lost").
S   After all the variants, the same parser instance parses the original
    output as it did first ("S_state").

I1 and I2 also run on token-boundary prefixes of the input ("_trunc"),
which is what a max_tokens cut produces.

The parser under test can raise anything. Such exceptions are findings,
so the handlers around parser calls catch Exception and record it.
"""

from __future__ import annotations

import json
import os
import random
import re

MAX_CUTS = int(os.environ.get("PC_MAX_CUTS", "48"))
PREFIX = "Sure, checking now. "
_TAG = re.compile(r"<[^<>\s\\]{1,40}>")


def _summary(info) -> dict:
    calls = [(c.function.name, c.function.arguments) for c in info.tool_calls or []]
    return {"tools_called": info.tools_called, "calls": calls, "content": info.content}


def _norm(text: str | None) -> str:
    return (text or "").strip()


def _markers(parser) -> list[str]:
    tokenizer = parser.model_tokenizer
    found: set[str] = set(getattr(tokenizer, "all_special_tokens", None) or [])
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        found.update(get_added_vocab().keys())
    for name, value in vars(parser).items():
        if isinstance(value, re.Pattern):
            value = value.pattern
        if not isinstance(value, str):
            continue
        if name.endswith("token") and value.strip():
            found.add(value)
        # Tags kept in regex sources, e.g. Hunyuan's r"\n</think>\n<answer>\n".
        found.update(_TAG.findall(value))
    return sorted(found, key=len, reverse=True)


def _strip(text: str | None, markers: list[str]) -> str:
    text = text or ""
    for marker in markers:
        text = text.replace(marker, "")
    return "".join(text.split())


def _canonical(args: str) -> str:
    """Compare JSON arguments by value. Invalid JSON is compared as written."""
    try:
        return json.dumps(json.loads(args), sort_keys=True)
    except json.JSONDecodeError:
        return args


def _calls_key(calls) -> list[tuple[str, str]]:
    return [
        (call.function.name, _canonical(call.function.arguments or "{}")) for call in calls or []
    ]


def _cut_policy(prefix: str, content: str | None, markers: list[str]) -> dict:
    """Record what non-streaming returns when a cut leaves no complete call.

    This is an observation, not a pass or fail: vLLM's parsers disagree here.
    "policy_keeps_text" returns the cut output as content, "policy_drops_call"
    keeps only the text before the first marker, "policy_drops_all" returns no
    content.
    """
    starts = [i for i in (prefix.find(m) for m in markers) if i >= 0]
    before = prefix[: min(starts)] if starts else prefix
    if _norm(content) == _norm(prefix):
        kind = "keeps_text"
    elif _norm(content) == _norm(before) and _norm(before):
        kind = "drops_call"
    elif not _norm(content):
        kind = "drops_all"
    else:
        kind = "other"
    return {"check": f"policy_{kind}", "input": prefix, "content": content}


def token_deltas(parser, text: str, findings: list[dict]) -> list[str] | None:
    """Split text into the token pieces that streaming would see.

    Returns None, and records why, when the test built the parser with a
    stand-in tokenizer. Such a tokenizer has no real vocabulary, so the
    token ids that streaming parsers match on are missing.
    """
    from tests.tool_parsers.utils import split_string_into_token_deltas

    vocab = getattr(parser, "vocab", None)
    if not isinstance(vocab, dict) or len(vocab) < 1000:
        findings.append({"check": "skip_mock_tokenizer", "input": text})
        return None
    try:
        deltas = split_string_into_token_deltas(parser.model_tokenizer, text)
    except Exception as e:  # noqa: BLE001  test-provided tokenizer, e.g. a Mock
        findings.append({"check": "skip_tokenize", "input": text, "error": repr(e)[:200]})
        return None
    if not all(isinstance(d, str) for d in deltas):
        findings.append({"check": "skip_tokenize", "input": text, "error": "non-str pieces"})
        return None
    return deltas


def check_seed(parser, text: str, request) -> list[dict]:
    from tests.tool_parsers.utils import run_tool_extraction_streaming

    findings: list[dict] = []
    markers = _markers(parser)

    def judge(source: str, info, tag: str) -> None:
        if info.tools_called != bool(info.tool_calls):
            findings.append({"check": f"I1_flag{tag}", "input": source, "result": _summary(info)})
        if not info.tool_calls and _norm(info.content) != _norm(source):
            lost = _strip(source, markers) != _strip(info.content, markers)
            findings.append(
                {
                    "check": f"I2_{'text' if lost else 'markers'}{tag}",
                    "input": source,
                    "result": _summary(info),
                }
            )

    try:
        base = parser.extract_tool_calls(text, request)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        return [{"check": "raises", "input": text, "error": repr(e)[:300]}]
    judge(text, base, "")

    # Text in front of the output must reach content and leave calls alone.
    prefixed = PREFIX + text
    try:
        info = parser.extract_tool_calls(prefixed, request)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        findings.append({"check": "raises_prefix", "input": prefixed, "error": repr(e)[:200]})
    else:
        if _calls_key(info.tool_calls) != _calls_key(base.tool_calls):
            findings.append({"check": "prefix_calls", "input": prefixed, "result": _summary(info)})
        if PREFIX.strip() not in (info.content or ""):
            findings.append({"check": "prefix_lost", "input": prefixed, "result": _summary(info)})

    deltas = token_deltas(parser, text, findings)
    if deltas is None:
        return findings

    cuts = list(range(1, len(deltas)))
    if len(cuts) > MAX_CUTS:
        cuts = sorted(random.Random(0).sample(cuts, MAX_CUTS))
    for k in cuts:
        prefix = "".join(deltas[:k])
        try:
            info = parser.extract_tool_calls(prefix, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": "raises_trunc", "input": prefix, "error": repr(e)[:200]})
            continue
        judge(prefix, info, "_trunc")
        if base.tool_calls and not info.tool_calls:
            findings.append(_cut_policy(prefix, info.content, markers))

    # Non-streaming parsing must not depend on earlier calls: after all the
    # variants, the same instance must parse the original output as before.
    try:
        again = parser.extract_tool_calls(text, request)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        findings.append({"check": "S_state", "input": text, "error": repr(e)[:200]})
    else:
        if (_calls_key(again.tool_calls), _norm(again.content)) != (
            _calls_key(base.tool_calls),
            _norm(base.content),
        ):
            findings.append({"check": "S_state", "input": text, "result": _summary(again)})

    if "_pc_init" not in vars(parser):
        findings.append({"check": "skip_fresh", "input": text})
        return findings
    args, kwargs = parser._pc_init
    fresh = type(parser)(*args, **kwargs)
    try:
        rec = run_tool_extraction_streaming(fresh, deltas, request, assert_one_tool_per_delta=False)
    except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
        findings.append({"check": "raises_stream", "input": text, "error": repr(e)[:300]})
        return findings
    stream_calls, base_calls = _calls_key(rec.tool_calls), _calls_key(base.tool_calls)
    if stream_calls and not base_calls:
        # Incremental streaming emits a call before it can see that the JSON
        # never closes, and it cannot take the call back. Report it apart
        # from real mismatches.
        findings.append({"check": "I3_lenient", "input": text, "stream": stream_calls})
        return findings
    if stream_calls != base_calls:
        findings.append(
            {"check": "I3_calls", "input": text, "stream": stream_calls, "nonstream": base_calls}
        )
    if _strip(rec.other_content, markers) != _strip(base.content, markers):
        findings.append(
            {
                "check": "I3_content",
                "input": text,
                "stream": _norm(rec.other_content)[:300],
                "nonstream": _norm(base.content)[:300],
            }
        )
    return findings
