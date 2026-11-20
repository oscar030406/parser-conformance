"""Conformance properties checked for one recorded tool parser input.

I1  tools_called is true exactly when tool_calls is non-empty.
I2  When no call is parsed, content keeps the model output.
    "I2_text" means non-marker text was lost. "I2_markers" means only
    special tokens or whitespace differ.
I3  Streaming the output token by token gives the same calls and content
    as non-streaming extraction ("I3_calls", "I3_content"). "I3_markers"
    means the content differs only in special tokens or whitespace.
    "I3_lenient" marks calls that only streaming returns, at the end, when
    the call never closes: the output does not end with a marker, or the
    arguments of every extra call are not complete JSON.
    "_chunked" repeats I3 with 2 to 8 tokens per delta, as speculative
    decoding produces. It runs only when the token-by-token check passes.

P   Putting plain text in front of the output keeps the same calls
    ("prefix_calls") and the text reaches content ("prefix_lost").
    "prefix_anchored" means the parser finds no call once text comes first.
S   One parser instance parses the output and all its variants in turn,
    and gives the same result as a fresh parser for each ("S_state").

I1 and I2 also run on token-boundary prefixes of the input ("_trunc"),
which is what a max_tokens cut produces. A cut inside a call that leaves
no complete call is recorded as a "policy_*" observation, not an I2 failure.

Every other parse uses a fresh parser built with the arguments of the test,
because vLLM builds a parser for each request.

The parser under test can raise anything. Such exceptions are findings,
so the handlers around parser calls catch Exception and record it.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import random
import re

MAX_CUTS = int(os.environ.get("PC_MAX_CUTS", "48"))
PREFIX = "Sure, checking now. "
_TAG = re.compile(r"<[^<>\s\\]{1,40}>")
_TAG_CHARS = re.compile(r"[<>\[\]|｜]")


def _closing(marker: str) -> bool:
    """A marker that ends a call or a section, e.g. </tool_call> or <｜tool_calls_end｜>."""
    return marker.startswith("</") or "end" in marker.lower()


def _closed(call, text: str, markers: list[str]) -> bool:
    """Whether a closing marker follows the call's name in the output.

    For formats without closing markers, such as pythonic calls, a call is
    closed when its arguments are complete JSON.
    """
    closing = [m for m in markers if _closing(m)]
    if not closing:
        return _complete(call.function.arguments)
    start = text.find(call.function.name or "")
    return start >= 0 and any(m in text[start:] for m in closing)


def parser_factory(parser):
    """A function that builds a parser as the test built this one, or None."""
    init = vars(parser).get("_pc_init")
    if init is None:
        return None
    args, kwargs = init
    return lambda: type(parser)(*args, **kwargs)


def _summary(info) -> dict:
    calls = [(c.function.name, c.function.arguments) for c in info.tool_calls or []]
    return {"tools_called": info.tools_called, "calls": calls, "content": info.content}


def _norm(text: str | None) -> str:
    return (text or "").strip()


@functools.cache
def _source_tags(cls) -> frozenset[str]:
    tags: set[str] = set()
    for c in cls.__mro__[:-1]:  # all but object
        try:
            source = inspect.getsource(c)
        except (OSError, TypeError):  # noqa: S112  classes made at run time by make_adapters
            continue
        tags.update(_TAG.findall(source))
    return frozenset(tags)


def _markers(parser) -> list[str]:
    tokenizer = parser.model_tokenizer
    vocab: set[str] = set(getattr(tokenizer, "all_special_tokens", None) or [])
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        vocab.update(get_added_vocab().keys())
    # ERNIE adds the digits and LFM2.5 adds plain words to the vocabulary; only
    # entries with tag characters are markers.
    found = {token for token in vocab if _TAG_CHARS.search(token)}
    # Instance attributes, then class attributes such as Step3's TOOL_CALLS_END.
    namespaces = [vars(parser), *(vars(c) for c in type(parser).__mro__ if c is not object)]
    for name, value in (item for space in namespaces for item in space.items()):
        if isinstance(value, re.Pattern):
            value = value.pattern
        if not isinstance(value, str) or name == "__doc__":
            continue
        if name.lower().endswith("token") and value.strip():
            found.add(value)
        # Tags kept in regex sources, e.g. Hunyuan's r"\n</think>\n<answer>\n".
        found.update(_TAG.findall(value))
    # Markers written inline, e.g. Llama 4's `startswith("<|python_start|>")`.
    found.update(_source_tags(type(parser)))
    # Parsers built on vLLM's parser engine keep their markers in its config.
    config = getattr(vars(parser).get("_parser_engine"), "parser_engine_config", None)
    terminals = getattr(config, "terminals", None) or {}
    found.update(v for v in terminals.values() if isinstance(v, str) and v.strip())
    return sorted(found, key=len, reverse=True)


def _strip(text: str | None, markers: list[str]) -> str:
    text = text or ""
    for marker in markers:
        text = text.replace(marker, "")
    return "".join(text.split())


def _canonical(args: str | None) -> str:
    """Compare JSON arguments by value. Invalid JSON, such as "", is compared as written."""
    try:
        return json.dumps(json.loads(args or ""), sort_keys=True)
    except json.JSONDecodeError:
        return args or ""


def _complete(args: str | None) -> bool:
    try:
        json.loads(args or "")
    except json.JSONDecodeError:
        return False
    return True


def _calls_key(calls) -> list[tuple[str, str]]:
    return [(call.function.name, _canonical(call.function.arguments)) for call in calls or []]


def _outcome(info) -> tuple:
    return (_calls_key(info.tool_calls), _norm(info.content))


def _name_starts(text: str, calls, markers: list[str]) -> list[int] | None:
    """Where each call's name starts in the output, or None when a name is not found.

    A name starts with the identifier around it: Kimi K2 writes
    `functions.get_weather:0` for the call `get_weather`.
    """
    starts = [i for i in (text.find(m) for m in markers) if i >= 0]
    pos, found = min(starts) if starts else 0, []
    for call in calls:
        at = text.find(call.function.name, pos)
        if at < 0:
            return None
        start = at
        while start > pos and (text[start - 1].isalnum() or text[start - 1] in "_.:-"):
            start -= 1
        found.append(start)
        pos = at + len(call.function.name)
    return found


def _cut_position(prefix: str, returned, base_calls, name_starts) -> str:
    """ "call" when the cut reaches the name of the first call not returned complete.

    "opener" when it stops before that name, inside the markup that opens the
    call. vLLM's DSML parsers treat the two positions differently (vllm#66474).
    """
    if name_starts is None:
        return "unknown"
    done = 0
    while done < min(len(returned), len(base_calls)) and returned[done] == base_calls[done]:
        done += 1
    if done >= len(name_starts):
        return "unknown"
    return "call" if len(prefix) > name_starts[done] else "opener"


def _served(parser, source: str, request) -> dict | None:
    """What the server returns for source: vLLM's Parser.parse with only this tool parser.

    Parser.parse decides the response from tools_called and, for legacy
    parsers, returns the raw output as content when no call is found. So it
    can differ from extract_tool_calls. None when vLLM has no Parser.parse.
    """
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.parser.abstract_parser import DelegatingParser
    except ImportError:
        return None
    served = type("Served", (DelegatingParser,), {"tool_parser_cls": None})(
        parser.model_tokenizer, None
    )
    served._tool_parser = parser
    served._engine_based = parser.engine_based_streaming
    if not isinstance(request, ChatCompletionRequest):
        request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    _, content, calls = served.parse(source, request, enable_auto_tools=True)
    return {
        "served_content": content,
        "served_calls": [(c.name, _canonical(c.arguments)) for c in calls or []],
    }


def _cut_policy(prefix: str, content: str | None, markers: list[str], position: str) -> dict:
    """Record what non-streaming returns when a cut leaves no complete call.

    This is an observation, not a pass or fail: vLLM's parsers disagree here.
    "policy_keeps_text" returns the cut output as content, with or without its
    markers. "policy_drops_call" keeps only the text before the first marker,
    and "policy_drops_all" returns no content. The suffix is the cut position
    from `_cut_position`.
    """
    starts = [i for i in (prefix.find(m) for m in markers) if i >= 0]
    before = prefix[: min(starts)] if starts else prefix
    if _norm(content) == _norm(prefix):
        kind = "keeps_text"
    elif _norm(content) == _norm(before) and _norm(before):
        kind = "drops_call"
    elif not _norm(content):
        kind = "drops_all"
    elif _strip(content, markers) == _strip(prefix, markers):
        kind = "keeps_text"  # e.g. Llama 4 returns "[" for "<|python_start|>["
    else:
        kind = "other"
    return {"check": f"policy_{kind}@{position}", "input": prefix, "content": content}


def token_pieces(tokenizer, text: str) -> list[str]:
    """Split text into the deltas that streaming would see, one per token.

    Like vLLM's detokenizer, a token that ends inside a character of several
    bytes adds no text until the character is complete, so no delta carries
    a replacement character.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces, sent = [], ""
    for i in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:i])
        if decoded.endswith("�") and i < len(ids):
            continue
        pieces.append(decoded[len(sent) :])
        sent = decoded
    return pieces


def token_deltas(parser, text: str, findings: list[dict]) -> list[str] | None:
    """The token pieces of text, or None, recording why, when they cannot be made.

    A test that builds the parser with a stand-in tokenizer gives it no real
    vocabulary, so the token ids that streaming parsers match on are missing.
    """
    vocab = getattr(parser, "vocab", None)
    if not isinstance(vocab, dict) or len(vocab) < 1000:
        findings.append({"check": "skip_mock_tokenizer", "input": text})
        return None
    try:
        deltas = token_pieces(parser.model_tokenizer, text)
    except Exception as e:  # noqa: BLE001  test-provided tokenizer, e.g. a Mock
        findings.append({"check": "skip_tokenize", "input": text, "error": repr(e)[:200]})
        return None
    if not all(isinstance(d, str) for d in deltas):
        findings.append({"check": "skip_tokenize", "input": text, "error": "non-str pieces"})
        return None
    return deltas


def chunked(deltas: list[str]) -> list[str]:
    """Merge the deltas into groups of 2 to 8, as when one step yields several tokens."""
    rng, out, i = random.Random(0), [], 0
    while i < len(deltas):
        n = rng.randint(2, 8)
        out.append("".join(deltas[i : i + n]))
        i += n
    return out


def compare_stream(rec, base, markers: list[str], text: str, tag: str) -> list[dict]:
    stream, nonstream = _calls_key(rec.tool_calls), _calls_key(base.tool_calls)
    findings = []
    if stream != nonstream:
        extra = rec.tool_calls[len(nonstream) :]
        unfinished = not any(_closed(call, text, markers) for call in extra)
        if stream[: len(nonstream)] == nonstream and unfinished:
            # Incremental streaming emits a call before it can see that the call
            # never closes, and it cannot take the call back.
            return [{"check": f"I3_lenient{tag}", "input": text, "stream": stream}]
        findings.append(
            {"check": f"I3_calls{tag}", "input": text, "stream": stream, "nonstream": nonstream}
        )
    if _norm(rec.other_content) != _norm(base.content):
        same = _strip(rec.other_content, markers) == _strip(base.content, markers)
        findings.append(
            {
                "check": f"I3_{'markers' if same else 'content'}{tag}",
                "input": text,
                "stream": _norm(rec.other_content)[:300],
                "nonstream": _norm(base.content)[:300],
            }
        )
    return findings


def check_seed(parser, text: str, request) -> list[dict]:
    from tests.tool_parsers.utils import run_tool_extraction_streaming

    make = parser_factory(parser)
    if make is None:
        return [{"check": "skip_fresh", "input": text}]
    findings: list[dict] = []
    markers = _markers(parser)
    expected: dict[str, tuple] = {}  # outcome of each input on a fresh parser

    def extract(source: str, check: str):
        fresh = make()
        try:
            info = fresh.extract_tool_calls(source, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": check, "input": source, "error": repr(e)[:300]})
            expected[source] = ("raises", type(e).__name__)
            return None
        expected[source] = _outcome(info)
        return info

    def judge(source: str, info, tag: str, content: bool = True) -> None:
        if info.tools_called != bool(info.tool_calls):
            findings.append({"check": f"I1_flag{tag}", "input": source, "result": _summary(info)})
        if content and not info.tool_calls and _norm(info.content) != _norm(source):
            lost = _strip(source, markers) != _strip(info.content, markers)
            findings.append(
                {
                    "check": f"I2_{'text' if lost else 'markers'}{tag}",
                    "input": source,
                    "result": _summary(info),
                }
            )

    base = extract(text, "raises")
    if base is None:
        return findings
    judge(text, base, "")
    # Markers of the calls: in the output but not in its content. ERNIE's
    # </think> comes before the call and stays in the content.
    call_markers = [m for m in markers if m in text and m not in (base.content or "")]

    # Text in front of the output must reach content and leave calls alone.
    prefixed = PREFIX + text
    info = extract(prefixed, "raises_prefix")
    if info is not None:
        if _calls_key(info.tool_calls) != _calls_key(base.tool_calls):
            anchored = base.tool_calls and not info.tool_calls and not _norm(base.content)
            check = "prefix_anchored" if anchored else "prefix_calls"
            findings.append({"check": check, "input": prefixed, "result": _summary(info)})
        if PREFIX.strip() not in (info.content or ""):
            findings.append({"check": "prefix_lost", "input": prefixed, "result": _summary(info)})

    deltas = token_deltas(parser, text, findings)
    cuts = list(range(1, len(deltas))) if deltas else []
    if len(cuts) > MAX_CUTS:
        cuts = sorted(random.Random(0).sample(cuts, MAX_CUTS))

    def served(source: str) -> dict:
        """The server's response, with the same kinds as the policy rows, or "call"."""
        try:
            got = _served(make(), source, request)
        except Exception as e:  # noqa: BLE001  parser under test; recorded with the row
            return {"served_error": repr(e)[:300]}
        if got is None:
            return {}
        policy = _cut_policy(source, got["served_content"], call_markers, "")["check"]
        kind = "call" if got["served_calls"] else policy[len("policy_") : -1]
        return {**got, "served_kind": kind}

    base_calls = _calls_key(base.tool_calls)
    name_starts = _name_starts(text, base.tool_calls or [], call_markers)
    for k in cuts:
        prefix = "".join(deltas[:k])
        info = extract(prefix, "raises_trunc")
        if info is None:
            continue
        returned = _calls_key(info.tool_calls)
        in_call = base.tool_calls and any(m in prefix for m in call_markers)
        if in_call and not info.tool_calls:
            # A cut inside a call: vLLM's parsers disagree on the content (vllm#67513).
            position = _cut_position(prefix, returned, base_calls, name_starts)
            judge(prefix, info, "_trunc", content=False)
            row = _cut_policy(prefix, info.content, call_markers, position)
            findings.append({**row, **served(prefix)})
        elif in_call and returned[: len(base_calls)] != base_calls[: len(returned)]:
            # A call that the full output does not have, e.g. a cut name with empty arguments.
            position = _cut_position(prefix, returned, base_calls, name_starts)
            judge(prefix, info, "_trunc")
            names = {name for name, _ in base_calls}
            findings.append(
                {
                    "check": f"policy_ghost_call@{position}",
                    "input": prefix,
                    "calls": returned,
                    "cut_name": any(name not in names for name, _ in returned),
                    **served(prefix),
                }
            )
        else:
            judge(prefix, info, "_trunc")

    # Non-streaming must not depend on earlier calls: one instance parses every
    # input in turn, then the original again, and must match a fresh parser.
    stateful = make()
    for source in [*expected, text]:
        try:
            got = _outcome(stateful.extract_tool_calls(source, request))
        except Exception as e:  # noqa: BLE001  parser under test; compared below
            got = ("raises", type(e).__name__)
        if got != expected[source]:
            findings.append({"check": "S_state", "input": source, "result": repr(got)[:300]})
            break

    if not deltas:
        return findings
    joined = "".join(deltas)
    target = base if joined == text else extract(joined, "raises")
    if target is None:
        return findings
    for tag, pieces in (("", deltas), ("_chunked", chunked(deltas))):
        try:
            rec = run_tool_extraction_streaming(
                make(), pieces, request, assert_one_tool_per_delta=False
            )
        except Exception as e:  # noqa: BLE001  parser under test; recorded as a finding
            findings.append({"check": f"raises_stream{tag}", "input": text, "error": repr(e)[:300]})
            return findings
        found = compare_stream(rec, target, markers, text, tag)
        findings += found
        if found:
            break
    return findings
