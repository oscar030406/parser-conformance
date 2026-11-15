"""The tool-call checks on the demo parsers in fake_vllm/vllm/tool_parsers/demo_tool_parser.py."""

import pytest
from vllm.fake_tokenizer import ByteTokenizer
from vllm.tool_parsers import demo_tool_parser as demo

from parser_conformance.checks import check_seed

TOKENIZER = ByteTokenizer()
CALL = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'


def checks(parser, text: str) -> set[str]:
    return {row["check"] for row in check_seed(parser, text, None)}


def failures(parser, text: str) -> set[str]:
    """The checks that failed, without the observations on cut calls."""
    return {check for check in checks(parser, text) if not check.startswith("policy_")}


@pytest.mark.parametrize(
    "text",
    [
        CALL,
        f"Hi {CALL}",
        f"Hi {CALL}\n{CALL} bye",
        # A character of four bytes, which the tokenizer splits over four tokens.
        '<tool_call>{"name": "f", "arguments": {"a": "\U0001f389"}}</tool_call>',
    ],
    ids=["call", "text-call", "two-calls", "multibyte"],
)
def test_correct_parser_passes(text):
    assert failures(demo.DemoToolParser(TOKENIZER), text) == set()


def test_state_does_not_leak_between_outputs():
    # One instance, as from a test fixture, records two outputs. The cuts of
    # the first leave it inside a call.
    shared = demo.StickyToolParser(TOKENIZER)
    check_seed(shared, f"Hi {CALL}", None)
    second = f"Bye {CALL}"
    assert failures(shared, second) == failures(demo.StickyToolParser(TOKENIZER), second)
    assert failures(shared, second) == {"S_state"}


def test_call_that_only_streaming_finds_is_a_mismatch():
    found = failures(demo.MissesCallToolParser(TOKENIZER), f"{CALL}\n")
    assert "I3_calls" in found
    assert "I3_lenient" not in found


def test_unfinished_call_that_only_streaming_returns_is_lenient():
    unfinished = '<tool_call>{"name": "f", "arguments": {"a": 1'
    assert failures(demo.PartialArgsToolParser(TOKENIZER), unfinished) == {"I3_lenient"}


def test_marker_in_streamed_content():
    assert failures(demo.LeakyToolParser(TOKENIZER), f"Hi {CALL}") == {"I3_markers"}


def test_empty_streamed_arguments():
    no_args = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert failures(demo.EmptyArgsToolParser(TOKENIZER), no_args) == {"I3_calls"}


def test_cut_inside_a_call_is_an_observation():
    found = checks(demo.DropsCutCallToolParser(TOKENIZER), f"Hi {CALL}")
    assert "policy_drops_call" in found
    assert "I2_text_trunc" not in found


def test_cut_before_the_call_is_still_checked():
    assert "I2_text_trunc" in failures(demo.NoContentToolParser(TOKENIZER), f"Hi there {CALL}")


def test_parser_that_needs_the_call_first():
    assert failures(demo.AnchoredToolParser(TOKENIZER), CALL) == {"prefix_anchored"}


def test_several_tokens_per_delta():
    found = failures(demo.PerTokenToolParser(TOKENIZER), f"Hi {CALL}")
    assert found == {"I3_calls_chunked", "I3_content_chunked"}
