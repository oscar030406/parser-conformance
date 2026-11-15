"""The reasoning checks on the demo parsers in fake_vllm/vllm/reasoning/demo_reasoning_parser.py."""

import pytest
from vllm.fake_tokenizer import ByteTokenizer
from vllm.reasoning import demo_reasoning_parser as demo

from parser_conformance.reasoning_checks import check_reasoning_seed

TOKENIZER = ByteTokenizer()


def checks(parser, text: str) -> set[str]:
    return {row["check"] for row in check_reasoning_seed(parser, text, None)}


@pytest.mark.parametrize(
    "text",
    ["<think>Plan</think>Answer", "Plan</think>Answer", "<think>Plan", "<think>\U0001f389</think>Ok"],
    ids=["both-markers", "end-marker", "no-end", "multibyte"],
)
def test_correct_parser_passes(text):
    assert checks(demo.DemoReasoningParser(TOKENIZER), text) == set()


def test_marker_in_streamed_content():
    found = checks(demo.LeakyReasoningParser(TOKENIZER), "<think>Plan</think>Answer")
    assert found == {"R2_stream_markers", "R2_trunc_markers"}


def test_each_variant_gets_a_fresh_parser():
    assert checks(demo.OnceReasoningParser(TOKENIZER), "<think>Plan</think>Answer") == set()
