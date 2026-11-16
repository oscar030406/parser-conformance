"""vLLM-style tests of the demo tool parsers. The plugin records the outputs they parse."""

import os
from unittest.mock import MagicMock

import pytest
from vllm.fake_tokenizer import ByteTokenizer
from vllm.protocol import ChatCompletionRequest
from vllm.tool_parsers.demo_tool_parser import DemoToolParser, NoContentToolParser

TOKENIZER = ByteTokenizer()
CALL = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'


@pytest.mark.parametrize("i", range(3))
def test_demo(i):
    assert DemoToolParser(TOKENIZER).extract_tool_calls(f"Hi {i} {CALL}", None).tools_called


def test_no_content():
    assert NoContentToolParser(TOKENIZER).extract_tool_calls(f"Hi {CALL}", None).tools_called


# The same output from two tests: one parser has a stand-in tokenizer, the other a real one.
def test_mock_tokenizer():
    assert DemoToolParser(MagicMock()).extract_tool_calls(f"Same {CALL}", None).tools_called


def test_real_tokenizer():
    assert DemoToolParser(TOKENIZER).extract_tool_calls(f"Same {CALL}", None).tools_called


class BrokenTokenizer(ByteTokenizer):
    def get_added_vocab(self):
        raise RuntimeError("broken tokenizer")


@pytest.mark.skipif(not os.environ.get("PC_TEST_BREAK"), reason="breaks the checks on purpose")
def test_broken_tokenizer():
    assert DemoToolParser(BrokenTokenizer()).extract_tool_calls(f"Bad {CALL}", None).tools_called


# Two tests parse the same output, each with a new request and its own random id.
@pytest.mark.parametrize("i", range(2))
def test_same_request_twice(i):
    request = ChatCompletionRequest(model="m")
    assert DemoToolParser(TOKENIZER).extract_tool_calls(f"Twice {CALL}", request).tools_called
