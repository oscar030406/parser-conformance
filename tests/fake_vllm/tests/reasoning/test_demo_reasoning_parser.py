"""vLLM-style tests of the demo reasoning parser. The plugin records the outputs it parses."""

import pytest
from vllm.fake_tokenizer import ByteTokenizer
from vllm.reasoning.demo_reasoning_parser import DemoReasoningParser

TOKENIZER = ByteTokenizer()


@pytest.mark.parametrize("i", range(2))
def test_demo(i):
    parser = DemoReasoningParser(TOKENIZER)
    assert parser.extract_reasoning(f"<think>Plan {i}</think>Answer", None)[1] == "Answer"
