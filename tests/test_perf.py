import time

from vllm.fake_tokenizer import ByteTokenizer
from vllm.tool_parsers import demo_tool_parser as demo

from parser_conformance import perf

CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


class SlowVocabTokenizer(ByteTokenizer):
    def get_vocab(self):
        time.sleep(0.5)
        return super().get_vocab()


def test_time_leaves_out_reading_the_vocabulary():
    row = perf.perf_seed(demo.DemoToolParser(SlowVocabTokenizer()), CALL, None)
    assert row["ms_256"] < 250


def test_padding_goes_into_an_argument_value():
    assert CALL[: perf.pad_position(CALL)].endswith('"city": "')


def test_padding_goes_in_front_without_a_string_value():
    assert perf.pad_position("[get_weather(city=1)]") == 0
