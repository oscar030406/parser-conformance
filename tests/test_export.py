from unittest.mock import MagicMock

from vllm.fake_tokenizer import ByteTokenizer
from vllm.tool_parsers import demo_tool_parser as demo

from parser_conformance.export import export_rows

TOKENIZER = ByteTokenizer()
CALL = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'


def test_every_output_is_exported():
    seeds = [
        ("tool", demo.DemoToolParser(MagicMock()), f"Mock {CALL}", None, "test_mock"),
        ("tool", demo.DemoToolParser(TOKENIZER), f"Real {CALL}", None, "test_real"),
        ("tool", demo.RaisingToolParser(TOKENIZER), CALL, None, "test_raises"),
    ]
    rows = export_rows(seeds)
    assert [row["tokenizer"] for row in rows] == [None, "fake/bytes", "fake/bytes"]
    assert rows[1]["parser_names"] == ["demo"]
    assert rows[1]["python"]["calls"] == [{"name": "f", "arguments": '{"a": 1}'}]
    assert "demo parser error" in rows[2]["python"]["error"]
