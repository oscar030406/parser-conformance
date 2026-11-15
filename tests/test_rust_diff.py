import pytest

from parser_conformance.rust_diff import classify

PYTHON = {"calls": [{"name": "f", "arguments": "{}"}], "content": "Hi"}


@pytest.mark.parametrize(
    ("rust", "kind"),
    [
        ({"error": "create: tool parser `x` is not registered (choose from: a, b)"}, "not in Rust"),
        ({"error": "create: failed to initialize tool parser `x`: bad schema"}, "Rust error"),
        ({"error": "tokenizer: no such file"}, "Rust error"),
        ({"calls": [{"name": "f", "arguments": "{ }"}], "content": "Hi "}, "same"),
        ({"calls": [{"name": "f", "arguments": ""}], "content": "Hi"}, "calls differ"),
        ({"calls": [{"name": "f", "arguments": "{}"}], "content": "Hi there"}, "content differs"),
    ],
)
def test_classify(rust, kind):
    assert classify(PYTHON, rust) == kind
