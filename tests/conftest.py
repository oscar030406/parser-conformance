"""Put the fake vLLM checkout first on sys.path, then wrap its parser classes as
the plugin does in a vLLM checkout, so that each parser records how it was built."""

import sys
from pathlib import Path

FAKE_VLLM = Path(__file__).parent / "fake_vllm"
sys.path.insert(0, str(FAKE_VLLM))

from parser_conformance import plugin  # noqa: E402  needs the fake vLLM on sys.path

plugin._STATE["recording"] = False
plugin.pytest_configure(None)

collect_ignore = ["fake_vllm"]
