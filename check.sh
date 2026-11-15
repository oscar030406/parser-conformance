#!/usr/bin/env bash
# Run every check of parser-conformance: lint, format, and the tests on the
# fake vLLM checkout in tests/fake_vllm. Needs uv on PATH.
#
#   bash check.sh
set -euo pipefail
cd "$(dirname "$0")"
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q -p no:cacheprovider
