"""Run the plugin on the fake vLLM checkout, as a user runs it on vLLM."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FAKE_VLLM = Path(__file__).parent / "fake_vllm"


@pytest.fixture
def checkout(tmp_path):
    return shutil.copytree(FAKE_VLLM, tmp_path / "checkout")


def run(checkout: Path, report: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "parser_conformance.plugin"]
    environment = {**os.environ, "PC_REPORT": str(report), "PYTHONDONTWRITEBYTECODE": "1", **env}
    return subprocess.run(
        [*command, "-p", "no:cacheprovider", *args],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def rows(report: Path) -> list[dict]:
    return [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]


def test_xdist_writes_the_same_report(checkout, tmp_path):
    single = run(checkout, tmp_path / "single.jsonl")
    assert single.returncode == 0, single.stdout
    parallel = run(checkout, tmp_path / "parallel.jsonl", "-n", "2")
    assert parallel.returncode == 0, parallel.stdout

    reports = sorted(path.name for path in tmp_path.iterdir() if path.is_file())
    assert reports == ["parallel.jsonl", "single.jsonl"]
    key = lambda row: json.dumps(row, sort_keys=True)  # noqa: E731
    assert sorted(map(key, rows(tmp_path / "single.jsonl"))) == sorted(
        map(key, rows(tmp_path / "parallel.jsonl"))
    )
    seeds = [row["input"] for row in rows(tmp_path / "single.jsonl") if row["check"] == "seed"]
    # Seven tool outputs and two reasoning outputs. "Same" comes from a parser
    # with a stand-in tokenizer and from one with a real one. "Twice" comes
    # from two requests that differ only in their random id.
    assert len(seeds) == 9
    assert sum(seed.startswith("Same ") for seed in seeds) == 2
    assert sum(seed.startswith("Twice ") for seed in seeds) == 1


def test_seed_limit_holds_for_the_whole_run(checkout, tmp_path):
    result = run(checkout, tmp_path / "report.jsonl", "-n", "2", PC_MAX_SEEDS="1")
    assert result.returncode == 0, result.stdout
    seeds = [row["parser"] for row in rows(tmp_path / "report.jsonl") if row["check"] == "seed"]
    assert sorted(seeds) == ["DemoReasoningParser", "DemoToolParser", "NoContentToolParser"]


def test_error_in_the_checks_fails_the_run(checkout, tmp_path):
    report = tmp_path / "report.jsonl"
    result = run(checkout, report, "-n", "2", PC_TEST_BREAK="1")
    assert result.returncode != 0
    assert "broken tokenizer" in result.stdout
    broken = [row["parser"] for row in rows(report) if row["check"] == "harness_error"]
    assert broken == ["DemoToolParser"]
