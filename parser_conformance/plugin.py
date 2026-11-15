"""Pytest plugin: record the inputs that vLLM's own tests feed to tool and
reasoning parsers, then re-check every recorded input against conformance
properties.

Run from a vLLM checkout:

    PC_REPORT=/tmp/pc.jsonl python -m pytest tests/tool_parsers tests/reasoning \\
        -p parser_conformance.plugin -q

With pytest-xdist, each worker checks the seeds it recorded and hands the
rows to the controller, which writes one report.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import time
import traceback
from collections import Counter

import pytest

MAX_SEEDS_PER_PARSER = int(os.environ.get("PC_MAX_SEEDS", "40"))
_KEY = "parser_conformance"

# (method name, family label, package, module suffix, base class path)
_FAMILIES = [
    (
        "extract_tool_calls",
        "tool",
        "vllm.tool_parsers",
        "_tool_parser",
        ("vllm.tool_parsers.abstract_tool_parser", "ToolParser"),
    ),
    (
        "extract_reasoning",
        "reasoning",
        "vllm.reasoning",
        "_reasoning_parser",
        ("vllm.reasoning.abs_reasoning_parsers", "ReasoningParser"),
    ),
]

_SEEDS: list[tuple] = []
_SEEN: set[tuple] = set()
_PER_PARSER: Counter[str] = Counter()
_IMPORT_ERRORS: list[str] = []
_STATE = {"depth": 0, "recording": True, "nodeid": None}
_RESULTS: list[dict] = []  # one entry per process that checked seeds


def _tokenizer_name(parser) -> str:
    name = getattr(parser.model_tokenizer, "name_or_path", None)
    return name if isinstance(name, str) else type(parser.model_tokenizer).__name__


def _request_key(request) -> str:
    dump = getattr(type(request), "model_dump_json", None)
    return dump(request) if dump else type(request).__name__


def _wrap_method(cls, method: str, family: str) -> None:
    orig = cls.__dict__.get(method)
    if orig is None or getattr(orig, "_pc_wrapped", False):
        return

    def wrapper(self, model_output, request, *args, **kwargs):
        name = type(self).__name__
        if (
            _STATE["recording"]
            and _STATE["depth"] == 0
            and isinstance(model_output, str)
            and _PER_PARSER[name] < MAX_SEEDS_PER_PARSER
        ):
            # The same text can mean something else with another tokenizer or request.
            key = (name, _tokenizer_name(self), _request_key(request), model_output)
            if key not in _SEEN:
                _SEEN.add(key)
                _PER_PARSER[name] += 1
                _SEEDS.append((family, self, model_output, request, _STATE["nodeid"], key))
        _STATE["depth"] += 1
        try:
            return orig(self, model_output, request, *args, **kwargs)
        finally:
            _STATE["depth"] -= 1

    wrapper._pc_wrapped = True
    setattr(cls, method, wrapper)


def _wrap_init(cls) -> None:
    """Keep the outermost constructor arguments, so the checks can build a
    fresh parser exactly as the test built this one."""
    orig = cls.__dict__.get("__init__")
    if orig is None or getattr(orig, "_pc_wrapped", False):
        return

    def __init__(self, *args, **kwargs):
        self.__dict__.setdefault("_pc_init", (args, kwargs))
        orig(self, *args, **kwargs)

    __init__._pc_wrapped = True
    cls.__init__ = __init__


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def pytest_configure(config):
    for method, family, package, suffix, (base_mod, base_name) in _FAMILIES:
        pkg = importlib.import_module(package)
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.endswith(suffix):
                try:
                    importlib.import_module(f"{package}.{mod.name}")
                except ImportError as e:  # parsers with optional dependencies
                    _IMPORT_ERRORS.append(f"{mod.name}: {e!r}")
        base = getattr(importlib.import_module(base_mod), base_name)
        for cls in [base, *_all_subclasses(base)]:
            _wrap_method(cls, method, family)
            _wrap_init(cls)


def pytest_runtest_setup(item):
    _STATE["nodeid"] = item.nodeid


def _line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, default=str)


def _check_seeds() -> dict:
    """Check the seeds this process recorded. Every value is JSON lines, so
    that a pytest-xdist worker can send the result to the controller."""
    from parser_conformance.checks import check_seed
    from parser_conformance.reasoning_checks import check_reasoning_seed

    checkers = {"tool": check_seed, "reasoning": check_reasoning_seed}
    _STATE["recording"] = False
    report: dict[str, list[str]] = {}  # seed key -> rows
    errors = []
    for family, parser, text, request, nodeid, key in _SEEDS:
        where = {"family": family, "parser": type(parser).__name__, "nodeid": nodeid}
        try:
            findings = checkers[family](parser, text, request)
        except Exception as e:  # noqa: BLE001  fails the run in pytest_sessionfinish
            errors.append(f"{where['parser']} {nodeid}\n{traceback.format_exc()}")
            findings = [{"check": "harness_error", "input": text, "error": repr(e)[:300]}]
        rows = [{"check": "seed", "input": text}, *findings]
        report[_line(key)] = [_line({**row, **where}) for row in rows]
    seeds = [seed[:5] for seed in _SEEDS]
    result = {"report": report, "errors": errors, "import_errors": _IMPORT_ERRORS}
    if os.environ.get("PC_SEEDS_OUT"):
        from parser_conformance.export import export_rows

        result["seeds"] = [_line(row) for row in export_rows(seeds)]
    if os.environ.get("PC_PERF"):
        from parser_conformance.perf import run_perf

        result["perf"] = [_line(row) for row in run_perf(seeds)]
    return result


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    result = getattr(node, "workeroutput", {}).get(_KEY)
    if result is None:
        worker = node.gateway.id
        result = {"report": {}, "errors": [f"worker {worker} sent no results: {error}"]}
    _RESULTS.append(result)


def _nodeid(lines: list[str]) -> str:
    return json.loads(lines[0])["nodeid"] or ""


def _unique_seeds(lines: list[str]) -> list[str]:
    """One exported row per input, whichever test recorded it."""
    rows: dict[str, str] = {}
    for line in lines:
        row = json.loads(line)
        row.pop("nodeid")
        rows.setdefault(_line(row), line)
    return list(rows.values())


def _write(path: str, lines: list[str]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
    return len(lines)


def _longest_per_parser(lines: list[str]) -> list[str]:
    best: dict[str, dict] = {}
    for row in map(json.loads, lines):
        if row["parser"] not in best or row["chars"] > best[row["parser"]]["chars"]:
            best[row["parser"]] = row
    rows = sorted(best.values(), key=lambda r: -r.get("ratio", 0))
    return [_line(row) for row in rows]


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if hasattr(config, "workerinput"):  # a pytest-xdist worker
        config.workeroutput[_KEY] = _check_seeds()
        return
    took = ""
    if config.pluginmanager.getplugin("dsession") is None:  # no workers
        start = time.time()
        _RESULTS.append(_check_seeds())
        took = f" in {time.time() - start:.0f}s"

    def merged(key: str) -> list[str]:
        return [line for result in _RESULTS for line in result.get(key, [])]

    # Two workers can record the same seed. Keep the rows from the earliest
    # test, as a run without workers would.
    groups: dict[str, list[str]] = {}
    for result in _RESULTS:
        for key, lines in result["report"].items():
            if key not in groups or _nodeid(lines) < _nodeid(groups[key]):
                groups[key] = lines
    out = os.environ.get("PC_REPORT", "parser_conformance.jsonl")
    _write(out, [line for lines in groups.values() for line in lines])
    seeds = len(groups)
    print(f"\n[parser-conformance] {seeds} seeds checked{took} -> {out}")
    for line in sorted(set(merged("import_errors"))):
        print(f"[parser-conformance] not imported: {line}")
    seeds_out = os.environ.get("PC_SEEDS_OUT")
    if seeds_out:
        count = _write(seeds_out, _unique_seeds(merged("seeds")))
        print(f"[parser-conformance] exported {count} tool seeds -> {seeds_out}")
    if os.environ.get("PC_PERF"):
        perf_out = os.environ.get("PC_PERF_OUT", "parser_conformance_perf.jsonl")
        _write(perf_out, _longest_per_parser(merged("perf")))
        print(f"[parser-conformance] streaming cost -> {perf_out}")
    errors = merged("errors")
    if errors:
        print("\n".join(errors))
        print(f"[parser-conformance] {len(errors)} seeds broke the checks")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
