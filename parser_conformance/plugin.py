"""Pytest plugin: record the inputs that vLLM's own tests feed to tool and
reasoning parsers, then re-check every recorded input against conformance
properties.

Run from a vLLM checkout:

    PYTHONPATH=$PWD:<tool dir> PC_REPORT=/tmp/pc.jsonl \
        python -m pytest tests/tool_parsers tests/reasoning \
        -p parser_conformance.plugin -q
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import time
import traceback
from collections import Counter

MAX_SEEDS_PER_PARSER = int(os.environ.get("PC_MAX_SEEDS", "40"))

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
_SEEN: set[tuple[str, str]] = set()
_PER_PARSER: Counter[str] = Counter()
_IMPORT_ERRORS: list[str] = []
_STATE = {"depth": 0, "recording": True, "nodeid": None}


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
            and (name, model_output) not in _SEEN
            and _PER_PARSER[name] < MAX_SEEDS_PER_PARSER
        ):
            _SEEN.add((name, model_output))
            _PER_PARSER[name] += 1
            _SEEDS.append((family, self, model_output, request, _STATE["nodeid"]))
        _STATE["depth"] += 1
        try:
            return orig(self, model_output, request, *args, **kwargs)
        finally:
            _STATE["depth"] -= 1

    wrapper._pc_wrapped = True
    setattr(cls, method, wrapper)


def _wrap_init(cls) -> None:
    """Keep the outermost constructor arguments, so the checks can build a
    fresh parser for streaming exactly as the test built this one."""
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


def pytest_sessionfinish(session, exitstatus):
    from parser_conformance.checks import check_seed
    from parser_conformance.reasoning_checks import check_reasoning_seed

    checkers = {"tool": check_seed, "reasoning": check_reasoning_seed}
    _STATE["recording"] = False
    out = os.environ.get("PC_REPORT", "parser_conformance.jsonl")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:  # one report per pytest-xdist worker
        out = f"{out}.{worker}"
    start = time.time()
    errors: list[str] = []
    with open(out, "w", encoding="utf-8") as f:
        for family, parser, text, request, nodeid in _SEEDS:
            try:
                findings = checkers[family](parser, text, request)
            except Exception:  # noqa: BLE001  re-raised below, after the report is written
                errors.append(f"{type(parser).__name__} {nodeid}\n{traceback.format_exc()}")
                continue
            for finding in findings:
                finding.update(family=family, parser=type(parser).__name__, nodeid=nodeid)
                f.write(json.dumps(finding, ensure_ascii=False, default=str) + "\n")
    print(
        f"\n[parser-conformance] {len(_SEEDS)} seeds from {len(_PER_PARSER)} parsers "
        f"checked in {time.time() - start:.0f}s -> {out}"
    )
    for line in _IMPORT_ERRORS:
        print(f"[parser-conformance] not imported: {line}")
    seeds_out = os.environ.get("PC_SEEDS_OUT")
    if seeds_out:
        from parser_conformance.export import export_seeds

        if worker:
            seeds_out = f"{seeds_out}.{worker}"
        count = export_seeds(_SEEDS, seeds_out)
        print(f"[parser-conformance] exported {count} tool seeds -> {seeds_out}")
    if os.environ.get("PC_PERF"):
        from parser_conformance.perf import run_perf

        perf_out = os.environ.get("PC_PERF_OUT", "parser_conformance_perf.jsonl")
        if worker:
            perf_out = f"{perf_out}.{worker}"
        with open(perf_out, "w", encoding="utf-8") as f:
            for row in run_perf(_SEEDS):
                f.write(json.dumps(row) + "\n")
        print(f"[parser-conformance] streaming cost -> {perf_out}")
    if errors:
        print("\n".join(errors))
        raise RuntimeError(f"parser-conformance: {len(errors)} seeds broke the harness")
