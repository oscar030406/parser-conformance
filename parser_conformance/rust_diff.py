"""Compare the Python tool parsers with vLLM's Rust tool parsers.

    python -m parser_conformance.rust_diff <rust binary> <seed files...>

The seed files come from a run with PC_SEEDS_OUT set. The binary is built from
`examples/parser_diff.rs` in vLLM's `rust/src/chat` crate. For every seed and
every name the Python parser is registered under, the script asks the Rust
parser of the same name to parse the same text, then prints the inputs where
the calls or the content differ. Tokenizers come from the local Hugging Face
cache, so run it after the Python tests have downloaded them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

from parser_conformance.checks import _canonical


def _tokenizer_file(name: str | None) -> str | None:
    if not name:
        return None
    if os.path.isdir(name):
        path = os.path.join(name, "tokenizer.json")
        return path if os.path.exists(path) else None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(name, "tokenizer.json")
    return path if isinstance(path, str) else None


def _key(calls: list[dict]) -> list[tuple[str, str]]:
    return [(c["name"] or "", _canonical(c["arguments"])) for c in calls]


def classify(python: dict, rust: dict) -> str:
    if "error" in rust:
        return "not in Rust" if "is not registered" in rust["error"] else "Rust error"
    if _key(rust["calls"]) != _key(python["calls"]):
        return "calls differ"
    if (rust["content"] or "").strip() != (python["content"] or "").strip():
        return "content differs"
    return "same"


def main(binary: str, paths: list[str]) -> None:
    rows, seeds = [], []
    skipped: Counter[str] = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                seed = json.loads(line)
                if "error" in seed["python"]:
                    skipped["Python parser raised"] += 1
                    continue
                tokenizer_file = _tokenizer_file(seed["tokenizer"])
                if tokenizer_file is None:
                    skipped["no tokenizer.json in cache"] += 1
                    continue
                for name in seed["parser_names"]:
                    rows.append(
                        {
                            "id": len(rows),
                            "parser": name,
                            "tokenizer_file": tokenizer_file,
                            "text": seed["text"],
                            "tools": seed["tools"],
                        }
                    )
                    seeds.append(seed)
    run = subprocess.run(
        [binary],
        input="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=True,
    )
    replies = [json.loads(line) for line in run.stdout.splitlines()]
    counts: Counter[tuple[str, str]] = Counter()
    for reply in replies:
        seed, name, rust = seeds[reply["id"]], reply["parser"], reply["rust"]
        kind = classify(seed["python"], rust)
        counts[(name, kind)] += 1
        if kind in ("same", "not in Rust"):
            continue
        detail = {"parser": name, "text": seed["text"], "rust": rust, "python": seed["python"]}
        print(json.dumps(detail, ensure_ascii=False))
    for (name, kind), n in sorted(counts.items()):
        print(f"{name}\t{kind}\t{n}", file=sys.stderr)
    for reason, n in skipped.items():
        print(f"skipped\t{reason}\t{n}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
