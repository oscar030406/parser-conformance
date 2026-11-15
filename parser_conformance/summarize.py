"""Summarize a parser-conformance JSONL report.

    python -m parser_conformance.summarize report.jsonl [more reports] [--examples]

Prints one row per parser: the number of recorded seeds, then the count of
findings per check. With --examples, it then prints the shortest input for
each (parser, check) pair.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main(paths: list[str], examples: bool) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    shortest: dict[tuple[str, str], dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            parser, check = row["parser"], row["check"]
            counts[parser][check] += 1
            best = shortest.get((parser, check))
            if best is None or len(row["input"]) < len(best["input"]):
                shortest[(parser, check)] = row
    checks = ["seed", *sorted({c for per in counts.values() for c in per} - {"seed"})]
    print("parser\t" + "\t".join(checks))
    for parser in sorted(counts):
        print(parser + "\t" + "\t".join(str(counts[parser][c]) for c in checks))
    if examples:
        for (parser, check), row in sorted(shortest.items()):
            if check == "seed" or check.startswith("skip"):
                continue
            detail = {k: v for k, v in row.items() if k not in ("parser", "check", "nodeid")}
            print(f"\n## {parser} {check}  ({row['nodeid']})")
            print(json.dumps(detail, ensure_ascii=False)[:600])


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if not a.startswith("--")], "--examples" in args)
