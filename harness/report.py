"""Turn results JSONs (spec 002) into one markdown table.

uv run python -m harness.report harness/results/*-baselines.json harness/results/*-eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def table(paths: list[Path]) -> str:
    """One row per source/split. Baselines come from the first file, then one column group per run."""
    runs = [json.loads(p.read_text()) for p in paths]
    keys = list(dict.fromkeys(k for r in runs for k in r["results"]))
    base = runs[0]["results"]
    models = [r for r in runs if r["predictor"]]

    head = ["source/split", "n", "random acc", "majority acc", "prior NLL", "prior ECE"]
    for r in models:
        name = r["run"] or r["predictor"]
        head += [f"{name} acc", f"{name} ECE", f"{name} NLL", f"{name} Brier"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in keys:
        b = base.get(key, {}).get("baselines", {})
        row = [
            key,
            str(base.get(key, {}).get("n", "")),
            _fmt(b.get("random", {}).get("accuracy")),
            _fmt(b.get("majority", {}).get("accuracy")),
            _fmt(b.get("prior", {}).get("nll")),
            _fmt(b.get("prior", {}).get("ece")),
        ]
        for r in models:
            m = (r["results"].get(key) or {}).get("metrics") or {}
            row += [
                _fmt(m.get("accuracy")),
                _fmt(m.get("ece")),
                _fmt(m.get("nll")),
                _fmt(m.get("brier")),
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    print(table(parser.parse_args().paths))


if __name__ == "__main__":
    main()
