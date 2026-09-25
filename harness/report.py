"""Turn results JSONs (spec 002) into markdown tables.

uv run python -m harness.report harness/results/*-baselines.json harness/results/*-eval.json
uv run python -m harness.report harness/results/<stamp>-calibration.json   # Phase 3, before/after
uv run python -m harness.report harness/results/<stamp>-compare.json       # Phase 5, B against A
uv run python -m harness.report harness/results/<stamp>-options.json       # Phase 5E, fewer options
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


def _delta(before: float, after: float) -> str:
    return f"{after - before:+.3f}"


def calibration_tables(result: dict) -> str:
    """Before/after ECE, NLL and Brier per file (training sources, then held-out apart) and ECE by K."""
    t = result["temperature"]
    before, after, change = result["before"], result["after"], result["ece_change"]
    lines = [
        f"T = {t['value']:.4f}, fitted on dev ({t['n']:,} questions from {len(t['sources'])} "
        f"sources); dev NLL {t['dev_before']['nll']:.4f} -> {t['dev_after']['nll']:.4f}, "
        f"dev ECE {t['dev_before']['ece']:.4f} -> {t['dev_after']['ece']:.4f}",
        "",
    ]
    head = ["source/split", "n", "ECE before", "ECE after", "Δ ECE (95% CI)",
            "NLL before", "NLL after", "Δ NLL", "Brier before", "Brier after"]  # fmt: skip
    for role, title in (("train", "Training sources (test)"), ("heldout", "Held-out datasets")):
        lines += [f"**{title}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for key, entry in before.items():
            if entry["role"] != role:
                continue
            b, a, c = entry["metrics"], after[key]["metrics"], change[key]
            lo, hi = c["ci95"]
            row = [key, str(b["n"]), _fmt(b["ece"]), _fmt(a["ece"]),
                   f"{c['delta']:+.3f} ({lo:+.3f}, {hi:+.3f})", _fmt(b["nll"]), _fmt(a["nll"]),
                   _delta(b["nll"], a["nll"]), _fmt(b["brier"]), _fmt(a["brier"])]  # fmt: skip
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ECE by K: pooled test (every K the model was trained on) and the held-out files apart.
    lines += ["**ECE by number of options**", "",
              "| set | K | n | ECE before | ECE after |", "|---|---|---|---|---|"]  # fmt: skip
    groups = [("pooled/test", before["pooled/test"], after["pooled/test"])] + [
        (key, entry, after[key]) for key, entry in before.items() if entry["role"] == "heldout"
    ]
    for key, b, a in groups:
        for k, g in sorted(b["metrics"]["by_k"].items(), key=lambda kv: int(kv[0])):
            ga = a["metrics"]["by_k"][k]
            lines.append(f"| {key} | {k} | {g['n']} | {_fmt(g['ece'])} | {_fmt(ga['ece'])} |")
    return "\n".join(lines)


def latency_tables(result: dict) -> str:
    """Per-request percentiles by K and state length, then the batch against one by one."""
    lines = ["| K | state tokens | longest sequence | p50 ms | p95 ms | mean ms |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for r in result["requests"]:
        lines.append(
            f"| {r['k']} | {r['state_tokens']} | {r['longest_sequence']} | {r['p50_ms']:.1f} "
            f"| {r['p95_ms']:.1f} | {r['mean_ms']:.1f} |"
        )
    lines += ["", "| questions | sequences | batch ms | one by one ms | batch ms/question "
              "| speedup | max diff |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for b in result["batches"]:
        lines.append(
            f"| {b['questions']} | {b['sequences']} | {b['batch_p50_ms']:.0f} "
            f"| {b['loop_p50_ms']:.0f} | {b['batch_ms_per_question']:.1f} | {b['speedup']:.2f}x "
            f"| {b['max_abs_diff']:.1e} |"
        )
    return "\n".join(lines)


def compare_tables(result: dict) -> str:
    """B - A per file, each difference with its paired 95% interval (training sources, held-out)."""
    a, b = result["a"], result["b"]
    lines = [f"A = {a['run']}/{a['checkpoint']} (T = {a['temperature']:.4f}), "
             f"B = {b['run']}/{b['checkpoint']} (T = {b['temperature']:.4f})", ""]  # fmt: skip
    head = ["source/split", "n", "acc A", "acc B", "Δ acc (95% CI)", "NLL A", "NLL B",
            "Δ NLL (95% CI)", "ECE A", "ECE B", "Δ ECE (95% CI)"]  # fmt: skip

    def cell(m: dict) -> list[str]:
        lo, hi = m["ci95"]
        return [_fmt(m["a"]), _fmt(m["b"]), f"{m['delta']:+.3f} ({lo:+.3f}, {hi:+.3f})"]

    for role, title in (("train", "Training sources (test)"), ("heldout", "Held-out datasets")):
        lines += [f"**{title}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for key, r in result["results"].items():
            if r["role"] == role:
                row = [key, str(r["n"]), *cell(r["accuracy"]), *cell(r["nll"]), *cell(r["ece"])]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def options_tables(result: dict) -> str:
    """Per model: each file at each option count, with confidence - accuracy and its interval."""
    lines = []
    head = ["file", "K", "n", "accuracy", "confidence", "conf - acc (95% CI)", "ECE"]
    for run, m in result["models"].items():
        lines += [f"**{run}** (T = {m['temperature']:.4f})", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]  # fmt: skip
        for file, cells in m["results"].items():
            for k, c in cells.items():
                lo, hi = c["gap_ci95"]
                row = [file, k, str(c["n"]), _fmt(c["accuracy"]), _fmt(c["confidence"]),
                       f"{c['gap']:+.3f} ({lo:+.3f}, {hi:+.3f})", _fmt(c["ece"])]  # fmt: skip
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    paths = parser.parse_args().paths
    kinds = [json.loads(p.read_text()).get("kind") for p in paths]
    special = ("calibration", "latency", "compare", "options")
    others = [p for p, k in zip(paths, kinds, strict=True) if k not in special]
    if others:
        print(table(others))
    for p, k in zip(paths, kinds, strict=True):
        if k == "calibration":
            print(f"\n### Calibration: {p.name}\n")
            print(calibration_tables(json.loads(p.read_text())))
        elif k == "latency":
            print(f"\n### Latency: {p.name}\n")
            print(latency_tables(json.loads(p.read_text())))
        elif k == "compare":
            print(f"\n### Comparison: {p.name}\n")
            print(compare_tables(json.loads(p.read_text())))
        elif k == "options":
            print(f"\n### Fewer options: {p.name}\n")
            print(options_tables(json.loads(p.read_text())))


if __name__ == "__main__":
    main()
