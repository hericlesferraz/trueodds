"""Reliability diagrams from a calibration result (Phase 3): before and after, on the same axes.

Drawn from the `reliability` bins already in the JSON, so no model and no scores are needed.
One PNG per set, next to the results: `harness/results/figures/<stamp>-reliability-<set>.png`.

    uv run --extra plots python -m harness.plots harness/results/<stamp>-calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trueodds import paths

SETS = ("pooled/test", "commonsense_qa/test", "dbpedia/test")
BEFORE, AFTER = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the dataviz reference palette
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MIN_BIN = 20  # a bin with fewer questions has an accuracy too noisy to draw as a point


def diagram(result: dict, key: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["temperature"]["value"]
    before = result["before"][key]["metrics"]
    after = result["after"][key]["metrics"]
    fig, (ax, bars) = plt.subplots(
        2, 1, figsize=(5.2, 6.0), height_ratios=[3, 1], sharex=True, facecolor=SURFACE
    )
    for a in (ax, bars):
        a.set_facecolor(SURFACE)
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=9)
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    for m, color, label in (
        (before, BEFORE, f"before (T = 1), ECE {before['ece']:.3f}"),
        (after, AFTER, f"after (T = {t:.3f}), ECE {after['ece']:.3f}"),
    ):
        pts = [(b["confidence"], b["accuracy"]) for b in m["reliability"] if b["n"] >= MIN_BIN]
        ax.plot(*zip(*pts, strict=True), color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy in bin", color=INK, fontsize=10)
    ax.set_title(f"{key}  (n = {before['n']:,})", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.text(0.99, 0.02, f"bins with fewer than {MIN_BIN} questions are not drawn",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)  # fmt: skip

    # Questions per bin (after), so a bin's point can be weighed by how many questions it holds.
    bins = after["reliability"]
    width = bins[0]["hi"] - bins[0]["lo"]
    bars.bar([b["lo"] + width / 2 for b in bins], [b["n"] for b in bins], width=width * 0.9,
             color=AFTER, alpha=0.35, linewidth=0)  # fmt: skip
    bars.set_ylabel("questions", color=INK, fontsize=10)
    bars.set_xlabel("confidence (top-1 probability)", color=INK, fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("--out-dir", type=Path, default=paths.RESULTS / "figures")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    stamp = args.result.name.removesuffix("-calibration.json")
    for key in SETS:
        if key in result["before"]:
            name = key.replace("/test", "").replace("/", "-")
            print(diagram(result, key, args.out_dir / f"{stamp}-reliability-{name}.png"))


if __name__ == "__main__":
    main()
