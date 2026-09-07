"""Plot all tables from the captured fixed-mask evaluations, without smoothing."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
points = json.loads((ROOT / "merged-periodic.json").read_text())
meta = json.loads((ROOT / "per-table-trends.json").read_text())
rounds = np.array([p["round"] for p in points])
base = next(p for p in points if p["round"] == 128)
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
colors = ["#2457a7", "#13866d", "#936330", "#8758a6"]
for ax, kind, color in zip(
    axes.flat, ["numeric", "binary", "ordinal", "categorical"], colors, strict=True
):
    names = [p["dataset"] for p in meta if p["target_type"] == kind]
    ratios = np.array(
        [[p["datasets"][n]["loss"] / base["datasets"][n]["loss"] for p in points] for n in names]
    )
    for row in ratios:
        ax.plot(rounds, row, color=("#b64242" if row[-1] >= 1 else color), alpha=0.27, lw=0.9)
    ax.plot(rounds, np.median(ratios, axis=0), color=color, lw=3, label="Median of table ratios")
    ax.axhline(1, color="#747b85", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_title(
        f"{kind.capitalize()} targets | {len(names)} tables", loc="left", fontsize=13, pad=12
    )
    ax.text(
        0.98,
        0.95,
        f"{int((ratios[:, -1] < 1).sum())}/{len(names)} lower at round 768 than 128",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color="#303740",
    )
    ax.grid(axis="y", alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([32, 128, 256, 768])
    ax.tick_params(labelsize=10)
    ax.set_ylabel("Loss / same-table loss at round 128", fontsize=10)
    axes[1, 0].set_xlabel("Training round")
    axes[1, 1].set_xlabel("Training round")
fig.suptitle(
    "One shared TAR model: learning across 120 fixed tables",
    fontsize=18,
    x=0.075,
    ha="left",
    y=0.985,
)
fig.text(
    0.075,
    0.942,
    "Complete evaluations through round 768 | Thin lines: individual tables; bold line: median",
    fontsize=11,
    color="#4b5563",
)
fig.text(
    0.075,
    0.022,
    "Fixed training-mask evaluation; no smoothing. Separate y-axis ranges. "
    "Local, unissued experimental evidence.",
    fontsize=10,
    color="#4b5563",
)
fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.10, hspace=0.32, wspace=0.24)
fig.savefig(ROOT / "loss-trends.png", dpi=190, facecolor="white")
fig.savefig(ROOT / "loss-trends.svg", facecolor="white")

# Keep generated SVG paths readable without trailing whitespace in Git.
svg = ROOT / "loss-trends.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
