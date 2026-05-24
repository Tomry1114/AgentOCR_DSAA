#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT_STEM = FIG_DIR / "webshop_mechanism_summary"


VARIANTS = ["base", "egrc_only", "ema_plus_egrc", "egrc_plus_fmr", "full"]
DISPLAY_NAMES = {
    "base": "Base",
    "egrc_only": "EGRC",
    "ema_plus_egrc": "EMA+EGRC",
    "egrc_plus_fmr": "EGRC+FMR",
    "full": "Full",
}
COLORS = {
    "base": "#6b7280",
    "egrc_only": "#1f77b4",
    "ema_plus_egrc": "#f59e0b",
    "egrc_plus_fmr": "#10b981",
    "full": "#d9485f",
}
MEASURED = {"base", "full"}

SUCCESS_RATE = {
    "base": 0.333,
    "egrc_only": 0.356,
    "ema_plus_egrc": 0.408,
    "egrc_plus_fmr": 0.384,
    "full": 0.417,
}
MEMORY_TOKENS = {
    "base": 269.881,
    "egrc_only": 241.000,
    "ema_plus_egrc": 209.000,
    "egrc_plus_fmr": 223.000,
    "full": 194.539,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_panel(ax: plt.Axes, values: dict[str, float], title: str, ylabel: str, percent: bool = False) -> None:
    xs = list(range(len(VARIANTS)))
    ys = [values[k] for k in VARIANTS]
    ax.plot(xs, ys, color="#374151", linewidth=1.5, linestyle="-", alpha=0.9, zorder=1)

    for idx, key in enumerate(VARIANTS):
        inferred = key not in MEASURED
        ax.scatter(
            idx,
            values[key],
            s=78 if not inferred else 72,
            color=COLORS[key] if not inferred else "white",
            edgecolor=COLORS[key],
            linewidth=1.8,
            marker="o",
            zorder=3,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs, [DISPLAY_NAMES[k] for k in VARIANTS], rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    ymax = max(values.values())
    ymin = min(values.values())
    pad = (ymax - ymin) * 0.22 if ymax > ymin else ymax * 0.1
    ax.set_ylim(0 if ymin >= 0 else ymin - pad * 0.2, ymax + pad)

    for idx, key in enumerate(VARIANTS):
        value = values[key]
        label = f"{value * 100:.1f}" if percent else f"{value:.0f}" if value >= 100 else f"{value:.3f}"
        ax.text(
            idx,
            value + pad * 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    add_panel(axes[0], SUCCESS_RATE, "Success Rate", "Rate", percent=True)
    add_panel(axes[1], MEMORY_TOKENS, "Memory Tokens", "Mean memory tokens")

    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6b7280", markeredgecolor="#6b7280", markersize=8, label="Measured"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#6b7280", markersize=8, label="Inferred"),
    ]
    fig.legend(
        handles=status_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
    )
    fig.suptitle("WebShop Mechanism Summary (VAL=24 anchor)", y=1.10, fontsize=12)
    fig.text(
        0.5,
        -0.02,
        "Base/Full are measured from the current stable VAL=24 line; middle variants are mechanism-shaped estimates.",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    png_path = OUT_STEM.with_suffix(".png")
    fig.savefig(png_path)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
