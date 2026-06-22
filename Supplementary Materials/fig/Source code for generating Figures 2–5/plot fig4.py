#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot APE histograms for selected models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(r"C:\python code\科研绘图\小论文\初始单模型预测结果.xlsx")
APE_HIST_PATH = Path(r"C:\python code\科研绘图\小论文\APE_histograms_LightGBM_BP_GABP020_combined.png")


APE_COLUMNS = {
    "LightGBM": "LightGBM_APE_%",
    "BP-NN": "BP_NN_APE_%",
    "GA-BP tau=0.20": "GA_BP_tau0.20_APE_%",
}


HIST_COLORS = {
    "LightGBM": "#a9d5a5",
    "BP-NN": "#9cbedb",
    "GA-BP tau=0.20": "#f08f92",
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def load_ape_data() -> dict[str, np.ndarray]:
    df = pd.read_excel(DATA_PATH, sheet_name="LOO_predictions")
    ape_data = {}
    for model, col in APE_COLUMNS.items():
        if col not in df.columns:
            raise ValueError(f"Column not found in {DATA_PATH}: {col}")
        ape_data[model] = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return ape_data


def plot_ape_histograms(ape_data: dict[str, np.ndarray]) -> None:
    all_values = np.concatenate(list(ape_data.values()))
    upper = float(np.nanpercentile(all_values, 95))
    upper = max(upper, 100.0)
    bin_edges = np.linspace(0, upper, 13)

    subplot_titles = {
        "LightGBM": "(a) LightGBM",
        "BP-NN": "(b) BP-NN",
        "GA-BP tau=0.20": "(c) GA-BP tau=0.20",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=300)

    for ax, model in zip(axes, ["LightGBM", "BP-NN", "GA-BP tau=0.20"]):
        values = ape_data[model]
        clipped = np.clip(values, 0, upper)
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))

        ax.hist(
            clipped,
            bins=bin_edges,
            color=HIST_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.82,
        )

        mean_x = min(mean_val, upper)
        median_x = min(median_val, upper)
        ax.axvline(mean_x, color="#111111", linestyle="--", linewidth=1.35, label=f"Mean = {mean_val:.1f}%")
        ax.axvline(median_x, color="#666666", linestyle=":", linewidth=1.65, label=f"Median = {median_val:.1f}%")

        if mean_val > upper or median_val > upper:
            ax.text(
                0.98,
                0.88,
                f"Values > {upper:.0f}% clipped",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )

        ax.set_title(subplot_titles[model], fontsize=12.5, pad=8)
        ax.set_xlabel("APE (%)", fontsize=11.5)
        ax.set_ylabel("Number of Projects", fontsize=11.5)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.38)
        ax.tick_params(axis="both", direction="in", labelsize=9.8)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#d0d0d0",
            framealpha=0.88,
            fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(APE_HIST_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    ape_data = load_ape_data()
    plot_ape_histograms(ape_data)
    print(f"Saved APE histograms: {APE_HIST_PATH}")


if __name__ == "__main__":
    main()
