#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw a combined RMSE-UR scatter plot for base, ensemble, and safety-margin methods."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_PATH = Path(r"C:\python code\科研绘图\小论文\RMSE_UR_all_methods_scatter.png")


# Base models: Table 1 single-model results, with GA-BP tau=0.20 and tau=0.50
# removed and GA-BP tau=0.35 kept from Table 2.
BASE_MODELS = [
    ("Ridge", 491.736, 50.000),
    ("SVR(RBF)", 527.437, 45.455),
    ("RF", 905.270, 40.909),
    ("GBT", 674.031, 43.182),
    ("XGBoost", 580.377, 61.364),
    ("LightGBM", 837.864, 56.818),
    ("BP-NN", 763.239, 45.455),
    ("PSO-BP", 568.618, 54.545),
    ("GA-BP tau=0.35", 613.358, 38.636),
]


# Table 3 upper-envelope pairwise combinations.
UPPER_ENVELOPE = [
    ("Ridge + SVR(RBF)", 498.079, 31.818),
    ("Ridge + RF", 537.884, 27.273),
    ("Ridge + GBT", 587.209, 31.818),
    ("Ridge + XGBoost", 547.341, 40.909),
    ("Ridge + LightGBM", 607.686, 36.364),
    ("Ridge + BP-NN", 588.044, 25.000),
    ("Ridge + PSO-BP", 503.664, 27.273),
    ("Ridge + GA-BP tau=0.35", 568.956, 25.000),
    ("SVR(RBF) + RF", 561.793, 25.000),
    ("SVR(RBF) + GBT", 621.347, 29.545),
    ("SVR(RBF) + XGBoost", 560.607, 34.091),
    ("SVR(RBF) + LightGBM", 656.707, 34.091),
    ("SVR(RBF) + BP-NN", 633.968, 20.455),
    ("SVR(RBF) + PSO-BP", 514.500, 34.091),
    ("SVR(RBF) + GA-BP tau=0.35", 564.154, 27.273),
    ("RF + GBT", 648.379, 36.364),
    ("RF + XGBoost", 616.159, 40.909),
    ("RF + LightGBM", 751.029, 36.364),
    ("RF + BP-NN", 708.742, 25.000),
    ("RF + PSO-BP", 527.361, 29.545),
    ("RF + GA-BP tau=0.35", 586.015, 15.909),
    ("GBT + XGBoost", 682.591, 43.182),
    ("GBT + LightGBM", 749.946, 40.909),
    ("GBT + BP-NN", 725.075, 22.727),
    ("GBT + PSO-BP", 646.702, 29.545),
    ("GBT + GA-BP tau=0.35", 688.779, 18.182),
    ("XGBoost + LightGBM", 658.033, 52.273),
    ("XGBoost + BP-NN", 681.350, 29.545),
    ("XGBoost + PSO-BP", 594.516, 34.091),
    ("XGBoost + GA-BP tau=0.35", 652.455, 27.273),
    ("LightGBM + BP-NN", 812.424, 27.273),
    ("LightGBM + PSO-BP", 691.566, 36.364),
    ("LightGBM + GA-BP tau=0.35", 718.473, 27.273),
    ("BP-NN + PSO-BP", 694.513, 34.091),
    ("BP-NN + GA-BP tau=0.35", 652.947, 18.182),
    ("PSO-BP + GA-BP tau=0.35", 563.939, 25.000),
]


# Table 4 residual-quantile safety-margin rows. The BP-NN baseline row is
# already included in BASE_MODELS, so only the five adjusted configurations
# are plotted here.
SAFETY_MARGIN = [
    ("q=0.50", 813.268, 29.545),
    ("q=0.70", 831.995, 25.000),
    ("q=0.85", 906.968, 13.636),
    ("q=0.90", 942.338, 9.091),
    ("q=0.95", 1096.362, 9.091),
]


BASE_STYLES = {
    "Ridge": ("#1f77b4", "o"),
    "SVR(RBF)": ("#d62728", "s"),
    "RF": ("#2ca02c", "^"),
    "GBT": ("#ff7f0e", "v"),
    "XGBoost": ("#9467bd", "D"),
    "LightGBM": ("#8c564b", "P"),
    "BP-NN": ("#17becf", "X"),
    "PSO-BP": ("#e377c2", "*"),
    "GA-BP tau=0.35": ("#bcbd22", "h"),
}


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    total_configs = len(BASE_MODELS) + len(UPPER_ENVELOPE) + len(SAFETY_MARGIN)
    if total_configs != 50:
        raise ValueError(f"Expected 50 plotted configurations, got {total_configs}.")

    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=300)

    # 36 upper-envelope combinations: same legend style.
    combo_rmse = [x[1] for x in UPPER_ENVELOPE]
    combo_ur = [x[2] for x in UPPER_ENVELOPE]
    ax.scatter(
        combo_ur,
        combo_rmse,
        s=58,
        marker="o",
        facecolor="#bdbdbd",
        edgecolor="#5f5f5f",
        linewidth=0.65,
        alpha=0.58,
        label="Upper-envelope ensembles",
        zorder=2,
    )

    # Safety-margin adjusted BP-NN results: same legend style.
    safety_rmse = [x[1] for x in SAFETY_MARGIN]
    safety_ur = [x[2] for x in SAFETY_MARGIN]
    ax.scatter(
        safety_ur,
        safety_rmse,
        s=88,
        marker="D",
        facecolor="#111111",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.82,
        label="Residual-quantile safety margin",
        zorder=3,
    )

    # Nine base machine-learning models.
    for name, rmse, ur in BASE_MODELS:
        color, marker = BASE_STYLES[name]
        ax.scatter(
            ur,
            rmse,
            s=82,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.75,
            alpha=0.95,
            label=name,
            zorder=4,
        )

    ax.set_xlabel("UR (%)", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_xlim(5, 65)
    ax.set_ylim(450, 1125)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.42)
    ax.tick_params(axis="both", direction="in", labelsize=11)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="#bdbdbd", markeredgecolor="#5f5f5f",
            label="Upper-envelope ensembles",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=7,
            markerfacecolor="#111111", markeredgecolor="white",
            label="Residual-quantile safety margin",
        ),
    ]
    for name, (color, marker) in BASE_STYLES.items():
        handles.append(
            Line2D(
                [0], [0], marker=marker, linestyle="none", markersize=6.2,
                markerfacecolor=color, markeredgecolor="black", label=name,
            )
        )

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.86,
        fontsize=7.8,
        labelspacing=0.46,
        handletextpad=0.45,
        borderpad=0.55,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
