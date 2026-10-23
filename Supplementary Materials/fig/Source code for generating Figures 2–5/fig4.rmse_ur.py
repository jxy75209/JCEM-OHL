#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw a combined RMSE-UR scatter plot for base, ensemble, and safety-margin methods."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parent / "RMSE_UR_all_methods_scatter.png"
RMSE_COMPRESSION_START = 900.0
RMSE_COMPRESSION_RATIO = 50.0
DATA_PATH = (
    Path(__file__).resolve().parent
    / "50随机种子_OOF_55候选嵌套决策结果"
    / "111多随机种子全部指标与汇总.xlsx"
)

# Each section in the multi-seed workbook supplies one scatter category.
DATA_SOURCES = [
    ("envelope", "Upper-envelope ensembles", "#7f7f7f", "o", 58),
    ("margin", "Residual-quantile safety margin", "#d62728", "D", 72),
    ("single", "Single models", "#1f77b4", "^", 66),
]

# All non-tau single models are displayed with distinct markers. Of the tau
# rows, only the tau=0.20 members of the Asym-BP and GA-BP series are retained.
SINGLE_MODEL_STYLES = {
    "Ridge": ("Ridge", "o"),
    "SVR_RBF": ("SVR(RBF)", "s"),
    "RandomForest": ("RF", "^"),
    "GBT": ("GBT", "v"),
    "XGBoost": ("XGBoost", "D"),
    "LightGBM": ("LightGBM", "P"),
    "BP_NN": ("BP-NN", "X"),
    "PSO_BP": ("PSO-BP", "*"),
    "GA_BP_pure": ("GA-BP", "h"),
    "Asym-BP_tau0.20": (r"Asym-BP $\tau{=}0.2$", ">"),
    "GA_BP_tau0.20": (r"GA-BP $\tau{=}0.2$", "<"),
}


def load_metrics(section: str) -> pd.DataFrame:
    """Average RMSE and UR_% across 50 distinct seeds for one method section."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Input workbook not found: {DATA_PATH}")

    data = pd.read_excel(DATA_PATH, sheet_name="All_seed_metrics")
    required_columns = ["Section", "Seed", "Model", "RMSE", "UR_%"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns {missing_columns} in All_seed_metrics of "
            f"{DATA_PATH.name}. Available columns: {list(data.columns)}"
        )

    metrics = data.loc[data["Section"].eq(section), ["Seed", "Model", "RMSE", "UR_%"]].copy()
    if metrics.empty:
        raise ValueError(f"No rows found for section: {section}")
    metrics["RMSE"] = pd.to_numeric(metrics["RMSE"], errors="coerce")
    metrics["UR_%"] = pd.to_numeric(metrics["UR_%"], errors="coerce")
    invalid_rows = metrics[["RMSE", "UR_%"]].isna().any(axis=1)
    if invalid_rows.any():
        excel_rows = (metrics.index[invalid_rows] + 2).tolist()
        raise ValueError(
            f"Non-numeric or missing RMSE/UR_% values in {DATA_PATH.name}, "
            f"Excel row(s): {excel_rows}"
        )

    seed_counts = metrics.groupby("Model", sort=False)["Seed"].agg(["size", "nunique"])
    invalid_models = seed_counts.loc[(seed_counts["size"] != 50) | (seed_counts["nunique"] != 50)]
    if not invalid_models.empty:
        raise ValueError(
            f"Models in section {section!r} do not contain exactly 50 distinct seeds:\n"
            + invalid_models.to_string()
        )

    return (
        metrics.groupby("Model", sort=False, as_index=False)[["RMSE", "UR_%"]]
        .mean()
    )


def plot_single_models(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    """Plot only the requested single models, each with a distinct marker."""
    for model, (display_name, marker) in SINGLE_MODEL_STYLES.items():
        row = metrics.loc[metrics["Model"] == model]
        if row.empty:
            raise ValueError(f"Required single-model row not found: {model}")
        ax.scatter(
            row["UR_%"], row["RMSE"], s=72, marker=marker,
            facecolor="#1f77b4", edgecolor="black", linewidth=0.65,
            alpha=0.82, label=display_name, zorder=4,
        )


def plot_margin_methods(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    """Plot fixed-amount and proportional safety-margin methods separately."""
    margin_styles = [
        ("BP_NN_OOF_fixed_", "BP-NN fixed-amount safety margin", "#d62728", "D", 72),
        ("SVR_RBF_OOF_fixed_", "SVR fixed-amount safety margin", "#ff7f0e", "P", 76),
        ("SVR_RBF_OOF_ratio_", "SVR proportional safety margin", "#9467bd", "s", 68),
    ]

    for model_prefix, label, color, marker, size in margin_styles:
        group = metrics.loc[metrics["Model"].str.startswith(model_prefix)].copy()
        if group.empty:
            raise ValueError(f"No safety-margin methods found for: {label}")
        ax.scatter(
            group["UR_%"], group["RMSE"],
            s=size, marker=marker, facecolor=color, edgecolor="black",
            linewidth=0.65, alpha=0.78, label=f"{label} (n={len(group)})", zorder=5,
        )


def compress_rmse(values):
    """Compress the RMSE axis 50:1 above 900 while preserving point order."""
    values = np.asarray(values, dtype=float)
    return np.where(
        values <= RMSE_COMPRESSION_START,
        values,
        RMSE_COMPRESSION_START + (values - RMSE_COMPRESSION_START) / RMSE_COMPRESSION_RATIO,
    )


def expand_rmse(values):
    """Inverse transform for the compressed RMSE axis."""
    values = np.asarray(values, dtype=float)
    return np.where(
        values <= RMSE_COMPRESSION_START,
        values,
        RMSE_COMPRESSION_START + (values - RMSE_COMPRESSION_START) * RMSE_COMPRESSION_RATIO,
    )


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=300)
    all_rmse = []
    all_ur = []

    for section, label, color, marker, size in DATA_SOURCES:
        metrics = load_metrics(section)
        if section == "single":
            metrics = metrics.loc[metrics["Model"].isin(SINGLE_MODEL_STYLES)].copy()
        elif section == "margin":
            # The two unadjusted baselines are already represented by the
            # corresponding points in the single-model category.
            metrics = metrics.loc[~metrics["Model"].str.endswith("_base")].copy()

        all_rmse.extend(metrics["RMSE"].tolist())
        all_ur.extend(metrics["UR_%"].tolist())
        if section == "single":
            plot_single_models(ax, metrics)
            continue
        if section == "margin":
            plot_margin_methods(ax, metrics)
            continue
        ax.scatter(
            metrics["UR_%"],
            metrics["RMSE"],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.75,
            label=f"{label} (n={len(metrics)})",
            zorder=3,
        )

    # Extend the upper/right sides more strongly to reserve a clear legend area.
    ur_span = max(all_ur) - min(all_ur)
    lower_ur_margin = max(2.0, ur_span * 0.06)
    upper_ur_margin = max(12.0, ur_span * 0.58)
    lower_rmse_margin = max(20.0, (RMSE_COMPRESSION_START - min(all_rmse)) * 0.06)
    upper_rmse_limit = float(expand_rmse(compress_rmse(max(all_rmse)) + 18.0))

    ax.set_xlabel("UR (%)", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_yscale("function", functions=(compress_rmse, expand_rmse))
    ax.set_xlim(max(0, min(all_ur) - lower_ur_margin), max(all_ur) + upper_ur_margin)
    ax.set_ylim(max(0, min(all_rmse) - lower_rmse_margin), upper_rmse_limit)
    lower_ticks = np.arange(500, RMSE_COMPRESSION_START + 1, 100, dtype=float)
    upper_ticks = np.arange(2000, max(all_rmse) + 1000, 2000, dtype=float)
    ax.set_yticks(np.concatenate([lower_ticks, upper_ticks]))
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.42)
    ax.axhline(
        RMSE_COMPRESSION_START, color="#777777", linestyle=":", linewidth=0.75,
        alpha=0.80, zorder=1,
    )
    ax.tick_params(axis="both", direction="in", labelsize=11)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.90,
        fontsize=7.8,
        labelspacing=0.45,
        handletextpad=0.45,
        borderpad=0.55,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_PATH}")
    for section, label, *_ in DATA_SOURCES:
        metrics = load_metrics(section)
        count = len(metrics)
        if section == "single":
            count = len(SINGLE_MODEL_STYLES)
        elif section == "margin":
            count = int((~metrics["Model"].str.endswith("_base")).sum())
        print(f"{label}: {count} points")


if __name__ == "__main__":
    main()
