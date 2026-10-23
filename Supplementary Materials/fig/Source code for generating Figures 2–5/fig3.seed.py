#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制单模型、Asym-BP和GA-BP的多随机种子RMSE与UR箱形图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "50随机种子_OOF_55候选嵌套决策结果" / "111多随机种子全部指标与汇总.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "多随机种子模型RMSE与UR箱形图.jpg"

TAU_VALUES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SINGLE_MODELS = [
    "Ridge", "SVR_RBF", "RandomForest", "GBT", "XGBoost",
    "LightGBM", "BP_NN", "PSO_BP", "GA_BP_pure",
]
SINGLE_LABELS = [
    "Ridge", "SVR (RBF)", "RF", "GBT", "XGBoost",
    "LightGBM", "BP-NN", "PSO-BP", "GA-BP",
]
ASYM_MODELS = [f"Asym-BP_tau{tau:.2f}" for tau in TAU_VALUES]
GA_TAU_MODELS = [f"GA_BP_tau{tau:.2f}" for tau in TAU_VALUES]
TAU_LABELS = [rf"$\tau$={tau:.2f}" for tau in TAU_VALUES]
ALL_MODELS = [*SINGLE_MODELS, *ASYM_MODELS, *GA_TAU_MODELS]


def configure_style() -> None:
    """设置全局Times New Roman字体和向内刻度。"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 600,
    })


def load_and_validate(path: Path) -> pd.DataFrame:
    """读取三类单模型结果，并确保每个模型恰有50个不同随机种子。"""
    data = pd.read_excel(path, sheet_name="All_seed_metrics")
    required = {"Section", "Seed", "Model", "RMSE", "UR_%"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"数据缺少必要列：{sorted(missing)}")

    selected = data.loc[
        data["Section"].eq("single") & data["Model"].isin(ALL_MODELS),
        ["Seed", "Model", "RMSE", "UR_%"],
    ].copy()
    selected["Model"] = pd.Categorical(selected["Model"], categories=ALL_MODELS, ordered=True)
    selected = selected.sort_values(["Model", "Seed"]).reset_index(drop=True)

    counts = selected.groupby("Model", observed=False)["Seed"].agg(["size", "nunique"])
    invalid = counts.loc[(counts["size"] != 50) | (counts["nunique"] != 50)]
    if not invalid.empty:
        raise ValueError("以下模型并非恰好包含50个不同随机种子：\n" + invalid.to_string())
    if selected[["RMSE", "UR_%"]].isna().any().any():
        raise ValueError("所选RMSE或UR_%数据中存在缺失值")
    if not np.isfinite(selected[["RMSE", "UR_%"]].to_numpy(dtype=float)).all():
        raise ValueError("所选RMSE或UR_%数据中存在非有限值")
    return selected


def single_model_colors() -> list:
    """为九个独立模型提供可区分的低饱和颜色。"""
    return [mpl.colormaps["Set2"](value) for value in np.linspace(0.0, 0.95, len(SINGLE_MODELS))]


def draw_box_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    model_order: list[str],
    display_labels: list[str],
    colors: list,
) -> None:
    values = [data.loc[data["Model"].eq(model), metric].to_numpy(dtype=float) for model in model_order]
    positions = np.arange(1, len(model_order) + 1)

    artists = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.35},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.9},
        capprops={"color": "#4A4A4A", "linewidth": 0.9},
        boxprops={"edgecolor": "#404040", "linewidth": 0.9},
    )
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.72)

    # 以Seed生成固定抖动，同一个点在重复运行时位置保持一致。
    for position, (model, color) in enumerate(zip(model_order, colors), start=1):
        group = data.loc[data["Model"].eq(model)].sort_values("Seed")
        rng = np.random.default_rng(20261004 + position)
        jitter = rng.uniform(-0.20, 0.20, size=len(group))
        ax.scatter(
            position + jitter,
            group[metric].to_numpy(dtype=float),
            s=14,
            facecolor=color,
            edgecolor="#202020",
            linewidth=0.30,
            alpha=0.78,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(display_labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center", fontweight="bold", pad=10)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    # 保留四周框线，但仅在下方和左侧显示刻度。
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.margins(x=0.02)


def create_figure(data: pd.DataFrame, output: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(3, 2, figsize=(16.0, 14.0), constrained_layout=True)
    single_colors = single_model_colors()
    asym_colors = [mpl.colormaps["Oranges"](value) for value in np.linspace(0.32, 0.88, len(TAU_VALUES))]
    ga_colors = [mpl.colormaps["Blues"](value) for value in np.linspace(0.32, 0.88, len(TAU_VALUES))]

    draw_box_panel(
        axes[0, 0], data, "RMSE", "RMSE",
        "(a) Multi-seed RMSE Distribution of Individual Models",
        SINGLE_MODELS, SINGLE_LABELS, single_colors,
    )
    draw_box_panel(
        axes[0, 1], data, "UR_%", "UR (%)",
        "(b) Multi-seed UR Distribution of Individual Models",
        SINGLE_MODELS, SINGLE_LABELS, single_colors,
    )
    draw_box_panel(
        axes[1, 0], data, "RMSE", "RMSE",
        "(c) Multi-seed RMSE Distribution of Asym-BP Models",
        ASYM_MODELS, TAU_LABELS, asym_colors,
    )
    draw_box_panel(
        axes[1, 1], data, "UR_%", "UR (%)",
        "(d) Multi-seed UR Distribution of Asym-BP Models",
        ASYM_MODELS, TAU_LABELS, asym_colors,
    )
    draw_box_panel(
        axes[2, 0], data, "RMSE", "RMSE",
        "(e) Multi-seed RMSE Distribution of GA-BP Models",
        GA_TAU_MODELS, TAU_LABELS, ga_colors,
    )
    draw_box_panel(
        axes[2, 1], data, "UR_%", "UR (%)",
        "(f) Multi-seed UR Distribution of GA-BP Models",
        GA_TAU_MODELS, TAU_LABELS, ga_colors,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="jpg", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制三类模型RMSE与UR的50随机种子箱形图")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = load_and_validate(args.input)
    create_figure(data, args.output)
    print(f"已输出：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
