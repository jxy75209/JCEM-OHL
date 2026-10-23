from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PATH = SCRIPT_DIR / "GA_BP_tau_sensitivity_plot.png"

TAU_METRICS = {
    "GA-BP": {
        "Tau": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "RMSE": [711.07, 689.19, 695.37, 694.10, 702.08, 689.92, 711.43, 711.39, 741.23],
        "UR_%": [47.91, 46.59, 47.27, 48.05, 48.55, 48.68, 50.00, 49.59, 49.82],
    },
    "Asym-BP": {
        "Tau": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "RMSE": [1013.70, 875.72, 785.24, 756.47, 750.19, 748.25, 748.57, 740.61, 754.84],
        "UR_%": [24.09, 29.09, 33.45, 38.50, 40.68, 42.09, 44.77, 46.59, 48.14],
    },
}


def load_tau_metrics(family: str) -> pd.DataFrame:
    """Return the Table 2/Table 3 mean RMSE and UR values for one model family."""
    if family not in TAU_METRICS:
        raise ValueError(f"Unknown model family: {family}")
    result = pd.DataFrame(TAU_METRICS[family]).sort_values("Tau").reset_index(drop=True)
    if result[["Tau", "RMSE", "UR_%"]].isna().any().any():
        raise ValueError(f"{family} contains missing or non-numeric Tau/RMSE/UR values")
    if result["Tau"].duplicated().any():
        raise ValueError(f"{family} contains duplicate tau values")
    return result


def draw_panel(ax_ur, data: pd.DataFrame, title: str) -> None:
    ax_rmse = ax_ur.twinx()

    ur_line, = ax_ur.plot(
        data["Tau"], data["UR_%"],
        marker="o", markersize=4.5, linestyle="-", linewidth=1.8,
        color="#2F6B9A", label="UR",
    )
    rmse_line, = ax_rmse.plot(
        data["Tau"], data["RMSE"],
        marker="s", markersize=4.2, linestyle="-", linewidth=1.8,
        color="#C44E52", label="RMSE",
    )

    ax_ur.set_title(title, fontsize=11, pad=10, loc="center")
    ax_ur.set_xlabel(r"$\tau$")
    ax_ur.set_ylabel("UR (%)", color="#2F6B9A")
    ax_rmse.set_ylabel("RMSE", color="#C44E52")
    ax_ur.set_xticks(data["Tau"])
    ax_ur.set_xticklabels(
        [f"{tau:.2f}" for tau in data["Tau"]],
        rotation=0, ha="center", rotation_mode="anchor",
    )
    ur_min, ur_max = data["UR_%"].min(), data["UR_%"].max()
    rmse_min, rmse_max = data["RMSE"].min(), data["RMSE"].max()
    ur_span = max(ur_max - ur_min, 1.0)
    rmse_span = max(rmse_max - rmse_min, 1.0)
    ax_ur.set_ylim(ur_min - 0.10 * ur_span, ur_max + 0.28 * ur_span)
    ax_rmse.set_ylim(rmse_min - 0.10 * rmse_span, rmse_max + 0.28 * rmse_span)
    ax_ur.grid(True, axis="both", linestyle="--", linewidth=0.55, alpha=0.45)
    ax_ur.tick_params(axis="both", direction="in")
    ax_ur.tick_params(axis="y", colors="#2F6B9A")
    ax_rmse.tick_params(axis="y", direction="in", colors="#C44E52")
    for axis in (ax_ur, ax_rmse):
        for spine in axis.spines.values():
            spine.set_visible(True)
    ax_ur.legend(
        [ur_line, rmse_line], ["UR", "RMSE"],
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
        frameon=False, handlelength=2.2, columnspacing=1.5,
    )


def main() -> None:
    ga_bp = load_tau_metrics("GA-BP")
    asym_bp = load_tau_metrics("Asym-BP")

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), dpi=300)
    draw_panel(axes[0], ga_bp, "(a) GA-BP Tau Sensitivity Analysis")
    draw_panel(axes[1], asym_bp, "(b) Asym-BP Tau Sensitivity Analysis")

    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.10, top=0.93, hspace=0.55)
    fig.savefig(PLOT_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
