from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_PATH = Path(r"C:\python code\科研绘图\小论文\GA_BP_tau_sensitivity_plot.png")


TAU_SENSITIVITY_DATA = [
    {"Tau": 0.10, "Low_penalty_coefficient": 9.0, "RMSE": 589.583, "MAE": 358.022, "MedAE": 218.047, "R2": 0.883, "EVS": 0.883, "MAPE_%": 46.446, "UR_%": 54.545},
    {"Tau": 0.15, "Low_penalty_coefficient": 5.7, "RMSE": 557.190, "MAE": 327.500, "MedAE": 160.866, "R2": 0.896, "EVS": 0.896, "MAPE_%": 41.108, "UR_%": 50.000},
    {"Tau": 0.20, "Low_penalty_coefficient": 4.0, "RMSE": 570.525, "MAE": 302.870, "MedAE": 150.301, "R2": 0.890, "EVS": 0.891, "MAPE_%": 40.185, "UR_%": 56.818},
    {"Tau": 0.25, "Low_penalty_coefficient": 3.0, "RMSE": 644.850, "MAE": 352.687, "MedAE": 162.952, "R2": 0.860, "EVS": 0.860, "MAPE_%": 35.718, "UR_%": 54.545},
    {"Tau": 0.30, "Low_penalty_coefficient": 2.3, "RMSE": 690.709, "MAE": 394.808, "MedAE": 179.078, "R2": 0.840, "EVS": 0.840, "MAPE_%": 64.985, "UR_%": 56.818},
    {"Tau": 0.35, "Low_penalty_coefficient": 1.9, "RMSE": 613.358, "MAE": 313.841, "MedAE": 111.151, "R2": 0.873, "EVS": 0.874, "MAPE_%": 27.522, "UR_%": 38.636},
    {"Tau": 0.40, "Low_penalty_coefficient": 1.5, "RMSE": 562.271, "MAE": 300.772, "MedAE": 127.279, "R2": 0.894, "EVS": 0.894, "MAPE_%": 40.029, "UR_%": 50.000},
    {"Tau": 0.45, "Low_penalty_coefficient": 1.2, "RMSE": 564.696, "MAE": 329.819, "MedAE": 157.544, "R2": 0.893, "EVS": 0.893, "MAPE_%": 34.099, "UR_%": 45.455},
    {"Tau": 0.50, "Low_penalty_coefficient": 1.0, "RMSE": 610.446, "MAE": 347.301, "MedAE": 172.221, "R2": 0.875, "EVS": 0.875, "MAPE_%": 32.471, "UR_%": 50.000},
]


def main() -> None:
    results_df = pd.DataFrame(TAU_SENSITIVITY_DATA)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    ax2 = ax1.twinx()

    line1, = ax1.plot(
        results_df["Tau"],
        results_df["UR_%"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#1f77b4",
        label="UR",
    )
    line2, = ax2.plot(
        results_df["Tau"],
        results_df["RMSE"],
        marker="s",
        linestyle="-",
        linewidth=2,
        color="#d62728",
        label="RMSE",
    )

    ax1.set_xlabel("Tau")
    ax1.set_ylabel("Underestimation Rate (%)")
    ax2.set_ylabel("RMSE")
    ax1.set_xticks(results_df["Tau"])
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax1.legend([line1, line2], ["UR", "RMSE"], loc="best", frameon=False)

    ax1.tick_params(axis="both", direction="in")
    ax2.tick_params(axis="both", direction="in")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
