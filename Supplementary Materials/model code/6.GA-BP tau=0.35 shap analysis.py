from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


BASE_DIR = Path(r"C:\python code\科研绘图\小论文")
MODEL_FILE = BASE_DIR / "generate_loo44_predictions0.35.py"
INPUT_FILE = BASE_DIR / "朱子豪-OHLs-数据.sav"

RANDOM_STATE = 42
FEATURE_COUNT = 15
MODEL_NAME = "GA_BP_tau0.35"
BACKGROUND_SIZE = 10
NSAMPLES = 300

SUMMARY_PNG = BASE_DIR / "GA_BP_tau035_SHAP_summary.png"
BAR_PNG = BASE_DIR / "GA_BP_tau035_SHAP_importance.png"
OUTPUT_XLSX = BASE_DIR / "GA_BP_tau035_SHAP_values.xlsx"


def load_prediction_module():
    spec = importlib.util.spec_from_file_location("loo44_model_module", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model file: {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_xy(module):
    df = module.read_table(str(INPUT_FILE))
    feature_cols = list(df.columns[:FEATURE_COUNT])
    target_col = "y" if "y" in df.columns else df.columns[FEATURE_COUNT]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    return X, y, feature_cols, target_col


def make_ga_bp_model(module):
    models = module.make_models(random_state=RANDOM_STATE)
    if MODEL_NAME not in models:
        available = ", ".join(models.keys())
        raise KeyError(f"{MODEL_NAME} was not found. Available models: {available}")
    return models[MODEL_NAME]


def configure_matplotlib():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600


def display_feature_names(feature_cols):
    corrected = {
        "Stie_grading": "Site_grading",
    }
    return [corrected.get(str(col), str(col)).replace("_", " ") for col in feature_cols]


def main():
    configure_matplotlib()

    module = load_prediction_module()
    X, y, feature_cols, target_col = load_xy(module)
    feature_labels = display_feature_names(feature_cols)
    model = make_ga_bp_model(module)

    print(f"[Info] Input file: {INPUT_FILE}")
    print(f"[Info] Samples: {len(X)}, features: {len(feature_cols)}, target: {target_col}")
    print(f"[Info] Fitting model: {MODEL_NAME}")
    model.fit(X, y)

    X_array = X.to_numpy(dtype=float)
    background_size = min(BACKGROUND_SIZE, len(X))
    background = shap.kmeans(X_array, background_size)

    def predict_fn(data):
        data_df = pd.DataFrame(np.asarray(data, dtype=float), columns=feature_cols)
        return model.predict(data_df)

    print(f"[Info] Calculating SHAP values with KernelExplainer, nsamples={NSAMPLES}")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_array, nsamples=NSAMPLES)
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance = (
        pd.DataFrame({"Feature": feature_labels, "MeanAbsSHAP": mean_abs})
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    shap_df = pd.DataFrame(shap_values, columns=feature_labels)
    shap_df.insert(0, "Sample", np.arange(1, len(shap_df) + 1))
    shap_df.insert(1, "y_true", y.to_numpy())
    shap_df.insert(2, "y_pred", model.predict(X))

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        shap_df.to_excel(writer, sheet_name="SHAP_values", index=False)
        importance.to_excel(writer, sheet_name="Mean_abs_SHAP", index=False)
        pd.DataFrame(
            {
                "Item": [
                    "Model",
                    "Input file",
                    "Feature rule",
                    "Ignored columns",
                    "Target",
                    "Background size",
                    "Kernel nsamples",
                ],
                "Value": [
                    MODEL_NAME,
                    str(INPUT_FILE),
                    "First 15 columns",
                    "Last 4 columns",
                    target_col,
                    background_size,
                    NSAMPLES,
                ],
            }
        ).to_excel(writer, sheet_name="Settings", index=False)

    plt.figure(figsize=(8.2, 5.8))
    X_display = X.copy()
    X_display.columns = feature_labels
    shap.summary_plot(
        shap_values,
        X_display,
        feature_names=feature_labels,
        show=False,
        plot_size=None,
        max_display=len(feature_cols),
    )
    ax = plt.gca()
    ax.tick_params(direction="in", length=4, width=1)
    ax.set_xlabel("SHAP value", fontsize=12)
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches="tight")
    plt.close()

    plot_data = importance.sort_values("MeanAbsSHAP", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.barh(
        plot_data["Feature"],
        plot_data["MeanAbsSHAP"],
        color="#4C78A8",
        edgecolor="#263238",
        linewidth=0.6,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(direction="in", length=4, width=1)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(BAR_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] SHAP summary plot: {SUMMARY_PNG}")
    print(f"[Done] SHAP importance plot: {BAR_PNG}")
    print(f"[Done] SHAP values workbook: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
