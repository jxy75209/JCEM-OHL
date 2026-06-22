#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP-NN residual-quantile safety-margin adjustment with full-sample LOOCV.

For each LOOCV fold:
    1. Train the BP-NN baseline model on the training samples.
    2. Compute training residuals e_i = f(x_i) - y_i.
    3. Collect underestimated residual magnitudes E- = {-e_i | e_i < 0}.
    4. Use s_q = quantile(E-, q) as the one-sided safety margin.
    5. Predict the held-out sample with y_safe = f(x) + s_q.

The safety margin is computed from the training set only in each fold.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DEFAULT_INPUT_PATH = r"C:\python code\科研绘图\小论文\朱子豪-OHLs-数据.sav"
DEFAULT_OUTPUT_PATH = r"C:\python code\科研绘图\小论文\残差分位数安全余量调整结果.xlsx"
DEFAULT_FEATURE_COUNT = 15
DEFAULT_Q_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if suffix == ".sav":
        try:
            import pyreadstat
        except ImportError as exc:
            raise ImportError("Reading .sav requires pyreadstat. Install it with: pip install pyreadstat") from exc
        df, _ = pyreadstat.read_sav(str(p))
        return df
    raise ValueError(f"Unsupported file type: {suffix}. Use .sav/.xlsx/.csv.")


def default_target_column(df: pd.DataFrame) -> str:
    if "y" in df.columns:
        return "y"
    if len(df.columns) > DEFAULT_FEATURE_COUNT:
        return str(df.columns[DEFAULT_FEATURE_COUNT])
    return str(df.columns[-1])


def parse_col_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s: Optional[str]) -> List[float]:
    if s is None or str(s).strip() == "":
        return DEFAULT_Q_VALUES
    values = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    invalid = [q for q in values if not 0 < q < 1]
    if invalid:
        raise ValueError(f"q values must be between 0 and 1: {invalid}")
    return values


def prepare_xy(
    df: pd.DataFrame,
    target: Optional[str],
    features: Optional[List[str]],
    id_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    if target is None:
        target = default_target_column(df)
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}\nAvailable columns: {list(df.columns)}")

    if features is None:
        feature_cols = list(df.columns[:DEFAULT_FEATURE_COUNT])
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        feature_cols = features

    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"ID column not found: {id_col}")
        project_id = df[id_col].copy()
    else:
        project_id = pd.Series(np.arange(1, len(df) + 1), name="project_id")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce").copy()
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    project_id = project_id.loc[valid].reset_index(drop=True)
    return X, y, project_id, feature_cols, target


def make_bp_nn(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(12, 6),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                random_state=random_state,
            ),
            transformer=StandardScaler(),
        )),
    ])


def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < y_true) * 100)


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(np.median(abs_err)),
        "R2": float(r2_score(y_true, y_pred)),
        "EVS": float(explained_variance_score(y_true, y_pred)),
        "MAPE_%": mape_safe(y_true, y_pred),
        "UR_%": underestimation_rate(y_true, y_pred),
    }


def residual_quantile_margin(train_y: np.ndarray, train_pred: np.ndarray, q: float) -> float:
    residuals = train_pred - train_y
    under_magnitudes = -residuals[residuals < 0]
    if len(under_magnitudes) == 0:
        return 0.0
    return float(np.quantile(under_magnitudes, q))


def add_prediction_columns(pred_df: pd.DataFrame, model_name: str, preds: np.ndarray) -> None:
    pred_df[f"{model_name}_pred"] = preds
    pred_df[f"{model_name}_error"] = preds - pred_df["y_true"].values
    pred_df[f"{model_name}_abs_error"] = np.abs(pred_df[f"{model_name}_error"])
    pred_df[f"{model_name}_APE_%"] = (
        pred_df[f"{model_name}_abs_error"] / np.maximum(np.abs(pred_df["y_true"].values), 1e-12) * 100
    )
    pred_df[f"{model_name}_underestimated"] = preds < pred_df["y_true"].values


def run_loocv_safety_margin(
    X: pd.DataFrame,
    y: pd.Series,
    project_id: pd.Series,
    q_values: List[float],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_cv = X.reset_index(drop=True)
    y_cv = y.reset_index(drop=True)
    id_cv = project_id.reset_index(drop=True)
    q_values = sorted(q_values)

    pred_df = pd.DataFrame({
        "project_id": id_cv.values,
        "original_row_number": np.arange(1, len(y_cv) + 1),
        "y_true": y_cv.values.astype(float),
    })

    base_preds = np.zeros(len(y_cv), dtype=float)
    safe_preds = {q: np.zeros(len(y_cv), dtype=float) for q in q_values}
    margins = {q: np.zeros(len(y_cv), dtype=float) for q in q_values}
    train_under_counts = np.zeros(len(y_cv), dtype=int)

    loo = LeaveOneOut()
    base_model = make_bp_nn(random_state=random_state)

    print(f"[信息] BP-NN LOOCV 使用全部有效样本数：{len(y_cv)}")
    print(f"[信息] 测试 q values：{q_values}")

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_cv), start=1):
        X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
        y_train = y_cv.iloc[train_idx].values.astype(float)

        est = clone(base_model)
        est.fit(X_train, y_train)

        train_pred = np.ravel(est.predict(X_train)).astype(float)
        test_pred = float(np.ravel(est.predict(X_test))[0])
        base_preds[test_idx[0]] = test_pred

        residuals = train_pred - y_train
        train_under_counts[test_idx[0]] = int(np.sum(residuals < 0))

        for q in q_values:
            margin = residual_quantile_margin(y_train, train_pred, q)
            margins[q][test_idx[0]] = margin
            safe_preds[q][test_idx[0]] = test_pred + margin

        margin_text = ", ".join(f"q={q:.2f}:s={margins[q][test_idx[0]]:.4f}" for q in q_values)
        print(
            f"  fold {fold:02d}/{len(y_cv)}: "
            f"test_id={id_cv.iloc[test_idx[0]]}, y={y_cv.iloc[test_idx[0]]:.4f}, "
            f"base_pred={test_pred:.4f}, {margin_text}"
        )

    pred_df["train_underestimated_residual_count"] = train_under_counts
    add_prediction_columns(pred_df, "BP_NN_base", base_preds)

    metric_rows = []
    base_metrics = compute_metrics(pred_df["y_true"].values, base_preds)
    base_metrics.update({
        "Model": "BP_NN_base",
        "q": np.nan,
        "Safety_margin_mean": 0.0,
        "Safety_margin_median": 0.0,
        "Safety_margin_min": 0.0,
        "Safety_margin_max": 0.0,
    })
    metric_rows.append(base_metrics)

    for q in q_values:
        model_name = f"BP_NN_safe_q{q:.2f}"
        pred_df[f"{model_name}_safety_margin"] = margins[q]
        add_prediction_columns(pred_df, model_name, safe_preds[q])

        metrics = compute_metrics(pred_df["y_true"].values, safe_preds[q])
        metrics.update({
            "Model": model_name,
            "q": q,
            "Safety_margin_mean": float(np.mean(margins[q])),
            "Safety_margin_median": float(np.median(margins[q])),
            "Safety_margin_min": float(np.min(margins[q])),
            "Safety_margin_max": float(np.max(margins[q])),
        })
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df[
        [
            "Model",
            "q",
            "RMSE",
            "MAE",
            "MedAE",
            "R2",
            "EVS",
            "MAPE_%",
            "UR_%",
            "Safety_margin_mean",
            "Safety_margin_median",
            "Safety_margin_min",
            "Safety_margin_max",
        ]
    ]
    return pred_df, metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="BP-NN residual-quantile safety-margin adjustment.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input file: .sav/.xlsx/.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output Excel file")
    parser.add_argument("--target", default=None, help="Target column name; default uses y or the 16th column")
    parser.add_argument("--features", default=None, help="Feature column names separated by commas")
    parser.add_argument("--id-col", default=None, help="Optional project ID column")
    parser.add_argument("--q-values", default=None, help="Comma-separated q values, e.g. 0.5,0.7,0.85,0.9,0.95")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--print-cols", action="store_true", help="Print columns and exit")
    args = parser.parse_args()

    df = read_table(args.input)
    if args.print_cols:
        for i, col in enumerate(df.columns, start=1):
            print(f"{i:02d}. {col} | dtype={df[col].dtype}")
        return

    X, y, project_id, feature_cols, target_col = prepare_xy(
        df=df,
        target=args.target,
        features=parse_col_list(args.features),
        id_col=args.id_col,
    )
    q_values = parse_float_list(args.q_values)

    pred_df, metrics_df = run_loocv_safety_margin(
        X=X,
        y=y,
        project_id=project_id,
        q_values=q_values,
        random_state=args.random_state,
    )

    settings = {
        "input": args.input,
        "target_col": target_col,
        "n_samples": int(len(y)),
        "n_loocv_folds": int(len(pred_df)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "q_values": q_values,
        "random_state": args.random_state,
        "note": "Safety margins are computed from underestimated training residuals within each LOOCV fold.",
    }
    settings_df = pd.DataFrame({
        "key": list(settings.keys()),
        "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for v in settings.values()],
    })

    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pred_df.to_excel(writer, index=False, sheet_name="BP_NN安全余量预测")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        pd.DataFrame({"feature_cols": feature_cols}).to_excel(writer, index=False, sheet_name="Feature_columns")
        settings_df.to_excel(writer, index=False, sheet_name="Settings")

    print("\n完成。输出文件：", out.resolve())
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
