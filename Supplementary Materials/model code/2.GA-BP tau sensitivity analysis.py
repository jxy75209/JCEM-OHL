#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GA-BP performance under different tau values with full-sample LOOCV.

Data rule:
    - Input file: C:\\python code\\科研绘图\\小论文\\朱子豪-OHLs-数据.sav
    - Features: first 15 columns
    - Target: column named "y" if present, otherwise the 16th column

Outputs:
    - tau敏感性分析结果.xlsx
    - GA_BP_tau_sensitivity_plot.png
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_PATH = Path(r"C:\python code\科研绘图\小论文\朱子豪-OHLs-数据.sav")
RESULT_PATH = Path(r"C:\python code\科研绘图\小论文\tau敏感性分析结果.xlsx")
PLOT_PATH = Path(r"C:\python code\科研绘图\小论文\GA_BP_tau_sensitivity_plot.png")
FEATURE_COUNT = 15
TAU_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


@dataclass
class NetworkShape:
    n_features: int
    hidden1: int = 12
    hidden2: int = 6

    @property
    def sizes(self) -> List[Tuple[int, int]]:
        return [(self.n_features, self.hidden1), (self.hidden1, self.hidden2), (self.hidden2, 1)]

    @property
    def n_params(self) -> int:
        total = 0
        for a, b in self.sizes:
            total += a * b + b
        return total


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def unpack_weights(vec: np.ndarray, shape: NetworkShape) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    coefs = []
    intercepts = []
    pos = 0
    for a, b in shape.sizes:
        w = vec[pos: pos + a * b].reshape(a, b)
        pos += a * b
        bias = vec[pos: pos + b]
        pos += b
        coefs.append(w)
        intercepts.append(bias)
    return coefs, intercepts


def forward_pass(X: np.ndarray, vec: np.ndarray, shape: NetworkShape) -> np.ndarray:
    coefs, intercepts = unpack_weights(vec, shape)
    h1 = relu(X @ coefs[0] + intercepts[0])
    h2 = relu(h1 @ coefs[1] + intercepts[1])
    out = h2 @ coefs[2] + intercepts[2]
    return out.ravel()


class GABPRegressor(BaseEstimator, RegressorMixin):
    """GA optimises initial weights, then MLPRegressor performs BP fine-tuning."""

    def __init__(
        self,
        hidden_layer_sizes=(12, 6),
        tau=0.2,
        population_size=50,
        generations=100,
        crossover_prob=0.7,
        mutation_prob=0.2,
        gene_mutation_prob=0.2,
        mutation_sigma=0.1,
        bp_max_iter=500,
        random_state=42,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tau = tau
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.mutation_sigma = mutation_sigma
        self.bp_max_iter = bp_max_iter
        self.random_state = random_state
        self.verbose = verbose

    def _loss(self, y_true, y_pred):
        err = y_pred - y_true
        penalty = self.tau * np.maximum(err, 0) + (1 - self.tau) * np.maximum(-err, 0)
        return float(np.mean(penalty))

    def _tournament_select(self, pop, losses, rng, k=3):
        idx = rng.choice(len(pop), size=k, replace=False)
        best = idx[np.argmin(losses[idx])]
        return pop[best].copy()

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        shape = NetworkShape(
            n_features=X.shape[1],
            hidden1=int(self.hidden_layer_sizes[0]),
            hidden2=int(self.hidden_layer_sizes[1]),
        )
        n_params = shape.n_params
        pop = rng.normal(0, 0.2, size=(self.population_size, n_params))

        def eval_pop(pop_arr):
            losses = np.empty(pop_arr.shape[0])
            for i in range(pop_arr.shape[0]):
                pred = forward_pass(X, pop_arr[i], shape)
                losses[i] = self._loss(y, pred)
            return losses

        losses = eval_pop(pop)
        best_idx = int(np.argmin(losses))
        best = pop[best_idx].copy()
        best_loss = losses[best_idx]

        for _ in range(self.generations):
            new_pop = [best.copy()]
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(pop, losses, rng)
                p2 = self._tournament_select(pop, losses, rng)

                if rng.random() < self.crossover_prob:
                    a, b = sorted(rng.choice(n_params, size=2, replace=False))
                    c1, c2 = p1.copy(), p2.copy()
                    c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
                else:
                    c1, c2 = p1.copy(), p2.copy()

                for child in (c1, c2):
                    if rng.random() < self.mutation_prob:
                        mask = rng.random(n_params) < self.gene_mutation_prob
                        child[mask] += rng.normal(0, self.mutation_sigma, size=mask.sum())
                    new_pop.append(child)
                    if len(new_pop) >= self.population_size:
                        break

            pop = np.asarray(new_pop)
            losses = eval_pop(pop)
            gen_best_idx = int(np.argmin(losses))
            if losses[gen_best_idx] < best_loss:
                best_loss = float(losses[gen_best_idx])
                best = pop[gen_best_idx].copy()

        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1,
            random_state=self.random_state,
            warm_start=True,
        )
        self.mlp_.fit(X, y)
        coefs, intercepts = unpack_weights(best, shape)
        self.mlp_.coefs_ = [c.copy() for c in coefs]
        self.mlp_.intercepts_ = [b.copy() for b in intercepts]
        self.mlp_.max_iter = self.bp_max_iter
        self.mlp_.fit(X, y)
        return self

    def predict(self, X):
        return self.mlp_.predict(np.asarray(X, dtype=float))


def read_sav(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError("Reading .sav requires pyreadstat. Install it with: pip install pyreadstat") from exc
    df, _ = pyreadstat.read_sav(str(path))
    return df


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < y_true) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(np.median(abs_err)),
        "R2": float(r2_score(y_true, y_pred)),
        "UR_%": underestimation_rate(y_true, y_pred),
        "MAPE_%": mape_safe(y_true, y_pred),
        "EVS": float(explained_variance_score(y_true, y_pred)),
    }


def make_ga_bp_model(tau: float, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=GABPRegressor(
                hidden_layer_sizes=(12, 6),
                tau=tau,
                population_size=50,
                generations=100,
                random_state=random_state,
            ),
            transformer=StandardScaler(),
        )),
    ])


def run_loocv_for_tau(X: pd.DataFrame, y: pd.Series, tau: float, random_state: int = 42) -> Tuple[Dict[str, float], pd.DataFrame]:
    model = make_ga_bp_model(tau=tau, random_state=random_state)
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)

    print(f"\n[GA-BP] tau={tau:.2f}")
    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        est = clone(model)
        est.fit(X.iloc[train_idx], y.iloc[train_idx].values.astype(float))
        pred = est.predict(X.iloc[test_idx])
        preds[test_idx[0]] = float(np.ravel(pred)[0])
        print(f"  fold {fold:02d}/{len(y)}: y={y.iloc[test_idx[0]]:.4f}, pred={preds[test_idx[0]]:.4f}")

    metrics = compute_metrics(y.values.astype(float), preds)
    metrics["Tau"] = tau

    pred_df = pd.DataFrame({
        "tau": tau,
        "original_row_number": np.arange(1, len(y) + 1),
        "y_true": y.values.astype(float),
        "y_pred": preds,
        "error": preds - y.values.astype(float),
        "abs_error": np.abs(preds - y.values.astype(float)),
        "APE_%": np.abs(preds - y.values.astype(float)) / np.maximum(np.abs(y.values.astype(float)), 1e-12) * 100,
        "underestimated": preds < y.values.astype(float),
    })
    return metrics, pred_df


def plot_results(results_df: pd.DataFrame, path: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = read_sav(DATA_PATH)
    feature_cols = list(df.columns[:FEATURE_COUNT])
    target_col = "y" if "y" in df.columns else df.columns[FEATURE_COUNT]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").copy()
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    print(f"Input: {DATA_PATH}")
    print(f"Samples: {len(y)}")
    print(f"Target: {target_col}")
    print(f"Features: {feature_cols}")

    metric_rows = []
    pred_frames = []
    for tau in TAU_VALUES:
        metrics, pred_df = run_loocv_for_tau(X, y, tau=tau)
        metric_rows.append(metrics)
        pred_frames.append(pred_df)

    results_df = pd.DataFrame(metric_rows)
    results_df = results_df[["Tau", "RMSE", "MAE", "MedAE", "R2", "UR_%", "MAPE_%", "EVS"]]
    predictions_df = pd.concat(pred_frames, ignore_index=True)

    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Tau_metrics")
        predictions_df.to_excel(writer, index=False, sheet_name="LOOCV_predictions")
        pd.DataFrame({"feature_cols": feature_cols}).to_excel(writer, index=False, sheet_name="Feature_columns")
        pd.DataFrame({"key": ["input", "target", "samples"], "value": [str(DATA_PATH), target_col, len(y)]}).to_excel(
            writer, index=False, sheet_name="Settings"
        )

    plot_results(results_df, PLOT_PATH)

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics: {RESULT_PATH}")
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
