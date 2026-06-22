#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_loo43_predictions.py

用途：
    对当前数据中的全部 OHLs 项目执行真正的 LOO-CV：
    每一折留出 1 个项目测试，其余项目训练，
    最终导出逐项目真实值、预测值、误差、APE，以及各模型的 RMSE/MAE/MedAE/R2/EVS/MAPE/UR。

重要说明：
    本程序不再删除极端项目，会直接使用输入文件中的全部有效样本进行 LOO-CV。

推荐运行：
    python generate_loo43_predictions.py \
        --input "朱子豪-OHLs-数据.sav" \
        --target "y" \
        --output "上包络集成预测.xlsx"

若不确定列名：
    python generate_loo43_predictions.py --input "朱子豪-OHLs-数据.sav" --print-cols

依赖：
    pip install numpy pandas scikit-learn scipy openpyxl
    如读取 .sav：pip install pyreadstat
    如使用 XGBoost/LightGBM：pip install xgboost lightgbm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


DEFAULT_INPUT_PATH = r"C:\python code\科研绘图\小论文\朱子豪-OHLs-数据.sav"
DEFAULT_OUTPUT_PATH = r"C:\python code\科研绘图\小论文\上包络集成预测.xlsx"
DEFAULT_FEATURE_COUNT = 15


def default_target_column(df: pd.DataFrame) -> str:
    if "y" in df.columns:
        return "y"
    if len(df.columns) > DEFAULT_FEATURE_COUNT:
        return df.columns[DEFAULT_FEATURE_COUNT]
    return df.columns[-1]


# ---------------------------------------------------------------------
# 1. 工具函数：读取数据、识别列
# ---------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    """Read CSV/XLSX/SAV file."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if suffix == ".sav":
        try:
            import pyreadstat
        except ImportError as e:
            raise ImportError(
                "读取 .sav 文件需要 pyreadstat。请先运行：pip install pyreadstat\n"
                "或者将 SPSS 文件另存为 .xlsx/.csv 后再运行。"
            ) from e
        df, meta = pyreadstat.read_sav(str(p))
        return df

    raise ValueError(f"暂不支持的文件格式：{suffix}. 请使用 .sav/.xlsx/.csv")


def guess_target_column(df: pd.DataFrame) -> str:
    """Try to infer target column. If uncertain, raise."""
    candidates = []
    key_words = [
        "造价", "总造价", "工程造价", "项目总造价",
        "cost", "Cost", "COST", "target", "Target", "y"
    ]
    for c in df.columns:
        name = str(c)
        if any(k in name for k in key_words):
            if pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(c)

    if len(candidates) == 1:
        return candidates[0]

    # fallback: choose numeric column with largest max value, but warn strongly
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        raise ValueError("没有找到数值型列，无法识别目标变量。请用 --target 指定。")

    max_col = max(numeric_cols, key=lambda c: np.nanmax(df[c].values))
    print(
        "\n[警告] 未能唯一识别目标变量。程序临时选择最大值最高的数值列作为 target："
        f"{max_col}\n"
        "请核对是否正确；更稳妥的做法是使用 --target 明确指定。\n"
    )
    return max_col


def parse_col_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def prepare_xy(
    df: pd.DataFrame,
    target: Optional[str],
    features: Optional[List[str]],
    drop_cols: Optional[List[str]],
    id_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """Prepare feature matrix and target vector."""
    if target is None:
        target = guess_target_column(df)
    if target not in df.columns:
        raise ValueError(f"target 列不存在：{target}\n可用列：{list(df.columns)}")

    drop_set = set(drop_cols or [])
    drop_set.add(target)

    if id_col and id_col not in df.columns:
        raise ValueError(f"id_col 列不存在：{id_col}")

    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"features 中存在不存在的列：{missing}")
        feature_cols = features
    else:
        # 默认使用所有数值列，排除 target、drop_cols、id_col
        feature_cols = []
        for c in df.columns:
            if c in drop_set:
                continue
            if id_col and c == id_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

    if len(feature_cols) == 0:
        raise ValueError("没有可用特征列。请使用 --features 指定。")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce").copy()

    if id_col:
        project_id = df[id_col].copy()
    else:
        project_id = pd.Series(np.arange(1, len(df) + 1), name="project_id")

    valid = y.notna()
    if valid.sum() < len(df):
        print(f"[提示] 删除 target 缺失的行：{len(df) - int(valid.sum())} 行")
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    project_id = project_id.loc[valid].reset_index(drop=True)

    return X, y, project_id, feature_cols, target


# ---------------------------------------------------------------------
# 2. 指标函数
# ---------------------------------------------------------------------

def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred < y_true))


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
        "UR_%": underestimation_rate(y_true, y_pred) * 100,
    }


def progressive_r2_removal(y_true: np.ndarray, y_pred: np.ndarray, max_remove: int = 7) -> pd.DataFrame:
    """Recompute R2 after progressively removing largest absolute residuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_resid = np.abs(y_true - y_pred)
    order = np.argsort(abs_resid)[::-1]

    rows = []
    n = len(y_true)
    for k in range(0, min(max_remove, n - 2) + 1):
        keep = np.ones(n, dtype=bool)
        if k > 0:
            keep[order[:k]] = False
        rows.append({
            "removed_top_k_residuals": k,
            "remaining_n": int(keep.sum()),
            "R2": float(r2_score(y_true[keep], y_pred[keep])),
            "RMSE": float(math.sqrt(mean_squared_error(y_true[keep], y_pred[keep]))),
            "MAE": float(mean_absolute_error(y_true[keep], y_pred[keep])),
            "MedAE": float(np.median(np.abs(y_true[keep] - y_pred[keep]))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. GA-BP / PSO-BP 简化但可复现实现
# ---------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


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


def unpack_weights(vec: np.ndarray, shape: NetworkShape) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Unpack flat chromosome to sklearn-like coefs_ and intercepts_."""
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
    """
    GA optimises initial weights, then MLPRegressor performs BP fine-tuning.
    Non-asymmetric part is in GA fitness by default.
    """

    def __init__(
        self,
        hidden_layer_sizes=(12, 6),
        tau=0.35,
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

        # Xavier-like init scale
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

        for gen in range(self.generations):
            new_pop = []
            # elitism
            new_pop.append(best.copy())
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

        # Initialise MLP and replace weights with GA best
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


class PSOBPRegressor(BaseEstimator, RegressorMixin):
    """
    PSO optimises initial weights, then MLPRegressor performs BP fine-tuning.
    用于生成 PSO-BP 对照表；若论文不需要，可在 --models 中不选它。
    """

    def __init__(
        self,
        hidden_layer_sizes=(12, 6),
        swarm_size=50,
        iterations=100,
        inertia=0.7,
        c1=1.5,
        c2=1.5,
        bp_max_iter=500,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.bp_max_iter = bp_max_iter
        self.random_state = random_state

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

        pos = rng.normal(0, 0.2, size=(self.swarm_size, n_params))
        vel = rng.normal(0, 0.02, size=(self.swarm_size, n_params))

        def mse_loss(vec):
            pred = forward_pass(X, vec, shape)
            return float(np.mean((pred - y) ** 2))

        pbest = pos.copy()
        pbest_loss = np.array([mse_loss(v) for v in pbest])
        gbest = pbest[int(np.argmin(pbest_loss))].copy()
        gbest_loss = float(np.min(pbest_loss))

        for _ in range(self.iterations):
            r1 = rng.random(size=pos.shape)
            r2 = rng.random(size=pos.shape)
            vel = self.inertia * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = pos + vel

            losses = np.array([mse_loss(v) for v in pos])
            improved = losses < pbest_loss
            pbest[improved] = pos[improved]
            pbest_loss[improved] = losses[improved]

            if np.min(losses) < gbest_loss:
                gbest_loss = float(np.min(losses))
                gbest = pos[int(np.argmin(losses))].copy()

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
        coefs, intercepts = unpack_weights(gbest, shape)
        self.mlp_.coefs_ = [c.copy() for c in coefs]
        self.mlp_.intercepts_ = [b.copy() for b in intercepts]
        self.mlp_.max_iter = self.bp_max_iter
        self.mlp_.fit(X, y)
        return self

    def predict(self, X):
        return self.mlp_.predict(np.asarray(X, dtype=float))


# ---------------------------------------------------------------------
# 4. 模型工厂
# ---------------------------------------------------------------------

def make_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    models: Dict[str, BaseEstimator] = {}

    models["Ridge"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    models["SVR_RBF"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=10.0, gamma=0.003, epsilon=0.1),
            transformer=StandardScaler(),
        )),
    ])

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    models["GBT"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=random_state,
        )),
    ])

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                objective="reg:squarederror",
            )),
        ])
    except Exception:
        print("[提示] 未安装 xgboost，跳过 XGBoost。")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=3,
                random_state=random_state,
                verbose=-1,
            )),
        ])
    except Exception:
        print("[提示] 未安装 lightgbm，跳过 LightGBM。")

    # Neural models: scale X and y.
    bp = Pipeline([
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
    models["BP_NN"] = bp

    pso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=PSOBPRegressor(
                hidden_layer_sizes=(12, 6),
                swarm_size=50,
                iterations=100,
                random_state=random_state,
            ),
            transformer=StandardScaler(),
        )),
    ])
    models["PSO_BP"] = pso

    gabp = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=GABPRegressor(
                hidden_layer_sizes=(12, 6),
                tau=0.35,
                population_size=50,
                generations=100,
                random_state=random_state,
            ),
            transformer=StandardScaler(),
        )),
    ])
    models["GA_BP_tau0.35"] = gabp

    return models


# ---------------------------------------------------------------------
# 5. LOO-CV 主函数
# ---------------------------------------------------------------------

def add_prediction_columns(pred_df: pd.DataFrame, model_name: str, preds: np.ndarray) -> None:
    pred_df[f"{model_name}_pred"] = preds
    pred_df[f"{model_name}_error"] = preds - pred_df["y_true"].values
    pred_df[f"{model_name}_abs_error"] = np.abs(pred_df[f"{model_name}_error"])
    pred_df[f"{model_name}_APE_%"] = (
        pred_df[f"{model_name}_abs_error"] / np.maximum(np.abs(pred_df["y_true"].values), 1e-12) * 100
    )
    pred_df[f"{model_name}_underestimated"] = preds < pred_df["y_true"].values


def add_upper_envelope_ensembles(pred_df: pd.DataFrame, base_model_names: List[str]) -> List[str]:
    ensemble_names = []
    for model_a, model_b in combinations(base_model_names, 2):
        ensemble_name = f"UpperEnvelope_{model_a}__{model_b}"
        preds = np.maximum(
            pred_df[f"{model_a}_pred"].values,
            pred_df[f"{model_b}_pred"].values,
        )
        add_prediction_columns(pred_df, ensemble_name, preds)
        ensemble_names.append(ensemble_name)
    return ensemble_names

def run_loocv(
    X: pd.DataFrame,
    y: pd.Series,
    project_id: pd.Series,
    selected_models: Optional[List[str]],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run LOO-CV on all valid samples in the input data."""
    X_cv = X.reset_index(drop=True)
    y_cv = y.reset_index(drop=True)
    id_cv = project_id.reset_index(drop=True)

    print(f"[信息] LOOCV 使用全部有效样本数：{len(y_cv)}")

    all_models = make_models(random_state=random_state)

    if selected_models:
        missing = [m for m in selected_models if m not in all_models]
        if missing:
            raise ValueError(f"--models 中存在未知模型：{missing}\n可选模型：{list(all_models.keys())}")
        models = {k: all_models[k] for k in selected_models}
    else:
        models = all_models

    pred_df = pd.DataFrame({
        "project_id": id_cv.values,
        "original_row_number": [int(i + 1) for i in range(len(y_cv))],
        "y_true": y_cv.values.astype(float),
    })

    loo = LeaveOneOut()

    for model_name, model in models.items():
        print(f"\n[LOO-CV] 正在运行模型：{model_name}")
        preds = np.zeros(len(y_cv), dtype=float)

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_cv), start=1):
            X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
            y_train = y_cv.iloc[train_idx].values.astype(float)

            est = clone(model)
            est.fit(X_train, y_train)
            pred = est.predict(X_test)
            preds[test_idx[0]] = float(np.ravel(pred)[0])

            print(f"  fold {fold:02d}/{len(y_cv)}: test_id={id_cv.iloc[test_idx[0]]}, y={y_cv.iloc[test_idx[0]]:.4f}, pred={preds[test_idx[0]]:.4f}")

        add_prediction_columns(pred_df, model_name, preds)

    base_model_names = list(models.keys())
    print(f"\n[上包络集成] 正在生成 {len(base_model_names)} 个模型的两两组合。")
    ensemble_names = add_upper_envelope_ensembles(pred_df, base_model_names)
    all_result_names = base_model_names + ensemble_names

    # Metrics
    metric_rows = []
    for model_name in all_result_names:
        y_pred = pred_df[f"{model_name}_pred"].values
        m = compute_metrics(pred_df["y_true"].values, y_pred)
        m["Model"] = model_name
        metric_rows.append(m)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df[["Model", "RMSE", "MAE", "MedAE", "R2", "EVS", "MAPE_%", "UR_%"]]

    # Progressive R2 removal for all models
    prog_rows = []
    for model_name in all_result_names:
        p = progressive_r2_removal(pred_df["y_true"].values, pred_df[f"{model_name}_pred"].values, max_remove=7)
        p.insert(0, "Model", model_name)
        prog_rows.append(p)
    prog_df = pd.concat(prog_rows, ignore_index=True)

    return pred_df, metrics_df, prog_df


# ---------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate full-sample LOO-CV predictions for OHLs cost models.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="输入文件：.sav/.xlsx/.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="输出 Excel 文件")
    parser.add_argument("--target", default=None, help="目标变量列名，例如：项目总造价")
    parser.add_argument("--features", default=None, help="特征列名，逗号分隔；不填则默认使用所有数值列（排除 target/drop/id）")
    parser.add_argument("--drop-cols", default=None, help="额外排除列，逗号分隔")
    parser.add_argument("--id-col", default=None, help="项目编号列名；不填则使用原始行号")
    parser.add_argument("--extreme-index", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--extreme-method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--models", default=None, help="要运行的模型，逗号分隔；不填则运行全部可用模型")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--print-cols", action="store_true", help="只打印列名并退出")
    args = parser.parse_args()

    df = read_table(args.input)

    if args.print_cols:
        print("\n数据列名：")
        for i, c in enumerate(df.columns, start=1):
            print(f"{i:02d}. {c}  | dtype={df[c].dtype}")
        return

    if args.target is None:
        args.target = default_target_column(df)

    features = parse_col_list(args.features)
    if features is None:
        features = list(df.columns[:DEFAULT_FEATURE_COUNT])

    drop_cols = parse_col_list(args.drop_cols)
    selected_models = parse_col_list(args.models)

    X, y, project_id, feature_cols, target_col = prepare_xy(
        df=df,
        target=args.target,
        features=features,
        drop_cols=drop_cols,
        id_col=args.id_col,
    )

    pred_df, metrics_df, prog_df = run_loocv(
        X=X,
        y=y,
        project_id=project_id,
        selected_models=selected_models,
        random_state=args.random_state,
    )

    # Settings sheet
    settings = {
        "input": args.input,
        "target_col": target_col,
        "n_samples": int(len(y)),
        "n_loocv_folds": int(len(pred_df)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "models": list(metrics_df["Model"]),
        "random_state": args.random_state,
        "note": "This file is generated by full-sample LOO-CV plus pairwise upper-envelope ensembles. No extreme project is removed."
    }
    settings_df = pd.DataFrame({
        "key": list(settings.keys()),
        "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for v in settings.values()],
    })

    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pred_df.to_excel(writer, index=False, sheet_name="上包络集成预测")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        prog_df.to_excel(writer, index=False, sheet_name="R2_progressive_removal")
        pd.DataFrame({"feature_cols": feature_cols}).to_excel(writer, index=False, sheet_name="Feature_columns")
        settings_df.to_excel(writer, index=False, sheet_name="Settings")

    print("\n完成。输出文件：", out.resolve())
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
