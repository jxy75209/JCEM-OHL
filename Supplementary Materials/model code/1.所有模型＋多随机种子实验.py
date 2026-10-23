# -*- coding: utf-8 -*-
"""OHL 50随机种子的嵌套评价、OOF安全余量与决策方案选择。

外层LeaveOneOut；内层KFold在外层训练项目中选择参数，重拟合后预测1个项目。
输出单模型、上包络、BP/SVR固定金额余量、SVR比例余量和嵌套决策结果。
同一外层折内，GA-BP固定使用BP选出的BP阶段参数；各τ模型再共享纯GA-BP选出的GA参数。
默认运行随机种子1到50；种子42作为主种子输出逐项目明细，其余种子用于稳健性分析。
嵌套决策使用55个预设候选，不包含PSO-BP、对称GA-BP和不同τ的GA-BP模型。
--profile smoke只用于程序检查，不用于论文结果。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, ParameterSampler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.extmath import safe_sparse_dot
from scipy.stats import wilcoxon

DEFAULT_INPUT = Path(r"C:\python code\科研绘图\小论文\朱子豪-OHLs-数据.sav")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "50随机种子_OOF_55候选嵌套决策结果"
DEFAULT_SEEDS = tuple(range(1, 51))
OUTPUT_NAMES = {
    "single": "111单模型预测结果.xlsx",
    "envelope": "111上包络集成预测结果.xlsx",
    "margin": "111残差分位数安全余量调整预测结果.xlsx",
    "decision": "111嵌套决策结果.xlsx",
    "oof": "111内层OOF预测汇总.xlsx",
    "seeds": "111多随机种子全部指标与汇总.xlsx",
}
DEFAULT_TAU_VALUES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
BASE_MODEL_NAMES = ["Ridge", "SVR_RBF", "RandomForest", "GBT", "XGBoost", "LightGBM",
                    "BP_NN", "PSO_BP", "GA_BP_pure"]


def tau_model_name(tau):
    return f"GA_BP_tau{tau:.2f}"


def asym_bp_model_name(tau):
    return f"Asym-BP_tau{tau:.2f}"


def available_model_names(config):
    return [*BASE_MODEL_NAMES,
            *(asym_bp_model_name(tau) for tau in config.tau_values),
            *(tau_model_name(tau) for tau in config.tau_values)]


def envelope_model_allowed(name):
    """基础模型全部参与；GA-BP和Asym-BP的固定tau模型均仅取0.20。"""
    if name.startswith("GA_BP_tau"):
        return name == "GA_BP_tau0.20"
    if name.startswith("Asym-BP_tau"):
        return name == "Asym-BP_tau0.20"
    return True


@dataclass
class RunConfig:
    seeds: tuple = DEFAULT_SEEDS
    primary_seed: int = 42
    inner_folds: int = 5
    inner_seed: int = 2026
    search_seed: int = 2026
    candidates: int = 4
    tuning_metric: str = "rmse"
    under_weight: float = 4.0
    tau_values: tuple = DEFAULT_TAU_VALUES
    q_values: tuple = (0.50, 0.70, 0.80, 0.90, 0.95)
    profile: str = "standard"
    models: tuple = ()
    allow_missing_models: bool = False
    search_overrides: dict = field(default_factory=dict)
    shap_enabled: bool = True
    shap_nsamples: int = 300


@dataclass
class ModelSpec:
    estimator: object
    search_space: dict


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    os.replace(temporary, path)


def read_data(path, target=None, features=None, id_col=None):
    path = Path(path)
    if path.suffix.lower() == ".sav":
        import pyreadstat
        frame, _ = pyreadstat.read_sav(str(path))
    elif path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("输入仅支持.sav、.xlsx、.xls或.csv")
    if frame.columns.duplicated().any():
        raise ValueError("输入含重复列名")
    target = target or ("y" if "y" in frame else frame.columns[15])
    if target not in frame:
        raise ValueError(f"目标列不存在：{target}")
    feature_cols = list(features) if features else list(frame.columns[:15])
    if target in feature_cols or (id_col is not None and id_col in feature_cols):
        raise ValueError("目标列或ID列进入特征；请用--features明确指定输入列")
    if len(feature_cols) != len(set(feature_cols)) or not feature_cols:
        raise ValueError("特征列为空或重复")
    X = frame[feature_cols].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame[target], errors="coerce")
    ids = frame[id_col].copy() if id_col else pd.Series(np.arange(1, len(frame) + 1))
    row_numbers = np.arange(1, len(frame) + 1)
    valid = y.notna()
    dropped = int((~valid).sum())
    X, y, ids = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), ids.loc[valid].reset_index(drop=True)
    row_numbers = row_numbers[valid]
    if len(y) < 4:
        raise ValueError("至少需要4个有效项目进行嵌套评估")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("project_id必须非空且唯一")
    if not np.isfinite(y).all() or (y <= 0).any() or np.isinf(X.to_numpy()).any():
        raise ValueError("成本必须为正有限值，特征不能含无穷值；特征缺失在训练折内插补")
    return X, y, ids, row_numbers, {"input": str(path.resolve()), "target": str(target),
        "features": feature_cols, "dropped_missing_target": dropped, "n_samples": len(y),
        "feature_missing_counts": X.isna().sum().to_dict()}


# GA/PSO仍采用原脚本的权重搜索和BP微调流程。
# GA新增fitness=mse作为普通GA-BP；asymmetric沿用论文公式。

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


class AsymmetricMLPRegressor(MLPRegressor):
    """与MLPRegressor相同的BP网络，但以固定tau非对称绝对误差反向传播。"""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        tau=0.5,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=None,
    ):
        self.tau = tau
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _backprop(self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads):
        """计算非对称绝对误差及其（在零点取0的）次梯度。"""
        if not 0 < self.tau < 1:
            raise ValueError("tau必须严格位于0和1之间")
        if self.activation != "relu" or self.solver != "adam":
            raise ValueError("AsymmetricMLPRegressor仅支持relu激活和adam优化器")

        n_samples = X.shape[0]
        activations = self._forward_pass(activations)
        error = activations[-1] - y
        element_loss = self.tau * np.maximum(error, 0) + (1 - self.tau) * np.maximum(-error, 0)
        if sample_weight is None:
            sw_sum = n_samples
            loss = float(np.sum(element_loss) / sw_sum)
        else:
            weights = np.asarray(sample_weight).reshape(-1, 1)
            sw_sum = float(np.sum(sample_weight))
            loss = float(np.sum(element_loss * weights) / sw_sum)

        l2_value = sum(float(np.dot(coef.ravel(), coef.ravel())) for coef in self.coefs_)
        loss += 0.5 * self.alpha * l2_value / sw_sum

        last = self.n_layers_ - 2
        deltas[last] = np.where(error > 0, self.tau,
                                np.where(error < 0, -(1 - self.tau), 0.0))
        if sample_weight is not None:
            deltas[last] *= weights
        self._compute_loss_grad(last, sw_sum, activations, deltas, coef_grads, intercept_grads)

        for i in range(last, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            # ReLU在非正区间的导数为0；activations[i]为该层ReLU输出。
            deltas[i - 1][activations[i] <= 0] = 0
            self._compute_loss_grad(i - 1, sw_sum, activations, deltas, coef_grads, intercept_grads)
        return loss, coef_grads, intercept_grads


class GABPRegressor(RegressorMixin, BaseEstimator):
    """
    GA optimises initial weights, then MLPRegressor performs BP fine-tuning.
    tau controls asymmetric absolute-error penalty in GA fitness:
    overestimation is weighted by tau, underestimation by 1 - tau.
    """

    def __init__(
        self,
        hidden_layer_sizes=(12, 6),
        tau=0.3,
        fitness="asymmetric",
        population_size=50,
        generations=100,
        crossover_prob=0.7,
        mutation_prob=0.2,
        gene_mutation_prob=0.2,
        mutation_sigma=0.1,
        bp_max_iter=500,
        learning_rate_init=0.001,
        bp_alpha=0.0001,
        random_state=42,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.tau = tau
        self.fitness = fitness
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.mutation_sigma = mutation_sigma
        self.bp_max_iter = bp_max_iter
        self.learning_rate_init = learning_rate_init
        self.bp_alpha = bp_alpha
        self.random_state = random_state
        self.verbose = verbose

    def _loss(self, y_true, y_pred):
        err = y_pred - y_true
        if self.fitness == "mse":
            return float(np.mean(err ** 2))
        if self.fitness != "asymmetric":
            raise ValueError("fitness must be mse or asymmetric")
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
            learning_rate_init=self.learning_rate_init,
            alpha=self.bp_alpha,
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


class PSOBPRegressor(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        hidden_layer_sizes=(12, 6),
        swarm_size=50,
        iterations=100,
        inertia=0.7,
        c1=1.5,
        c2=1.5,
        bp_max_iter=500,
        learning_rate_init=0.001,
        bp_alpha=0.0001,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.bp_max_iter = bp_max_iter
        self.learning_rate_init = learning_rate_init
        self.bp_alpha = bp_alpha
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
            learning_rate_init=self.learning_rate_init,
            alpha=self.bp_alpha,
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





def scaled_model(regressor, scale_y=True, scale_x=True):
    # 保留特征列名，避免LightGBM训练时有列名、预测时收到无列名数组的警告。
    # 数值插补、特征顺序和模型计算逻辑均不改变。
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.set_output(transform="pandas")
    steps = [("imputer", imputer)]
    if scale_x:
        steps.append(("scaler", StandardScaler()))
    model = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler()) if scale_y else regressor
    return Pipeline(steps + [("model", model)])


def make_specs(seed, config):
    """每个模型都有预先指定的搜索空间；不访问任何项目值或测试结果。"""
    smoke = config.profile == "smoke"
    specs, unavailable = {}, {}
    all_model_names = available_model_names(config)
    requested = set(config.models or all_model_names)
    if "BP_NN" not in requested:
        raise ValueError("三部分评估必须包含BP_NN，第三部分需要复用其预测")
    unknown = requested - set(all_model_names)
    if unknown:
        raise ValueError(f"未知模型：{sorted(unknown)}")

    def put(name, est, space):
        if name in requested:
            specs[name] = ModelSpec(est, space)

    put("Ridge", scaled_model(Ridge(alpha=1.0), scale_y=False),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
    put("SVR_RBF", scaled_model(SVR(C=10, gamma=0.003, epsilon=0.1)),
        {"model__regressor__C": [1.0, 10.0, 100.0], "model__regressor__gamma": [0.003, 0.03, "scale"],
         "model__regressor__epsilon": [0.01, 0.1, 0.2]})
    nt = [8, 12] if smoke else [200, 500]
    put("RandomForest", scaled_model(RandomForestRegressor(n_estimators=nt[-1], min_samples_leaf=2,
          random_state=seed, n_jobs=1), scale_y=False, scale_x=False),
        {"model__n_estimators": nt, "model__min_samples_leaf": [1, 2, 4],
         "model__max_depth": [None, 3, 6], "model__max_features": [0.7, 1.0]})
    ng = [8, 12] if smoke else [100, 300]
    put("GBT", scaled_model(GradientBoostingRegressor(n_estimators=ng[-1], learning_rate=0.03,
          max_depth=3, random_state=seed), scale_y=False, scale_x=False),
        {"model__n_estimators": ng, "model__learning_rate": [0.01, 0.03, 0.1],
         "model__max_depth": [1, 2, 3], "model__min_samples_leaf": [1, 2, 4]})
    if "XGBoost" in requested:
        try:
            from xgboost import XGBRegressor
            put("XGBoost", scaled_model(XGBRegressor(n_estimators=nt[-1], learning_rate=0.03,
                max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=1, objective="reg:squarederror"), scale_y=False, scale_x=False),
                {"model__n_estimators": nt, "model__learning_rate": [0.01, 0.03, 0.1],
                 "model__max_depth": [2, 3, 4], "model__reg_lambda": [1.0, 10.0],
                 "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0]})
        except ImportError as exc:
            unavailable["XGBoost"] = str(exc)
    if "LightGBM" in requested:
        try:
            from lightgbm import LGBMRegressor
            put("LightGBM", scaled_model(LGBMRegressor(n_estimators=nt[-1], learning_rate=0.03,
                num_leaves=15, min_child_samples=3, random_state=seed, n_jobs=1, verbose=-1),
                scale_y=False, scale_x=False),
                {"model__n_estimators": nt, "model__learning_rate": [0.01, 0.03, 0.1],
                 "model__num_leaves": [3, 7, 15], "model__min_child_samples": [2, 3, 5],
                 "model__reg_lambda": [0.0, 1.0, 10.0]})
        except ImportError as exc:
            unavailable["LightGBM"] = str(exc)
    if unavailable and not config.allow_missing_models:
        raise ImportError("缺少模型依赖，停止以避免遗漏两两组合：" + ", ".join(unavailable)
                          + "。请在当前Python环境安装xgboost/lightgbm，或明确使用--allow-missing-models。")

    hidden = [(4, 2), (6, 3)] if smoke else [(6, 3), (12, 6), (16, 8)]
    base_hidden = hidden[0] if smoke else (12, 6)
    epochs = [12, 20] if smoke else [300, 500]
    p = "model__regressor__"
    bp_space = {p + "hidden_layer_sizes": hidden, p + "alpha": [0.0001, 0.01, 0.1],
                p + "learning_rate_init": [0.0005, 0.001, 0.005], p + "max_iter": epochs}
    put("BP_NN", scaled_model(MLPRegressor(hidden_layer_sizes=base_hidden, activation="relu", solver="adam",
        max_iter=epochs[-1], random_state=seed)), bp_space)
    for tau in config.tau_values:
        name = asym_bp_model_name(tau)
        put(name, scaled_model(AsymmetricMLPRegressor(hidden_layer_sizes=base_hidden, tau=tau,
            activation="relu", solver="adam", max_iter=epochs[-1], random_state=seed)), dict(bp_space))
    populations = [4, 6] if smoke else [30, 50]
    generations = [1, 2] if smoke else [50, 100]
    shared = {p + "hidden_layer_sizes": hidden, p + "bp_alpha": [0.0001, 0.01, 0.1],
              p + "learning_rate_init": [0.0005, 0.001, 0.005], p + "bp_max_iter": epochs}
    put("PSO_BP", scaled_model(PSOBPRegressor(hidden_layer_sizes=base_hidden,
        swarm_size=populations[-1], iterations=generations[-1], bp_max_iter=epochs[-1],
        random_state=seed)),
        {**shared, p + "swarm_size": populations, p + "iterations": generations,
         p + "inertia": [0.5, 0.7, 0.9], p + "c1": [1.0, 1.5, 2.0], p + "c2": [1.0, 1.5, 2.0]})
    ga_space = {**shared, p + "population_size": populations, p + "generations": generations,
                p + "crossover_prob": [0.6, 0.7, 0.9], p + "mutation_prob": [0.1, 0.2, 0.3],
                p + "gene_mutation_prob": [0.1, 0.2], p + "mutation_sigma": [0.05, 0.1, 0.2]}
    ga_models = [("GA_BP_pure", "mse", 0.5)] + [
        (tau_model_name(tau), "asymmetric", tau) for tau in config.tau_values]
    for name, fitness, tau in ga_models:
        put(name, scaled_model(GABPRegressor(hidden_layer_sizes=base_hidden, fitness=fitness, tau=tau,
            population_size=populations[-1], generations=generations[-1], bp_max_iter=epochs[-1], random_state=seed)),
            dict(ga_space))
    for name, space in config.search_overrides.items():
        if name not in all_model_names:
            raise ValueError(f"搜索配置含未知模型{name}")
        if name not in specs:
            continue
        space = {k: [tuple(v) if k.endswith("hidden_layer_sizes") else v for v in values]
                 for k, values in space.items()}
        for key, values in space.items():
            if not values or key.endswith(("random_state", "fitness", "tau")):
                raise ValueError("搜索值不可为空；随机种子、适应度和τ均为固定实验设置")
            for value in values:
                clone(specs[name].estimator).set_params(**{key: value})
        specs[name].search_space.update(space)
    return specs, unavailable


def candidate_parameters(name, spec, config, fixed_params=None):
    """保留基准配置，再抽取固定数量的参数候选；fixed_params在所有候选中保持不变。"""
    fixed_params = dict(fixed_params or {})
    unknown = set(fixed_params) - set(spec.search_space)
    if unknown:
        raise ValueError(f"{name}固定参数不在搜索空间中：{sorted(unknown)}")
    base = spec.estimator.get_params(deep=True)
    baseline = {key: base[key] for key in spec.search_space}
    baseline.update(fixed_params)
    # 随机抽样次序与模型随机种子、项目数据无关；可固定参数由本外层折的BP调参结果传入。
    variable_space = {key: values for key, values in spec.search_space.items() if key not in fixed_params}
    space_size = math.prod(len(values) for values in variable_space.values())
    pool = list(ParameterSampler(variable_space, n_iter=min(space_size, max(config.candidates * 3, 8)),
                                 random_state=config.search_seed))
    candidates, seen = [], set()
    for params in [baseline, *pool]:
        params = {**params, **fixed_params}
        serial = json_text(params)
        if serial not in seen:
            seen.add(serial)
            candidates.append(params)
        if len(candidates) == config.candidates:
            break
    if len(candidates) < 2:
        raise ValueError(f"{name}至少需要两个不同候选才能进行调参")
    return candidates


def ga_bp_params_from_bp(bp_params):
    """把本外层折选出的普通BP参数映射为GA-BP的BP阶段参数。"""
    bp_prefix = "model__regressor__"
    mapping = {
        bp_prefix + "hidden_layer_sizes": bp_prefix + "hidden_layer_sizes",
        bp_prefix + "alpha": bp_prefix + "bp_alpha",
        bp_prefix + "learning_rate_init": bp_prefix + "learning_rate_init",
        bp_prefix + "max_iter": bp_prefix + "bp_max_iter",
    }
    missing = set(mapping) - set(bp_params)
    if missing:
        raise RuntimeError(f"BP最优参数不完整，无法传递给GA-BP：{sorted(missing)}")
    return {ga_key: bp_params[bp_key] for bp_key, ga_key in mapping.items()}


def objective(y, pred, config):
    error = np.asarray(pred) - np.asarray(y)
    if config.tuning_metric == "rmse":
        return float(np.sqrt(np.mean(error ** 2)))
    if config.tuning_metric == "mae":
        return float(np.mean(np.abs(error)))
    if config.tuning_metric == "under_cost":
        return float(np.mean(config.under_weight * np.maximum(-error, 0) + np.maximum(error, 0)))
    raise ValueError(f"未知调参指标{config.tuning_metric}")


def fit_logged(estimator, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return len(caught), messages


def tune_and_fit(spec, candidates, X_train, y_train, inner_splits, config):
    """此函数只接收外层训练数据，接口不接收外层测试项目/标签。"""
    records, best_loss, best_params, best_oof = [], math.inf, None, None
    for candidate_id, params in enumerate(candidates):
        oof = np.full(len(y_train), np.nan)
        warn_count, messages = 0, set()
        for fit_idx, valid_idx in inner_splits:
            if np.intersect1d(fit_idx, valid_idx).size:
                raise RuntimeError("内层训练验证索引重叠")
            model = clone(spec.estimator).set_params(**params)
            count, warnings_list = fit_logged(model, X_train.iloc[fit_idx], y_train[fit_idx])
            warn_count += count
            messages.update(warnings_list)
            oof[valid_idx] = model.predict(X_train.iloc[valid_idx])
        if not np.isfinite(oof).all():
            raise RuntimeError("内层预测含非有限值；停止而不是跳过失败候选")
        loss = objective(y_train, oof, config)
        records.append({"Candidate": candidate_id, "Inner_loss": loss, "Parameters": json_text(params),
                        "Warning_count": warn_count, "Warnings": " | ".join(sorted(messages))})
        if loss < best_loss:
            best_loss, best_params, best_oof = loss, params, oof.copy()
    if best_oof is None or len(best_oof) != len(y_train) or not np.isfinite(best_oof).all():
        raise RuntimeError("未生成完整的最佳候选内层折外预测")
    final = clone(spec.estimator).set_params(**best_params)
    count, messages = fit_logged(final, X_train, y_train)
    return final, best_params, best_loss, records, count, messages, best_oof


def crossfit_with_params(spec, params, X_train, y_train, inner_splits):
    """用固定参数生成内层折外预测；每个训练项目恰好由未见过它的模型预测一次。"""
    oof = np.full(len(y_train), np.nan)
    validation_count = np.zeros(len(y_train), dtype=int)
    for fit_idx, valid_idx in inner_splits:
        if np.intersect1d(fit_idx, valid_idx).size:
            raise RuntimeError("内层训练验证索引重叠")
        model = clone(spec.estimator).set_params(**params)
        fit_logged(model, X_train.iloc[fit_idx], y_train[fit_idx])
        oof[valid_idx] = model.predict(X_train.iloc[valid_idx])
        validation_count[valid_idx] += 1
    if not np.all(validation_count == 1):
        raise RuntimeError("每个外层训练项目必须恰好获得一次内层折外预测")
    if not np.isfinite(oof).all():
        raise RuntimeError("固定参数内层折外预测含非有限值")
    return oof


def compute_metrics(y_true, y_pred):
    y, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("预测形状错误或存在非有限值")
    error = pred - y
    absolute = np.abs(error)
    shortfall = np.maximum(-error, 0)
    overage = np.maximum(error, 0)
    under = shortfall > 0
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(absolute)),
        "MedAE": float(np.median(absolute)), "R2": float(r2_score(y, pred)),
        "EVS": float(explained_variance_score(y, pred)), "MAPE_%": float(np.mean(absolute / y) * 100),
        "UR_%": float(np.mean(under) * 100), "Under_count": int(under.sum()),
        "Under_Magnitude": float(np.mean(shortfall[under])) if under.any() else np.nan,
        "Under_Mean_All": float(np.mean(shortfall)), "Cumulative_Underestimation": float(shortfall.sum()),
        "Cumulative_Overestimation": float(overage.sum()),
        "Under_Magnitude_%": float(np.mean(shortfall[under] / y[under]) * 100) if under.any() else np.nan,
        "Cost_Weighted_Under_%": float(shortfall.sum() / y.sum() * 100)}


METRIC_DEFINITIONS = pd.DataFrame([
    ("UR_%", "低估项目数/项目总数×100"),
    ("Under_Magnitude", "低估项目的平均低估金额；无低估时空值"),
    ("Under_Mean_All", "全部项目的平均低估金额，未低估项目计0"),
    ("Cumulative_Underestimation", "全部项目低估金额之和，不与高估抵销"),
    ("Cumulative_Overestimation", "全部项目高估金额之和，即决策函数中的CO，不与低估抵销"),
    ("Under_Magnitude_%", "低估项目的平均相对低估幅度×100；无低估时空值"),
    ("Cost_Weighted_Under_%", "累计低估金额/实际总成本×100"),
    ("金额单位", "与输入目标成本单位一致；原论文数据为万元CNY"),
    ("每项目under_amount", "max(y_true-y_pred,0)"),
    ("每项目cumulative_under_amount", "按输入项目原顺序累计低估金额；不是时间序列累计"),
], columns=["Metric", "Definition"])


def prediction_table(y, ids, row_numbers, predictions):
    columns = {"project_id": ids.to_numpy(), "original_row_number": row_numbers, "y_true": y}
    for name, pred in predictions.items():
        error = pred - y
        shortfall = np.maximum(-error, 0)
        columns.update({f"{name}_pred": pred, f"{name}_error": error, f"{name}_abs_error": np.abs(error),
            f"{name}_APE_%": np.abs(error) / y * 100, f"{name}_underestimated": pred < y,
            f"{name}_under_amount": shortfall, f"{name}_under_ratio_%": shortfall / y * 100,
            f"{name}_cumulative_under_amount": np.cumsum(shortfall)})
    return pd.DataFrame(columns)


def diagnostic_model_allowed(name):
    return (not name.startswith("Asym-BP_tau")
            and (not name.startswith("GA_BP_tau") or name in ("GA_BP_tau0.20", "GA_BP_tau0.50")))


def wilcoxon_model_allowed(name):
    """Wilcoxon检验额外纳入tau=0.20和0.50的Asym-BP。"""
    return (diagnostic_model_allowed(name)
            or name in ("Asym-BP_tau0.20", "Asym-BP_tau0.50"))


def single_model_diagnostics(table, model_names):
    """主种子LOO结果的描述性残差剔除分析和配对绝对误差检验。"""
    y = table["y_true"].to_numpy(float)
    base_error = np.abs(table["BP_NN_pred"].to_numpy(float) - y)
    progressive, comparisons = [], []
    for name in model_names:
        if not wilcoxon_model_allowed(name):
            continue
        pred = table[f"{name}_pred"].to_numpy(float)
        absolute = np.abs(pred - y)
        if diagnostic_model_allowed(name):
            order = np.argsort(-absolute, kind="stable")
            for k in range(min(7, len(y) - 2) + 1):
                keep = order[k:]
                metrics = compute_metrics(y[keep], pred[keep])
                progressive.append({"Model": name, "removed_top_k_residuals": k,
                    "remaining_n": len(keep), **{key: metrics[key] for key in ("R2", "RMSE", "MAE", "MedAE")}})
        if name == "BP_NN":
            continue
        difference = base_error - absolute
        nonzero = int(np.count_nonzero(difference))
        if nonzero:
            result = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        else:
            statistic, pvalue = 0.0, 1.0
        baseline_mae, compared_mae = float(base_error.mean()), float(absolute.mean())
        better = "Tie" if baseline_mae == compared_mae else ("BP_NN" if baseline_mae < compared_mae else name)
        significant = pvalue < 0.05
        interpretation = (f"Significant paired absolute-error difference between BP_NN and {name}; "
                          f"lower MAE: {better}." if significant else
                          f"No significant paired absolute-error difference between BP_NN and {name}.")
        comparisons.append({"Baseline": "BP_NN", "Compared_model": name, "n_pairs": len(y),
            "n_nonzero_differences": nonzero, "BP_NN_MAE": baseline_mae, "Compared_MAE": compared_mae,
            "Mean_abs_error_difference_BP_minus_compared": float(difference.mean()),
            "Median_abs_error_difference_BP_minus_compared": float(np.median(difference)),
            "Wilcoxon_statistic": statistic, "p_value": pvalue, "alpha": 0.05,
            "Significant": significant, "Better_model_by_MAE": better, "Interpretation": interpretation})
    comparison_table = pd.DataFrame(comparisons)
    if not comparison_table.empty:
        raw = comparison_table["p_value"].to_numpy(float)
        order = np.argsort(raw, kind="stable")
        adjusted_sorted = np.maximum.accumulate(
            np.array([(len(raw) - rank) * raw[index] for rank, index in enumerate(order)]))
        adjusted = np.empty(len(raw), dtype=float)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
        comparison_table["p_value_Holm"] = adjusted
        comparison_table["Significant_Holm"] = adjusted < 0.05
        comparison_table["Holm_family_size"] = len(raw)
    return {"R2_progressive_removal": pd.DataFrame(progressive),
            "Wilcoxon_BP_NN_Holm": comparison_table}


def positive_under_residuals(y_true, prediction):
    y_true, prediction = np.asarray(y_true, float), np.asarray(prediction, float)
    return (y_true - prediction)[prediction < y_true]


def fixed_margin(y_true, oof_prediction, q):
    residuals = positive_under_residuals(y_true, oof_prediction)
    return float(np.quantile(residuals, q)) if len(residuals) else 0.0


def proportional_margin(y_true, oof_prediction, q):
    """相对低估比例=(实际值-OOF预测)/OOF预测；仅适用于严格为正的OOF预测。"""
    y_true, oof_prediction = np.asarray(y_true, float), np.asarray(oof_prediction, float)
    valid = (oof_prediction > 0) & (oof_prediction < y_true)
    if not np.any(valid):
        return 0.0
    ratios = (y_true[valid] - oof_prediction[valid]) / oof_prediction[valid]
    return float(np.quantile(ratios, q))


def build_decision_candidates(model_predictions, model_oof, y_train, q_values):
    """构造55个预设候选；不同tau的GA-BP、对称GA-BP和PSO-BP不进入决策层。"""
    traditional = ["Ridge", "SVR_RBF", "RandomForest", "GBT", "XGBoost", "LightGBM"]
    asym_all = [asym_bp_model_name(tau) for tau in DEFAULT_TAU_VALUES]
    required = [*traditional, "BP_NN", *asym_all]
    missing = [name for name in required if name not in model_predictions or name not in model_oof]
    if missing:
        raise RuntimeError(f"嵌套决策缺少预设候选的预测：{missing}")

    outer, inner = {}, {}
    for name in required:
        outer[name], inner[name] = float(model_predictions[name]), np.asarray(model_oof[name], float)
    for a, b in combinations(traditional, 2):
        name = f"MAX_{a}__{b}"
        outer[name] = max(model_predictions[a], model_predictions[b])
        inner[name] = np.maximum(model_oof[a], model_oof[b])
    for base in ("Ridge", "SVR_RBF", "RandomForest"):
        for tau in (0.10, 0.20, 0.30):
            asym = asym_bp_model_name(tau)
            name = f"MAX_{base}__{asym}"
            outer[name] = max(model_predictions[base], model_predictions[asym])
            inner[name] = np.maximum(model_oof[base], model_oof[asym])
    for base in ("BP_NN", "SVR_RBF"):
        for q in q_values:
            margin = fixed_margin(y_train, model_oof[base], q)
            name = f"{base}_OOF_fixed_q{q:.2f}"
            outer[name] = model_predictions[base] + margin
            inner[name] = model_oof[base] + margin
    for q in q_values:
        ratio = proportional_margin(y_train, model_oof["SVR_RBF"], q)
        name = f"SVR_RBF_OOF_ratio_q{q:.2f}"
        outer[name] = model_predictions["SVR_RBF"] * (1 + ratio)
        inner[name] = model_oof["SVR_RBF"] * (1 + ratio)
    if len(outer) != 55 or set(outer) != set(inner):
        raise RuntimeError(f"预设嵌套决策候选应为55个，实际为{len(outer)}个")
    return outer, inner


def choose_decision_candidate(y_true, candidate_oof, k, candidate_order):
    """平局规则：J最小、CU最小、RMSE最小、预设候选顺序最前。"""
    rows = []
    y_true = np.asarray(y_true, float)
    for order, name in enumerate(candidate_order):
        pred = np.asarray(candidate_oof[name], float)
        cu = float(np.maximum(y_true - pred, 0).sum())
        co = float(np.maximum(pred - y_true, 0).sum())
        rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        rows.append({"Candidate": name, "k": k, "J": k * cu + co,
                     "CU": cu, "CO": co, "RMSE": rmse, "Candidate_order": order})
    ranking = pd.DataFrame(rows).sort_values(
        ["J", "CU", "RMSE", "Candidate_order"], kind="stable", ignore_index=True)
    return ranking.iloc[0], ranking


def build_final_decision_candidates(predictions, safety_predictions, q_values):
    """在44个最终外层预测上重建同一组55候选，仅用于标注清楚的事后对照。"""
    traditional = ["Ridge", "SVR_RBF", "RandomForest", "GBT", "XGBoost", "LightGBM"]
    required = [*traditional, "BP_NN", *(asym_bp_model_name(tau) for tau in DEFAULT_TAU_VALUES)]
    candidates = {name: np.asarray(predictions[name], float) for name in required}
    for a, b in combinations(traditional, 2):
        candidates[f"MAX_{a}__{b}"] = np.maximum(predictions[a], predictions[b])
    for base in ("Ridge", "SVR_RBF", "RandomForest"):
        for tau in (0.10, 0.20, 0.30):
            asym = asym_bp_model_name(tau)
            candidates[f"MAX_{base}__{asym}"] = np.maximum(predictions[base], predictions[asym])
    for base in ("BP_NN", "SVR_RBF"):
        for q in q_values:
            name = f"{base}_OOF_fixed_q{q:.2f}"
            candidates[name] = np.asarray(safety_predictions[name], float)
    for q in q_values:
        name = f"SVR_RBF_OOF_ratio_q{q:.2f}"
        candidates[name] = np.asarray(safety_predictions[name], float)
    if len(candidates) != 55:
        raise RuntimeError(f"事后对照候选应为55个，实际为{len(candidates)}个")
    return candidates


def build_candidate_manifest(candidate_names, decision_records):
    """列出所有预设候选，并汇总各k下的嵌套选择次数和事后选择结果。"""
    rows = []
    for order, name in enumerate(candidate_names, 1):
        if name.startswith("SVR_RBF_OOF_ratio_q"):
            category, base = "OOF proportional margin", "SVR_RBF"
            q = float(name.rsplit("q", 1)[1])
        elif "_OOF_fixed_q" in name:
            category, base = "OOF fixed-amount margin", name.split("_OOF_", 1)[0]
            q = float(name.rsplit("q", 1)[1])
        elif name.startswith("MAX_"):
            category, base, q = "Upper-envelope ensemble", name.removeprefix("MAX_"), np.nan
        else:
            category, base, q = "Single model", name, np.nan
        rows.append({"Candidate_order": order, "Candidate": name, "Candidate_type": category,
                     "Base_model_or_pair": base, "q": q,
                     "GA_BP_tau_candidate": name.startswith("GA_BP_tau")})
    manifest = pd.DataFrame(rows)
    records = pd.DataFrame(decision_records)
    nested = records.loc[records["Selection_scope"].eq("nested")]
    posthoc = records.loc[records["Selection_scope"].eq("posthoc")]
    for k in (1, 2, 4, 9):
        counts = nested.loc[nested["k"].eq(k), "Selected_candidate"].value_counts()
        manifest[f"Nested_k{k}_selected_count"] = manifest["Candidate"].map(counts).fillna(0).astype(int)
        selected = set(posthoc.loc[posthoc["k"].eq(k), "Selected_candidate"])
        manifest[f"Posthoc_k{k}_selected"] = manifest["Candidate"].isin(selected)
    return manifest


def build_sections(y, ids, row_numbers, predictions, safety_predictions, safety_metadata,
                   decision_predictions, decision_records):
    single = prediction_table(y, ids, row_numbers, predictions)
    single_metrics = pd.DataFrame([{"Model": name, **compute_metrics(y, pred)} for name, pred in predictions.items()])
    envelope_predictions = {name: pred for name, pred in predictions.items()
                            if envelope_model_allowed(name)}
    ensemble = {f"MAX_{a}__{b}": np.maximum(envelope_predictions[a], envelope_predictions[b])
                for a, b in combinations(envelope_predictions, 2)}
    if len(ensemble) != len(envelope_predictions) * (len(envelope_predictions) - 1) // 2:
        raise RuntimeError("两两组合数量不正确")
    envelope = prediction_table(y, ids, row_numbers, ensemble)
    ensemble_metrics = pd.DataFrame([{"Model": name, **compute_metrics(y, pred)} for name, pred in ensemble.items()])
    ensemble_metrics["Rank_by_RMSE"] = ensemble_metrics["RMSE"].rank(method="min").astype(int)
    ensemble_metrics["Is_champion"] = ensemble_metrics["Rank_by_RMSE"].eq(1)
    safe = {"BP_NN_base": predictions["BP_NN"], "SVR_RBF_base": predictions["SVR_RBF"],
            **safety_predictions}
    safety = prediction_table(y, ids, row_numbers, safe)
    for name, meta in safety_metadata.items():
        suffix = "ratio" if meta["Margin_type"] == "proportional" else "amount"
        safety[f"{name}_margin_{suffix}"] = np.asarray(meta["Margin_values"], float)
    safety_metrics = []
    for name, pred in safe.items():
        meta = safety_metadata.get(name, {"Base_model": name.replace("_base", ""),
                                          "Margin_type": "none", "q": np.nan,
                                          "Margin_values": np.zeros(len(y))})
        margin = np.asarray(meta["Margin_values"], float)
        safety_metrics.append({"Model": name, "Base_model": meta["Base_model"],
            "Margin_type": meta["Margin_type"], "q": meta["q"], **compute_metrics(y, pred),
            "Safety_margin_mean": float(np.mean(margin)), "Safety_margin_median": float(np.median(margin)),
            "Safety_margin_min": float(np.min(margin)), "Safety_margin_max": float(np.max(margin))})
    decision = prediction_table(y, ids, row_numbers, decision_predictions)
    decision_metrics = pd.DataFrame(
        [{"Model": name, **compute_metrics(y, pred)} for name, pred in decision_predictions.items()])
    return {"single": (single, single_metrics), "envelope": (envelope, ensemble_metrics),
            "margin": (safety, pd.DataFrame(safety_metrics)),
            "decision": (decision, decision_metrics)}


def evaluate_seed(X, y, ids, row_numbers, config, seed, checkpoint_dir=None, resume=False, progress=True):
    specs, unavailable = make_specs(seed, config)
    asym_names = [name for name in specs if name.startswith("Asym-BP_tau")]
    tau_names = [name for name in specs if name.startswith("GA_BP_tau")]
    if tau_names and "GA_BP_pure" not in specs:
        raise ValueError("固定τ模型需要同时包含GA_BP_pure，以共享其内层调参结果")
    tunable_names = [name for name in specs if name not in tau_names and name not in asym_names]
    candidates = {name: candidate_parameters(name, specs[name], config) for name in tunable_names}
    y = np.asarray(y, float)
    predictions = {name: np.full(len(y), np.nan) for name in specs}
    safety_predictions = {}
    safety_metadata = {}
    for base in ("BP_NN", "SVR_RBF"):
        for q in config.q_values:
            name = f"{base}_OOF_fixed_q{q:.2f}"
            safety_predictions[name] = np.full(len(y), np.nan)
            safety_metadata[name] = {"Base_model": base, "Margin_type": "fixed_amount",
                                     "q": q, "Margin_values": np.full(len(y), np.nan)}
    for q in config.q_values:
        name = f"SVR_RBF_OOF_ratio_q{q:.2f}"
        safety_predictions[name] = np.full(len(y), np.nan)
        safety_metadata[name] = {"Base_model": "SVR_RBF", "Margin_type": "proportional",
                                 "q": q, "Margin_values": np.full(len(y), np.nan)}
    decision_predictions = {f"Nested_k{k:g}": np.full(len(y), np.nan) for k in (1, 2, 4, 9)}
    decision_records, oof_audit = [], []
    tuning, search, split_audit = [], [], []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X), 1):
        held_out = int(test_idx[0])
        # 以下训练/调参/裕度只读train_idx，y[held_out]仅在最后计算指标。
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
        inner = list(KFold(n_splits=config.inner_folds, shuffle=True, random_state=config.inner_seed).split(X_train))
        inner_fold_id = np.full(len(train_idx), -1, dtype=int)
        for inner_fold, (fit_idx, valid_idx) in enumerate(inner, 1):
            inner_fold_id[valid_idx] = inner_fold
            split_audit.append({"Outer_fold": fold, "Test_project_id": ids.iloc[held_out], "Inner_fold": inner_fold,
                "Inner_fit_ids": json_text(ids.iloc[train_idx[fit_idx]].tolist()),
                "Inner_validation_ids": json_text(ids.iloc[train_idx[valid_idx]].tolist())})
        if not np.all(inner_fold_id > 0):
            raise RuntimeError("内层折未覆盖全部外层训练项目")
        fitted_records = {}
        fold_model_oof = {}
        fold_model_predictions = {}
        for name, spec in specs.items():
            path = Path(checkpoint_dir) / f"seed_{seed}" / f"fold_{fold:03d}_{name}.json" if checkpoint_dir else None
            cached = resume and path is not None and path.exists()
            if cached:
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                if name in asym_names:
                    if "BP_NN" not in fitted_records:
                        raise RuntimeError("Asym-BP必须在BP_NN完成本折调参后运行")
                    params = dict(fitted_records["BP_NN"]["params"])
                    final = clone(spec.estimator).set_params(**params)
                    warn_count, warn_messages = fit_logged(final, X_train, y_train)
                    best_oof = crossfit_with_params(spec, params, X_train, y_train, inner)
                    loss, trials = None, []
                    parameter_source = "shared_from_BP_NN"
                elif name in tau_names:
                    pure = fitted_records["GA_BP_pure"]
                    params = dict(pure["params"])
                    final = clone(spec.estimator).set_params(**params)
                    warn_count, warn_messages = fit_logged(final, X_train, y_train)
                    best_oof = None
                    # JSON检查点禁止NaN；None导入DataFrame后会自然显示为空值。
                    loss, trials = None, []
                    parameter_source = "shared_from_GA_BP_pure"
                elif name == "GA_BP_pure":
                    if "BP_NN" not in fitted_records:
                        raise RuntimeError("GA_BP_pure必须在BP_NN完成本折调参后运行")
                    fixed_bp_params = ga_bp_params_from_bp(fitted_records["BP_NN"]["params"])
                    ga_candidates = candidate_parameters(name, spec, config, fixed_params=fixed_bp_params)
                    final, params, loss, trials, warn_count, warn_messages, best_oof = tune_and_fit(
                        spec, ga_candidates, X_train, y_train, inner, config)
                    parameter_source = "inner_CV_GA_only_with_BP_shared"
                else:
                    final, params, loss, trials, warn_count, warn_messages, best_oof = tune_and_fit(
                        spec, candidates[name], X_train, y_train, inner, config)
                    parameter_source = "inner_CV"
                prediction = float(final.predict(X.iloc[test_idx])[0])
                if not np.isfinite(prediction):
                    raise RuntimeError(f"seed={seed},fold={fold},model={name}输出非有限预测")
                tau = (final.named_steps["model"].regressor_.tau
                       if name in tau_names or name in asym_names else None)
                record = {"prediction": prediction, "params": params, "tau": tau, "inner_loss": loss,
                    "parameter_source": parameter_source,
                    "trials": trials, "warning_count": warn_count, "warnings": warn_messages}
                record["best_oof"] = best_oof.tolist() if best_oof is not None else None
                if path:
                    atomic_json(path, record)
            predictions[name][held_out] = record["prediction"]
            fold_model_predictions[name] = float(record["prediction"])
            fitted_records[name] = record
            if record.get("best_oof") is not None:
                best_oof = np.asarray(record["best_oof"], float)
                if len(best_oof) != len(train_idx) or not np.isfinite(best_oof).all():
                    raise RuntimeError(f"seed={seed},fold={fold},model={name}的OOF不完整")
                fold_model_oof[name] = best_oof
                for local_idx, oof_value in enumerate(best_oof):
                    project_position = train_idx[local_idx]
                    oof_audit.append({"Seed": seed, "Outer_fold": fold,
                        "Outer_test_project": ids.iloc[held_out], "Model": name,
                        "Inner_fold": int(inner_fold_id[local_idx]),
                        "Project_ID": ids.iloc[project_position],
                        "Original_row_number": int(row_numbers[project_position]),
                        "y_true": float(y_train[local_idx]), "OOF_prediction": float(oof_value),
                        "Residual_y_minus_pred": float(y_train[local_idx] - oof_value),
                        "Under_residual": float(max(y_train[local_idx] - oof_value, 0)),
                        "Selected_parameters": json_text(record["params"])})
            tuning.append({"Seed": seed, "Outer_fold": fold, "Test_project_id": ids.iloc[held_out],
                "Train_n": len(train_idx), "Model": name, "Tuning_metric": config.tuning_metric,
                "Inner_loss": record["inner_loss"], "Selected_tau": record["tau"],
                "Parameter_source": record.get("parameter_source", "inner_CV"),
                "Best_parameters": json_text(record["params"]), "Refit_warning_count": record["warning_count"],
                "Refit_warnings": " | ".join(record["warnings"])})
            search.extend({"Seed": seed, "Outer_fold": fold, "Model": name, **trial} for trial in record["trials"])
            if progress:
                # 继承其他模型参数、未独立进行内层搜索的模型没有inner_loss。
                loss_text = ("shared" if record["inner_loss"] is None
                             else f"{record['inner_loss']:.6g}")
                print(f"seed={seed} fold={fold}/{len(y)} {name} inner_{config.tuning_metric}={loss_text}"
                      + (" [缓存]" if cached else ""), flush=True)

        for base in ("BP_NN", "SVR_RBF"):
            if base not in fold_model_oof:
                raise RuntimeError(f"安全余量缺少{base}的内层OOF预测")
            for q in config.q_values:
                margin = fixed_margin(y_train, fold_model_oof[base], q)
                safe_name = f"{base}_OOF_fixed_q{q:.2f}"
                safety_predictions[safe_name][held_out] = fold_model_predictions[base] + margin
                safety_metadata[safe_name]["Margin_values"][held_out] = margin
        if np.any(fold_model_oof["SVR_RBF"] <= 0):
            raise RuntimeError("SVR比例余量要求全部内层OOF预测严格为正")
        for q in config.q_values:
            ratio = proportional_margin(y_train, fold_model_oof["SVR_RBF"], q)
            safe_name = f"SVR_RBF_OOF_ratio_q{q:.2f}"
            safety_predictions[safe_name][held_out] = fold_model_predictions["SVR_RBF"] * (1 + ratio)
            safety_metadata[safe_name]["Margin_values"][held_out] = ratio

        candidate_outer, candidate_oof = build_decision_candidates(
            fold_model_predictions, fold_model_oof, y_train, config.q_values)
        candidate_order = list(candidate_outer)
        for k in (1, 2, 4, 9):
            selected, ranking = choose_decision_candidate(y_train, candidate_oof, k, candidate_order)
            selected_name = str(selected["Candidate"])
            decision_predictions[f"Nested_k{k:g}"][held_out] = candidate_outer[selected_name]
            decision_records.append({"Selection_scope": "nested", "Seed": seed, "Outer_fold": fold,
                "Outer_test_project": ids.iloc[held_out], "k": k,
                "Selected_candidate": selected_name, "Inner_J": float(selected["J"]),
                "Inner_CU": float(selected["CU"]), "Inner_CO": float(selected["CO"]),
                "Inner_RMSE": float(selected["RMSE"]),
                "Outer_prediction": float(candidate_outer[selected_name]),
                "Candidate_count": len(ranking)})

    for name, values in {**safety_predictions, **decision_predictions}.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"结果{name}存在缺失或非有限值")
    posthoc_candidates = build_final_decision_candidates(predictions, safety_predictions, config.q_values)
    posthoc_order = list(posthoc_candidates)
    for k in (1, 2, 4, 9):
        selected, _ = choose_decision_candidate(y, posthoc_candidates, k, posthoc_order)
        selected_name = str(selected["Candidate"])
        decision_predictions[f"Posthoc_k{k:g}"] = posthoc_candidates[selected_name]
        decision_records.append({"Selection_scope": "posthoc", "Seed": seed, "Outer_fold": "ALL",
            "Outer_test_project": "ALL", "k": k, "Selected_candidate": selected_name,
            "Inner_J": float(selected["J"]), "Inner_CU": float(selected["CU"]),
            "Inner_CO": float(selected["CO"]), "Inner_RMSE": float(selected["RMSE"]),
                "Outer_prediction": np.nan, "Candidate_count": 55})

    candidate_manifest = build_candidate_manifest(posthoc_order, decision_records)
    sections = build_sections(y, ids, row_numbers, predictions, safety_predictions, safety_metadata,
                              decision_predictions, decision_records)
    return (sections, pd.DataFrame(tuning), pd.DataFrame(search), pd.DataFrame(split_audit),
            pd.DataFrame(oof_audit), pd.DataFrame(decision_records), candidate_manifest, unavailable)


def settings_table(settings):
    return pd.DataFrame({"key": list(settings), "value": [json_text(v) if isinstance(v, (list, tuple, dict)) else v
                                                        for v in settings.values()]})


def summarize_across_seeds(all_metrics):
    """按结果部分和模型汇总跨随机种子的中心、离散程度及95%经验区间。"""
    metric_columns = [
        "RMSE", "MAE", "MedAE", "R2", "EVS", "MAPE_%", "UR_%", "Under_count",
        "Under_Magnitude", "Under_Mean_All", "Cumulative_Underestimation",
        "Cumulative_Overestimation", "Under_Magnitude_%", "Cost_Weighted_Under_%",
    ]
    available = [name for name in metric_columns if name in all_metrics.columns]
    rows = []
    for (section, model), group in all_metrics.groupby(["Section", "Model"], sort=False):
        for metric in available:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append({
                "Section": section, "Model": model, "Metric": metric,
                "Seed_count": int(group["Seed"].nunique()), "Valid_count": int(len(values)),
                "Mean": float(np.mean(values)),
                "Std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "Median": float(np.median(values)), "Min": float(np.min(values)),
                "Q2.5%": float(np.quantile(values, 0.025)),
                "Q97.5%": float(np.quantile(values, 0.975)), "Max": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def rank_models_across_seeds(all_metrics):
    """给出各部分模型的跨种子均值排序；RMSE/UR/CU/CO均按越小越前。"""
    ranking_metrics = ["RMSE", "UR_%", "Cumulative_Underestimation", "Cumulative_Overestimation"]
    rows = []
    for section, section_rows in all_metrics.groupby("Section", sort=False):
        for metric in ranking_metrics:
            if metric not in section_rows:
                continue
            table = (section_rows.groupby("Model", sort=False)[metric]
                     .agg(Mean="mean", Std="std", Median="median", Min="min", Max="max",
                          Seed_count="count")
                     .reset_index())
            table["Rank_by_mean"] = table["Mean"].rank(method="min", ascending=True).astype(int)
            table.insert(0, "Metric", metric)
            table.insert(0, "Section", section)
            rows.append(table.sort_values(["Rank_by_mean", "Model"], kind="stable"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_decision_selection(all_decision_records, n_projects):
    """统计50个种子中嵌套逐折选择和事后选择的候选稳定性。"""
    if all_decision_records.empty:
        return pd.DataFrame(), pd.DataFrame()
    nested = all_decision_records.loc[
        all_decision_records["Selection_scope"].eq("nested")].copy()
    nested_summary = (nested.groupby(["k", "Selected_candidate"], as_index=False)
                      .agg(Selected_count=("Selected_candidate", "size"),
                           Seed_count_selected=("Seed", "nunique")))
    seed_count = int(all_decision_records["Seed"].nunique())
    nested_summary["Possible_fold_selections"] = seed_count * n_projects
    nested_summary["Selected_proportion"] = (
        nested_summary["Selected_count"] / nested_summary["Possible_fold_selections"])
    nested_summary = nested_summary.sort_values(
        ["k", "Selected_count", "Selected_candidate"], ascending=[True, False, True], kind="stable")

    posthoc = all_decision_records.loc[
        all_decision_records["Selection_scope"].eq("posthoc")].copy()
    posthoc_summary = (posthoc.groupby(["k", "Selected_candidate"], as_index=False)
                       .agg(Selected_seed_count=("Seed", "nunique")))
    posthoc_summary["Total_seed_count"] = seed_count
    posthoc_summary["Selected_seed_proportion"] = (
        posthoc_summary["Selected_seed_count"] / posthoc_summary["Total_seed_count"])
    posthoc_summary = posthoc_summary.sort_values(
        ["k", "Selected_seed_count", "Selected_candidate"],
        ascending=[True, False, True], kind="stable")
    return nested_summary, posthoc_summary


def save_workbook(path, sheets):
    """原pandas/openpyxl结果导出形式；临时文件写完后原子替换。"""
    path = Path(path)
    temporary = path.with_name(path.stem + ".writing.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if len(frame) > 1048575 or len(frame.columns) > 16384:
                raise ValueError(f"{name}超过Excel工作表容量；请减少自定义搜索范围")
            frame.to_excel(writer, sheet_name=name, index=False)
            writer.sheets[name].freeze_panes = "A2"
    os.replace(temporary, path)


def run_shap_analysis(X, y, ids, row_numbers, config, output_dir):
    """全数据重拟合模型的解释；不作为LOO性能估计或测试集SHAP。"""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("SHAP：主种子全数据调参并重训练BP_NN。", flush=True)
    specs, _ = make_specs(config.primary_seed, config)
    inner = list(KFold(n_splits=config.inner_folds, shuffle=True,
                       random_state=config.inner_seed).split(X))
    bp = specs["BP_NN"]
    model, bp_params, bp_loss, _, warning_count, warning_messages, _ = tune_and_fit(
        bp, candidate_parameters("BP_NN", bp, config), X, y, inner, config)

    # 所有项目都参与解释和背景分布；缺失特征使用最终模型的训练中位数插补。
    explain_X = np.asarray(model.named_steps["imputer"].transform(X), dtype=float)
    features = list(X.columns)

    def predict_fn(values):
        return model.predict(pd.DataFrame(np.asarray(values, float), columns=features))

    state = np.random.get_state()
    try:
        np.random.seed(config.primary_seed)
        explainer = shap.KernelExplainer(predict_fn, explain_X)
        values = np.asarray(explainer.shap_values(
            explain_X, nsamples=config.shap_nsamples, l1_reg=0.0), dtype=float)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.shape != explain_X.shape or not np.isfinite(values).all():
            raise RuntimeError("SHAP输出维度错误或包含非有限值")
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        predictions = predict_fn(explain_X)
        reconstructed = expected + values.sum(axis=1)
        if not np.allclose(reconstructed, predictions, rtol=1e-5, atol=1e-5):
            raise RuntimeError("SHAP加和与模型预测不一致")
        folder = Path(output_dir) / "SHAP_BP_NN"
        folder.mkdir(parents=True, exist_ok=True)
        importance = pd.DataFrame({"Feature": features, "Mean_abs_SHAP": np.abs(values).mean(axis=0)})
        importance = importance.sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)
        shap_table = pd.DataFrame(values, columns=[f"SHAP_{f}" for f in features])
        shap_table.insert(0, "project_id", ids.to_numpy())
        shap_table.insert(1, "original_row_number", row_numbers)
        shap_table.insert(2, "y_true", y)
        shap_table.insert(3, "full_fit_prediction", predictions)
        shap_table.insert(4, "expected_value", expected)
        shap_table.insert(5, "SHAP_reconstructed_prediction", reconstructed)
        metadata = {"model": "BP_NN", "seed": config.primary_seed,
            "smoke_only": config.profile == "smoke",
            "scope": "full-data refit explanation, not held-out LOO predictions",
            "background": "all training projects, median imputation from final model",
            "n_explained": len(X), "nsamples": config.shap_nsamples,
            "units": "original target cost units", "BP_parameters": bp_params,
            "BP_inner_loss": bp_loss,
            "refit_warning_count": warning_count, "refit_warnings": warning_messages}
        save_workbook(folder / "BP_NN_SHAP_values.xlsx", {
            "SHAP_values": shap_table, "Mean_abs_SHAP": importance,
            "Explained_features": pd.DataFrame(explain_X, columns=features).assign(project_id=ids.to_numpy()),
            "Settings": settings_table(metadata)})
        with plt.rc_context({"font.family": "Times New Roman", "font.size": 11,
                             "xtick.direction": "in", "ytick.direction": "in"}):
            shap.summary_plot(values, explain_X, feature_names=features,
                              max_display=len(features), show=False, plot_size=(10, 8))
            fig = plt.gcf()
            fig.axes[0].set_title("BP-NN: SHAP contributions")
            fig.axes[0].set_xlabel("SHAP value (cost units)")
            fig.savefig(folder / "BP_NN_SHAP_summary.jpg", dpi=600,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)
            ordered = importance.iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
            ax.barh(ordered["Feature"], ordered["Mean_abs_SHAP"], color="#517EA5")
            ax.set_xlabel("Mean absolute SHAP value (cost units)")
            ax.set_title("BP-NN: feature importance")
            ax.tick_params(direction="in", top=False, right=False)
            fig.savefig(folder / "BP_NN_SHAP_importance.jpg", dpi=600,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)
    finally:
        np.random.set_state(state)
    print(f"SHAP分析已保存：{folder.resolve()}", flush=True)


def package_versions():
    versions = {"python": sys.version.split()[0]}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "openpyxl", "pyreadstat", "xgboost", "lightgbm", "shap", "matplotlib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def validate_config(config, n_samples):
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("随机种子须非空且不能重复")
    if config.primary_seed not in config.seeds:
        raise ValueError("primary-seed必须在seeds中")
    if any(not 0 <= int(s) < 2**32 for s in (*config.seeds, config.inner_seed, config.search_seed)):
        raise ValueError("种子必须在0到2**32-1之间")
    if not 2 <= config.inner_folds <= n_samples - 1:
        raise ValueError("内层折数须介于2与外层训练项目数之间")
    if config.candidates < 2:
        raise ValueError("每个模型至少两个候选参数配置")
    for name, values in [("q", config.q_values), ("tau", config.tau_values)]:
        if not values or any(not 0 < x < 1 for x in values) or len(set(values)) != len(values):
            raise ValueError(f"{name}必须为0到1之间不重复的数值")
        if any(abs(v - round(v, 2)) > 1e-9 for v in values):
            raise ValueError(f"{name}保留至多两位小数，避免输出模型名冲突")
    if len(config.tau_values) < 2:
        raise ValueError("τ敏感性分析至少需要两个固定τ值")
    if tuple(config.q_values) != (0.50, 0.70, 0.80, 0.90, 0.95):
        raise ValueError("55候选嵌套决策固定使用q={0.50,0.70,0.80,0.90,0.95}")
    if config.under_weight <= 0:
        raise ValueError("低估成本权重必须为正")
    if config.shap_nsamples < 1:
        raise ValueError("SHAP采样次数必须为正")


def run_experiment(X, y, ids, row_numbers, config, output_dir, data_info=None, resume=False, progress=True):
    validate_config(config, len(y))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, unavailable = make_specs(config.primary_seed, config)
    if config.shap_enabled:
        if "BP_NN" not in specs:
            raise ValueError("SHAP需要BP_NN；子集运行可使用--skip-shap")
        # 长时间训练开始前检查解释依赖，避免训练结束后才发现缺包。
        import shap
        import matplotlib
    counts = {name: (0 if name.startswith(("GA_BP_tau", "Asym-BP_tau"))
                     else len(candidate_parameters(name, spec, config)))
              for name, spec in specs.items()}
    versions = package_versions()
    data_digest = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()
        + pd.util.hash_pandas_object(pd.Series(y), index=True).values.tobytes()
        + json_text(ids.tolist()).encode() + json_text(list(X.columns)).encode()
        + np.asarray(row_numbers).tobytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json_text({"data": data_digest, "config": asdict(config),
        "code": code_hash, "versions": versions}).encode()).hexdigest()[:20]
    checkpoint_dir = output_dir / "checkpoints" / fingerprint
    settings = {**(data_info or {}), **asdict(config), "data_sha256": data_digest, "code_sha256": code_hash,
        "run_id": fingerprint, "versions": versions, "available_models": list(specs),
        "unavailable_models": unavailable, "candidate_counts": counts,
        "outer_protocol": "LeaveOneOut; all selection confined to outer training projects",
        "inner_selection": "pooled inner out-of-fold RMSE (default), original cost scale; tau is not selected",
        "ga_bp_control_protocol": "within each outer fold all GA-BP models inherit BP_NN hidden layers, BP learning rate, L2 alpha and BP iterations; GA_BP_pure tunes GA-only parameters",
        "tau_protocol": "fixed tau models share each outer fold's BP settings, GA settings and initialization seed from GA_BP_pure; only fitness tau differs",
        "asym_bp_protocol": "each fixed-tau Asym-BP inherits the outer fold's BP_NN hyperparameters and initialization seed; only the BP loss and its gradient differ",
        "calibration": "positive inner out-of-fold underestimation residuals within each outer training fold; empirical only",
        "fixed_margin_models": ["BP_NN", "SVR_RBF"],
        "proportional_margin_definition": "SVR only: (y-oof)/oof for underestimated samples with oof>0",
        "decision_candidate_count": 55,
        "decision_candidate_exclusions": ["PSO_BP", "GA_BP_pure", "all GA_BP_tau models"],
        "decision_k_values": [1, 2, 4, 9],
        "decision_tie_break": "minimum J, then CU, then RMSE, then prespecified candidate order",
        "primary_tables": "primary_seed only; not selected by test performance",
        "diagnostic_models": [name for name in specs if diagnostic_model_allowed(name)],
        "wilcoxon_models": [name for name in specs if wilcoxon_model_allowed(name)],
        "progressive_removal_protocol": "remove 0 through 7 largest absolute LOO residuals per model; descriptive sensitivity only; no refitting",
        "wilcoxon_protocol": "paired project absolute LOO errors vs BP_NN; two-sided; Holm familywise correction",
        "shap_protocol": "primary seed full-data CV selection then refit BP_NN; all projects explained, not held-out predictions",
        "envelope_models": [name for name in specs if envelope_model_allowed(name)],
        "envelope_excluded_models": [name for name in specs if not envelope_model_allowed(name)],
        "envelope_champion_selection": "post-hoc minimum RMSE on final outer-LOO predictions; descriptive winner",
        "seed_protocol": "every model with a stochastic training mechanism uses the current evaluation seed; deterministic Ridge and SVR have no random_state parameter",
        "multi_seed_analysis": "seeds 1-50; report every seed, mean, sample SD, median, min, max and empirical 2.5%-97.5% interval; rank by mean",
        "smoke_only": config.profile == "smoke"}
    atomic_json(checkpoint_dir / "run_settings.json", settings)
    asym_crossfits = sum(name.startswith("Asym-BP_tau") for name in specs) * config.inner_folds
    fits = len(config.seeds) * len(y) * (
        config.inner_folds * sum(counts.values()) + asym_crossfits + len(specs))
    envelope_n = len([name for name in specs if envelope_model_allowed(name)])
    print(f"{len(y)}项目；{len(specs)}单模型；{envelope_n*(envelope_n-1)//2}个max组合；"
          f"{len(config.q_values)}个q；{len(config.seeds)}个种子；计划约{fits}次模型拟合。", flush=True)
    if unavailable:
        print("明确允许跳过的缺失模型：" + ", ".join(unavailable), flush=True)
    print(f"结果目录：{output_dir.resolve()}；主结果种子：{config.primary_seed}", flush=True)
    all_metrics = []
    all_decision_records = []
    # 先运行主种子42并写出逐项目明细，再完成其余49个种子的稳健性分析。
    ordered_seeds = [config.primary_seed, *(s for s in config.seeds if s != config.primary_seed)]
    for seed in ordered_seeds:
        sections, tuning, search, splits, oof_audit, decision_records, candidate_manifest, _ = evaluate_seed(X, y, ids, row_numbers, config, seed,
            checkpoint_dir=checkpoint_dir, resume=resume, progress=progress)
        all_decision_records.append(decision_records)
        for section, (_, metrics) in sections.items():
            rows = metrics.copy()
            rows.insert(0, "Seed", seed)
            rows.insert(0, "Section", section)
            all_metrics.append(rows)
        if seed == config.primary_seed:
            common = {"Feature_columns": pd.DataFrame({"feature_cols": X.columns}),
                      "Metric_definitions": METRIC_DEFINITIONS, "Settings": settings_table(settings)}
            for section, (predictions, metrics) in sections.items():
                sheet = {"single": "LOO_predictions", "envelope": "上包络集成预测",
                         "margin": "OOF安全余量预测", "decision": "嵌套决策预测"}[section]
                sheets = {sheet: predictions, "Metrics": metrics}
                if section == "single":
                    sheets.update(single_model_diagnostics(predictions, metrics["Model"]))
                elif section == "margin":
                    sheets["BP_tuning_details"] = tuning.loc[tuning["Model"] == "BP_NN"]
                elif section == "envelope":
                    sheets["Champion"] = metrics.loc[metrics["Is_champion"]].copy()
                elif section == "decision":
                    sheets["Selection_records"] = decision_records
                    sheets["Candidate_manifest"] = candidate_manifest
                    nested_records = decision_records.loc[
                        decision_records["Selection_scope"].eq("nested")].copy()
                    stability = (nested_records.groupby(["k", "Selected_candidate"], as_index=False)
                                 .size().rename(columns={"size": "Selected_count"}))
                    stability["Selected_proportion"] = stability["Selected_count"] / len(y)
                    sheets["Selection_stability"] = stability
                sheets.update(common)
                save_workbook(output_dir / OUTPUT_NAMES[section], sheets)
            save_workbook(output_dir / OUTPUT_NAMES["oof"], {
                "Inner_OOF_predictions": oof_audit,
                "Split_audit": splits,
                "Tuning_details": tuning,
                "Settings": settings_table(settings)})
        aggregate = pd.concat(all_metrics, ignore_index=True)
        decision_aggregate = pd.concat(all_decision_records, ignore_index=True)
        nested_stability, posthoc_stability = summarize_decision_selection(decision_aggregate, len(y))
        save_workbook(output_dir / OUTPUT_NAMES["seeds"], {
            "All_seed_metrics": aggregate,
            "Across_seed_summary": summarize_across_seeds(aggregate),
            "Across_seed_ranking": rank_models_across_seeds(aggregate),
            "Nested_selection_stability": nested_stability,
            "Posthoc_selection_stability": posthoc_stability,
            "Settings": settings_table(settings),
        })
        atomic_json(checkpoint_dir / "completed_seeds.json", ordered_seeds[:ordered_seeds.index(seed) + 1])
        print(f"种子{seed}完成，已保存{len(aggregate)}行多种子指标。", flush=True)
    if config.shap_enabled:
        run_shap_analysis(X, np.asarray(y, float), ids, row_numbers, config, output_dir)
    return pd.concat(all_metrics, ignore_index=True)


def comma_values(text, cast):
    return tuple(cast(v.strip()) for v in text.split(",") if v.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description="OHL 50随机种子评估：内层OOF＋安全余量＋55候选嵌套决策")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", default=None)
    parser.add_argument("--features", default=None, help="逗号分隔列名，省略则沿用前15列")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-seed", type=int, default=2026)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--candidates", type=int, default=4, help="每个可调模型的结构参数候选数；固定τ模型共享纯GA-BP参数")
    parser.add_argument("--tuning-metric", choices=["rmse", "mae", "under_cost"], default="rmse")
    parser.add_argument("--under-weight", type=float, default=4.0)
    parser.add_argument("--tau-values", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--q-values", default="0.50,0.70,0.80,0.90,0.95")
    parser.add_argument("--models", default=None, help="默认全部基础模型和各固定τ模型；自定义必须包含BP_NN")
    parser.add_argument("--allow-missing-models", action="store_true", help="明确允许跳过未安装的XGBoost/LightGBM并记录")
    parser.add_argument("--search-config", type=Path, help="JSON覆盖各模型搜索空间，键使用Pipeline参数名")
    parser.add_argument("--profile", choices=["standard", "smoke"], default="standard")
    parser.add_argument("--resume", action="store_true", help="仅复用相同数据/代码/依赖/配置指纹的已完成折")
    parser.add_argument("--quiet", action="store_true", help="不打印每折每模型进度")
    parser.add_argument("--dry-run", action="store_true", help="核查数据/依赖并显示拟合次数，不训练")
    parser.add_argument("--skip-shap", action="store_true", help="跳过预测完成后的全数据SHAP分析")
    parser.add_argument("--shap-nsamples", type=int, default=300, help="Kernel SHAP每个项目的采样预算，默认300")
    args = parser.parse_args(argv)
    config = RunConfig(seeds=comma_values(args.seeds, int), primary_seed=args.primary_seed,
        inner_folds=args.inner_folds, inner_seed=args.inner_seed, search_seed=args.search_seed,
        candidates=args.candidates, tuning_metric=args.tuning_metric, under_weight=args.under_weight,
        tau_values=comma_values(args.tau_values, float), q_values=tuple(sorted(comma_values(args.q_values, float))),
        models=comma_values(args.models, str) if args.models else (), profile=args.profile,
        allow_missing_models=args.allow_missing_models,
        shap_enabled=not args.skip_shap, shap_nsamples=args.shap_nsamples,
        search_overrides=json.loads(args.search_config.read_text(encoding="utf-8-sig")) if args.search_config else {})
    X, y, ids, rows, data_info = read_data(args.input, args.target,
        comma_values(args.features, str) if args.features else None, args.id_col)
    validate_config(config, len(y))
    if args.dry_run:
        specs, unavailable = make_specs(config.primary_seed, config)
        counts = {name: (0 if name.startswith(("GA_BP_tau", "Asym-BP_tau"))
                         else len(candidate_parameters(name, spec, config)))
                  for name, spec in specs.items()}
        print(json_text({"data": data_info, "config": asdict(config), "candidate_counts": counts,
            "unavailable": unavailable,
            "pair_count": sum(envelope_model_allowed(name) for name in specs)
                          * (sum(envelope_model_allowed(name) for name in specs) - 1) // 2,
            "planned_fits": len(y)*len(config.seeds)*(
                config.inner_folds*sum(counts.values())
                + config.inner_folds*sum(name.startswith("Asym-BP_tau") for name in specs)
                + len(specs))}))
        return 0
    if config.profile == "smoke":
        print("注意：smoke为缩减训练预算的软件检查，不是正式论文结果。", flush=True)
    run_experiment(X, y, ids, rows, config, args.output_dir, data_info, args.resume, not args.quiet)
    print("完成：单模型、上包络、安全余量、嵌套决策、OOF汇总及50随机种子分析均已保存。"
          + ("SHAP图与数值已另存。" if config.shap_enabled else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
