# -*- coding: utf-8 -*-
"""单独运行四模型统一脚本中的 BP-NN SHAP 分析。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SOURCE_FILE = Path(__file__).resolve().parent / "所有模型＋多随机种子实验.py"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "SHAP单独运行结果"
FEATURE_DISPLAY_NAMES = {
    "Stie_grading": "Site grading",
}


def load_source_module():
    """载入原脚本中的模型、数据读取和调参工具，但不执行其 main。"""
    spec = importlib.util.spec_from_file_location("four_model_core", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法载入源脚本：{SOURCE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_shap_analysis(core, X, y, ids, row_numbers, config, output_dir):
    """全数据重拟合 BP-NN 并进行 SHAP 解释，不运行外层 LOO 实验。"""
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("SHAP：全数据调参并重训练 BP-NN。", flush=True)
    specs, _ = core.make_specs(config.primary_seed, config)
    inner = list(
        KFold(
            n_splits=config.inner_folds,
            shuffle=True,
            random_state=config.inner_seed,
        ).split(X)
    )
    bp = specs["BP_NN"]
    model, bp_params, bp_loss, _, warning_count, warning_messages, _ = core.tune_and_fit(
        bp,
        core.candidate_parameters("BP_NN", bp, config),
        X,
        y,
        inner,
        config,
    )

    explain_X = np.asarray(model.named_steps["imputer"].transform(X), dtype=float)
    features = list(X.columns)
    display_features = [
        FEATURE_DISPLAY_NAMES.get(str(name), str(name).replace("_", " "))
        for name in features
    ]

    def predict_fn(values):
        frame = pd.DataFrame(np.asarray(values, dtype=float), columns=features)
        return model.predict(frame)

    random_state = np.random.get_state()
    try:
        np.random.seed(config.primary_seed)
        explainer = shap.KernelExplainer(predict_fn, explain_X)
        values = np.asarray(
            explainer.shap_values(
                explain_X,
                nsamples=config.shap_nsamples,
                l1_reg=0.0,
            ),
            dtype=float,
        )
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
        importance = pd.DataFrame(
            {"Feature": display_features, "Mean_abs_SHAP": np.abs(values).mean(axis=0)}
        ).sort_values("Mean_abs_SHAP", ascending=False, ignore_index=True)

        shap_table = pd.DataFrame(values, columns=[f"SHAP {name}" for name in display_features])
        shap_table.insert(0, "project_id", ids.to_numpy())
        shap_table.insert(1, "original_row_number", row_numbers)
        shap_table.insert(2, "y_true", y)
        shap_table.insert(3, "full_fit_prediction", predictions)
        shap_table.insert(4, "expected_value", expected)
        shap_table.insert(5, "SHAP_reconstructed_prediction", reconstructed)

        metadata = {
            "model": "BP_NN",
            "seed": config.primary_seed,
            "smoke_only": config.profile == "smoke",
            "scope": "full-data refit explanation, not held-out LOO predictions",
            "background": "all training projects, median imputation from final model",
            "n_explained": len(X),
            "nsamples": config.shap_nsamples,
            "units": "original target cost units",
            "BP_parameters": bp_params,
            "BP_inner_loss": bp_loss,
            "refit_warning_count": warning_count,
            "refit_warnings": warning_messages,
        }
        core.save_workbook(
            folder / "BP_NN_SHAP_values.xlsx",
            {
                "SHAP_values": shap_table,
                "Mean_abs_SHAP": importance,
                "Explained_features": pd.DataFrame(explain_X, columns=display_features).assign(
                    project_id=ids.to_numpy()
                ),
                "Settings": core.settings_table(metadata),
            },
        )

        with plt.rc_context(
            {
                "font.family": "Times New Roman",
                "font.size": 11,
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
        ):
            shap.summary_plot(
                values,
                explain_X,
                feature_names=display_features,
                max_display=len(display_features),
                show=False,
                plot_size=(10, 8),
            )
            figure = plt.gcf()
            figure.axes[0].set_xlabel("SHAP value (cost units)")
            figure.savefig(
                folder / "BP_NN_SHAP_summary.jpg",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(figure)

            ordered = importance.iloc[::-1]
            figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
            axis.barh(ordered["Feature"], ordered["Mean_abs_SHAP"], color="#517EA5")
            axis.set_xlabel("Mean absolute SHAP value (cost units)")
            axis.tick_params(direction="in", top=False, right=False)
            figure.savefig(
                folder / "BP_NN_SHAP_importance.jpg",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(figure)
    finally:
        np.random.set_state(random_state)

    print(f"SHAP分析已保存：{folder.resolve()}", flush=True)


def comma_values(text, cast):
    return tuple(cast(value.strip()) for value in text.split(",") if value.strip())


def main(argv=None):
    core = load_source_module()
    parser = argparse.ArgumentParser(description="单独运行 BP-NN 的全数据 SHAP 分析")
    parser.add_argument("--input", type=Path, default=core.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", default=None)
    parser.add_argument("--features", default=None, help="逗号分隔列名；省略则沿用前15列")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-seed", type=int, default=2026)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--tuning-metric",
        choices=["rmse", "mae", "under_cost"],
        default="rmse",
    )
    parser.add_argument("--under-weight", type=float, default=4.0)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--profile", choices=["standard", "smoke"], default="standard")
    parser.add_argument("--shap-nsamples", type=int, default=300)
    args = parser.parse_args(argv)

    search_overrides = {}
    if args.search_config:
        search_overrides = json.loads(args.search_config.read_text(encoding="utf-8-sig"))

    config = core.RunConfig(
        primary_seed=args.primary_seed,
        inner_folds=args.inner_folds,
        inner_seed=args.inner_seed,
        search_seed=args.search_seed,
        candidates=args.candidates,
        tuning_metric=args.tuning_metric,
        under_weight=args.under_weight,
        models=("BP_NN",),
        profile=args.profile,
        shap_enabled=True,
        shap_nsamples=args.shap_nsamples,
        search_overrides=search_overrides,
    )
    X, y, ids, rows, _ = core.read_data(
        args.input,
        args.target,
        comma_values(args.features, str) if args.features else None,
        args.id_col,
    )
    core.validate_config(config, len(y))
    run_shap_analysis(core, X, np.asarray(y, dtype=float), ids, rows, config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
