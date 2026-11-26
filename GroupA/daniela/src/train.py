#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible training entry point for one experiment run.

Everything is driven by ``experiments/config.yaml``::

    python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift

For every target the script

* builds a leakage-free pipeline
  ``SimpleImputer(mean) -> FeatureSelector -> StandardScaler -> model``;
* tunes each enabled model with nested CV (outer 5-fold / inner 4-fold
  ``GridSearchCV``, scoring ``neg_mean_absolute_error``);
* selects the best model by outer-CV MAE;
* refits it on all available children and writes the run artifacts.

Artifacts in ``experiments/runs/<run_id>/``::

    data/                    copy of the cleaned + merged + feature CSVs (gitignored)
    metrics.json             MAE, RMSE, R2, Pearson, Spearman, cv_mae_std per model
    params.json              best hyperparameters per target/model
    model.pkl                dict {target: fitted pipeline} (joblib)
    models/<target>.pkl      one fitted pipeline per target (joblib)
    predictions.csv          out-of-fold predictions, compatible with evaluate_model.py
    plots/pred_vs_actual_<target>.png
    plots/residuals_<target>.png
    plots/shap_summary_<target>.png
    selected_features.json   features kept by the best pipeline per target
    best_models.json         best model name per target
    manifest.json            config hash, git commit, timestamp, environment

``predictions.csv`` uses the columns expected by the root ``evaluate_model.py``:
``child_id, repeat, fold, true_<target>, predicted_<target>`` (a ``pred_<target>``
alias is written too for the shorthand used in the restructure spec).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from .config import (
        CLEAN_FILES,
        data_dir,
        environment_info,
        find_project_root,
        get_id_columns,
        get_run,
        get_targets,
        git_commit,
        is_id_column,
        load_config,
        run_dir,
        slugify,
        source_dir,
        write_json,
    )
    from .feature_selection import FeatureSelector
except ImportError:  # direct script execution
    from config import (  # type: ignore
        CLEAN_FILES,
        data_dir,
        environment_info,
        find_project_root,
        get_id_columns,
        get_run,
        get_targets,
        git_commit,
        is_id_column,
        load_config,
        run_dir,
        slugify,
        source_dir,
        write_json,
    )
    from feature_selection import FeatureSelector  # type: ignore

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# model catalogue (kept in sync with the old notebooks)
# ---------------------------------------------------------------------------
def build_model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the enabled model specifications with their notebook grids."""
    enabled = config.get("models", {}) or {}
    specs: list[dict[str, Any]] = []

    if enabled.get("ridge", True):
        specs.append({
            "name": "Ridge",
            "estimator": Ridge(random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        })
    if enabled.get("lasso", True):
        specs.append({
            "name": "Lasso",
            "estimator": Lasso(random_state=RANDOM_STATE, max_iter=10000),
            "param_grid": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        })
    if enabled.get("elasticnet", True):
        specs.append({
            "name": "ElasticNet",
            "estimator": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
            "param_grid": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        })
    if enabled.get("bayesian_ridge", True):
        specs.append({
            "name": "BayesianRidge",
            "estimator": BayesianRidge(),
            "param_grid": {},
        })
    if enabled.get("random_forest", True):
        specs.append({
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
        })
    if enabled.get("lightgbm", True):
        try:
            from lightgbm import LGBMRegressor

            specs.append({
                "name": "LightGBM",
                "estimator": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1,
                                           n_jobs=1),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__num_leaves": [31, 63],
                    "model__learning_rate": [0.05, 0.1],
                },
            })
        except ImportError:
            print("WARNING: lightgbm is not installed; skipping LightGBM.")
    if enabled.get("xgboost", True):
        try:
            from xgboost import XGBRegressor

            specs.append({
                "name": "XGBoost",
                "estimator": XGBRegressor(random_state=RANDOM_STATE, verbosity=0,
                                          n_jobs=1),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 6],
                    "model__learning_rate": [0.05, 0.1],
                },
            })
        except ImportError:
            print("WARNING: xgboost is not installed; skipping XGBoost.")
    if enabled.get("svr", True):
        specs.append({
            "name": "SVR",
            "estimator": SVR(),
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        })
    return specs


# ---------------------------------------------------------------------------
# data loading / column handling
# ---------------------------------------------------------------------------
def load_features(path: str | Path, targets: list[str],
                  id_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run python -m src.preprocessing, "
            f"python -m src.merge and python -m src.features first."
        )
    df = pd.read_csv(path, sep="\t", encoding="utf-8")

    present_targets = [t for t in targets if t in df.columns]
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Targets missing from {path}: {missing}")
    if not present_targets:
        raise ValueError(f"No configured target found in {path}")

    id_cols = [c for c in df.columns if is_id_column(c, id_columns)]
    if "Numéro" in df.columns and "Numéro" not in id_cols:
        id_cols.insert(0, "Numéro")

    # All WPPSI columns are targets or target sub-scores: never use them as
    # features.  This is the leakage fix that was applied ad hoc in notebooks.
    wppsi_cols = [c for c in df.columns if "WPPSI" in c]
    feature_cols = [c for c in df.columns if c not in id_cols and c not in wppsi_cols]

    for column in feature_cols + present_targets:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove columns that are entirely empty after coercion.
    all_empty = [c for c in feature_cols if df[c].notna().sum() == 0]
    if all_empty:
        df = df.drop(columns=all_empty)
        feature_cols = [c for c in feature_cols if c not in all_empty]

    return df, feature_cols, present_targets


# ---------------------------------------------------------------------------
# cross-validation helpers
# ---------------------------------------------------------------------------
def make_splits(n_rows: int, outer_folds: int, repeats: int,
                random_state: int) -> list[dict[str, Any]]:
    splitter = RepeatedKFold(n_splits=outer_folds, n_repeats=repeats,
                             random_state=random_state)
    splits = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(n_rows))):
        splits.append({
            "repeat": i // outer_folds + 1,
            "fold": i % outer_folds + 1,
            "train": train_idx,
            "test": test_idx,
        })
    return splits


def build_pipeline(estimator, config: dict[str, Any]) -> Pipeline:
    fs = config.get("feature_selection", {}) or {}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("selector", FeatureSelector(
            k_best=int(fs.get("select_k_best", 20)),
            embedded=str(fs.get("embedded_estimator", "lasso")),
            shap_refinement=bool(fs.get("shap_refinement", True)),
            shap_top_k=int(fs.get("shap_top_k", 20)),
            shap_tree_max_depth=int(fs.get("shap_tree_max_depth", 5)),
            random_state=int(config.get("cv", {}).get("random_state", RANDOM_STATE)),
        )),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _safe_corr(kind: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy import stats

    if len(y_true) < 3 or np.ptp(y_pred) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = stats.pearsonr(y_true, y_pred)[0] if kind == "pearson" \
            else stats.spearmanr(y_true, y_pred)[0]
    return float(value)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    r2 = r2_score(y_true, y_pred) if np.ptp(y_true) > 0 else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2),
        "pearson": _safe_corr("pearson", y_true, y_pred),
        "spearman": _safe_corr("spearman", y_true, y_pred),
        "n": int(len(y_true)),
        "residual_mean": float(error.mean()),
        "residual_std": float(error.std(ddof=1)) if len(error) > 1 else float("nan"),
    }


def _mean_ignore_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


# ---------------------------------------------------------------------------
# evaluation of one target / model
# ---------------------------------------------------------------------------
def evaluate_model_for_target(
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    valid: np.ndarray,
    splits: list[dict[str, Any]],
    config: dict[str, Any],
    n_jobs: int,
    verbose: bool = True,
) -> dict[str, Any]:
    cv_cfg = config.get("cv", {}) or {}
    inner_cv = KFold(n_splits=int(cv_cfg.get("inner_folds", 4)), shuffle=True,
                     random_state=int(cv_cfg.get("random_state", RANDOM_STATE)))
    scoring = str(cv_cfg.get("scoring", "neg_mean_absolute_error"))

    repeats = sorted({s["repeat"] for s in splits})
    oof = {r: np.full(len(y), np.nan) for r in repeats}
    fold_mae: list[float] = []
    fold_r2: list[float] = []
    fold_rows: list[dict[str, Any]] = []

    for split in splits:
        train_idx = split["train"][valid[split["train"]]]
        test_idx = split["test"][valid[split["test"]]]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        pipeline = build_pipeline(spec["estimator"], config)
        search = GridSearchCV(
            pipeline,
            spec["param_grid"],
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = search.predict(X.iloc[test_idx])
        oof[split["repeat"]][test_idx] = pred

        mae = float(mean_absolute_error(y.iloc[test_idx], pred))
        fold_mae.append(mae)
        if np.ptp(y.iloc[test_idx].to_numpy()) > 0:
            fold_r2.append(float(r2_score(y.iloc[test_idx], pred)))
        fold_rows.append({
            "repeat": split["repeat"],
            "fold": split["fold"],
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "MAE": mae,
        })
        if verbose:
            print(f"      repeat {split['repeat']} fold {split['fold']}: "
                  f"train {len(train_idx)} / test {len(test_idx)} -> MAE {mae:.3f}")

    # Metrics per repeat, then averaged (the same convention as evaluate_model).
    per_repeat: list[dict[str, float]] = []
    for repeat in repeats:
        mask = valid & np.isfinite(oof[repeat])
        if mask.sum() < 2:
            continue
        per_repeat.append(compute_metrics(y.to_numpy()[mask], oof[repeat][mask]))

    metrics: dict[str, float] = {}
    for key in ("MAE", "RMSE", "R2", "pearson", "spearman"):
        metrics[key] = _mean_ignore_nan([m[key] for m in per_repeat])
    metrics["cv_mae_mean"] = float(np.mean(fold_mae)) if fold_mae else float("nan")
    metrics["cv_mae_std"] = float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 \
        else float("nan")
    metrics["cv_r2_mean"] = float(np.mean(fold_r2)) if fold_r2 else float("nan")
    metrics["cv_r2_std"] = float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 \
        else float("nan")
    metrics["n"] = int(valid.sum())
    metrics["n_repeats"] = int(len(repeats))

    return {"metrics": metrics, "oof": oof, "fold_rows": fold_rows}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, target: str,
                         out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="k")
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], "r--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs actual - {target} (out-of-fold)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, target: str,
                    out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.65, edgecolors="k")
    ax.axhline(0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - predicted)")
    ax.set_title(f"Residuals - {target}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transform_through_pipeline(pipeline: Pipeline, X: pd.DataFrame
                                ) -> tuple[np.ndarray, list[str]]:
    transformed: Any = X
    names = list(X.columns)
    for name, step in pipeline.steps[:-1]:
        if name == "selector":
            names = step.get_selected_features()
        transformed = step.transform(transformed)
    return np.asarray(transformed, dtype=float), names


def _shap_values(estimator, X_transformed: np.ndarray) -> np.ndarray | None:
    try:
        import shap
    except ImportError:
        return None

    try:
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        tree_types = (RandomForestRegressor, LGBMRegressor, XGBRegressor)
    except ImportError:  # pragma: no cover
        tree_types = (RandomForestRegressor,)

    from sklearn.linear_model import LinearRegression
    linear_types = (Ridge, Lasso, ElasticNet, BayesianRidge, LinearRegression)

    try:
        if isinstance(estimator, tree_types):
            values = shap.TreeExplainer(estimator).shap_values(X_transformed)
        elif isinstance(estimator, linear_types):
            values = shap.LinearExplainer(estimator, X_transformed).shap_values(
                X_transformed)
        else:
            background = shap.sample(X_transformed, min(50, len(X_transformed)),
                                     random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(estimator.predict, background)
            n_explain = min(100, len(X_transformed))
            values = explainer.shap_values(X_transformed[:n_explain], nsamples=100,
                                           silent=True)
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 0]
        return values
    except Exception as exc:  # pragma: no cover - robust fallback
        print(f"      SHAP explainer failed ({type(exc).__name__}: {exc}); "
              f"falling back to permutation importance.")
        return None


def _plot_shap(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, target: str,
               out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = pipeline.named_steps["model"]
    transformed, names = _transform_through_pipeline(pipeline, X)
    values = _shap_values(estimator, transformed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if values is not None and values.shape[1] == len(names):
        try:
            import shap
            shap.summary_plot(values, pd.DataFrame(transformed, columns=names),
                              plot_type="bar", show=False, max_display=20)
            plt.title(f"SHAP summary - {target}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            return
        except Exception as exc:  # pragma: no cover
            print(f"      SHAP plot failed ({type(exc).__name__}: {exc}); "
                  f"falling back to permutation importance.")
            plt.close("all")

    # Fallback: permutation importance (still a useful feature ranking).
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            estimator, transformed, y, n_repeats=5, random_state=RANDOM_STATE)
        importance = result.importances_mean
    except Exception:
        # Last resort: absolute coefficients when available.
        importance = np.abs(np.ravel(getattr(estimator, "coef_", np.zeros(len(names)))))
        if len(importance) != len(names):
            importance = np.zeros(len(names))

    order = np.argsort(importance)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.asarray(names)[order][::-1], importance[order][::-1])
    ax.set_xlabel("Permutation importance (SHAP unavailable)")
    ax.set_title(f"Feature importance - {target}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main training routine
# ---------------------------------------------------------------------------
def run_training(config_path: str | Path, run_id: str, source_override: str | None = None,
                 targets_override: list[str] | None = None,
                 models_override: list[str] | None = None,
                 n_jobs: int | None = None, verbose: bool = True) -> Path:
    cfg = load_config(config_path)
    run = get_run(cfg, run_id)
    source = source_override or str(run.get("source", "clean_data_auto"))
    root = Path(cfg["_root"])
    out_dir = run_dir(cfg, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    targets = list(targets_override or get_targets(cfg))
    id_columns = get_id_columns(cfg)
    feature_path = source_dir(cfg, source) / "merged_dataset_features.csv"
    df, feature_cols, present_targets = load_features(feature_path, targets, id_columns)

    if models_override:
        wanted = {m.lower() for m in models_override}
        specs = [s for s in build_model_specs(cfg) if s["name"].lower() in wanted]
    else:
        specs = build_model_specs(cfg)
    if not specs:
        raise ValueError("No model is enabled in the config / --models selection.")

    n_jobs = int(n_jobs if n_jobs is not None else
                 cfg.get("cv", {}).get("n_jobs", -1))
    cv_cfg = cfg.get("cv", {}) or {}
    splits = make_splits(
        n_rows=len(df),
        outer_folds=int(cv_cfg.get("outer_folds", 5)),
        repeats=int(cv_cfg.get("repeats", 1)),
        random_state=int(cv_cfg.get("random_state", RANDOM_STATE)),
    )

    say = print if verbose else (lambda *a, **k: None)
    say(f"Run: {run_id}")
    say(f"Source: {feature_path}")
    say(f"Rows: {len(df)} | features: {len(feature_cols)} | targets: {present_targets}")
    say(f"Models: {[s['name'] for s in specs]}")

    # Iterator used to refit each model on all children after CV.
    inner_cv = KFold(n_splits=int(cv_cfg.get("inner_folds", 4)), shuffle=True,
                     random_state=int(cv_cfg.get("random_state", RANDOM_STATE)))
    scoring = str(cv_cfg.get("scoring", "neg_mean_absolute_error"))

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    params: dict[str, dict[str, dict[str, Any]]] = {}
    oof_by_target: dict[str, dict[str, np.ndarray]] = {}
    best_models: dict[str, str] = {}
    fitted: dict[str, Pipeline] = {}
    selected_features: dict[str, list[str]] = {}

    for target in present_targets:
        target_slug = slugify(target)
        say(f"\n{'=' * 70}\nTarget: {target}\n{'=' * 70}")
        valid = df[target].notna().to_numpy()
        y = df[target]
        metrics[target] = {}
        params[target] = {}
        oof_by_target[target] = {}

        for spec in specs:
            say(f"  {spec['name']}: nested CV")
            result = evaluate_model_for_target(
                spec, df[feature_cols], y, valid, splits, cfg, n_jobs=n_jobs,
                verbose=verbose)
            metrics[target][spec["name"]] = result["metrics"]
            oof_by_target[target][spec["name"]] = result["oof"]
            say(f"    MAE {result['metrics']['MAE']:.4f} | "
                f"R2 {result['metrics']['R2']:.4f} | "
                f"outer-fold MAE {result['metrics']['cv_mae_mean']:.4f} "
                f"+/- {result['metrics']['cv_mae_std']:.4f}")

        # Best model by outer-CV MAE (mean fold MAE), as in the notebooks.
        best_name = min(
            metrics[target],
            key=lambda name: (metrics[target][name]["cv_mae_mean"]
                              if np.isfinite(metrics[target][name]["cv_mae_mean"])
                              else np.inf),
        )
        best_models[target] = best_name
        say(f"  Best model: {best_name}")

        # Refit every model on all children to record honest full-data params.
        best_estimator: Pipeline | None = None
        for spec in specs:
            pipeline = build_pipeline(spec["estimator"], cfg)
            search = GridSearchCV(
                pipeline, spec["param_grid"], cv=inner_cv, scoring=scoring,
                n_jobs=n_jobs, refit=True, error_score="raise")
            search.fit(df.loc[valid, feature_cols], y.loc[valid])
            best_params = search.best_params_
            params[target][spec["name"]] = {
                "best_params": _jsonable(best_params),
                "best_cv_mae": float(-search.best_score_),
            }
            if spec["name"] == best_name:
                best_estimator = search.best_estimator_
                selector = best_estimator.named_steps.get("selector")
                if selector is not None:
                    selected_features[target] = selector.get_selected_features()

        assert best_estimator is not None
        fitted[target] = best_estimator

        # Artifacts for this target (plots + per-target pickle).
        best_oof = oof_by_target[target][best_name]
        prediction_mask = valid & np.isfinite(best_oof[sorted(best_oof)[0]])
        # Use the first repeat for the diagnostic plots.
        first_repeat = sorted(best_oof)[0]
        y_true_plot = y.to_numpy()[prediction_mask]
        y_pred_plot = best_oof[first_repeat][prediction_mask]
        _plot_pred_vs_actual(y_true_plot, y_pred_plot, target,
                             plots_dir / f"pred_vs_actual_{target_slug}.png")
        _plot_residuals(y_true_plot, y_pred_plot, target,
                        plots_dir / f"residuals_{target_slug}.png")
        _plot_shap(best_estimator, df.loc[valid, feature_cols],
                   y.to_numpy()[valid], target,
                   plots_dir / f"shap_summary_{target_slug}.png")

        model_dir = out_dir / "models"
        model_dir.mkdir(exist_ok=True)
        import joblib
        joblib.dump(best_estimator, model_dir / f"{target_slug}.pkl")
        say(f"  Saved model + plots for {target}")

    # ------------------------------------------------------------------
    # predictions.csv (complete cases across all targets)
    # ------------------------------------------------------------------
    import joblib

    complete = np.ones(len(df), dtype=bool)
    for target in present_targets:
        complete &= df[target].notna().to_numpy()
        best_name = best_models[target]
        for repeat_values in oof_by_target[target][best_name].values():
            complete &= np.isfinite(repeat_values)

    fold_lookup: dict[int, dict[int, int]] = {}
    for split in splits:
        repeat = split["repeat"]
        fold_lookup.setdefault(repeat, {})
        for idx in split["test"]:
            fold_lookup[repeat][int(idx)] = split["fold"]

    rows = []
    for repeat in sorted({s["repeat"] for s in splits}):
        for idx in np.flatnonzero(complete):
            row = {
                "child_id": str(df.iloc[idx]["Numéro"]) if "Numéro" in df.columns
                else str(idx),
                "repeat": repeat,
                "fold": fold_lookup.get(repeat, {}).get(int(idx), 0),
            }
            for target in present_targets:
                best_name = best_models[target]
                true_value = float(df.iloc[idx][target])
                pred_value = float(oof_by_target[target][best_name][repeat][idx])
                row[f"true_{target}"] = true_value
                row[f"predicted_{target}"] = pred_value
                row[f"pred_{target}"] = pred_value
            rows.append(row)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    say(f"\nSaved {len(predictions)} prediction rows -> {out_dir / 'predictions.csv'}")

    # ------------------------------------------------------------------
    # JSON artifacts + manifest
    # ------------------------------------------------------------------
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "params.json", params)
    write_json(out_dir / "best_models.json", best_models)
    write_json(out_dir / "selected_features.json", selected_features)
    joblib.dump(fitted, out_dir / "model.pkl")

    manifest = {
        "run_id": run_id,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "config_hash": cfg.get("_config_hash"),
        "config_path": str(cfg.get("_config_path")),
        "source": source,
        "feature_file": str(feature_path),
        "n_subjects": int(len(df)),
        "n_features": int(len(feature_cols)),
        "targets": present_targets,
        "models": [s["name"] for s in specs],
        "best_models": best_models,
        "predictions_rows": int(len(predictions)),
        "complete_case_subjects": int(complete.sum()),
        "cv": {
            "outer_folds": int(cv_cfg.get("outer_folds", 5)),
            "inner_folds": int(cv_cfg.get("inner_folds", 4)),
            "repeats": int(cv_cfg.get("repeats", 1)),
            "scoring": scoring,
        },
        "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)

    # Copy the derived data next to the run (gitignored), so a run is self-contained.
    run_data = out_dir / "data"
    run_data.mkdir(exist_ok=True)
    src = source_dir(cfg, source)
    for name in CLEAN_FILES + ["merged_dataset.csv", "merged_dataset_features.csv"]:
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, run_data / name)

    say(f"\nRun artifacts written to: {out_dir}")
    return out_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="Config file (default: GroupA/daniela/experiments/config.yaml)")
    ap.add_argument("--run-id", required=True,
                    help="Run id defined under runs: in the config")
    ap.add_argument("--source", default=None,
                    help="Override the run's cleaned-data source directory")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Override the configured targets (useful for quick tests)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only run these models, e.g. --models Ridge SVR")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="Parallel jobs for GridSearchCV (default: config cv.n_jobs)")
    args = ap.parse_args(argv)

    config_path = Path(args.config) if args.config else (
        Path(__file__).resolve().parent.parent / "experiments" / "config.yaml")
    try:
        run_training(config_path, args.run_id, source_override=args.source,
                     targets_override=args.targets,
                     models_override=args.models, n_jobs=args.n_jobs)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
