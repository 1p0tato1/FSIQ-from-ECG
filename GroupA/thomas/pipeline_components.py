"""Reusable building blocks for the HRV -> WPPSI-III IQ regression study.

The module is deliberately kept outside the notebook so that:
  * pipelines saved with joblib can be reloaded in any other script (pickles store
    the *import path* of custom classes, which must therefore be importable), and
  * joblib worker processes can import the fold-evaluation code by reference.

Contents
--------
CorrelationPruner / LowVarianceFilter : NaN-tolerant, fit-on-training-data filters
build_preprocessor / assemble_pipeline : the leakage-free preprocessing chain
MethodSpec / build_method_registry     : one entry per regression model + its search space
make_search                            : picks an exhaustive or randomised hyper-parameter search
fit_and_score_fold                     : tunes and scores one outer cross-validation fold
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

# Name given to the constant-prediction reference model.
BASELINE_LABEL = "MeanBaseline"

# Prefix that scikit-learn uses to address the final estimator inside a Pipeline.
MODEL_STEP = "model"


# --------------------------------------------------------------------------------------
# Column filters that learn what to drop from the training data only
# --------------------------------------------------------------------------------------
class CorrelationPruner(TransformerMixin, BaseEstimator):
    """Drop every column that is strongly correlated with a column located before it.

    Correlations are computed pairwise, so missing values are tolerated. Only the upper
    triangle of the |r| matrix is inspected, which means that for each highly correlated
    pair the column appearing later is the one removed.
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        self.columns_seen_ = np.array([str(name) for name in frame.columns], dtype=object)
        abs_corr = np.nan_to_num(frame.corr().abs().to_numpy(), nan=0.0)
        # A column is redundant if any earlier column exceeds the threshold against it.
        redundant = (np.triu(abs_corr, k=1) > self.threshold).any(axis=0)
        self.keep_mask_ = ~redundant
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.keep_mask_.size:
            raise ValueError("Number of columns differs from the data seen during fit.")
        return values[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        names = self.columns_seen_ if input_features is None else np.asarray(input_features, dtype=object)
        return names[self.keep_mask_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class LowVarianceFilter(TransformerMixin, BaseEstimator):
    """Remove columns whose sample variance (NaN ignored) is below ``threshold``."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
            spread = np.nanvar(values, axis=0, ddof=1)
        # Undefined variances (e.g. a single observation) are kept rather than dropped.
        self.keep_mask_ = ~(spread < self.threshold)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.keep_mask_.size:
            raise ValueError("Number of columns differs from the data seen during fit.")
        return values[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_mask_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.keep_mask_.size)]
        return np.asarray(input_features, dtype=object)[self.keep_mask_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


# --------------------------------------------------------------------------------------
# Preprocessing chain and full pipeline
# --------------------------------------------------------------------------------------
def build_preprocessor(corr_threshold=0.95, variance_threshold=0.01, lasso_alpha=0.01):
    """Return a fresh, unfitted preprocessing pipeline.

    Order of operations (every statistic is learned from the data passed to ``fit``):
      1. drop near-duplicate columns (|r| above ``corr_threshold``)
      2. drop near-constant columns (variance below ``variance_threshold``)
      3. fill remaining gaps with the column mean
      4. standardise to zero mean / unit variance
      5. keep only columns that receive a non-zero Lasso coefficient
    """
    return Pipeline(
        steps=[
            ("corr_filter", CorrelationPruner(threshold=corr_threshold)),
            ("variance_filter", LowVarianceFilter(threshold=variance_threshold)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "lasso_screen",
                # 1e-12 behaves as "coefficient is not exactly zero"; the generous iteration
                # cap lets the coordinate-descent solver converge on collinear predictors.
                SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=10000), threshold=1e-12),
            ),
        ]
    )


def assemble_pipeline(estimator):
    """Chain the preprocessing steps and a regressor into a single estimator."""
    steps = list(build_preprocessor().steps) + [(MODEL_STEP, estimator)]
    return Pipeline(steps=steps)


# --------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodSpec:
    """A regression method together with the hyper-parameter values to explore."""

    label: str
    factory: Callable[[], BaseEstimator]
    search_space: Dict[str, Sequence]
    uses_preprocessing: bool = True
    tree_based: bool = False

    def build(self):
        """Create a new unfitted estimator (wrapped in the preprocessing pipeline if needed)."""
        model = self.factory()
        return assemble_pipeline(model) if self.uses_preprocessing else model

    def prefixed_space(self):
        """Search space keyed the way scikit-learn expects for the built estimator."""
        prefix = f"{MODEL_STEP}__" if self.uses_preprocessing else ""
        return {prefix + name: list(values) for name, values in self.search_space.items()}

    def n_candidates(self):
        """Number of distinct hyper-parameter combinations in the space."""
        return prod(len(values) for values in self.search_space.values()) if self.search_space else 1


def build_baseline_spec():
    """Reference model that always predicts the mean of the training targets."""
    return MethodSpec(
        label=BASELINE_LABEL,
        factory=lambda: DummyRegressor(strategy="mean"),
        search_space={},
        uses_preprocessing=False,
    )


def build_method_registry(seed=42, verbose=True):
    """Create the METHOD_REGISTRY: model label -> MethodSpec.

    LightGBM and XGBoost are optional dependencies; when one of them is not installed the
    corresponding entry is skipped and a message is printed.
    """
    registry: Dict[str, MethodSpec] = {}

    registry["ElasticNet"] = MethodSpec(
        label="ElasticNet",
        factory=lambda: ElasticNet(max_iter=10000, random_state=seed),
        search_space={
            "alpha": np.logspace(-4, 1, 20),
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    )

    registry["RandomForest"] = MethodSpec(
        label="RandomForest",
        factory=lambda: RandomForestRegressor(random_state=seed, n_jobs=1),
        search_space={
            "n_estimators": [200, 500],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [5, 10, 20],
            "max_features": ["sqrt", "log2"],
        },
        tree_based=True,
    )

    try:
        from lightgbm import LGBMRegressor

        registry["LightGBM"] = MethodSpec(
            label="LightGBM",
            factory=lambda: LGBMRegressor(
                objective="regression", random_state=seed, n_jobs=1, verbosity=-1
            ),
            search_space={
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, -1],
                "min_child_samples": [10, 20],
                "reg_alpha": [0.001, 0.1, 0.5],
                "reg_lambda": [0.0, 0.1, 1.0],
            },
            tree_based=True,
        )
    except ImportError:
        if verbose:
            print("lightgbm is not installed: LightGBM will be skipped.")

    try:
        from xgboost import XGBRegressor

        registry["XGBoost"] = MethodSpec(
            label="XGBoost",
            factory=lambda: XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            search_space={
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 5],
                "subsample": [0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10.0],
            },
            tree_based=True,
        )
    except ImportError:
        if verbose:
            print("xgboost is not installed: XGBoost will be skipped.")

    registry["SVR"] = MethodSpec(
        label="SVR",
        factory=lambda: SVR(),
        search_space={
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 1.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    )

    return registry


# --------------------------------------------------------------------------------------
# Hyper-parameter search and per-fold evaluation
# --------------------------------------------------------------------------------------
def make_search(spec, inner_cv, n_iter, seed):
    """Build the inner-loop hyper-parameter search for ``spec``.

    When the space holds no more combinations than ``n_iter`` every combination is tried
    (GridSearchCV); otherwise ``n_iter`` combinations are sampled (RandomizedSearchCV).
    Models are always ranked by mean absolute error.
    """
    common = dict(scoring="neg_mean_absolute_error", cv=inner_cv, n_jobs=1, error_score="raise")
    if spec.n_candidates() <= n_iter:
        return GridSearchCV(spec.build(), spec.prefixed_space(), **common)
    return RandomizedSearchCV(
        spec.build(), spec.prefixed_space(), n_iter=n_iter, random_state=seed, **common
    )


def regression_scores(y_true, y_pred):
    """MAE, MSE, RMSE and R2 for one set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


def fit_and_score_fold(spec, X, y, train_idx, test_idx, inner_splits, n_iter, seed):
    """Tune ``spec`` on the training rows only, then score it on the held-out rows.

    The inner search, the preprocessing statistics and the final fit all see the training
    rows exclusively. Warnings raised during the work are tallied by category so that
    convergence problems can be reviewed afterwards.
    """
    started = time.perf_counter()
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if spec.search_space:
            inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
            search = make_search(spec, inner_cv, n_iter, seed)
            search.fit(X_train, y_train)
            fitted, best_params = search.best_estimator_, search.best_params_
        else:
            fitted, best_params = spec.build().fit(X_train, y_train), {}
        predictions = np.asarray(fitted.predict(X_test), dtype=float)

    return {
        "scores": regression_scores(y_test, predictions),
        "best_params": best_params,
        "warnings": dict(Counter(item.category.__name__ for item in caught)),
        "test_idx": np.asarray(test_idx),
        "predictions": predictions,
        "seconds": time.perf_counter() - started,
    }
