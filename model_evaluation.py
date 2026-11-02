"""
test_for_model.py
=================

Base script for evaluating all regression methods in the same way so they
can be compared fairly.

Project: FSIQ-from-ECG — regression of Full-Scale IQ (WPPSI-III) from
neonatal ECG / HRV features (AuBE cohort, CHU Saint-Étienne).
Research only, not a diagnostic tool.

Usage
-----
    python test_for_model.py

Each regression method should register a ``build_model()`` callable in the
``METHOD_REGISTRY`` below. The script will then:

1. Load the data from ``data/``
2. Build a consistent preprocessing + model pipeline for each method
3. Evaluate on the same train / test split
4. Report RMSE, MAE, R² and MAPE for every method
5. Save a comparison table to ``results/model_comparison.csv``

"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Target column name — update to match your WPPSI-III results file.
TARGET_COLUMN = "FSIQ"  # <-- TODO: set the actual target column name

# ID / non-feature columns to drop from X.
DROP_COLUMNS: list[str] = []  # <-- TODO: e.g. ["subject_id", "date"]

# Directory where data files live.
DATA_DIR = Path(__file__).resolve().parent / "data"

# Directory where comparison results will be written.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
#
# Each entry maps a human-readable method name to a factory that returns an
# unfitted scikit-learn regressor.  The test harness wraps every regressor in
# the same preprocessing pipeline so all methods are evaluated identically.
#
# Example:
#
#     from sklearn.ensemble import RandomForestRegressor
#
#     def build_random_forest() -> RandomForestRegressor:
#         return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
#
#     METHOD_REGISTRY["Random Forest"] = build_random_forest
#
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, Callable[[], object]] = {}

# TODO: Register your group's regression methods here.
# Example entries (uncomment and adapt):

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import ElasticNet, BayesianRidge
# from sklearn.gaussian_process import GaussianProcessRegressor


# def build_random_forest() -> RandomForestRegressor:
#     return RandomForestRegressor(
#         n_estimators=300,
#         max_depth=None,
#         min_samples_leaf=2,
#         random_state=RANDOM_STATE,
#         n_jobs=-1,
#     )


# def build_elastic_net() -> ElasticNet:
#     return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000)


# def build_bayesian_ridge() -> BayesianRidge:
#     return BayesianRidge()


# def build_gaussian_process() -> GaussianProcessRegressor:
#     return GaussianProcessRegressor(
#         kernel=None,
#         alpha=1e-2,
#         normalize_y=True,
#         random_state=RANDOM_STATE,
#     )


# METHOD_REGISTRY["Random Forest"] = build_random_forest
# METHOD_REGISTRY["Elastic Net"] = build_elastic_net
# METHOD_REGISTRY["Bayesian Ridge"] = build_bayesian_ridge
# METHOD_REGISTRY["Gaussian Process"] = build_gaussian_process


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and merge the raw data files from ``data/``.

    The exact merge keys depend on the file schemas — update this function to
    match your HRV and WPPSI-III CSVs.  Do **not** commit data files; they
    contain children's health information.
    """
    raise NotImplementedError(
        "Implement load_data() to read and merge the files in data/ "
        "(HRV_Lineaire_Full.csv, HRV_NonLineaire_Full.csv, Résultats WPPSI III.csv)."
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a column-wise preprocessor: impute + scale numeric, impute + one-hot categorical."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dictionary of standard regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:  # division by zero when y_true contains 0
        mape = float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    name: str,
    build_model: Callable[[], object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Fit one method and return its test-set metrics."""
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", build_model()),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred)
    metrics["method"] = name
    return metrics


def main() -> None:
    if not METHOD_REGISTRY:
        print("No methods registered in METHOD_REGISTRY. "
              "Add your regressors to test_for_model.py before running.")
        return

    print("Loading data ...")
    df = load_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # Split into features and target.
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS + [TARGET_COLUMN]]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # Single shared train / test split so every method is evaluated identically.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(
        f"Data: {X.shape[0]} samples, {X.shape[1]} features | "
        f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
    )

    results: list[dict[str, float]] = []
    for name, build_model in METHOD_REGISTRY.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_method(name, build_model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        print(
            f"  RMSE={metrics['RMSE']:.4f}  "
            f"MAE={metrics['MAE']:.4f}  "
            f"R2={metrics['R2']:.4f}  "
            f"MAPE={metrics['MAPE']:.4f}"
        )

    # --- comparison table -------------------------------------------------
    results_df = pd.DataFrame(results)
    cols = ["method", "RMSE", "MAE", "R2", "MAPE"]
    results_df = results_df[cols].sort_values("RMSE").reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by RMSE, lower is better)")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- save results ------------------------------------------------------
    out_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nComparison table saved to: {out_path}")


if __name__ == "__main__":
    main()
