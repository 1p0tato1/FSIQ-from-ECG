"""
model_evaluation.py
===================

Base script for evaluating all regression methods in the same way so they
can be compared fairly.

Project: FSIQ-from-ECG — regression of WPPSI-III IQ scores from neonatal
ECG / HRV features (AuBE cohort, CHU Saint-Étienne).
Research only, not a diagnostic tool.

Data loading & preprocessing follow `Group A/dataset.md` (§5.2, §7):

1. HRV files: skip row 2 (units row) → skiprows=[1]
2. WPPSI-III: header on row 2, tab-separated → header=1, sep="\t"
3. Fix comma decimal separator in WPPSI-III ("," -> ".")
4. Filter both HRV files to Passage == "M0" (birth recordings only)
5. Drop WPPSI rows missing the target IQ for the current run
6. Merge linear + nonlinear HRV on Monogramme + DDN + Passage
7. Merge HRV + WPPSI on Monogramme + DDN
8. Drop fully-empty feature columns
9. Impute remaining NaN with column mean
10. Standardize features (StandardScaler, mean 0, SD 1) — fit on train only
11. Evaluate with MAE and R² (dataset.md §9)

Usage
-----
    python model_evaluation.py

Each regression method registers a ``build_model()`` callable in
``METHOD_REGISTRY``. The script wraps every regressor in the same
preprocessing and evaluates on the same train / test split for each
target IQ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration (per Group A/dataset.md)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Target IQ variables to try (dataset.md §3)
# ---------------------------------------------------------------------------
#
# WPPSI-III columns available:
#   QIT  — Full-Scale IQ (primary target)
#   QIV  — Verbal IQ
#   QIP  — Performance IQ
#   COM, INF, DIM, CUB, AOB — subtest scores
#   NCL  — Percentile rank
#
# Edit TARGET_COLUMNS to the IQs you want to compare. The script will run
# every registered method once per target.
#
TARGET_COLUMNS: list[str] = ["QIT", "QIV", "QIP"]

# Identifier / metadata columns that are NOT features (dataset.md §1–§3).
ID_COLS: list[str] = [
    "Numéro",
    "Monogramme",
    "DDN",
    "Sexe",
    "Passage",
    "Date",
    "Recorder",
]

# Merge keys (dataset.md §4).
HRV_MERGE_KEYS = ["Monogramme", "DDN", "Passage"]
WPPSI_MERGE_KEYS = ["Monogramme", "DDN"]

# Data directory (sibling of this script).
DATA_DIR = Path(__file__).resolve().parent / "data"

LINEAR_PATH = DATA_DIR / "HRV_Lineaire_Full.csv"
NONLINEAR_PATH = DATA_DIR / "HRV_NonLineaire_Full.csv"
WPPSI_PATH = DATA_DIR / "Résultats WPPSI III.csv"

# Results directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
#
# Each entry maps a method name to a factory returning an unfitted
# scikit-learn regressor. All methods are evaluated on identical
# preprocessed data and the same train / test split.
#
# Example:
#
#     from sklearn.linear_model import Ridge, ElasticNet
#     from sklearn.ensemble import RandomForestRegressor
#
#     def build_ridge() -> Ridge:
#         return Ridge(alpha=1.0, random_state=RANDOM_STATE)
#
#     METHOD_REGISTRY["Ridge"] = build_ridge
#
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, Callable[[], object]] = {}

# TODO: Register your group's regression methods here.
# Suggested for small n / high p (~189 samples, ~100 features):
#   - Ridge, Lasso, ElasticNet, BayesianRidge (regularized linear)
#   - RandomForestRegressor, GradientBoostingRegressor (with depth limits)
#
# Example entries (uncomment and adapt):

# from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
# from sklearn.ensemble import RandomForestRegressor


# def build_ridge() -> Ridge:
#     return Ridge(alpha=1.0, random_state=RANDOM_STATE)


# def build_elastic_net() -> ElasticNet:
#     return ElasticNet(
#         alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000
#     )


# def build_bayesian_ridge() -> BayesianRidge:
#     return BayesianRidge()


# def build_random_forest() -> RandomForestRegressor:
#     return RandomForestRegressor(
#         n_estimators=300,
#         max_depth=None,
#         min_samples_leaf=2,
#         random_state=RANDOM_STATE,
#         n_jobs=-1,
#     )


# METHOD_REGISTRY["Ridge"] = build_ridge
# METHOD_REGISTRY["Elastic Net"] = build_elastic_net
# METHOD_REGISTRY["Bayesian Ridge"] = build_bayesian_ridge
# METHOD_REGISTRY["Random Forest"] = build_random_forest


# ---------------------------------------------------------------------------
# Data loading (dataset.md §7, steps 1–8)
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and merge the three CSVs following dataset.md.

    Returns the merged DataFrame (HRV linear + nonlinear + WPPSI) filtered to
    M0 passages, with fully-empty columns dropped.  Target-specific NA rows
    are dropped later in ``preprocess()`` per target column.
    """
    # --- Step 1: linear HRV, skip units row (row 2) -----------------------
    df_linear = pd.read_csv(LINEAR_PATH, skiprows=[1])

    # --- Step 2: nonlinear HRV, skip units row (row 2) --------------------
    df_nonlinear = pd.read_csv(NONLINEAR_PATH, skiprows=[1])

    # --- Step 3: WPPSI-III, header on row 2, tab-separated ----------------
    df_wppsi = pd.read_csv(WPPSI_PATH, header=1, sep="\t")

    # --- Step 4: fix comma decimal separator in WPPSI ----------------------
    df_wppsi = df_wppsi.replace({",": "."}, regex=True)
    for col in df_wppsi.columns:
        df_wppsi[col] = pd.to_numeric(df_wppsi[col], errors="ignore")

    # --- Step 5: filter both HRV to Passage == "M0" -----------------------
    df_linear_m0 = df_linear[df_linear["Passage"] == "M0"].copy()
    df_nonlinear_m0 = df_nonlinear[df_nonlinear["Passage"] == "M0"].copy()

    # --- Steps 6–8: merges ------------------------------------------------
    # linear + nonlinear on Monogramme + DDN + Passage
    df_hrv = df_linear_m0.merge(
        df_nonlinear_m0,
        on=HRV_MERGE_KEYS,
        how="inner",
        suffixes=("_lin", "_nonlin"),
    )

    # HRV + WPPSI on Monogramme + DDN (keep all WPPSI IQ columns)
    df = df_hrv.merge(df_wppsi, on=WPPSI_MERGE_KEYS, how="inner")

    # --- Step 9: drop fully-empty feature columns --------------------------
    df = df.dropna(axis=1, how="all")

    return df


# ---------------------------------------------------------------------------
# Preprocessing (dataset.md §5.2, steps 9–10)
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame, target_column: str):
    """Split into X/y for a given target, drop rows missing that target,
    impute NaN with column mean, train/test split, and standardize
    features (fit on train only).

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    """
    id_cols = [c for c in ID_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols + TARGET_COLUMNS]

    # Work on a copy and keep only rows with a valid target value.
    df_t = df.dropna(subset=[target_column]).copy()

    X = df_t[feature_cols].copy()
    y = df_t[target_column].copy()

    # --- Step 10a: impute remaining NaN with column mean -------------------
    X = X.fillna(X.mean())

    # --- Train / test split (shared across all methods for this target) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- Step 10b: StandardScaler fit on train only ------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=feature_cols, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Metrics (dataset.md §9)
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, R² and RMSE. Per dataset.md §9 the headline metrics are
    MAE and R² (R² > 0 → better than predicting the mean)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"MAE": mae, "R2": r2, "RMSE": rmse}


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
    """Fit one method on scaled training data and return test-set metrics."""
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred)
    metrics["method"] = name
    return metrics


def main() -> None:
    if not METHOD_REGISTRY:
        print(
            "No methods registered in METHOD_REGISTRY. "
            "Add your regressors to model_evaluation.py before running."
        )
        return

    # Validate that all requested targets exist after merge.
    print("Loading and merging data (per Group A/dataset.md) ...")
    df = load_data()

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Target column(s) {missing} not found in merged data. "
            f"Available WPPSI columns: {[c for c in df.columns if c not in ID_COLS]}"
        )

    print(f"Merged dataset: {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"Targets to evaluate: {TARGET_COLUMNS}")

    all_results: list[dict] = []

    for target in TARGET_COLUMNS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        # Preprocess once per target: impute mean, split, standardize (fit on train).
        X_train, X_test, y_train, y_test, feature_cols = preprocess(df, target)

        print(
            f"Features: {len(feature_cols)} | "
            f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | "
            f"Target mean={y_train.mean():.2f}, SD={y_train.std():.2f}"
        )

        for name, build_model in METHOD_REGISTRY.items():
            print(f"\nEvaluating: {name}")
            metrics = evaluate_method(
                name, build_model, X_train, X_test, y_train, y_test
            )
            metrics["target"] = target
            all_results.append(metrics)
            print(
                f"  MAE={metrics['MAE']:.4f}  "
                f"R²={metrics['R2']:.4f}  "
                f"RMSE={metrics['RMSE']:.4f}"
            )

    # --- Comparison table -------------------------------------------------
    results_df = pd.DataFrame(all_results)
    if "R2" in results_df.columns:
        results_df = results_df.rename(columns={"R2": "R²"})

    cols = ["target", "method", "MAE", "R²", "RMSE"]
    results_df = results_df[cols].sort_values(["target", "MAE"]).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by target, then MAE — lower is better)")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Save results ------------------------------------------------------
    out_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nComparison table saved to: {out_path}")


if __name__ == "__main__":
    main()
