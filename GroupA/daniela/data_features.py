#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature processing on the merged WPPSI + HRV dataset.

Reads:  data/<source>/merged_dataset.csv
Writes: data/<source>/merged_dataset_features.csv

Does NOT modify merged_dataset.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ----------------------------- CONFIGURATION ------------------------------
SEPARATOR = "\t"
ENCODING = "utf-8"

# Columns that are identifiers / metadata, not features
ID_COLUMNS = ["Numéro", "Monogramme_WPPSI", "Monogramme_Linear", "Monogramme_NonLinear"]

# Target: choose which WPPSI score to predict (adjust to your needs)
# Available: "Note composite_WPPSI", "NS Verb._WPPSI", "NS Perf._WPPSI", "Rang percentile_WPPSI"
TARGET_COL = "Note composite_WPPSI"

# Correlation threshold for dropping redundant features
CORR_THRESHOLD = 0.95
# --------------------------------------------------------------------------


def load_merged(source_dir):
    path = source_dir / "merged_dataset.csv"
    if not path.exists():
        sys.exit(f"Merged file not found: {path}")
    return pd.read_csv(path, sep=SEPARATOR, encoding=ENCODING, dtype=str)


def detect_feature_columns(df):
    """Separate ID, target, and feature columns."""
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    target_col = TARGET_COL if TARGET_COL in df.columns else None

    # Everything else is a candidate feature
    feature_cols = [c for c in df.columns if c not in id_cols and c != target_col]
    return id_cols, target_col, feature_cols


def to_numeric_safe(df, cols):
    """Convert columns to numeric, coercing errors to NaN."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def drop_constant_features(df, feature_cols):
    """Remove features with zero or near-zero variance."""
    to_drop = []
    for c in feature_cols:
        if df[c].nunique(dropna=True) <= 1:
            to_drop.append(c)
    if to_drop:
        print(f"Dropping {len(to_drop)} constant features")
        df = df.drop(columns=to_drop)
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return df, feature_cols


def drop_highly_correlated(df, feature_cols, threshold=CORR_THRESHOLD):
    """Drop one of each pair of features with |corr| > threshold."""
    if len(feature_cols) < 2:
        return df, feature_cols
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features (> {threshold})")
        df = df.drop(columns=to_drop)
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return df, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["clean_data_shifted", "clean_data_m_shifted"],
                    required=True)
    ap.add_argument("--folder", default=str(Path(__file__).resolve().parent.parent.parent))
    args = ap.parse_args()

    folder = Path(args.folder)
    src_dir = folder / "data" / args.source
    if not src_dir.is_dir():
        sys.exit(f"Source directory not found: {src_dir}")

    out_dir = src_dir
    out_dir.mkdir(exist_ok=True)

    # --- Load ---
    df = load_merged(src_dir)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from merged_dataset.csv")

    id_cols, target_col, feature_cols = detect_feature_columns(df)
    print(f"ID columns: {id_cols}")
    print(f"Target: {target_col}")
    print(f"Candidate features: {len(feature_cols)}")

    # --- Convert features to numeric ---
    df = to_numeric_safe(df, feature_cols)

    if target_col:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # --- Drop rows missing target ---
    if target_col:
        before = len(df)
        df = df.dropna(subset=[target_col])
        print(f"Dropped {before - len(df)} rows missing target '{target_col}'")

    # --- Feature processing ---
    df, feature_cols = drop_constant_features(df, feature_cols)
    df, feature_cols = drop_highly_correlated(df, feature_cols)

    # --- Save processed table ---
    keep_cols = id_cols + ([target_col] if target_col else []) + feature_cols
    out = df[keep_cols].copy()
    out_path = out_dir / "merged_dataset_features.csv"
    out.to_csv(out_path, sep=SEPARATOR, index=False, encoding=ENCODING)

    print(f"\nSaved {out.shape[0]} rows x {out.shape[1]} columns -> {out_path}")
    print(f"Features kept: {len(feature_cols)}")


if __name__ == "__main__":
    sys.exit(main())