#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature cleaning for the merged WPPSI + HRV table.

Reads ``<root>/data/<source>/merged_dataset.csv`` and writes
``merged_dataset_features.csv`` next to it.  The merged table is never modified.

Steps:
1. Detect ID columns, WPPSI target columns and candidate HRV/clinical features.
2. Coerce features and targets to numeric (errors -> NaN).
3. Drop rows that have no target at all.
4. Drop constant features (<= 1 unique non-null value).
5. Drop one of each feature pair with ``|corr| > corr_threshold``.
6. Keep every WPPSI column in the output (targets are needed by the multi-target
   trainer) but exclude **all** of them from the feature matrix to prevent
   target leakage.

Usage (from ``GroupA/daniela``)::

    python -m src.features --source clean_data_auto
    python -m src.features --source clean_data_manual
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        DEFAULT_TARGETS,
        find_project_root,
        get_id_columns,
        get_targets,
        is_id_column,
        load_config,
    )
except ImportError:  # direct script execution
    from config import (  # type: ignore
        DEFAULT_TARGETS,
        find_project_root,
        get_id_columns,
        get_targets,
        is_id_column,
        load_config,
    )

# ----------------------------- CONFIGURATION ------------------------------
SEPARATOR = "\t"
ENCODING = "utf-8"
DEFAULT_CORR_THRESHOLD = 0.95
IN_FILENAME = "merged_dataset.csv"
OUT_FILENAME = "merged_dataset_features.csv"
# --------------------------------------------------------------------------


def load_merged(src_dir: str | Path) -> pd.DataFrame:
    path = Path(src_dir) / IN_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Merged file not found: {path}. Run python -m src.merge first."
        )
    return pd.read_csv(path, sep=SEPARATOR, encoding=ENCODING, dtype=str,
                       keep_default_na=False)


def detect_columns(df: pd.DataFrame, id_columns: list[str],
                   target_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return ``(id_cols, wppsi_cols, feature_cols)``.

    ``wppsi_cols`` is every column that belongs to the WPPSI source.  They are
    preserved in the output but never used as features (target leakage).
    """
    id_cols = [c for c in df.columns if is_id_column(c, id_columns)]
    if "Numéro" in df.columns and "Numéro" not in id_cols:
        id_cols.insert(0, "Numéro")

    wppsi_cols = [c for c in df.columns if "WPPSI" in c]
    # Explicitly configured targets must survive even if their name does not
    # contain "WPPSI" (e.g. a renamed target).
    for target in target_columns:
        if target in df.columns and target not in wppsi_cols:
            wppsi_cols.append(target)

    feature_cols = [c for c in df.columns
                    if c not in id_cols and c not in wppsi_cols]
    return id_cols, wppsi_cols, feature_cols


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to numeric, coercing errors to NaN."""
    for column in cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def drop_constant_features(df: pd.DataFrame,
                           feature_cols: list[str], verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    to_drop = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if to_drop and verbose:
        print(f"Dropping {len(to_drop)} constant / all-missing features")
    if to_drop:
        df = df.drop(columns=to_drop)
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return df, feature_cols


def drop_highly_correlated(df: pd.DataFrame, feature_cols: list[str],
                           threshold: float = DEFAULT_CORR_THRESHOLD,
                           verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Drop one of each pair of features with ``|corr| > threshold``."""
    if len(feature_cols) < 2:
        return df, feature_cols
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop and verbose:
        print(f"Dropping {len(to_drop)} highly correlated features (> {threshold})")
    if to_drop:
        df = df.drop(columns=to_drop)
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return df, feature_cols


def run(folder: str | Path, source: str, config_path: str | Path | None = None,
        verbose: bool = True) -> Path:
    folder = Path(folder).expanduser().resolve()
    src_dir = folder / "data" / source
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    cfg = load_config(config_path) if config_path else (
        load_config() if (find_project_root() / "GroupA" / "daniela" /
                          "experiments" / "config.yaml").exists() else {}
    )
    # ``load_config`` resolves the default config; if it is unavailable we still
    # run with the documented defaults.
    if cfg:
        targets = get_targets(cfg)
        id_columns = get_id_columns(cfg)
        corr_threshold = float(cfg.get("features", {}).get(
            "corr_threshold", DEFAULT_CORR_THRESHOLD))
    else:
        targets = list(DEFAULT_TARGETS)
        id_columns = []
        corr_threshold = DEFAULT_CORR_THRESHOLD

    say = print if verbose else (lambda *a, **k: None)
    df = load_merged(src_dir)
    say(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {IN_FILENAME}")

    id_cols, wppsi_cols, feature_cols = detect_columns(df, id_columns, targets)
    say(f"ID columns ({len(id_cols)}): {id_cols}")
    say(f"WPPSI columns kept as targets, excluded from features "
        f"({len(wppsi_cols)}): {wppsi_cols}")
    say(f"Candidate features: {len(feature_cols)}")

    df = to_numeric_safe(df, feature_cols)
    present_targets = [c for c in wppsi_cols if c in df.columns]
    df = to_numeric_safe(df, present_targets)

    # Drop rows with no target at all (not rows missing one particular target).
    if present_targets:
        before = len(df)
        df = df.dropna(subset=present_targets, how="all")
        say(f"Dropped {before - len(df)} rows with no WPPSI target")
    for target in targets:
        if target in df.columns:
            say(f"  target '{target}': {int(df[target].isna().sum())} missing")

    df, feature_cols = drop_constant_features(df, feature_cols, verbose=verbose)
    df, feature_cols = drop_highly_correlated(
        df, feature_cols, threshold=corr_threshold, verbose=verbose)

    keep_cols = list(dict.fromkeys(id_cols + present_targets + feature_cols))
    out = df[keep_cols].copy()
    out_path = src_dir / OUT_FILENAME
    out.to_csv(out_path, sep=SEPARATOR, index=False, encoding=ENCODING)

    say(f"\nSaved {out.shape[0]} rows x {out.shape[1]} columns -> {out_path}")
    say(f"Features kept: {len(feature_cols)}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="Cleaned-data subdirectory under data/ "
                         "(e.g. clean_data_auto or clean_data_manual)")
    ap.add_argument("--folder", default=str(find_project_root()),
                    help="Project root folder containing data/ (default: repository root)")
    ap.add_argument("--config", default=None,
                    help="Config file (default: GroupA/daniela/experiments/config.yaml)")
    args = ap.parse_args(argv)
    try:
        run(args.folder, args.source, args.config)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
