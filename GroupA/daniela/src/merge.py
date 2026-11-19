#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the three cleaned sources into one table.

Rows are matched on the subject NUMBER (column ``Numéro``), never on initials.
Non-key columns are suffixed with their source (``_WPPSI``, ``_Linear``,
``_NonLinear``) so that identical header names do not collide.

Usage (from ``GroupA/daniela``)::

    python -m src.merge --source clean_data_auto
    python -m src.merge --source clean_data_manual --how outer

Reads ``<root>/data/<source>/Résultats_WPPSI_III.csv``, ``HRV_Lineaire_Full.csv``
and ``HRV_NonLineaire_Full.csv``; writes ``merged_dataset.csv`` next to them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from .config import (
        CLEAN_FILES,
        LINEAR_FILE,
        NONLINEAR_FILE,
        RESULTS_FILE,
        find_project_root,
    )
except ImportError:  # direct script execution
    from config import (  # type: ignore
        CLEAN_FILES,
        LINEAR_FILE,
        NONLINEAR_FILE,
        RESULTS_FILE,
        find_project_root,
    )

# ----------------------------- CONFIGURATION ------------------------------
SEPARATOR = "\t"
ENCODING = "utf-8"
WPPSI_HEADER_ROWS = 2   # WPPSI has 2 header rows
HRV_HEADER_ROWS = 1     # HRV has 1 header row (+ 1 coding row kept below it)
COL_NUMBER = 0          # subject number = merge key (by position)
COL_NAME = 1            # initials (kept for reference, not used for matching)
COL_DOB = 2             # date of birth (consistency check)
KEY_COLUMN_NAME = "Numéro"   # standard name for the merge key column
OUT_FILENAME = "merged_dataset.csv"
# --------------------------------------------------------------------------


def read_raw(path: str | Path, header_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read everything as text, no automatic NaN. Returns (headers, data)."""
    df = pd.read_csv(path, sep=SEPARATOR, header=None, dtype=str,
                     keep_default_na=False, encoding=ENCODING)
    headers = df.iloc[:header_rows].copy()
    data = df.iloc[header_rows:].copy()
    return headers, data


def save_with_headers(headers: pd.DataFrame, data: pd.DataFrame,
                      path: str | Path) -> None:
    """Write headers + data to CSV, preserving tab separation."""
    out = pd.concat([headers, data])
    out.to_csv(path, sep=SEPARATOR, header=False, index=False,
               encoding=ENCODING, lineterminator="\n")


def prepare_dataframe(headers: pd.DataFrame, data: pd.DataFrame, source_tag: str,
                      skip_coding_row: bool = False) -> pd.DataFrame:
    """Turn raw headers + data into a suffixed DataFrame.

    ``source_tag`` is appended to every non-key column.  For HRV files
    ``skip_coding_row`` removes the first data row (the "M/F", "0/1" coding row).
    """
    df = data.copy()
    if skip_coding_row:
        df = df.iloc[1:].copy()

    # Use column 0 (subject number) as the key; keep it as a regular column too.
    df[COL_NUMBER] = df[COL_NUMBER].str.strip()

    header_row = [str(h).strip() for h in headers.iloc[0]] if len(headers) > 0 else []
    names = []
    for i in range(df.shape[1]):
        if i < len(header_row) and header_row[i] != "":
            names.append(header_row[i])
        else:
            names.append(f"Col{i}")
    df.columns = names

    # Standardise the key column name so all sources merge on the same column.
    df = df.rename(columns={df.columns[COL_NUMBER]: KEY_COLUMN_NAME})

    # Add suffix to non-key columns to avoid collisions during merge.
    df = df.rename(columns={
        col: (col if col == KEY_COLUMN_NAME else f"{col}{source_tag}")
        for col in df.columns
    })
    return df


def check_dob(merged: pd.DataFrame, verbose: bool = True) -> int:
    """Compare the DOB columns that survived the merge; return mismatch count.

    HRV sources name the column ``DDN_*``.  The WPPSI file has a two-row header
    (group labels first, sub-labels second); the merge uses the group-label row,
    so its DOB column is ``Col2_WPPSI``.  Include it explicitly.
    """
    dob_cols = [c for c in merged.columns
                if c == "DDN" or c.startswith("DDN_") or c.endswith("_DDN")]
    if "Col2_WPPSI" in merged.columns and "Col2_WPPSI" not in dob_cols:
        dob_cols.insert(0, "Col2_WPPSI")
    mismatches = 0
    for i in range(len(dob_cols)):
        for j in range(i + 1, len(dob_cols)):
            c1, c2 = dob_cols[i], dob_cols[j]
            mask = merged[c1].fillna("") != merged[c2].fillna("")
            if mask.any():
                mismatches += int(mask.sum())
                if verbose:
                    print(f"  DOB mismatch {c1} vs {c2}: {int(mask.sum())} rows")
    if verbose and len(dob_cols) >= 2 and mismatches == 0:
        print("DOB consistent across all sources.")
    if verbose and len(dob_cols) < 2:
        print("NOTE: fewer than two DOB (_DDN) columns found; DOB check skipped.")
    return mismatches


def run(folder: str | Path, source: str, how: str = "inner",
        verbose: bool = True) -> Path:
    folder = Path(folder).expanduser().resolve()
    src_dir = folder / "data" / source
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    missing = [name for name in CLEAN_FILES if not (src_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing cleaned file(s) in {src_dir}: {missing}. "
            f"Run python -m src.preprocessing --shift-repair "
            f"{'auto' if 'auto' in source else 'manual'} first."
        )

    say = print if verbose else (lambda *a, **k: None)

    w_header, w_data = read_raw(src_dir / RESULTS_FILE, WPPSI_HEADER_ROWS)
    wppi = prepare_dataframe(w_header, w_data, source_tag="_WPPSI")

    l_header, l_data = read_raw(src_dir / LINEAR_FILE, HRV_HEADER_ROWS)
    linear = prepare_dataframe(l_header, l_data, source_tag="_Linear",
                               skip_coding_row=True)

    n_header, n_data = read_raw(src_dir / NONLINEAR_FILE, HRV_HEADER_ROWS)
    nonlinear = prepare_dataframe(n_header, n_data, source_tag="_NonLinear",
                                  skip_coding_row=True)

    merged = wppi.merge(linear, on=KEY_COLUMN_NAME, how=how, suffixes=("", ""))
    merged = merged.merge(nonlinear, on=KEY_COLUMN_NAME, how=how, suffixes=("", ""))

    check_dob(merged, verbose=verbose)

    out_path = src_dir / OUT_FILENAME
    header_out = pd.DataFrame([merged.columns], columns=merged.columns)
    save_with_headers(header_out, merged, out_path)

    say(f"\nMerged {len(merged)} subjects -> {out_path}")
    say(f"Columns: {len(merged.columns)}")
    say(f"Saved in: {src_dir}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    default_folder = find_project_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=str(default_folder),
                    help="Project root folder containing data/ (default: repository root)")
    ap.add_argument("--source", required=True,
                    help="Cleaned-data subdirectory under data/ "
                         "(e.g. clean_data_auto or clean_data_manual)")
    ap.add_argument("--how", choices=["inner", "outer"], default="inner",
                    help="inner: subjects present in ALL files (default); "
                         "outer: all subjects, missing cells left empty")
    args = ap.parse_args(argv)
    try:
        run(args.folder, args.source, args.how)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
