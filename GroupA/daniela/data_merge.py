import argparse
import sys
from pathlib import Path

import pandas as pd

# ----------------------------- CONFIGURATION ------------------------------
WPPSI_FILE = "Résultats_WPPSI_III.csv"
LINEAR_FILE = "HRV_Lineaire_Full.csv"
NONLINEAR_FILE = "HRV_NonLineaire_Full.csv"
SEPARATOR = "\t"
ENCODING = "utf-8"

WPPSI_HEADER_ROWS = 2   # WPPSI has 2 header rows
HRV_HEADER_ROWS = 1     # HRV has 1 header row (+ 1 coding row kept below it)
COL_NUMBER = 0          # subject number = merge key
COL_NAME = 1            # initials (kept for reference, not used for matching)
COL_DOB = 2             # date of birth (consistency check)

# Merge strategy: "inner" = keep only subjects in ALL three files;
# "outer" = keep all subjects, fill missing with empty
MERGE_HOW = "inner"
# --------------------------------------------------------------------------


def read_raw(path, header_rows):
    """Read everything as text, no automatic NaN. Returns (headers, data)."""
    df = pd.read_csv(path, sep=SEPARATOR, header=None, dtype=str,
                     keep_default_na=False, encoding=ENCODING)
    headers = df.iloc[:header_rows].copy()
    data = df.iloc[header_rows:].copy()
    return headers, data


def save_with_headers(headers, data, path):
    """Write headers + data to CSV, preserving tab separation."""
    out = pd.concat([headers, data])
    out.to_csv(path, sep=SEPARATOR, header=False, index=False,
               encoding=ENCODING, lineterminator="\n")


def prepare_dataframe(headers, data, source_tag, skip_coding_row=False):
    """
    Convert raw headers+data into a clean DataFrame indexed by subject number.
    - source_tag: suffix added to non-key columns (e.g. '_WPPSI')
    - skip_coding_row: for HRV files, drop the first data row (coding row)
    """
    df = data.copy()
    if skip_coding_row:
        df = df.iloc[1:].copy()

    # Use column 0 (subject number) as the key; keep it as a regular column too
    df[COL_NUMBER] = df[COL_NUMBER].str.strip()

    # Build column names: use header row values where available, else generic
    if len(headers) > 0:
        header_row = [str(h).strip() for h in headers.iloc[0]]
    else:
        header_row = []

    # Ensure we have a name for every column
    n_cols = df.shape[1]
    names = []
    for i in range(n_cols):
        if i < len(header_row) and header_row[i] != "":
            names.append(header_row[i])
        else:
            names.append(f"Col{i}")

    df.columns = names

    # Add suffix to non-key columns to avoid collisions during merge
    new_cols = {}
    for col in df.columns:
        if col == df.columns[COL_NUMBER]:
            new_cols[col] = col  # keep key column name unchanged
        else:
            new_cols[col] = f"{col}{source_tag}"
    df = df.rename(columns=new_cols)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--how", choices=["inner", "outer"], default=MERGE_HOW,
                    help='inner: subjects present in ALL files; '
                         'outer: all subjects, missing cells left empty')
    args = ap.parse_args()

    folder = Path(args.folder)
    merged_table = folder / "merged_table"
    out_dir = folder / "merged"
    out_dir.mkdir(exist_ok=True)

    # --- Read WPPSI ---
    w_header, w_data = read_raw(merged_table / WPPSI_FILE, WPPSI_HEADER_ROWS)
    wppsI = prepare_dataframe(w_header, w_data, source_tag="_WPPSI")

    # --- Read Linear HRV ---
    l_header, l_data = read_raw(merged_table / LINEAR_FILE, HRV_HEADER_ROWS)
    linear = prepare_dataframe(l_header, l_data, source_tag="_Linear",
                               skip_coding_row=True)

    # --- Read Non-linear HRV ---
    n_header, n_data = read_raw(merged_table / NONLINEAR_FILE, HRV_HEADER_ROWS)
    nonlinear = prepare_dataframe(n_header, n_data, source_tag="_NonLinear",
                                  skip_coding_row=True)

    # --- Merge on subject number ---
    key_col = wppsI.columns[COL_NUMBER]

    merged = wppsI.merge(linear, on=key_col, how=args.how, suffixes=("", ""))
    merged = merged.merge(nonlinear, on=key_col, how=args.how, suffixes=("", ""))

    # --- Consistency check: DOB across sources ---
    dob_cols = [c for c in merged.columns if c.endswith("_DDN") or c == "DDN"]
    if len(dob_cols) >= 2:
        mismatches = 0
        for i in range(len(dob_cols)):
            for j in range(i + 1, len(dob_cols)):
                c1, c2 = dob_cols[i], dob_cols[j]
                mask = merged[c1].fillna("") != merged[c2].fillna("")
                if mask.any():
                    mismatches += int(mask.sum())
                    print(f"  DOB mismatch {c1} vs {c2}: {int(mask.sum())} rows")
        if mismatches == 0:
            print("DOB consistent across all sources.")

    # --- Save ---
    out_path = out_dir / OUT_FILE
    # Reconstruct headers for output: one row with column names
    header_out = pd.DataFrame([merged.columns], columns=merged.columns)
    save_with_headers(header_out, merged, out_path)

    print(f"\nMerged {len(merged)} subjects -> {out_path}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Files saved in: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())