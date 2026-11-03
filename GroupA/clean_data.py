#!/usr/bin/env python3
"""
1) Removes from the WPPSI results file the rows with empty cells, and removes
   from the two HRV files (linear and non-linear) the rows of the subjects
   that were removed.
2) Keeps only the HRV rows whose "Passage" column equals "M0".
3) Finally removes from the WPPSI file the subjects that do not appear in
   either of the two HRV files (after cleaning).

Rows are matched on the subject NUMBER (first column of all files), NOT on the
initials alone (second column): initials are repeated across different
subjects (e.g. "DIM" appears twice in the WPPSI file), so deleting by initials
would also remove valid subjects. As a consistency check, the date of birth is
used (if it does not match, the row is NOT removed); differences in the
initials are only reported.

Usage:
    python clean_data.py                      # looks for the files in the script's folder
    python clean_data.py --folder /path       # different folder
    python clean_data.py --mode all_empty     # remove only completely empty rows
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# ----------------------------- CONFIGURATION ------------------------------
RESULTS_FILE = "../data/Résultats WPPSI III.csv"
HRV_FILES = ["../data/HRV_Lineaire_Full.csv", "../data/HRV_NonLineaire_Full.csv"]
SEPARATOR = "\t"
ENCODING = "utf-8"
RESULTS_HEADER_ROWS = 2   # the WPPSI file has 2 header rows
COL_NUMBER = 0            # first column: subject number
COL_NAME = 1              # second column: initials
COL_DOB = 2               # third column: date of birth (consistency check)
PASSAGE_HEADER = "Passage"  # HRV column used to filter the visits
PASSAGE_VALUE = "M0"        # only rows with this value are kept
# --------------------------------------------------------------------------


def read(path, header_rows):
    """Reads everything as text, without interpreting anything (no automatic NaN)."""
    df = pd.read_csv(path, sep=SEPARATOR, header=None, dtype=str,
                     keep_default_na=False, encoding=ENCODING)
    return df.iloc[:header_rows].copy(), df.iloc[header_rows:].copy()


def save(header, data, path):
    out = pd.concat([header, data])
    out.to_csv(path, sep=SEPARATOR, header=False, index=False,
               encoding=ENCODING, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--mode", choices=["any_empty", "all_empty"], default="any_empty",
                    help="any_empty (default): remove a row if it has AT LEAST ONE empty cell; "
                         "all_empty: remove it only if ALL data cells are empty")
    args = ap.parse_args()
    folder = Path(args.folder)
    out_dir = (folder / "../data/cleaned").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Results file ----------------------------------------------------
    results_header, results = read(folder / RESULTS_FILE, RESULTS_HEADER_ROWS)
    empty = results.apply(lambda c: c.str.strip() == "")
    if args.mode == "any_empty":
        to_remove = empty.any(axis=1)
    else:
        to_remove = empty.iloc[:, 3:].all(axis=1)  # from the Age column onwards

    removed = results[to_remove]
    kept = results[~to_remove]
    removed_numbers = set(removed[COL_NUMBER].str.strip())
    name_by_number = dict(zip(removed[COL_NUMBER].str.strip(),
                              removed[COL_NAME].str.strip()))
    dob_by_number = dict(zip(removed[COL_NUMBER].str.strip(),
                             removed[COL_DOB].str.strip()))

    print(f"{RESULTS_FILE}: {len(removed)} rows removed because of empty cells")
    present_in_hrv = []   # sets of (number, DOB) present in each HRV file

    # --- 2) HRV files -------------------------------------------------------
    for file_name in HRV_FILES:
        # 1 header row; the next row ("M/F", "0/1"...) is a coding row
        # and must be kept
        header, data = read(folder / file_name, 1)
        coding = data.iloc[:1]
        data = data.iloc[1:]

        number = data[COL_NUMBER].str.strip()
        mask = number.isin(removed_numbers)

        # consistency check on the date of birth
        different_dob = mask & (data[COL_DOB].str.strip() != number.map(dob_by_number))
        if different_dob.any():
            print(f"  WARNING {file_name}: rows with the same number but a different "
                  f"date of birth, NOT removed:")
            print(data[different_dob].iloc[:, :3].drop_duplicates().to_string(header=False))
            mask &= ~different_dob

        # report different initials (e.g. "M S" in WPPSI vs "MBS" in HRV)
        different_name = mask & (data[COL_NAME].str.strip() != number.map(name_by_number))
        if different_name.any():
            print(f"  NOTE {file_name}: same number and date of birth but different "
                  f"initials (removed anyway):")
            for n_, g in data[different_name].groupby(COL_NUMBER):
                print(f"    no. {n_}: WPPSI='{name_by_number[n_.strip()]}' "
                      f"HRV='{g[COL_NAME].iloc[0]}'")

        remaining = data[~mask]

        # keep only the rows with Passage == "M0" (column found by header name)
        names = [str(h).strip() for h in header.iloc[0]]
        if PASSAGE_HEADER not in names:
            sys.exit(f"Column '{PASSAGE_HEADER}' not found in {file_name}")
        col_passage = names.index(PASSAGE_HEADER)
        is_m0 = remaining[col_passage].str.strip() == PASSAGE_VALUE
        n_not_m0 = int((~is_m0).sum())
        remaining = remaining[is_m0]
        present_in_hrv.append(set(zip(remaining[COL_NUMBER].str.strip(),
                                      remaining[COL_DOB].str.strip())))
        save(pd.concat([header, coding]), remaining, out_dir / Path(file_name).name)
        print(f"{file_name}: {len(data)} rows -> {len(remaining)} "
              f"({mask.sum()} removed for the WPPSI cleaning, "
              f"{n_not_m0} removed because Passage != {PASSAGE_VALUE})")

    # --- 3) Subjects present only in the WPPSI file -------------------------
    in_hrv = set().union(*present_in_hrv)
    keys = list(zip(kept[COL_NUMBER].str.strip(), kept[COL_DOB].str.strip()))
    wppsi_only = pd.Series([k not in in_hrv for k in keys], index=kept.index)
    if wppsi_only.any():
        print(f"{RESULTS_FILE}: subjects present only here (removed):")
        print(kept[wppsi_only].iloc[:, :3].to_string(header=False))
    kept = kept[~wppsi_only]
    save(results_header, kept, out_dir / Path(RESULTS_FILE).name)
    print(f"{RESULTS_FILE}: {len(results)} rows -> {len(kept)}")

    # warning if the two HRV files do not contain the same subjects
    if present_in_hrv[0] != present_in_hrv[1]:
        print("WARNING: the two HRV files do not contain the same subjects; "
              "a subject is kept if it appears in at least one of them.")

    print(f"\nFiles saved in: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
