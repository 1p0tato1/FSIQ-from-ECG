#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline (all files are read as text and nothing is modified in place):

 1) WPPSI results file: remove the rows with empty cells; remove the same
    subjects from the two HRV files (linear and non-linear).
 2) HRV files: repair the rows whose values are shifted to the left
    (the last cells are empty because the values moved left by SHIFT columns).
 3) HRV files: keep only the rows whose "Passage" column equals "M0".
 4) HRV files: if a row has empty cells in EITHER HRV file, the corresponding
    subject is removed from BOTH HRV files and from the WPPSI file.
 5) WPPSI file: remove the subjects that do not appear in any HRV file.

Rows are matched on the subject NUMBER (first column of all files), NOT on the
initials alone (second column): initials are repeated across different
subjects (e.g. "DIM" appears twice in the WPPSI file). As a consistency check,
the date of birth is used (if it does not match, the row is NOT removed);
differences in the initials are only reported.

Usage:
    python pulisci_righe.py                      # looks for the files in the script's folder
    python pulisci_righe.py --folder /path       # different folder
    python pulisci_righe.py --mode all_empty     # WPPSI: remove only completely empty rows
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# ----------------------------- CONFIGURATION ------------------------------
RESULTS_FILE = "Résultats_WPPSI_III.csv"
HRV_FILES = ["HRV_Lineaire_Full.csv", "HRV_NonLineaire_Full.csv"]
SEPARATOR = "\t"
ENCODING = "utf-8"
RESULTS_HEADER_ROWS = 2   # the WPPSI file has 2 header rows
COL_NUMBER = 0            # first column: subject number
COL_NAME = 1              # second column: initials
COL_DOB = 2               # third column: date of birth (consistency check)
PASSAGE_HEADER = "Passage"  # HRV column used to filter the visits
PASSAGE_VALUE = "M0"        # only rows with this value are kept
SHIFT = 2                 # number of columns by which broken rows are shifted left
MISSING = "-"             # value used for missing data in the HRV files
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


def is_empty(df):
    return df.apply(lambda c: c.str.strip() == "")


def cell_kind(v):
    """'d' = missing marker, 'n' = number, 't' = text."""
    v = v.strip()
    if v == MISSING:
        return "d"
    try:
        float(v)
        return "n"
    except ValueError:
        return "t"


def fix_shifted_rows(data, file_name):
    """
    Repairs the rows whose last SHIFT cells are empty while everything else is
    filled (values shifted left by SHIFT columns). For each such row, the
    position where SHIFT missing cells must be re-inserted is the one that makes
    the row best match the kind of value (missing / number / text) usually found
    in each column of the well-formed rows. When several positions are equally
    good (a run of missing values) they all give the same row.
    """
    empty = is_empty(data)
    broken = empty.iloc[:, -SHIFT:].all(axis=1) & ~empty.iloc[:, :-SHIFT].any(axis=1)
    if not broken.any():
        return data

    reference = data[~empty.any(axis=1)].map(cell_kind)
    freq = {c: reference[c].value_counts(normalize=True).to_dict()
            for c in reference.columns}

    def score(values):
        return sum(freq[i].get(cell_kind(v), 0.0) for i, v in enumerate(values))

    print(f"  {file_name}: {int(broken.sum())} shifted rows repaired:")
    for idx in data.index[broken]:
        values = list(data.loc[idx].iloc[:-SHIFT])
        candidates = [values[:p] + [MISSING] * SHIFT + values[p:]
                      for p in range(len(values) + 1)]
        scores = [score(c) for c in candidates]
        best = max(scores)
        tied = [p for p, s in enumerate(scores) if abs(s - best) < 1e-9]
        distinct = {tuple(candidates[p]) for p in tied}
        data.loc[idx, :] = candidates[tied[0]]
        note = "" if len(distinct) == 1 else \
            f"  (WARNING: {len(distinct)} equally good versions, first one used)"
        print(f"    no. {data.loc[idx, COL_NUMBER]}: {SHIFT} missing cells inserted at "
              f"column {tied[0]}{'' if tied[0] == tied[-1] else '-' + str(tied[-1])}"
              f"{note}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--mode", choices=["any_empty", "all_empty"], default="any_empty",
                    help="any_empty (default): remove a WPPSI row if it has AT LEAST ONE "
                         "empty cell; all_empty: remove it only if ALL data cells are empty")
    args = ap.parse_args()
    folder = Path(args.folder)
    out_dir = folder / "puliti"
    out_dir.mkdir(exist_ok=True)

    # --- 1) WPPSI results file: empty rows ----------------------------------
    results_header, results = read(folder / RESULTS_FILE, RESULTS_HEADER_ROWS)
    empty = is_empty(results)
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

    # --- 2) + 3) HRV files: subjects, shifted rows, Passage -----------------
    hrv = {}   # file name -> (header + coding rows, data rows)
    for file_name in HRV_FILES:
        # 1 header row; the next row ("M/F", "0/1"...) is a coding row, kept as is
        header, data = read(folder / file_name, 1)
        coding = data.iloc[:1]
        data = data.iloc[1:]
        total = len(data)

        # subjects removed because of the WPPSI cleaning
        number = data[COL_NUMBER].str.strip()
        mask = number.isin(removed_numbers)

        different_dob = mask & (data[COL_DOB].str.strip() != number.map(dob_by_number))
        if different_dob.any():
            print(f"  WARNING {file_name}: rows with the same number but a different "
                  f"date of birth, NOT removed:")
            print(data[different_dob].iloc[:, :3].drop_duplicates().to_string(header=False))
            mask &= ~different_dob

        different_name = mask & (data[COL_NAME].str.strip() != number.map(name_by_number))
        if different_name.any():
            print(f"  NOTE {file_name}: same number and date of birth but different "
                  f"initials (removed anyway):")
            for n_, g in data[different_name].groupby(COL_NUMBER):
                print(f"    no. {n_}: WPPSI='{name_by_number[n_.strip()]}' "
                      f"HRV='{g[COL_NAME].iloc[0]}'")
        data = data[~mask]
        n_wppsi = int(mask.sum())

        # repair the rows shifted to the left
        data = fix_shifted_rows(data, file_name)

        # keep only Passage == M0 (column found by header name)
        names = [str(h).strip() for h in header.iloc[0]]
        if PASSAGE_HEADER not in names:
            sys.exit(f"Column '{PASSAGE_HEADER}' not found in {file_name}")
        col_passage = names.index(PASSAGE_HEADER)
        is_m0 = data[col_passage].str.strip() == PASSAGE_VALUE
        n_not_m0 = int((~is_m0).sum())
        data = data[is_m0]

        hrv[file_name] = (pd.concat([header, coding]), data)
        print(f"{file_name}: {total} rows -> {len(data)} "
              f"({n_wppsi} removed for the WPPSI cleaning, "
              f"{n_not_m0} removed because Passage != {PASSAGE_VALUE})")

    # --- 4) Empty cells in either HRV file -> remove the subject everywhere --
    empty_subjects = set()
    for file_name, (_, data) in hrv.items():
        rows_with_empty = is_empty(data).any(axis=1)
        if rows_with_empty.any():
            subjects = set(data[rows_with_empty][COL_NUMBER].str.strip())
            print(f"{file_name}: {len(subjects)} subjects with empty cells: "
                  f"{sorted(subjects, key=int)}")
            empty_subjects |= subjects
    if not empty_subjects:
        print("No HRV rows with empty cells (after repairing the shifted rows).")
    for file_name in hrv:
        top, data = hrv[file_name]
        hrv[file_name] = (top, data[~data[COL_NUMBER].str.strip().isin(empty_subjects)])
    kept = kept[~kept[COL_NUMBER].str.strip().isin(empty_subjects)]

    # --- 5) Subjects present only in the WPPSI file -------------------------
    present_in_hrv = [set(zip(d[COL_NUMBER].str.strip(), d[COL_DOB].str.strip()))
                      for _, d in hrv.values()]
    in_hrv = set().union(*present_in_hrv)
    keys = list(zip(kept[COL_NUMBER].str.strip(), kept[COL_DOB].str.strip()))
    wppsi_only = pd.Series([k not in in_hrv for k in keys], index=kept.index)
    if wppsi_only.any():
        print(f"{RESULTS_FILE}: subjects present only here (removed):")
        print(kept[wppsi_only].iloc[:, :3].to_string(header=False))
    kept = kept[~wppsi_only]

    # --- Save ---------------------------------------------------------------
    for file_name, (top, data) in hrv.items():
        save(top, data, out_dir / file_name)
    save(results_header, kept, out_dir / RESULTS_FILE)
    print(f"{RESULTS_FILE}: {len(results)} rows -> {len(kept)}")

    if present_in_hrv[0] != present_in_hrv[1]:
        print("WARNING: the two HRV files do not contain the same subjects; "
              "a subject is kept if it appears in at least one of them.")

    print(f"\nFiles saved in: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
