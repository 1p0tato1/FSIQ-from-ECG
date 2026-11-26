#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean the WPPSI + HRV inputs (single replacement for the two old scripts).

The pipeline reads the three raw files as text and never modifies them in place:

1. WPPSI results file: remove rows with empty cells (``--mode`` decides whether
   one empty cell or a completely empty row is enough).
2. Remove the same subject numbers from both HRV files, using the date of birth
   (``DDN``) as a consistency check.  If the DOB differs, the row is *not*
   removed and a warning is printed.  Rows are matched on the subject NUMBER,
   never on the initials.
3. ``--shift-repair auto``: repair left-shifted rows (last ``SHIFT`` cells empty
   while the rest is filled) with :func:`fix_shifted_rows` before the visit
   filter.  ``manual``: leave the rows untouched so they can be repaired by hand
   in the output directory.
4. Keep only rows whose ``Passage`` column equals ``M0`` in both HRV files.
5. ``auto`` only: remove subjects that still have empty cells in either HRV file
   from both HRV files *and* from the WPPSI file.  In ``manual`` mode the
   shifted rows are kept on purpose: removing them would make the hand repair
   impossible.
6. Remove WPPSI subjects that do not appear in any HRV file.

Outputs (tab-separated, UTF-8, headers preserved) go to::

    data/clean_data_auto/      # --shift-repair auto
    data/clean_data_manual/    # --shift-repair manual

Usage (from ``GroupA/daniela``)::

    python -m src.preprocessing --shift-repair auto
    python -m src.preprocessing --shift-repair manual
    python -m src.preprocessing --shift-repair auto --mode all_empty
    python -m src.preprocessing --shift-repair manual --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:  # ``python -m src.preprocessing`` from GroupA/daniela
    from .config import (
        CLEAN_FILES,
        LINEAR_FILE,
        NONLINEAR_FILE,
        RESULTS_FILE,
        SOURCE_DIRS,
        find_project_root,
    )
except ImportError:  # ``python GroupA/daniela/src/preprocessing.py``
    from config import (  # type: ignore
        CLEAN_FILES,
        LINEAR_FILE,
        NONLINEAR_FILE,
        RESULTS_FILE,
        SOURCE_DIRS,
        find_project_root,
    )

# ----------------------------- CONFIGURATION ------------------------------
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
HRV_FILES = [LINEAR_FILE, NONLINEAR_FILE]
# --------------------------------------------------------------------------


def read(path: str | Path, header_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read everything as text, without interpreting anything (no automatic NaN)."""
    df = pd.read_csv(path, sep=SEPARATOR, header=None, dtype=str,
                     keep_default_na=False, encoding=ENCODING)
    return df.iloc[:header_rows].copy(), df.iloc[header_rows:].copy()


def save(header: pd.DataFrame, data: pd.DataFrame, path: str | Path) -> None:
    out = pd.concat([header, data])
    out.to_csv(path, sep=SEPARATOR, header=False, index=False,
               encoding=ENCODING, lineterminator="\n")


def is_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: c.str.strip() == "")


def cell_kind(value: str) -> str:
    """``'d'`` = missing marker, ``'n'`` = number, ``'t'`` = text."""
    value = value.strip()
    if value == MISSING:
        return "d"
    try:
        float(value)
        return "n"
    except ValueError:
        return "t"


def _score(freq: dict, values: list[str]) -> float:
    return sum(freq[i].get(cell_kind(v), 0.0) for i, v in enumerate(values))


def fix_shifted_rows(data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Repair rows whose last ``SHIFT`` cells are empty.

    For each broken row the position where ``SHIFT`` missing cells must be
    re-inserted is the one that makes the row best match the kind of value
    (missing / number / text) usually found in each column of the well-formed
    rows.  When several positions are equally good (a run of missing values)
    they all give the same row.
    """
    empty = is_empty(data)
    broken = empty.iloc[:, -SHIFT:].all(axis=1) & ~empty.iloc[:, :-SHIFT].any(axis=1)
    if not broken.any():
        return data

    reference = data[~empty.any(axis=1)].map(cell_kind)
    freq = {c: reference[c].value_counts(normalize=True).to_dict()
            for c in reference.columns}

    print(f"  {file_name}: {int(broken.sum())} shifted rows repaired:")
    for idx in data.index[broken]:
        values = list(data.loc[idx].iloc[:-SHIFT])
        candidates = [values[:p] + [MISSING] * SHIFT + values[p:]
                      for p in range(len(values) + 1)]
        scores = [_score(freq, c) for c in candidates]
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


def resolve_input_dir(folder: Path, input_dir: str | Path | None = None) -> Path:
    """Prefer ``data/raw/`` when it exists, otherwise use the legacy ``data/``."""
    if input_dir:
        return Path(input_dir).expanduser().resolve()
    raw = folder / "data" / "raw"
    if raw.is_dir() and any((raw / name).exists() for name in CLEAN_FILES):
        return raw
    return folder / "data"


def run(folder: str | Path, shift_repair: str = "auto", mode: str = "any_empty",
        input_dir: str | Path | None = None, overwrite: bool = False,
        verbose: bool = True) -> Path:
    """Run the cleaning pipeline and return the output directory."""
    folder = Path(folder).expanduser().resolve()
    in_dir = resolve_input_dir(folder, input_dir)
    out_dir = folder / "data" / SOURCE_DIRS[shift_repair]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manual output is hand-edited.  Never clobber it by accident.
    if shift_repair == "manual" and not overwrite and all(
            (out_dir / name).exists() for name in CLEAN_FILES):
        print(f"Reusing manually repaired files in {out_dir}")
        print("Pass --overwrite to regenerate them from the raw inputs.")
        return out_dir

    say = print if verbose else (lambda *a, **k: None)

    # --- 1) WPPSI results file: empty rows ----------------------------------
    results_header, results = read(in_dir / RESULTS_FILE, RESULTS_HEADER_ROWS)
    empty = is_empty(results)
    if mode == "any_empty":
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
    say(f"{RESULTS_FILE}: {len(removed)} rows removed because of empty cells")

    # --- 2) + 3) + 4) HRV files: subjects, shifted rows, Passage -------------
    hrv: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for file_name in HRV_FILES:
        # 1 header row; the next row ("M/F", "0/1"...) is a coding row, kept as is
        header, data = read(in_dir / file_name, 1)
        coding = data.iloc[:1]
        data = data.iloc[1:]
        total = len(data)

        # subjects removed because of the WPPSI cleaning
        number = data[COL_NUMBER].str.strip()
        mask = number.isin(removed_numbers)

        different_dob = mask & (data[COL_DOB].str.strip() != number.map(dob_by_number))
        if different_dob.any():
            say(f"  WARNING {file_name}: rows with the same number but a different "
                f"date of birth, NOT removed:")
            say(data[different_dob].iloc[:, :3].drop_duplicates().to_string(header=False))
            mask &= ~different_dob

        different_name = mask & (data[COL_NAME].str.strip() != number.map(name_by_number))
        if different_name.any():
            say(f"  NOTE {file_name}: same number and date of birth but different "
                f"initials (removed anyway):")
            for n_, group in data[different_name].groupby(COL_NUMBER):
                say(f"    no. {n_}: WPPSI='{name_by_number[n_.strip()]}' "
                    f"HRV='{group[COL_NAME].iloc[0]}'")
        data = data[~mask]
        n_wppsi = int(mask.sum())

        # repair the rows shifted to the left (automatic mode only)
        if shift_repair == "auto":
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
        say(f"{file_name}: {total} rows -> {len(data)} "
            f"({n_wppsi} removed for the WPPSI cleaning, "
            f"{n_not_m0} removed because Passage != {PASSAGE_VALUE})")

    # --- 5) Empty cells in either HRV file -> remove the subject everywhere --
    # Automatic mode repairs the shifted rows first, so any remaining empty cell
    # is a genuine problem.  Manual mode deliberately keeps shifted rows for the
    # hand repair; running this step would delete exactly those subjects.
    if shift_repair == "auto":
        empty_subjects: set[str] = set()
        for file_name, (_, data) in hrv.items():
            rows_with_empty = is_empty(data).any(axis=1)
            if rows_with_empty.any():
                subjects = set(data[rows_with_empty][COL_NUMBER].str.strip())
                say(f"{file_name}: {len(subjects)} subjects with empty cells: "
                    f"{sorted(subjects, key=int)}")
                empty_subjects |= subjects
        if not empty_subjects:
            say("No HRV rows with empty cells (after repairing the shifted rows).")
        for file_name in hrv:
            top, data = hrv[file_name]
            hrv[file_name] = (top, data[~data[COL_NUMBER].str.strip().isin(empty_subjects)])
        kept = kept[~kept[COL_NUMBER].str.strip().isin(empty_subjects)]
    else:
        say("Manual mode: shifted rows are kept for hand repair "
            "(empty-cell removal is skipped).")

    # --- 6) Subjects present only in the WPPSI file -------------------------
    present_in_hrv = [set(zip(d[COL_NUMBER].str.strip(), d[COL_DOB].str.strip()))
                      for _, d in hrv.values()]
    in_hrv = set().union(*present_in_hrv)
    keys = list(zip(kept[COL_NUMBER].str.strip(), kept[COL_DOB].str.strip()))
    wppsi_only = pd.Series([k not in in_hrv for k in keys], index=kept.index)
    if wppsi_only.any():
        say(f"{RESULTS_FILE}: subjects present only here (removed):")
        say(kept[wppsi_only].iloc[:, :3].to_string(header=False))
    kept = kept[~wppsi_only]

    # --- Save ---------------------------------------------------------------
    for file_name, (top, data) in hrv.items():
        save(top, data, out_dir / file_name)
    save(results_header, kept, out_dir / RESULTS_FILE)
    say(f"{RESULTS_FILE}: {len(results)} rows -> {len(kept)}")

    if present_in_hrv[0] != present_in_hrv[1]:
        say("WARNING: the two HRV files do not contain the same subjects; "
            "a subject is kept if it appears in at least one of them.")

    say(f"\nFiles saved in: {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=None,
                    help="Project root (default: repository root, the parent of GroupA/)")
    ap.add_argument("--shift-repair", choices=["auto", "manual"], default="auto",
                    help="auto: repair left-shifted HRV rows in-script; "
                         "manual: keep them for hand repair")
    ap.add_argument("--mode", choices=["any_empty", "all_empty"], default="any_empty",
                    help="any_empty (default): remove a WPPSI row if it has AT LEAST "
                         "ONE empty cell; all_empty: remove it only if ALL data cells "
                         "are empty")
    ap.add_argument("--input-dir", default=None,
                    help="Directory with the raw CSV files (default: <root>/data/raw "
                         "if it exists, else <root>/data)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Manual mode only: regenerate data/clean_data_manual/ even "
                         "if hand-repaired files already exist")
    args = ap.parse_args(argv)
    folder = Path(args.folder).resolve() if args.folder else find_project_root()
    run(folder=folder, shift_repair=args.shift_repair, mode=args.mode,
        input_dir=args.input_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
