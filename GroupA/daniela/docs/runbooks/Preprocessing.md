# Preprocessing

Single entry point: `src/preprocessing.py`. It replaces the old
`data_preprocessing.py` (automatic shift repair) and `data_preprocessing_m.py`
(no in-script repair). Both modes share the same matching, visit filtering and
subject bookkeeping; only the shift step differs.

## Usage

Run from `GroupA/daniela`:

```shell
# Automatic shift repair -> data/clean_data_auto/
python -m src.preprocessing --shift-repair auto

# No in-script repair, for hand repair afterwards -> data/clean_data_manual/
python -m src.preprocessing --shift-repair manual

# Options
python -m src.preprocessing --shift-repair auto --mode all_empty
python -m src.preprocessing --shift-repair manual --overwrite
python -m src.preprocessing --shift-repair auto --input-dir /path/to/raw
```

From the repository root:

```shell
python GroupA/daniela/src/preprocessing.py --shift-repair auto
```

## Arguments

| Argument | Choices | Default | Meaning |
|---|---|---|---|
| `--folder` | path | repository root | Project root that owns `data/` |
| `--shift-repair` | `auto`, `manual` | `auto` | Repair left-shifted HRV rows in-script (`auto`) or leave them for hand repair (`manual`) |
| `--mode` | `any_empty`, `all_empty` | `any_empty` | WPPSI: remove a row if **any** cell is empty, or only if **all** data cells are empty |
| `--input-dir` | path | `data/raw/` if present, else `data/` | Directory containing the three raw CSVs |
| `--overwrite` | flag | off | Manual mode only: regenerate `data/clean_data_manual/` even if hand-repaired files exist |

## Input files

The script looks in `data/raw/` first and falls back to the legacy `data/`
location:

| File | Description |
|---|---|
| `Résultats_WPPSI_III.csv` | WPPSI-III results (2 header rows) |
| `HRV_Lineaire_Full.csv` | Linear HRV features (1 header row + 1 coding row) |
| `HRV_NonLineaire_Full.csv` | Non-linear HRV features (1 header row + 1 coding row) |

Raw files are never modified in place.

## Pipeline

1. **WPPSI empty rows** — remove rows with empty cells. `--mode any_empty`
   removes a row if at least one cell is empty; `all_empty` only removes rows
   whose data cells (from the Age column onwards) are all empty.
2. **Match removed subjects in both HRV files** — the subject NUMBER
   (first column) is the key, never the initials. The date of birth is a
   consistency check: if the DOB differs, the HRV row is **not** removed and a
   warning is printed. Initial differences are reported but do not prevent
   removal.
3. **Shift repair** — `auto` only: `fix_shifted_rows()` re-inserts `SHIFT = 2`
   missing cells (`-`) into rows whose last two cells are empty. The insertion
   position is chosen to best match the typical value kind (missing / number /
   text) of each column. `manual` leaves the rows untouched so they can be
   repaired by hand in `data/clean_data_manual/`.
4. **Visit filter** — keep only `Passage == "M0"` in both HRV files.
5. **Empty-cell subjects** — `auto` only: after repair, any subject with a
   remaining empty cell in either HRV file is removed from both HRV files and
   from WPPSI. `manual` skips this step on purpose: the shifted rows are
   exactly the ones that need hand repair.
6. **WPPSI-only subjects** — remove WPPSI subjects that do not appear in any
   HRV file.

## Outputs

```text
data/clean_data_auto/
  Résultats_WPPSI_III.csv
  HRV_Lineaire_Full.csv
  HRV_NonLineaire_Full.csv

data/clean_data_manual/
  Résultats_WPPSI_III.csv
  HRV_Lineaire_Full.csv
  HRV_NonLineaire_Full.csv
```

Tab-separated, UTF-8, original header rows preserved. The directories are
gitignored.

## Manual mode and reproducibility

`manual` writes the cleaned files **before** any shift repair. Repair them by
hand, then run merge/features/train. On later runs the script detects the three
existing files and reuses them instead of overwriting the hand edits. Use
`--overwrite` only when you deliberately want a fresh unrepaired copy.

The legacy hand-repaired files from `data/clean_data_m_shifted/` were migrated
to `data/clean_data_manual/` during the restructure, so `run_002_manual_shift`
reproduces the previous manual-shift experiment.

## Functions preserved

`read()`, `save()`, `is_empty()`, `cell_kind()`, `fix_shifted_rows()` and the
internal scoring helper are kept in `src/preprocessing.py`. Behaviour of the
automatic mode is byte-identical to the old `data_preprocessing.py`; this was
verified by re-running both on the same raw inputs.

## Related

- [Data Merge](Data%20Merge.md)
- [Feature Cleaning](Feature%20Cleaning.md)
- [Experiment Tracking](Experiment%20Tracking.md)
- Historical notes: [Important milestones](Important%20milestones.md)
