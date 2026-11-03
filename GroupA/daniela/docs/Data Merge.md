---
tags:
  - documentation
---
## Usage

Run from the **repo root**:

```shell
# Merge the automatically shifted cleaned data
python GroupA/daniela/data_merge.py --source clean_data_shifted

# Merge the manually shifted cleaned data
python GroupA/daniela/data_merge.py --source clean_data_m_shifted
```

# Data Merge

Merges cleaned WPPSI and HRV files into a single dataset per subject.

## Script

`GroupA/daniela/data_merge.py`

## Purpose

Reads the three cleaned CSVs (WPPSI results, linear HRV, non-linear HRV) from a
cleaned-data directory under `data/` and merges them into one table saved as
`merged_dataset.csv` in that same directory.

Rows are matched on subject **NUMBER** (column `Numéro`), NOT on initials, because
initials repeat across different subjects (e.g. "DIM" appears twice in the WPPSI file).
### Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--source` | `clean_data_shifted`, `clean_data_m_shifted` | (required) | Which cleaned-data subdirectory under `data/` to read from and save to |
| `--how` | `inner`, `outer` | `inner` | Merge strategy (see below) |
| `--folder` | path | repo root | Project root folder containing `data/` |

## Merge strategy (`--how`)

- **`inner`** (default): keeps only subjects present in **all three** files
  (WPPSI + linear HRV + non-linear HRV). No missing values from the merge itself.
- **`outer`**: keeps **all** subjects from any of the three files. Cells are left
  empty where a subject has no data in one of the files.

## Input files expected

All read from `data/<source>/`:

| File | Description |
|------|-------------|
| `Résultats_WPPSI_III.csv` | WPPSI IQ results (2 header rows) |
| `HRV_Lineaire_Full.csv` | Linear HRV features (1 header row + 1 coding row) |
| `HRV_NonLineaire_Full.csv` | Non-linear HRV features (1 header row + 1 coding row) |

## Output

Saved to `data/<source>/merged_dataset.csv`:

- Tab-separated, UTF-8
- First column: `Numéro` (subject number, the merge key)
- WPPSI columns get suffix `_WPPSI`
- Linear HRV columns get suffix `_Linear`
- Non-linear HRV columns get suffix `_NonLinear`
- One header row with column names, then data rows

## Column name collisions

Non-key columns are suffixed by source (`_WPPSI`, `_Linear`, `_NonLinear`) so that
columns with the same name across files (e.g. `Numéro`, `Monogramme`, `DDN`) do not
collide during the merge. The key column `Numéro` is kept without a suffix.

## Consistency checks

After merging, the script checks the date of birth (`DDN`) columns across the three
sources and prints the number of mismatches if any are found.

## Example output

```
Merged 190 subjects -> /.../data/clean_data_shifted/merged_dataset.csv
Columns: 299
Saved in: /.../data/clean_data_shifted
```

## Notes

- The script reads everything as text (no automatic NaN interpretation) to preserve
  exact cell contents.
- The WPPSI file is expected to have 2 header rows; HRV files have 1 header row plus
  1 coding row (e.g. M/F, 0/1) which is dropped before merging.
- Default `--folder` resolves to the repository root so the script works when run
  from anywhere as long as `--source` is given.