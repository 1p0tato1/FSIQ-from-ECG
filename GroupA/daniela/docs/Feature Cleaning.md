---
tags:
  - documentation
---

## Usage

Run from the **repo root**:

```shell
# Automatically shifted data
python3 GroupA/daniela/data_features.py --source clean_data_shifted

# Manually shifted data
python3 GroupA/daniela/data_features.py --source clean_data_m_shifted
```

# Feature Cleaning

Documents `GroupA/daniela/data_features.py` — feature processing on the merged WPPSI + HRV dataset.

## Purpose

Reads `merged_dataset.csv` (produced by `data_merge.py`) and produces a processed copy
`merged_dataset_features.csv` ready for modeling.

**Important:** This script does **NOT** modify `merged_dataset.csv`. Your central merged
table stays intact as the clean source; the processed table is a separate file.

## Pipeline relationship

```
data_preprocessing.py   →  data/clean_data_*/puliti/          (cleaned files)
data_merge.py           →  data/clean_data_*/merged_dataset.csv  (central merged table)
data_features.py        →  data/clean_data_*/merged_dataset_features.csv  (processed copy)
evaluate_model.py       ←  reads merged_dataset_features.csv
```
### Arguments

| Argument | Choices | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| `--source` | `clean_data_shifted`, `clean_data_m_shifted` | Yes | — | Which cleaned-data subdirectory under `data/` to read from and save to |
| `--folder` | path | No | repo root | Project root folder containing `data/` |

## Input

All read from `data/<source>/`:

| File | Description |
|------|-------------|
| `merged_dataset.csv` | Central merged table (WPPSI + linear HRV + non-linear HRV) |

Required: you must run `data_merge.py` first to generate this file.

## Configuration

Edit the top of `GroupA/daniela/data_features.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TARGET_COL` | `"Note composite_WPPSI"` | WPPSI score to predict. Options: `"Note composite_WPPSI"`, `"NS Verb._WPPSI"`, `"NS Perf._WPPSI"`, `"Rang percentile_WPPSI"` |
| `CORR_THRESHOLD` | `0.95` | Drop one of each feature pair with \|corr\| above this |
| `ID_COLUMNS` | `["Numéro", "Monogramme_WPPSI", "Monogramme_Linear", "Monogramme_NonLinear"]` | Identifier columns excluded from features |
| `SEPARATOR` | `"\t"` | Field delimiter (tab) |
| `ENCODING` | `"utf-8"` | File encoding |

## What the script does

1. **Load** `merged_dataset.csv` as text (no automatic NaN interpretation).
2. **Detect columns**: separates ID columns, target column, and candidate feature columns.
3. **Convert to numeric**: all feature columns and the target are coerced to numeric (errors become `NaN`).
4. **Drop rows missing the target**: subjects without a WPPSI score are removed.
5. **Drop constant features**: features with zero or near-zero variance (≤ 1 unique value) are removed.
6. **Drop highly correlated features**: for each pair of features with \|correlation\| > `CORR_THRESHOLD`, one is removed.
7. **Save** `merged_dataset_features.csv` in the same directory, tab-separated, UTF-8, no index.

## Output

| File | Description |
|------|-------------|
| `merged_dataset_features.csv` | Processed table with ID columns, target, and kept features |

Columns kept:
- `Numéro` (subject number, merge key)
- `Monogramme_Linear`, `Monogramme_NonLinear` (initials, for reference)
- Target column (e.g. `Note composite_WPPSI`)
- All remaining feature columns after constant/correlation filtering

## Example output

```shell
Loaded 190 rows x 299 columns from merged_dataset.csv
ID columns: ['Numéro', 'Monogramme_Linear', 'Monogramme_NonLinear']
Target: Note composite_WPPSI
Candidate features: 295
Dropped 0 rows missing target 'Note composite_WPPSI'
Dropping 185 constant features
Dropping 40 highly correlated features (> 0.95)

Saved 190 rows x 74 columns -> /path/to/data/clean_data_shifted/merged_dataset_features.csv
Features kept: 70
```

```shell
python3 GroupA/daniela/data_features.py --source clean_data_m_shifted
Loaded 191 rows x 299 columns from merged_dataset.csv
ID columns: ['Numéro', 'Monogramme_Linear', 'Monogramme_NonLinear']
Target: Note composite_WPPSI
Candidate features: 295
Dropped 0 rows missing target 'Note composite_WPPSI'
Dropping 185 constant features
Dropping 40 highly correlated features (> 0.95)

Saved 191 rows x 74 columns -> /path/to/data/clean_data_m_shifted/merged_dataset_features.csv
Features kept: 70
```

## Notes

- Row counts and feature counts may vary slightly depending on your data and `CORR_THRESHOLD`.
- The target column is included in the output but not counted as a feature.
- If you change `TARGET_COL`, make sure the column exists in `merged_dataset.csv`.
- Scaling/normalization (z-score, min-max) is intentionally **not** done here — it should be
  applied inside your model pipeline (e.g. in `evaluate_model.py`) to avoid data leakage.