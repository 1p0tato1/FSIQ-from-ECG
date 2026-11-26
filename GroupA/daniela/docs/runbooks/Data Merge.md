# Data Merge

`src/merge.py` merges the three cleaned CSVs into one table per subject. It
replaces the old `data_merge.py`.

## Usage

Run from `GroupA/daniela`:

```shell
# Automatically shifted data
python -m src.merge --source clean_data_auto

# Manually repaired data
python -m src.merge --source clean_data_manual

# Keep subjects that are missing from one source
python -m src.merge --source clean_data_auto --how outer
```

From the repository root:

```shell
python GroupA/daniela/src/merge.py --source clean_data_auto
```

## Arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--source` | `clean_data_auto`, `clean_data_manual` (any directory under `data/` with the three cleaned files) | required | Cleaned-data directory to read from and write to |
| `--how` | `inner`, `outer` | `inner` | `inner`: subjects present in all three files; `outer`: union, missing cells left empty |
| `--folder` | path | repository root | Project root containing `data/` |

## Inputs

Read from `data/<source>/`:

| File | Description |
|---|---|
| `Résultats_WPPSI_III.csv` | WPPSI results (2 header rows) |
| `HRV_Lineaire_Full.csv` | Linear HRV (1 header row + 1 coding row) |
| `HRV_NonLineaire_Full.csv` | Non-linear HRV (1 header row + 1 coding row) |

Run [Preprocessing](Preprocessing.md) first.

## Merge key

Rows are matched on the subject **NUMBER**, stored as `Numéro`. Initials are
kept for reference but are never used for matching: the same initials occur for
different children.

## Column naming

Non-key columns receive a source suffix:

- `_WPPSI` — WPPSI columns
- `_Linear` — linear HRV columns
- `_NonLinear` — non-linear HRV columns

The merge key `Numéro` is not suffixed. This prevents collisions such as
`Monogramme`, `DDN`, `Date` or `Recorder` across sources.

## Output

```text
data/<source>/merged_dataset.csv
```

Tab-separated, UTF-8, one header row followed by one row per subject. The file
is gitignored.

The script prints the number of subjects, the number of columns and the output
path.

## Consistency checks

After the merge, the date-of-birth columns are compared:

- `Col2_WPPSI` — WPPSI DOB (the two-row WPPSI header means this column is named
  by position after the group-label header row is used)
- `DDN_Linear`
- `DDN_NonLinear`

If the dates differ for a subject, the mismatch is reported. A missing DOB
column is reported as a skipped check, not treated as an error.

## Related

- [Preprocessing](Preprocessing.md)
- [Feature Cleaning](Feature%20Cleaning.md)
- [Experiment Tracking](Experiment%20Tracking.md)
