# Feature Cleaning

`src/features.py` turns `merged_dataset.csv` into a modelling table. It replaces
the old `data_features.py`. The merged table is never modified.

## Usage

Run from `GroupA/daniela`:

```shell
python -m src.features --source clean_data_auto
python -m src.features --source clean_data_manual
```

From the repository root:

```shell
python GroupA/daniela/src/features.py --source clean_data_auto
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--source` | yes | — | Cleaned-data directory under `data/` |
| `--folder` | no | repository root | Project root containing `data/` |
| `--config` | no | `experiments/config.yaml` | Config with targets, ID columns and correlation threshold |

## Configuration

Read from `experiments/config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `features.targets` | four WPPSI scores | Targets kept in the output and used by the trainer |
| `features.target_col` | `Note composite_WPPSI` | Fallback when `targets` is absent |
| `features.corr_threshold` | `0.95` | Drop one of each feature pair with `|corr|` above this |
| `features.id_columns` | `Numéro`, `Monogramme_*` | Identifier columns preserved but never used as features |

Additional per-source metadata columns (`DDN_*`, `Date_*`, `Filename_*`,
`Recorder_*`, `Passage_*`) are treated as IDs by name, even if they are not
listed explicitly.

## Pipeline

1. Load `merged_dataset.csv` as text.
2. Detect:
   - ID / metadata columns;
   - all WPPSI columns (targets and sub-scores);
   - candidate HRV / clinical features.
3. Coerce features and targets to numeric; non-numeric cells become `NaN`.
4. Drop rows that have no WPPSI target at all. A row missing one target is kept;
   the trainer drops it only for that target.
5. Drop constant / all-missing features (`<= 1` unique non-null value).
6. Drop one of each pair of features with `|corr| > corr_threshold`.
7. Write the cleaned table.

**Leakage guard:** every WPPSI column is excluded from the feature matrix. The
targets stay in the output so that the multi-target trainer can use them, but
they are never features. The old notebooks applied this fix ad hoc; it is now
enforced in the pipeline.

## Output

```text
data/<source>/merged_dataset_features.csv
```

Columns: ID/metadata columns, all WPPSI columns (targets), then the selected
HRV/clinical features. Tab-separated, UTF-8. Gitignored.

The script prints the detected columns, the number of dropped targets-missing
rows, constant features, correlated features, and the final shape.

## Notes

- Missing feature values stay as `NaN` on purpose. Imputation happens inside the
  CV folds in `src/train.py` (`SimpleImputer(strategy="mean")`), not here, so no
  information leaks across folds.
- The feature count is data-dependent. On the current cohort it is 59 features
  for both auto and manual shift.

## Related

- [Data Merge](Data%20Merge.md)
- [Training](Training.md)
- [Experiment Tracking](Experiment%20Tracking.md)
