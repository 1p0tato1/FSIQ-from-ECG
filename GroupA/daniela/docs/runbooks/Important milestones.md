---
tags:
  - documentation
  - history
---
# Important milestones

This note records how the cleaning pipeline evolved. The executable source of
truth is now `src/`; this page explains why two shift strategies exist.

## Data cleaning

The data was cleaned in two ways.

### First version (no shift repair)

1. WPPSI results file: remove rows with empty cells; remove the same subjects
   from both HRV files.
2. HRV files: keep only `Passage == "M0"`.
3. HRV files: if a row has empty cells in either file, remove the subject from
   both HRV files and from WPPSI.
4. WPPSI file: remove subjects that do not appear in any HRV file.

### After discovering the column shift

1. WPPSI results file: remove rows with empty cells; remove the same subjects
   from both HRV files.
2. **HRV files: repair rows whose values are shifted left** (the last cells are
   empty because values moved left by `SHIFT` columns).
3. HRV files: keep only `Passage == "M0"`.
4. HRV files: remove subjects with remaining empty cells from both HRV files and
   from WPPSI.
5. WPPSI file: remove subjects that do not appear in any HRV file.

## Both versions were kept

The manual version was cleaned with the old `data_preprocessing_m.py`, then the
shift was repaired by hand on the saved CSVs (`data/clean_data_m_shifted/`). The
automatic version used `data_preprocessing.py`, which repaired the shift
in-script (`data/clean_data_shifted/`). Both were kept in case the automatic
repair altered the values.

During the restructure, both are reproduced by the single
`src/preprocessing.py`:

| Aspect | `--shift-repair manual` | `--shift-repair auto` |
|---|---|---|
| Repairs shifted HRV columns | **No** — the shift is left untouched for hand repair | **Yes** — `fix_shifted_rows()` re-inserts `SHIFT=2` missing cells (`-`) at the best-fitting column |
| Order of shift repair | N/A | Before `Passage == "M0"` filtering |
| Output directory | `data/clean_data_manual/` | `data/clean_data_auto/` |
| Shift config | None | `SHIFT = 2`, `MISSING = "-"` |
| Empty-cell removal across HRV | Skipped (the shifted rows are the ones to repair) | Applied after repair |
| Column shift repair | Performed manually on the cleaned CSVs afterwards | Performed automatically in-script |

Both modes share the same cleaning steps (WPPSI empty-row removal, subject
matching by NUMBER with a DOB consistency check, `Passage == "M0"` filtering,
and removal of WPPSI-only subjects) and neither modifies the input files in
place.

Current documentation: [Preprocessing](Preprocessing.md).

## Data merge

`src/merge.py` merges the three cleaned sources into one table on `Numéro`.
Details: [Data Merge](Data%20Merge.md).

## Clean features

`src/features.py` performs numeric coercion, constant-feature removal,
correlation pruning, and excludes every WPPSI column from the feature matrix.
Details: [Feature Cleaning](Feature%20Cleaning.md).

## Model exploration

The first experiments lived in `regularizedModels.ipynb` (Ridge, Lasso,
ElasticNet, BayesianRidge) and `otherModels.ipynb` (RandomForest, LightGBM,
XGBoost, SVR), with per-target copies under `with_auto_shift/` and
`with_manual_shift/`. Those notebooks were exploratory and selected features
before cross-validation.

The restructure replaced them with:

- `src/train.py` — one config-driven training entry point, nested CV, feature
  selection inside each fold;
- `experiments/runs/<run_id>/` — machine-readable metrics, params, predictions,
  models and plots;
- `notebooks/` — exploration only, never the source of truth;
- `results/SUMMARY.md` — aggregated comparison across runs.

Details: [Training](Training.md), [Experiment Tracking](Experiment%20Tracking.md).

## Next steps

1. Inspect `results/SUMMARY.md` and compare auto vs manual shift.
2. Use the notebooks for correlation and model-comparison exploration.
3. If new features or models are tried, add a run in `experiments/config.yaml`
   instead of editing a notebook.
4. Compare against the shared cross-group evaluation with the root
   `evaluate_model.py` and a run's `predictions.csv`.
