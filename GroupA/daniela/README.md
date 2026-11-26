# GroupA/daniela — FSIQ from neonatal HRV

Reproducible experiment workspace for predicting WPPSI-III IQ scores from
neonatal linear and non-linear HRV features (AuBE cohort, `Passage == "M0"`).

## Layout

```text
GroupA/daniela/
├── src/                     # all pipeline code
│   ├── preprocessing.py     # clean raw WPPSI + HRV files
│   ├── merge.py             # merge the three cleaned sources
│   ├── features.py          # numeric cleaning, leakage guard
│   ├── feature_selection.py # filter + embedded + SHAP selection inside CV
│   ├── train.py             # nested CV, metrics, models, plots
│   ├── summarize.py         # results/SUMMARY.md
│   └── config.py            # shared config/path helpers
├── experiments/
│   ├── config.yaml
│   └── runs/<run_id>/       # metrics.json, params.json, predictions.csv, ...
├── notebooks/               # exploration only
├── docs/
│   ├── runbooks/
│   └── reports/Antwort/     # German theory / math documents
├── results/SUMMARY.md
└── ../../data/              # raw + derived data (local only, gitignored)
```

## Quick start

From `GroupA/daniela`:

```shell
# Auto shift experiment
python -m src.preprocessing --shift-repair auto
python -m src.merge --source clean_data_auto
python -m src.features --source clean_data_auto
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift

# Manual shift experiment (hand-repaired files live in data/clean_data_manual/)
python -m src.preprocessing --shift-repair manual
python -m src.merge --source clean_data_manual
python -m src.features --source clean_data_manual
python -m src.train --config experiments/config.yaml --run-id run_002_manual_shift

# Aggregate
python -m src.summarize
```

The scripts also work from the repository root:

```shell
python GroupA/daniela/src/preprocessing.py --shift-repair auto
```

## Targets

- `Note composite_WPPSI` — Full-scale IQ
- `NS Verb._WPPSI` — Verbal IQ
- `NS Perf._WPPSI` — Performance IQ
- `Rang percentile_WPPSI` — Percentile rank

## Models

Ridge, Lasso, ElasticNet, BayesianRidge, RandomForest, LightGBM, XGBoost and SVR.
Each is tuned with nested CV (outer 5-fold, inner 4-fold GridSearchCV) and
leakage-free feature selection. The best model per target is selected by outer-CV
MAE.

## Results

All models are close to or worse than the mean predictor (negative R²). That is
the known weak-signal result for this cohort, not a pipeline failure. The
restructure makes experiments cheaper and honest; it does not promise better
accuracy.

See `results/SUMMARY.md` for current numbers and
`docs/reports/archive/legacy_results/` for the old (optimistic) tables.

## Data warning

Raw files (`Résultats_WPPSI_III.csv`, `HRV_Lineaire_Full.csv`,
`HRV_NonLineaire_Full.csv`) contain children's health data. They stay local and
are never committed. All derived CSVs, models and plots are gitignored.

## Documentation

- [Preprocessing](docs/runbooks/Preprocessing.md)
- [Data Merge](docs/runbooks/Data%20Merge.md)
- [Feature Cleaning](docs/runbooks/Feature%20Cleaning.md)
- [Training](docs/runbooks/Training.md)
- [Experiment Tracking](docs/runbooks/Experiment%20Tracking.md)
- [Important milestones](docs/runbooks/Important%20milestones.md)
