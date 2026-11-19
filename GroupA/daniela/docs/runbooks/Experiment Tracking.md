# Experiment Tracking

Every training invocation is one run. Code lives in `src/`, configuration in
`experiments/config.yaml`, artifacts in `experiments/runs/<run_id>/`, and the
aggregated comparison in `results/SUMMARY.md`.

## Layout

```text
GroupA/daniela/
├── src/
│   ├── preprocessing.py
│   ├── merge.py
│   ├── features.py
│   ├── feature_selection.py
│   ├── train.py
│   └── summarize.py
├── experiments/
│   ├── config.yaml
│   └── runs/
│       ├── run_001_auto_shift/
│       └── run_002_manual_shift/
├── notebooks/
│   ├── exploratory_correlation.ipynb
│   └── model_comparison.ipynb
├── docs/
│   ├── runbooks/
│   └── reports/Antwort/
├── results/
│   └── SUMMARY.md
└── .gitignore
```

## Defining a run

Add an entry under `runs:` in `experiments/config.yaml`:

```yaml
runs:
  - id: run_003_my_idea
    preprocessing:
      shift_repair: auto
    source: clean_data_auto
```

`source` must point to a directory under `data/` that contains
`merged_dataset_features.csv`. Use `src.preprocessing`, `src.merge` and
`src.features` to produce it.

## Running

```shell
python -m src.preprocessing --shift-repair auto
python -m src.merge --source clean_data_auto
python -m src.features --source clean_data_auto
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift

python -m src.summarize
```

`src.summarize` reads all `metrics.json` files and regenerates
`results/SUMMARY.md`. It is safe to run after each experiment.

## What is tracked vs ignored

Tracked:

- all code in `src/`
- `experiments/config.yaml`
- `results/SUMMARY.md`
- notebooks, runbooks and reports

Ignored (derived data):

- `data/clean_data_*/`
- `data/**/merged_dataset*.csv`
- `experiments/runs/*/data/`
- `experiments/runs/*/*.pkl`
- `experiments/runs/*/*.csv`
- `experiments/runs/*/plots/*.png`

Raw children's health data is never committed. The root `.gitignore` also
excludes CSV/PDF/PKL files globally.

## Adding a model

1. Add the estimator and its grid in `build_model_specs()` in `src/train.py`.
2. Add a boolean under `models:` in `experiments/config.yaml`.
3. Add the dependency to `requirements.txt` if it is a new package.
4. Re-run the pipeline and `src.summarize`.

Feature-selection changes go through `src/feature_selection.py` and the
`feature_selection:` block of the config.

## Reproducibility notes

`manifest.json` records:

- config file path and SHA-256 hash
- git commit at training time
- timestamp (UTC)
- source directory and feature file
- number of subjects/features, targets, models, best models
- CV settings
- Python and package versions

Results are deterministic for the same data, config and package versions:
random seeds are fixed, and the inner `GridSearchCV` parallelism does not change
the selected candidates.

## Legacy documents

The old result markdowns are kept for provenance, not as a source of truth:

```text
docs/reports/archive/legacy_results/
docs/reports/archive/legacy_docs/
```

Their numbers were produced with feature selection outside the CV and are not
directly comparable to current metrics.

## Related

- [Preprocessing](Preprocessing.md)
- [Data Merge](Data%20Merge.md)
- [Feature Cleaning](Feature%20Cleaning.md)
- [Training](Training.md)
