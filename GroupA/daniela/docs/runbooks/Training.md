# Training

`src/train.py` is the reproducible training entry point. One invocation runs one
experiment defined in `experiments/config.yaml` and writes a self-contained run
directory.

## Quick start

From `GroupA/daniela`:

```shell
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift
python -m src.train --config experiments/config.yaml --run-id run_002_manual_shift
```

Useful overrides for quick checks:

```shell
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift \
    --targets "Note composite_WPPSI" --models Ridge SVR --n-jobs 4

python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift \
    --source clean_data_manual          # override the run's source directory
```

## Prerequisites

Run the upstream steps first:

```shell
python -m src.preprocessing --shift-repair auto
python -m src.merge --source clean_data_auto
python -m src.features --source clean_data_auto
```

`src/train.py` reads:

```text
data/<source>/merged_dataset_features.csv
```

## What the trainer does

For **each target** in `features.targets`:

1. Drops rows where that target is missing (the other targets keep their rows).
2. Builds the pipeline:

   ```text
   SimpleImputer(strategy="mean")
       -> FeatureSelector
       -> StandardScaler()
       -> model
   ```

3. Runs **nested cross-validation**:
   - outer: `KFold(n_splits=5, shuffle=True, random_state=42)` (configurable,
     optionally repeated);
   - inner: `KFold(n_splits=4, shuffle=True, random_state=42)` with
     `GridSearchCV(scoring="neg_mean_absolute_error")`.
4. Evaluates every enabled model and selects the best by mean outer-fold MAE
   (`cv_mae_mean`).
5. Refits every model on all children to record full-data hyperparameters, and
   refits the best one as the saved estimator.
6. Writes out-of-fold predictions, metrics, plots and models.

The fold assignment is generated once for the whole table, so every target is
evaluated on the same children/folds as far as its valid rows allow.

## Feature selection (inside every fold)

`src/feature_selection.py` implements the old notebook steps as a transformer:

1. `SelectKBest(f_regression, k=min(20, n_features))`.
2. `SelectFromModel` with `Lasso(alpha=0.1)` (or a Random Forest if configured),
   `threshold="median"`. If it selects zero features, the KBest set is kept.
3. Optional SHAP refinement: fit a shallow `DecisionTreeRegressor(max_depth=5)`,
   rank features by mean `|SHAP|`, keep the top `shap_top_k` (default 20).

Because the selector is a pipeline step, it is re-fitted on each training fold.
This is the main methodological change from the old notebooks, which selected
features once on the full dataset and therefore leaked test information. New
metrics are more conservative and are **not directly comparable** to the old
tables.

## Models and grids

Enabled per `models:` in the config:

| Model | Hyperparameters searched |
|---|---|
| Ridge | `alpha` in `[0.01, 0.1, 1, 10, 100]` |
| Lasso | `alpha` in `[0.001, 0.01, 0.1, 1, 10]` |
| ElasticNet | `alpha` in `[0.001, 0.01, 0.1, 1]`, `l1_ratio` in `[0.1, 0.5, 0.9]` |
| BayesianRidge | none |
| RandomForest | `n_estimators` in `[100, 200]`, `max_depth` in `[None, 10, 20]`, `min_samples_split` in `[2, 5]` |
| LightGBM | `n_estimators` in `[100, 200]`, `num_leaves` in `[31, 63]`, `learning_rate` in `[0.05, 0.1]` |
| XGBoost | `n_estimators` in `[100, 200]`, `max_depth` in `[3, 6]`, `learning_rate` in `[0.05, 0.1]` |
| SVR | `C` in `[0.1, 1, 10]`, `kernel` in `[rbf, linear]`, `gamma` in `[scale, auto]` |

All models are fitted inside the same preprocessing/selection pipeline and use
`random_state=42` where supported.

## Metrics

For each target/model, `metrics.json` contains:

| Key | Meaning |
|---|---|
| `MAE` | Mean absolute error on pooled out-of-fold predictions |
| `RMSE` | Root mean squared error on the same predictions |
| `R2` | Coefficient of determination; negative = worse than the mean predictor |
| `pearson`, `spearman` | Correlation between true and predicted scores |
| `cv_mae_mean`, `cv_mae_std` | Mean and standard deviation of the outer-fold MAEs |
| `cv_r2_mean`, `cv_r2_std` | Same for R² |
| `n` | Number of children used for the target |

## Run artifacts

```text
experiments/runs/<run_id>/
  data/                       copy of cleaned + merged + feature CSVs (gitignored)
  metrics.json                metrics per target/model
  params.json                 best hyperparameters per target/model + CV MAE
  best_models.json            selected model per target
  selected_features.json      features kept by the best pipeline per target
  model.pkl                   dict {target: fitted pipeline}
  models/<target>.pkl         one fitted pipeline per target
  predictions.csv             out-of-fold predictions
  plots/pred_vs_actual_<target>.png
  plots/residuals_<target>.png
  plots/shap_summary_<target>.png
  manifest.json               config hash, git commit, timestamp, environment
```

Target names in filenames are slugified (spaces and punctuation become `_`).

### predictions.csv

```text
child_id,repeat,fold,true_<target>,predicted_<target>,pred_<target>,...
```

- One row per child per repeat, using only children that have a true value and a
  prediction for **every** configured target. On the current data this excludes
  the two children without a percentile rank.
- `predicted_<target>` is the column name expected by the root
  `evaluate_model.py`. `pred_<target>` is written as an alias for the shorthand
  used in the restructure specification; `evaluate_model.py` ignores it.
- Fold numbering is 1-based; repeats start at 1.

Cross-group compatibility was verified by feeding a run's `predictions.csv`
(plus a folds/cohort table derived from it) to `evaluate_model.evaluate()`.

Note: `NS Verb._WPPSI` and `NS Perf._WPPSI` are scaled WPPSI sub-scores (roughly
7–16), not IQ-scale values. `evaluate_model.py` emits a heuristic warning that
their predictions are not around 100; that warning is expected for these two
targets.

### Loading a saved model

The pipeline contains `src.feature_selection.FeatureSelector`, so Python needs
`GroupA/daniela` on `sys.path` when unpickling. Run from `GroupA/daniela`:

```python
from src.models import load_model, load_target_model

models = load_model("experiments/runs/run_001_auto_shift/model.pkl")
best_full_scale = load_target_model(
    "experiments/runs/run_001_auto_shift/model.pkl", "Note composite_WPPSI")
predictions = best_full_scale.predict(X_new)
```

`model.pkl` is a dict `{target: pipeline}`; `models/<target>.pkl` contains one
pipeline directly.

## Known result

All models are close to or worse than the mean baseline (negative R²). This is
the known weak-signal result, not a bug in the restructure. The point of the new
layout is to make experiments cheap and honest, not to promise higher accuracy.

For reference, the old auto-shift notebooks reported (with feature selection
outside CV):

| Target | Model | MAE |
|---|---|---:|
| `Note composite_WPPSI` | BayesianRidge | 11.64 |
| `NS Verb._WPPSI` | BayesianRidge | 2.16 |
| `NS Perf._WPPSI` | BayesianRidge | 1.72 |
| `Rang percentile_WPPSI` | BayesianRidge | 23.66 |

The first value is essentially the mean-predictor baseline. See
`results/SUMMARY.md` for the current, leakage-free numbers.

## Related

- [Feature Cleaning](Feature%20Cleaning.md)
- [Experiment Tracking](Experiment%20Tracking.md)
- [Preprocessing](Preprocessing.md)
