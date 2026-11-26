# Predicting WPPSI-III IQ from neonatal HRV

This folder holds a leakage-free regression pipeline that tries to predict a child's WPPSI-III
Full-Scale IQ (**QIT**) from heart-rate-variability (HRV) features recorded at birth (period `M0`).
It compares five regression methods against a mean-predictor baseline using repeated nested
cross-validation, then produces plots, an error analysis, stored models and a written summary.

> **Research use only.** The source files contain health data. They, and every file derived from
> individual children, must stay on the local machine and must never be committed or sent to
> external tools.

- [Folder contents](#folder-contents)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Data preparation](#data-preparation)
- [Preprocessing pipeline](#preprocessing-pipeline)
- [Models and search spaces](#models-and-search-spaces)
- [Evaluation protocol](#evaluation-protocol)
- [Outputs](#outputs)
- [Diagnostics and error analysis](#diagnostics-and-error-analysis)
- [SHAP feature-count study](#shap-feature-count-study)
- [Reading the results](#reading-the-results)
- [Limitations](#limitations)
- [Adding the other IQ scores](#adding-the-other-iq-scores)

## Folder contents

| File | Role |
|---|---|
| `iq_pipeline.ipynb` | The analysis, from loading to `SUMMARY.md`. Run it top to bottom. |
| `pipeline_components.py` | Importable helpers: the two column filters, the preprocessing chain, the model registry (`METHOD_REGISTRY` factory) and the per-fold evaluation routine. |
| `env.example` | Template for the `.env` file that holds all paths and run options. |
| `readme.md` | This document. |

`pipeline_components.py` lives in its own file on purpose. A pipeline saved with joblib stores the
import path of its custom classes, so those classes must be importable wherever a model is
reloaded. The same module is also what the parallel worker processes import.

## Quick start

1. Install the dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`,
   `python-dotenv`, plus `lightgbm`, `xgboost` and `shap`. LightGBM and XGBoost are skipped
   automatically when missing; the SHAP study is skipped when `shap` is missing.
2. Copy `env.example` to `.env` and fill in the paths.
3. Start Jupyter (or Colab) **from this folder**, so that `.env` and `pipeline_components.py` are
   found, and run the notebook from the first cell.

A fast smoke test: set `OUTER_REPEATS=1` and `SEARCH_ITERATIONS=5` in `.env`. The full protocol
(5 folds × 10 repeats, budget 50) is much heavier; Random Forest dominates the run time. Finished
model/target runs are cached, so an interrupted session continues where it stopped
(`RESUME_RUNS=1`). The cache key covers the settings, the column layout, the preprocessing chain
and the search space, so changing any of them triggers a recomputation.

## Configuration

All settings are read from `.env` (see `env.example`).

| Variable | Meaning | Default |
|---|---|---|
| `HRV_LINEAR_FILE` | Linear HRV export (tab-separated) | required |
| `HRV_NONLINEAR_FILE` | Non-linear HRV export (tab-separated) | required |
| `WPPSI_FILE` | WPPSI-III results export (tab-separated) | required |
| `PROCESSED_DIR` | Local modelling tables, CV cache, fold-level best hyper-parameters | required |
| `MODELS_DIR` | Stored pipelines and scaler | required |
| `PLOTS_DIR` | Figures | required |
| `RESULTS_DIR` | Aggregated, shareable outputs | required |
| `TARGETS` | Comma-separated list among `QIT`, `QIV`, `QIP`, `NCL` | `QIT` |
| `RANDOM_SEED` | Seed for all splits and stochastic models | `42` |
| `OUTER_SPLITS` / `OUTER_REPEATS` | Outer folds and number of repetitions | `5` / `10` |
| `INNER_SPLITS` | Folds of the inner tuning loop | `5` |
| `SEARCH_ITERATIONS` | Maximum hyper-parameter combinations per fit | `50` |
| `N_JOBS` | Workers used across outer folds (`-1` = all cores) | `-1` |
| `RESUME_RUNS` | Reuse cached runs (`1`) or recompute (`0`) | `1` |
| `RUN_SHAP` | Run the SHAP study (`1`) or skip it (`0`) | `1` |
| `SHAP_MAX_FEATURES` | Largest predictor count examined by the SHAP study | `8` |

## Data preparation

The three input files are tab-separated, although they carry a `.csv` extension.

**HRV files.** Two header lines precede the data: feature names, then units. The unit line has no
entries for the three leading identifier columns, so it is shifted right by three positions. Each
column becomes a single label of the form `name_(unit)`. Known broken accents in the header
(`Numéro`, `Diabète`, `Prématuré`) are repaired.

**WPPSI file.** Two header lines again: a group label (written only on the first column of each
group and therefore propagated to the right) and a variable name. Labels take the form
`group_(variable)`, for instance `Note composite_(QIT)`. Column 2 is the child identifier
(`Monogramme`) and column 3 the date of birth (`DDN`); the loader checks this layout before using it.

**Modelling table.** For each requested target the notebook:

1. keeps only `M0` rows in each HRV file,
2. keeps WPPSI rows that have the requested score,
3. inner-joins the two HRV tables on `Monogramme` + `DDN` + `Passage`, then joins the score on
   `Monogramme` + `DDN` (only that one score column is brought in, so no other WPPSI value can
   leak into the predictors),
4. converts placeholder strings (`-`, `NaN`, blanks) to real missing values,
5. drops empty columns and columns with more than 80 % missing values,
6. converts everything to numbers and discards columns that are not numeric at all (text, dates),
7. removes identifier columns (`Numéro_x`, `Numéro_y`) and any remaining `Note composite` column.

Only shapes are printed while loading; individual rows are never displayed. The resulting table is
saved locally as `modeling_table_<TARGET>.csv` in `PROCESSED_DIR`.

Steps 1–7 depend neither on the outcome values nor on any train/test split (apart from discarding
children without a score), so they are applied once to the whole table. Everything that is *learned
from data* is deferred to the pipeline below.

## Preprocessing pipeline

`build_preprocessor()` returns five chained steps. Because they sit inside the scikit-learn
`Pipeline`, each cross-validation fit relearns them from its training rows only; the held-out rows
are only transformed.

| # | Step | Setting | Effect |
|---|---|---|---|
| 1 | `CorrelationPruner` | \|r\| > 0.95 | For every strongly correlated pair the later column is removed. Pairwise correlations tolerate missing values. |
| 2 | `LowVarianceFilter` | variance < 0.01 | Removes near-constant columns (variance ignores missing values). |
| 3 | `SimpleImputer` | mean | Fills the remaining gaps with the training-column mean. |
| 4 | `StandardScaler` | – | Zero mean, unit variance, statistics from training rows. |
| 5 | `SelectFromModel(Lasso)` | `alpha = 0.01` | Keeps columns that receive a non-zero Lasso coefficient. |

The notebook also fits this chain once on the full table and prints how many predictors survive each
stage. That report is descriptive only and is not used for evaluation.

A final pipeline therefore accepts a raw numeric feature table with the original columns and returns
predictions; there is no separate manual scaling step.

## Models and search spaces

Each entry of `METHOD_REGISTRY` bundles an estimator factory with its hyper-parameter values. The
estimators follow the scikit-learn interface (`fit`, `predict`, `get_params`, `set_params`).

| Model | Hyper-parameter values |
|---|---|
| ElasticNet | `alpha` = 20 values log-spaced from 1e-4 to 10; `l1_ratio` ∈ {0.1, 0.3, 0.5, 0.7, 0.9} |
| Random Forest | `n_estimators` ∈ {200, 500}; `max_depth` ∈ {3, 5, 7, None}; `min_samples_leaf` ∈ {5, 10, 20}; `max_features` ∈ {sqrt, log2} |
| LightGBM | `n_estimators` ∈ {100, 300}; `learning_rate` ∈ {0.01, 0.05, 0.1}; `max_depth` ∈ {3, 5, −1}; `min_child_samples` ∈ {10, 20}; `reg_alpha` ∈ {0.001, 0.1, 0.5}; `reg_lambda` ∈ {0, 0.1, 1} |
| XGBoost | `n_estimators` ∈ {100, 300}; `learning_rate` ∈ {0.01, 0.05, 0.1}; `max_depth` ∈ {3, 5, 7}; `min_child_weight` ∈ {1, 5}; `subsample` ∈ {0.8, 1.0}; `reg_alpha` ∈ {0, 0.1, 1}; `reg_lambda` ∈ {0.1, 1, 10} |
| SVR | `C` ∈ {0.1, 1, 10, 100}; `epsilon` ∈ {0.01, 0.1, 1}; `kernel` ∈ {rbf, linear}; `gamma` ∈ {scale, auto} |
| MeanBaseline | none: always predicts the mean of the training targets |

`make_search()` uses an exhaustive `GridSearchCV` when a space holds no more combinations than
`SEARCH_ITERATIONS`, and `RandomizedSearchCV` with that many samples otherwise. Candidates are
ranked by mean absolute error.

## Evaluation protocol

- **Outer loop:** `RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS)`. The splits are
  generated once per target and reused for every model, so models and baseline are compared on
  identical folds.
- **Inner loop:** shuffled `KFold(INNER_SPLITS)` on the outer training rows selects the
  hyper-parameters.
- **Metrics per outer fold:** MAE, MSE, RMSE and R², computed on the held-out rows.
- **Aggregation:** mean and standard deviation (population SD) over all outer folds × repeats.
- **Parallelism:** outer folds run in separate worker processes; each fold's inner search is
  single-process.
- **Warnings:** warnings raised during every fold are counted by category and stored with the fold
  scores, so convergence problems are visible afterwards.

## Outputs

| File | Folder | Content | Individual-level? |
|---|---|---|---|
| `model_comparison.csv` | `RESULTS_DIR` | `model, target, mae_mean, mae_std, mse_mean, mse_std, rmse_mean, rmse_std, r2_mean, r2_std` (baseline included) | no |
| `fold_scores.csv` | `RESULTS_DIR` | Metrics, warning counts and timing of every outer fold | no |
| `SUMMARY.md` | `RESULTS_DIR` | Best model per target, baseline comparison, limitations, next steps | no |
| `selected_features_<T>.csv` | `RESULTS_DIR` | Predictors kept by the Lasso screen when fitted on all children | no |
| `shap_*.csv` | `RESULTS_DIR` | Feature-count results and mean-\|SHAP\| rankings | no |
| `model_comparison_mae.png`, `model_comparison_r2.png`, `model_comparison_boxplots_<T>.png`, `error_analysis_<model>_<T>.png`, `shap_feature_count_mae_<T>.png` | `PLOTS_DIR` | Figures | no (but keep local) |
| `modeling_table_<T>.csv` | `PROCESSED_DIR` | Cleaned, unscaled table with gaps left in place | **yes – local only** |
| `fold_best_params.csv`, `cv_cache/` | `PROCESSED_DIR` | Chosen hyper-parameters per fold; cached runs including out-of-fold predictions | **yes – local only** |
| `best_<model>_<T>.joblib`, `scaler_<T>.joblib` | `MODELS_DIR` | Final fitted pipelines (filters → imputer → scaler → Lasso → regressor) and the fitted scaler | **yes – local only** |

The last notebook cell lists any data-like or model file that git tracks outside `RESULTS_DIR`.

## Diagnostics and error analysis

- **Best model per target:** lowest mean MAE and highest mean R² (baseline excluded), with a flag
  when mean R² is above zero.
- **Paired baseline comparison:** because folds are shared, the MAE difference to the baseline is
  computed fold by fold, together with the share of folds in which the model wins.
- **Stability review:** total warnings per model and the smallest/largest fold MAE.
- **Residual analysis of the best model:** each child is a test case once per repeat, so predictions
  are averaged over repeats. The figure shows predicted vs. actual, residuals vs. predicted and the
  residual distribution; a table lists the ten largest errors by row position (no identifiers), and
  the output reports how many children lie beyond two residual standard deviations.

## SHAP feature-count study

An exploratory analysis of how few predictors suffice. Within every training fold of a 5-fold split:

1. the preprocessing chain is refitted and a ranking model is trained on the retained predictors,
2. predictors are ranked by mean absolute SHAP value computed on the **training rows only**,
3. models restricted to the top 1…`SHAP_MAX_FEATURES` predictors are trained and scored on the
   untouched test rows.

Tree models use `TreeExplainer`; SVR uses a model-agnostic explainer with a 100-row background.
ElasticNet borrows the Random Forest ranking and is re-tuned for each predictor count. The
hyper-parameters of these models come from the final refit on all children, so this study stays
exploratory; the nested cross-validation is the reference performance estimate.

## Reading the results

- **R² below zero** means a model predicts worse than a constant equal to the mean of the test
  fold. The mean-predictor baseline is evaluated on the same folds, so MAE and R² can be compared
  with it directly. When no model clearly beats the baseline, the honest conclusion is that the
  features do not demonstrate useful predictive power under this protocol.
- Differences between models should be judged against the fold-to-fold standard deviation, which is
  large with a sample of this size.
- The figures in `SUMMARY.md` are produced automatically from the last full run; re-run the
  notebook after any change to the data or settings.

## Limitations

- Small sample (a couple of hundred children) with many correlated candidate predictors.
- One cohort and no external validation: cross-validation estimates are optimistic about
  generalisation to a new recruitment.
- Mean imputation and a linear screening step can hide non-linear or missingness-related signal.
- Pre-computed HRV summary indices only; no raw ECG and few clinical covariates.

## Adding the other IQ scores

`TARGET_COLUMNS` already maps `QIT`, `QIV`, `QIP` and `NCL` to their merged column labels. Set
`TARGETS=QIT,QIV,QIP,NCL` in `.env` to evaluate several scores in one run; tables, comparison
files, plots (grouped by target) and the summary are produced for every listed target. The
cross-validation cache is kept per target and model.
