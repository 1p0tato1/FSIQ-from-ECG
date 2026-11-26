Project Restructure Prompt: GroupA/daniela/
# 1. Context & Goal
Restructure the ML exploration workspace at GroupA/daniela/ into a clean, reproducible, experiment-tracking layout. The current state is a flat directory with duplicated preprocessing scripts, result markdowns full of copy-pasted terminal output, per-target notebooks scattered in date-named subfolders, and no machine-readable experiment tracking.

The goal is a single source of truth per experiment: code in src/, configuration in experiments/config.yaml, per-run artifacts in experiments/runs/<run_id>/, exploration notebooks in notebooks/, and docs split between runbooks and reports. All derived artifacts (cleaned CSVs, merged tables, pickles, plots) must be gitignored.

# 2. Current State (Reference)
Location: GroupA/daniela/
## Existing files to migrate or consolidate:
- data_preprocessing.py — cleaning pipeline with automatic left-shift repair (fix_shifted_rows), output data/clean_data_shifted/
- data_preprocessing_m.py — nearly identical cleaning pipeline WITHOUT shift repair, output data/clean_data_m_shifted/ (shift repaired manually on CSVs)
- data_merge.py — merges WPPSI + linear HRV + nonlinear HRV on subject Numéro, output merged_dataset.csv
- data_features.py — numeric coercion, drop constant features, drop \|corr\| > 0.95, output merged_dataset_features.csv
- otherModels.ipynb — RF, LightGBM, XGBoost, SVR with nested CV
- regularizedModels.ipynb — Ridge, Lasso, ElasticNet, BayesianRidge with nested CV
- with_auto_shift/ / with_manual_shift/ — each contains other/ and regularized/ with 4 per-target notebooks (full-scale_IQ, verbal_IQ, performance_IQ, percentile_rank)
- docs/ — 12 markdown runbooks + Antwort/ folder with German theory, math, correlation CSVs, PDF
## Pipeline logic currently:
1. Preprocess (remove empty rows, match on subject NUMBER + DOB check, keep Passage == "M0", drop empty-cell subjects, optional shift repair)
2. Merge three sources into one table
3. Feature clean (numeric coercion, constant drop, correlation drop)
4. Train models via nested CV (outer 5-fold × inner 4-fold GridSearch), pipeline SimpleImputer(mean) → StandardScaler → Model, feature selection via SelectKBest + SelectFromModel + SHAP refinement
5. Save regression_model_{TARGET}.pkl
## Known issues to fix in restructure:
- data_preprocessing.py and data_preprocessing_m.py are ~80% duplicated code
- Result metrics live only in copy-pasted markdown tables, not machine-readable
- All models show negative R² (worse than mean predictor) — feature signal is weak; restructure should make experimentation cheaper, not promise better accuracy
- Must remain compatible with root-level evaluate_model.py for cross-group comparison (shared folds, bootstrapped CIs)
# 3. Target Directory Structure
GroupA/daniela/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # --shift-repair {auto,manual}; output data/clean_data_{auto,manual}/
│   ├── merge.py              # --source {clean_data_auto,clean_data_manual}; output merged_dataset.csv
│   ├── features.py           # --source ...; output merged_dataset_features.csv
│   └── train.py              # reproducible training entry point; reads experiments/config.yaml
├── experiments/
│   ├── config.yaml           # targets, models, CV, feature-selection, paths
│   └── runs/
│       ├── run_001_auto_shift/
│       │   ├── data/                 # gitignored: cleaned + merged + features
│       │   ├── metrics.json          # machine-readable metrics per target & model
│       │   ├── params.json           # best hyperparameters
│       │   ├── model.pkl             # refit best estimator (joblib)
│       │   ├── predictions.csv       # out-of-fold predictions (child_id, true, pred per target)
│       │   └── plots/                # pred_vs_actual.png, residuals.png, shap_summary.png
│       └── run_002_manual_shift/
├── notebooks/
│   ├── exploratory_correlation.ipynb
│   └── model_comparison.ipynb
├── docs/
│   ├── runbooks/
│   │   ├── Data Merge.md
│   │   ├── Preprocessing.md
│   │   └── Feature Cleaning.md
│   └── reports/
│       └── Antwort/                  # existing German theory/math docs moved here
├── results/
│   └── SUMMARY.md                    # auto-generated aggregated comparison across runs
└── .gitignore                        # updated to ignore derived artifacts

# 4. Module Specifications
## 4.1 src/preprocessing.py
Purpose: Single script replacing both data_preprocessing.py and data_preprocessing_m.py.
### CLI:
python -m src.preprocessing --folder <project_root> --shift-repair {auto,manual} --mode {any_empty,all_empty}
- --folder: default = repo root (parent of GroupA/)
- --shift-repair: auto → runs fix_shifted_rows() (SHIFT=2, MISSING="-") before Passage filtering; manual → skips shift repair (user repairs CSVs by hand afterwards)
- --mode: any_empty (default) or all_empty for WPPSI empty-row removal
### Behavior:
- Read Résultats_WPPSI_III.csv, HRV_Lineaire_Full.csv, HRV_NonLineaire_Full.csv from "folder"/data/raw/ (or existing data/ location)
- Remove WPPSI rows with empty cells (per --mode)
- Remove same subject numbers from both HRV files (DOB consistency check; if DOB differs, do NOT remove; report initial differences only)
- If --shift-repair auto: repair left-shifted rows via fix_shifted_rows() before Passage filter
- Keep only Passage == "M0" rows in HRV files
- Remove subjects with empty cells in EITHER HRV file from BOTH HRV files and WPPSI
- Remove WPPSI-only subjects (not in any HRV file)
- Write outputs to "folder"/data/clean_data_auto/ or "folder"/data/clean_data_manual/ depending on flag
- Do NOT modify input files in place
- Print summary counts per file
- 
Functions to preserve/extract: read(), save(), is_empty(), cell_kind(), fix_shifted_rows(), score() (only needed for auto).

## 4.2 src/merge.py
Purpose: Replace data_merge.py.
### CLI:
python -m src.merge --folder <project_root> --source {clean_data_auto,clean_data_manual} --how {inner,outer}
### Behavior:
- Read the three cleaned CSVs from data/<source>/
- Match on Numéro (subject number), NOT initials
- Suffix non-key columns: _WPPSI, _Linear, _NonLinear
- Check DOB consistency across sources after merge
- Write merged_dataset.csv to data/<source>/
- Print subject count and column count
## 4.3 src/features.py
Purpose: Replace data_features.py.
### CLI:
python -m src.features --folder <project_root> --source {clean_data_auto,clean_data_manual}
### Behavior:
- Read merged_dataset.csv from data/<source>/
- Detect ID columns, target column, feature columns
- Coerce features and target to numeric (errors → NaN)
- Drop rows missing target
- Drop constant features (≤1 unique value)
- Drop feature pairs with |corr| > threshold (configurable)
- Write merged_dataset_features.csv to data/<source>/
- ID columns preserved but not used as features
- Target included in output but excluded from features
- All WPPSI columns excluded from features (target leakage prevention — keep the fix already applied in notebooks)

Configurable constants (read from config.yaml or defaults): TARGET_COL, CORR_THRESHOLD, ID_COLUMNS.

## 4.4 src/train.py
Purpose: Reproducible training entry point. Reads experiments/config.yaml, runs one experiment, writes artifacts to experiments/runs/<run_id>/.
### CLI:
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift
### Behavior:
1. Load config (targets, models, CV splits, feature-selection flags, paths)
2. Load features from data/<source>/merged_dataset_features.csv
3. For each target in config:
	- Exclude all WPPSI columns + ID columns from feature matrix
	- Impute missing values with SimpleImputer(strategy="mean") inside CV folds only
	- Feature selection: SelectKBest (ANOVA f_regression, k=min(20, n_features)) → SelectFromModel (Lasso or RF, threshold="median") → fallback to SelectKBest if zero selected → optional SHAP refinement (top 20 by mean |SHAP|, shallow DecisionTree max_depth=5)
	- Nested CV: outer 5-fold KFold (shuffle, random_state=42), inner 4-fold KFold + GridSearchCV (scoring=neg_mean_absolute_error)
	- Pipeline: SimpleImputer(mean) → StandardScaler → Model
	- Models: Ridge, Lasso, ElasticNet, BayesianRidge, RandomForest, LightGBM, XGBoost, SVR (per config enable/disable)
	- Select best model by outer-CV MAE
	- Refit best model on all data via GridSearchCV, save as model.pkl
	- Generate out-of-fold predictions via cross_val_predict on outer folds → predictions.csv with columns: child_id, true_"target", pred_"target" (compatible with root evaluate_model.py format)
	- Compute metrics: MAE, RMSE, R², Pearson, Spearman per target/model → metrics.json
	- Save best hyperparameters → params.json
	- Save plots: predicted vs actual, residuals, SHAP summary → plots/
4. Write a run manifest recording config hash, git commit, timestamp

Output format for predictions.csv (to be compatible with root evaluate_model.py):

child_id,repeat,fold,true_Note composite_WPPSI,pred_Note composite_WPPSI,true_NS Verb._WPPSI,pred_NS Verb._WPPSI,...

One row per child per repeat of outer CV.

# 5. experiments/config.yaml Schema
```yaml
project:
  root: ".."
  data_dir: "data"

preprocessing:
  shift_repair: auto          # auto | manual
  mode: any_empty             # any_empty | all_empty

merge:
  how: inner                  # inner | outer

features:
  target_col: "Note composite_WPPSI"   # or list of targets
  corr_threshold: 0.95
  id_columns:
    - "Numéro"
    - "Monogramme_WPPSI"
    - "Monogramme_Linear"
    - "Monogramme_NonLinear"

cv:
  outer_folds: 5
  inner_folds: 4
  random_state: 42
  scoring: "neg_mean_absolute_error"
  repeats: 1

feature_selection:
  select_k_best: 20
  embedded_estimator: "lasso"     # lasso | random_forest | none
  shap_refinement: true
  shap_top_k: 20
  shap_tree_max_depth: 5

models:
  ridge: true
  lasso: true
  elasticnet: true
  bayesian_ridge: true
  random_forest: true
  lightgbm: true
  xgboost: true
  svr: true

runs:
  - id: run_001_auto_shift
    preprocessing:
      shift_repair: auto
    source: clean_data_auto
  - id: run_002_manual_shift
    preprocessing:
      shift_repair: manual
    source: clean_data_manual
```

# 6. Experiment Run Artifacts Spec
For each `experiments/runs/<run_id>/`:

| File | Format | Contents |
|------|--------|----------|
| `data/` | directory | Gitignored copy/symlink of cleaned + merged + features CSVs |
| `metrics.json` | JSON | `{ "target": { "model": { "MAE": float, "RMSE": float, "R2": float, "pearson": float, "spearman": float, "cv_mae_std": float }, ... }, ... }` |
| `params.json` | JSON | `{ "target": { "model": { "best_params": {...}, "best_cv_mae": float }, ... }, ... }` |
| `model.pkl` | joblib | Refit best estimator pipeline per target |
| `predictions.csv` | CSV | Out-of-fold predictions, child_id-indexed, compatible with `evaluate_model.py` |
| `plots/pred_vs_actual_"target".png` | PNG | Scatter plot predicted vs actual (out-of-fold) |
| `plots/residuals_"target".png` | PNG | Residual plot |
| `plots/shap_summary_"target".png` | PNG | SHAP summary bar plot |
# 7. Notebooks Requirements
- notebooks/exploratory_correlation.ipynb: correlation analysis, feature-target correlations, linear operations exploration (move large CSVs out of docs, keep only summaries)
- notebooks/model_comparison.ipynb: visual comparison across runs, load metrics.json from multiple run dirs
- Notebooks must call functions from src/, not duplicate logic
- Notebooks are for exploration only; they are not the source of truth for results
# 8. Docs Reorganization
- docs/runbooks/: move Data Merge.md, Data Preprocessing and automatic shifting.md, Data Preprocessing and manual shifting.md, Feature Cleaning.md, Training.md here. Update them to reference new CLI commands and directory layout.
- docs/reports/Antwort/: move existing Antwort/ folder here (German theory, math, Regression und Maschinelles Lernen.md, Methoden in Modell.md, correlation docs, PDF)
- Remove Correlation Coefficient.md, Other models + Auto shifting.md, etc. from root docs/ — results now live in experiments/runs/*/metrics.json and results/SUMMARY.md
- Important milestones.md: either move to docs/runbooks/ or archive; its content is now embodied in code structure
# 9. .gitignore Updates
Add:
```gitignore
# Derived data (gitignored - raw data stays local only, never committed)
GroupA/daniela/data/clean_data_*/
GroupA/daniela/data/**/merged_dataset*.csv
GroupA/daniela/experiments/runs/*/data/
GroupA/daniela/experiments/runs/*/*.pkl
GroupA/daniela/experiments/runs/*/*.csv
GroupA/daniela/experiments/runs/*/plots/*.png

# Large analysis CSVs not needed in git
GroupA/daniela/docs/reports/Antwort/*.csv

# Python
__pycache__/
*.pyc
.venv/
.idea/
.DS_Store
```

**Important:** Raw data files (HRV_Lineaire_Full.csv, HRV_NonLineaire_Full.csv, Résultats_WPPSI_III.csv) must remain in data/ locally but never committed per README warning about children's health data.

# 10. Migration Steps (Step-by-Step)
 10. Create directory structure (src/, experiments/runs/, notebooks/, docs/runbooks/, docs/reports/, results/)
 11. Write src/preprocessing.py by merging data_preprocessing.py + data_preprocessing_m.py; add --shift-repair flag
 12. Write src/merge.py from data_merge.py
 13. Write src/features.py from data_features.py; add config-driven constants
 14. Write src/train.py implementing nested CV, feature selection, metrics export, artifact writing
 15. Write experiments/config.yaml with run definitions for auto and manual shift
 16. Run the pipeline end-to-end for run_001_auto_shift and run_002_manual_shift; verify outputs match previous results (RandomForest/BayesianRidge MAE values should be comparable)
 17. Convert notebooks: move per-target notebook content into notebooks/ as exploration notebooks calling src/ functions; retire otherModels.ipynb and regularizedModels.ipynb from root
 18. Reorganize docs: move runbooks to docs/runbooks/, move Antwort/ to docs/reports/Antwort/, archive or remove result-specific markdowns
19. Write results/SUMMARY.md generator script (or manually generate first version) comparing all runs
20. Update .gitignore and verify git status shows no derived artifacts tracked
21. Verify compatibility with root evaluate_model.py: ensure predictions.csv format matches expected schema (child_id, true_"T", pred_"T")
# 11. End-to-End Data Pipeline Flow
raw data (data/*.csv, local only)
    │
    ▼
src/preprocessing.py --shift-repair auto|manual
    │
    ▼
data/clean_data_auto/ or data/clean_data_manual/  [gitignored]
    │
    ▼
src/merge.py --source clean_data_auto|clean_data_manual
    │
    ▼
data/clean_data_*/merged_dataset.csv              [gitignored]
    │
    ▼
src/features.py --source clean_data_auto|clean_data_manual
    │
    ▼
data/clean_data_*/merged_dataset_features.csv     [gitignored]
    │
    ▼
src/train.py --config experiments/config.yaml --run-id run_00X
    │
    ▼
experiments/runs/run_00X/{data/, metrics.json, params.json, model.pkl, predictions.csv, plots/}
    │
    ▼
results/SUMMARY.md (aggregated across runs)
# 12. Example Commands After Restructure

```shell
# Full pipeline for auto-shift experiment
python -m src.preprocessing --shift-repair auto
python -m src.merge --source clean_data_auto
python -m src.features --source clean_data_auto
python -m src.train --config experiments/config.yaml --run-id run_001_auto_shift
```

```shell
# Full pipeline for manual-shift experiment
python -m src.preprocessing --shift-repair manual
python -m src.merge --source clean_data_manual
python -m src.features --source clean_data_manual
python -m src.train --config experiments/config.yaml --run-id run_002_manual_shift
```

```shell
# Compare with root evaluate_model.py (cross-group compatible)
python evaluate_model.py --folds data/cleaned/folds.csv --cohort data/cleaned/cohort.csv \
    --predictions experiments/runs/run_001_auto_shift/predictions.csv
```



# 13. Acceptance Criteria
- data_preprocessing.py and data_preprocessing_m.py are removed; replaced by single src/preprocessing.py
- data_merge.py, data_features.py removed; replaced by src/merge.py, src/features.py
- otherModels.ipynb, regularizedModels.ipynb removed from root; notebooks live only in notebooks/
- with_auto_shift/ and with_manual_shift/ directories removed
- Running src/train.py produces metrics.json, params.json, model.pkl, predictions.csv, plots/ for each run
- results/SUMMARY.md exists and compares all runs
- git status is clean of derived artifacts (cleaned CSVs, merged CSVs, pickles, plots, large CSVs)
- predictions.csv format is compatible with root evaluate_model.py
- All docs in docs/runbooks/ reference the new CLI commands and directory layout
---
Please execute this restructure, creating all files, migrating logic, and verifying the pipeline runs end-to-end. Ask if any path or column name needs confirmation before overwriting existing files.