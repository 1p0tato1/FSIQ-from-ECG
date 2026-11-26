# Archive

Historical material kept for provenance. It is **not** the source of truth for
current results.

## `legacy_results/`

The old result markdowns with copy-pasted terminal output:

- `Correlation Coefficient.md`
- `Other models + Auto shifting.md`
- `Other Models + Manual shifting.md`
- `Regularized Models + Auto shifting.md`
- `Regularized Models + Manual shifting.md`

Their numbers were produced with feature selection performed once on the full
dataset, before cross-validation. That leaks test information, so the values are
optimistic and not directly comparable to the current leakage-free metrics in
`experiments/runs/*/metrics.json`.

For context, the old best auto-shift MAEs were approximately:

| Target | Model | MAE |
|---|---|---:|
| `Note composite_WPPSI` | BayesianRidge | 11.64 |
| `NS Verb._WPPSI` | BayesianRidge | 2.16 |
| `NS Perf._WPPSI` | BayesianRidge | 1.72 |
| `Rang percentile_WPPSI` | BayesianRidge | 23.66 |

The first value is close to the mean-predictor baseline.

## `legacy_docs/`

- `Data Preprocessing and manual shifting.md` — the old documentation for
  `data_preprocessing_m.py`. Its content is now in
  `docs/runbooks/Preprocessing.md`.

## `legacy_notebooks/`

- `otherModels.ipynb` — the old aggregated notebook for RandomForest, LightGBM,
  XGBoost and SVR.
- `regularizedModels.ipynb` — the old aggregated notebook for Ridge, Lasso,
  ElasticNet and BayesianRidge.

These were exploratory: feature selection ran once on the full dataset before
cross-validation. The per-target copies under `with_auto_shift/` and
`with_manual_shift/` were removed; their logic now lives in `src/train.py` and
`src/feature_selection.py`.
