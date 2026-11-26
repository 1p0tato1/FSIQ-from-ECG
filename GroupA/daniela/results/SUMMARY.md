# Experiment summary

Generated: 2026-11-18 11:53

Metrics are computed on out-of-fold predictions with feature selection **inside** each CV fold. They are therefore more conservative than the old notebook tables, which selected features once on the full dataset.

Negative R² means the model is worse than predicting the training-fold mean — the HRV signal is weak for this cohort.

## Runs

| Run | Source | Children | Features | Best models | Commit |
|---|---|---:|---:|---|---|
| `run_001_auto_shift` | `clean_data_auto` | 190 | 59 | Note composite_WPPSI: Lasso, NS Verb._WPPSI: Lasso, NS Perf._WPPSI: SVR, Rang percentile_WPPSI: ElasticNet | `0d93d250` |
| `run_002_manual_shift` | `clean_data_manual` | 191 | 59 | Note composite_WPPSI: SVR, NS Verb._WPPSI: SVR, NS Perf._WPPSI: Lasso, Rang percentile_WPPSI: SVR | `0d93d250` |

## Best model per target

Selection criterion: lowest mean outer-fold MAE (`cv_mae_mean`).

### Note composite_WPPSI

| Run | Best model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std |
|---|---|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | Lasso | 11.730 | 14.525 | -0.028 | -0.029 | -0.138 | 0.807 |
| `run_002_manual_shift` | SVR | 11.625 | 14.368 | 0.001 | 0.030 | 0.049 | 1.025 |

### NS Verb._WPPSI

| Run | Best model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std |
|---|---|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | Lasso | 2.164 | 2.780 | -0.005 | -0.096 | -0.086 | 0.227 |
| `run_002_manual_shift` | SVR | 2.185 | 2.811 | -0.016 | -0.096 | -0.099 | 0.149 |

### NS Perf._WPPSI

| Run | Best model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std |
|---|---|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | SVR | 1.694 | 2.267 | -0.020 | 0.011 | -0.070 | 0.277 |
| `run_002_manual_shift` | Lasso | 1.695 | 2.256 | -0.006 | 0.028 | -0.029 | 0.212 |

### Rang percentile_WPPSI

| Run | Best model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std |
|---|---|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | ElasticNet | 23.698 | 27.611 | -0.038 | 0.009 | -0.034 | 0.663 |
| `run_002_manual_shift` | SVR | 23.525 | 27.620 | -0.036 | -0.030 | 0.004 | 1.700 |

## All models

### Note composite_WPPSI

| Run | Model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | Lasso | 11.730 | 14.525 | -0.028 | -0.029 | -0.138 | 0.807 | 190 |
| `run_001_auto_shift` | ElasticNet | 11.794 | 14.655 | -0.046 | -0.045 | -0.087 | 0.832 | 190 |
| `run_001_auto_shift` | BayesianRidge | 11.868 | 14.714 | -0.055 | -0.049 | -0.083 | 0.821 | 190 |
| `run_001_auto_shift` | Ridge | 11.950 | 14.825 | -0.071 | -0.044 | -0.074 | 0.843 | 190 |
| `run_001_auto_shift` | SVR | 12.045 | 14.842 | -0.073 | -0.102 | -0.159 | 0.944 | 190 |
| `run_001_auto_shift` | XGBoost | 12.366 | 15.328 | -0.144 | 0.010 | 0.011 | 0.654 | 190 |
| `run_001_auto_shift` | RandomForest | 12.839 | 15.822 | -0.219 | -0.099 | -0.124 | 0.902 | 190 |
| `run_001_auto_shift` | LightGBM | 13.156 | 15.874 | -0.227 | -0.068 | -0.097 | 0.571 | 190 |
| `run_002_manual_shift` | SVR | 11.625 | 14.368 | 0.001 | 0.030 | 0.049 | 1.025 | 191 |
| `run_002_manual_shift` | Lasso | 11.660 | 14.436 | -0.009 | -0.126 | -0.079 | 0.955 | 191 |
| `run_002_manual_shift` | ElasticNet | 11.696 | 14.538 | -0.023 | -0.022 | 0.005 | 0.860 | 191 |
| `run_002_manual_shift` | BayesianRidge | 11.712 | 14.496 | -0.017 | -0.009 | 0.010 | 0.861 | 191 |
| `run_002_manual_shift` | Ridge | 11.720 | 14.533 | -0.022 | 0.016 | 0.028 | 0.837 | 191 |
| `run_002_manual_shift` | LightGBM | 12.557 | 15.539 | -0.169 | -0.065 | -0.081 | 0.957 | 191 |
| `run_002_manual_shift` | XGBoost | 12.604 | 15.580 | -0.175 | -0.062 | -0.074 | 0.923 | 191 |
| `run_002_manual_shift` | RandomForest | 12.752 | 15.948 | -0.231 | -0.093 | -0.087 | 1.141 | 191 |

### NS Verb._WPPSI

| Run | Model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | Lasso | 2.164 | 2.780 | -0.005 | -0.096 | -0.086 | 0.227 | 190 |
| `run_001_auto_shift` | ElasticNet | 2.164 | 2.780 | -0.005 | -0.096 | -0.086 | 0.227 | 190 |
| `run_001_auto_shift` | SVR | 2.178 | 2.796 | -0.017 | -0.129 | -0.128 | 0.212 | 190 |
| `run_001_auto_shift` | BayesianRidge | 2.238 | 2.868 | -0.070 | -0.196 | -0.188 | 0.239 | 190 |
| `run_001_auto_shift` | Ridge | 2.286 | 2.925 | -0.113 | -0.196 | -0.187 | 0.236 | 190 |
| `run_001_auto_shift` | RandomForest | 2.392 | 3.060 | -0.218 | -0.146 | -0.135 | 0.256 | 190 |
| `run_001_auto_shift` | XGBoost | 2.441 | 3.143 | -0.285 | -0.142 | -0.129 | 0.387 | 190 |
| `run_001_auto_shift` | LightGBM | 2.448 | 3.109 | -0.258 | -0.115 | -0.110 | 0.207 | 190 |
| `run_002_manual_shift` | SVR | 2.185 | 2.811 | -0.016 | -0.096 | -0.099 | 0.149 | 191 |
| `run_002_manual_shift` | Lasso | 2.186 | 2.811 | -0.016 | -0.167 | -0.127 | 0.144 | 191 |
| `run_002_manual_shift` | ElasticNet | 2.186 | 2.811 | -0.016 | -0.167 | -0.127 | 0.144 | 191 |
| `run_002_manual_shift` | BayesianRidge | 2.202 | 2.827 | -0.027 | -0.112 | -0.098 | 0.126 | 191 |
| `run_002_manual_shift` | Ridge | 2.223 | 2.856 | -0.049 | -0.087 | -0.079 | 0.124 | 191 |
| `run_002_manual_shift` | RandomForest | 2.383 | 3.088 | -0.225 | -0.132 | -0.141 | 0.188 | 191 |
| `run_002_manual_shift` | XGBoost | 2.405 | 3.069 | -0.211 | -0.042 | -0.101 | 0.148 | 191 |
| `run_002_manual_shift` | LightGBM | 2.450 | 3.116 | -0.248 | -0.095 | -0.127 | 0.249 | 191 |

### NS Perf._WPPSI

| Run | Model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | SVR | 1.694 | 2.267 | -0.020 | 0.011 | -0.070 | 0.277 | 190 |
| `run_001_auto_shift` | Lasso | 1.698 | 2.275 | -0.028 | -0.222 | -0.214 | 0.251 | 190 |
| `run_001_auto_shift` | ElasticNet | 1.702 | 2.283 | -0.035 | -0.231 | -0.195 | 0.252 | 190 |
| `run_001_auto_shift` | BayesianRidge | 1.766 | 2.328 | -0.076 | -0.045 | -0.024 | 0.249 | 190 |
| `run_001_auto_shift` | Ridge | 1.775 | 2.329 | -0.077 | -0.034 | -0.010 | 0.249 | 190 |
| `run_001_auto_shift` | LightGBM | 1.862 | 2.403 | -0.147 | 0.045 | 0.062 | 0.312 | 190 |
| `run_001_auto_shift` | RandomForest | 1.896 | 2.539 | -0.280 | -0.098 | -0.016 | 0.248 | 190 |
| `run_001_auto_shift` | XGBoost | 1.914 | 2.511 | -0.252 | -0.034 | -0.003 | 0.264 | 190 |
| `run_002_manual_shift` | Lasso | 1.695 | 2.256 | -0.006 | 0.028 | -0.029 | 0.212 | 191 |
| `run_002_manual_shift` | ElasticNet | 1.698 | 2.256 | -0.006 | 0.032 | -0.007 | 0.211 | 191 |
| `run_002_manual_shift` | SVR | 1.702 | 2.261 | -0.011 | 0.057 | 0.067 | 0.229 | 191 |
| `run_002_manual_shift` | BayesianRidge | 1.720 | 2.249 | -0.000 | 0.108 | 0.092 | 0.169 | 191 |
| `run_002_manual_shift` | Ridge | 1.726 | 2.248 | 0.001 | 0.133 | 0.118 | 0.170 | 191 |
| `run_002_manual_shift` | RandomForest | 1.853 | 2.431 | -0.169 | 0.003 | 0.059 | 0.162 | 191 |
| `run_002_manual_shift` | LightGBM | 1.859 | 2.368 | -0.109 | 0.088 | 0.075 | 0.181 | 191 |
| `run_002_manual_shift` | XGBoost | 1.875 | 2.464 | -0.201 | 0.012 | 0.056 | 0.170 | 191 |

### Rang percentile_WPPSI

| Run | Model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_001_auto_shift` | ElasticNet | 23.698 | 27.611 | -0.038 | 0.009 | -0.034 | 0.663 | 188 |
| `run_001_auto_shift` | Ridge | 23.760 | 27.768 | -0.050 | 0.011 | -0.034 | 0.596 | 188 |
| `run_001_auto_shift` | BayesianRidge | 23.900 | 27.827 | -0.055 | 0.004 | -0.046 | 0.463 | 188 |
| `run_001_auto_shift` | Lasso | 24.050 | 27.879 | -0.059 | -0.033 | -0.170 | 0.629 | 188 |
| `run_001_auto_shift` | SVR | 24.378 | 28.339 | -0.094 | -0.103 | -0.159 | 1.390 | 188 |
| `run_001_auto_shift` | XGBoost | 24.506 | 29.418 | -0.179 | 0.008 | 0.028 | 0.688 | 188 |
| `run_001_auto_shift` | RandomForest | 24.873 | 29.368 | -0.175 | -0.036 | -0.054 | 1.077 | 188 |
| `run_001_auto_shift` | LightGBM | 25.180 | 29.694 | -0.201 | -0.069 | -0.084 | 1.571 | 188 |
| `run_002_manual_shift` | SVR | 23.525 | 27.620 | -0.036 | -0.030 | 0.004 | 1.700 | 189 |
| `run_002_manual_shift` | Lasso | 23.573 | 27.275 | -0.010 | 0.008 | -0.032 | 1.647 | 189 |
| `run_002_manual_shift` | ElasticNet | 23.700 | 27.493 | -0.026 | 0.022 | 0.026 | 1.745 | 189 |
| `run_002_manual_shift` | Ridge | 23.700 | 27.532 | -0.029 | 0.030 | 0.033 | 1.782 | 189 |
| `run_002_manual_shift` | BayesianRidge | 23.740 | 27.534 | -0.029 | 0.013 | 0.021 | 1.747 | 189 |
| `run_002_manual_shift` | RandomForest | 24.959 | 29.600 | -0.190 | -0.092 | -0.083 | 2.346 | 189 |
| `run_002_manual_shift` | LightGBM | 25.147 | 29.708 | -0.198 | -0.104 | -0.100 | 2.537 | 189 |
| `run_002_manual_shift` | XGBoost | 25.486 | 30.881 | -0.295 | -0.103 | -0.109 | 2.176 | 189 |

## Legacy notebook reference

The old result documents (now in `docs/reports/archive/legacy_results/`) reported, for the auto-shift data with feature selection done **before** CV:

| Target | Best legacy model | Legacy MAE |
|---|---|---:|
| `Note composite_WPPSI` | BayesianRidge | 11.64 |
| `NS Verb._WPPSI` | BayesianRidge | 2.16 |
| `NS Perf._WPPSI` | BayesianRidge | 1.72 |
| `Rang percentile_WPPSI` | BayesianRidge | 23.66 |

Use these as a sanity floor, not as a target: the new numbers cannot be directly compared because the leakage was removed.

