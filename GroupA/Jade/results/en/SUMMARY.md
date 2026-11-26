# Predicting IQ at age 3 from neonatal HRV: summary
*Group A · Jade · generated on 26/11/2026 by `01_prediction_qit_en.ipynb` · 10 × 5 folds*

## Conclusion
**No model does better than predicting the mean score, for any target.** Yet HRV does recognise prematurity (AUC 0.79 [0.67, 0.93]): the data and the method can see a link when there is one.

## Data and method
- 189 children of the AuBE cohort (recording at birth, complete WPPSI-III scores); 25 linear HRV indices, 45 non-linear, 6 clinical covariates.
- Models: Decision tree, Elastic Net, Kernel ridge, LightGBM, Linear regression, Multiple kernel ridge, Random Forest, SVR, XGBoost; feature reduction within each fold; same folds for all models (10 × 5), evaluation with `evaluate_model.py` (the team's shared tool).
- Targets (French codes of the data): QIT = Full Scale IQ, QIV = Verbal IQ, QIP = Performance IQ, NCL = General Language Composite.

## Regression (best model per target)
| Target | Model | MAE ± std | R² | MAE of the mean | Gain vs mean [95% CI] |
|---|---|---|---|---|---|
| QIT | Elastic Net · all · Clinical only | 10.33 ± 0.69 | -0.007 | 10.39 | +0.06 [-0.12, 0.23] (not demonstrated) |
| QIV | Elastic Net · all · Clinical only | 11.68 ± 1.20 | -0.010 | 11.66 | -0.02 [-0.22, 0.16] (not demonstrated) |
| QIP | Elastic Net · all · Clinical only | 9.75 ± 0.94 | -0.017 | 9.75 | +0.00 [-0.16, 0.17] (not demonstrated) |
| NCL | Elastic Net · all · Clinical only | 11.64 ± 1.23 | -0.021 | 11.62 | -0.01 [-0.15, 0.13] (not demonstrated) |

- Adding HRV to the clinical model (Elastic Net, FSIQ): MAE goes from 10.33 to 10.47 (worse).
- Signal in the variables: the strongest correlation with FSIQ is 0.16, versus 0.17 on average by chance (up to 0.21): nothing beats chance, no link to learn.
- Detection limit (simulation): with these children, the method sees a link as soon as it explains about 20% of the IQ variance; if HRV explains part of it, that part is smaller.
- Bound (indices chosen in advance: RMSSD (TD24h), HF (FT24h), SDNN (TD24h), 1st HRV component): no link demonstrated; none explains more than 6% of the FSIQ variance (6% beyond the clinical covariates), simultaneous 95% intervals; a link of 6% would have been detected 8 times out of 10.
- 90% prediction interval (conformal prediction): ± 21.3 points, versus ± 20.9 around the mean FSIQ.

## All models (FSIQ)
MAE and R²: mean ± standard deviation over the folds; gain over the mean with its 95% CI (`evaluate_model.py`).

| model | MAE | R² | gain vs mean [95% CI] | gain demonstrated |
|---|---|---|---|---|
| Elastic Net · all · Clinical only | 10.33 ± 0.69 | -0.007 ± 0.041 | +0.06 [-0.12, 0.23] | no |
| Multiple kernel ridge · all · Clinical + HRV | 10.38 ± 0.73 | -0.020 ± 0.064 | +0.02 [-0.30, 0.32] | no |
| mean score (baseline) | 10.39 ± 0.66 | -0.005 ± 0.004 |  |  |
| Elastic Net · all · Clinical + HRV | 10.47 ± 0.72 | -0.025 ± 0.045 | -0.08 [-0.14, -0.02] | no, worse |
| SVR · PCA 90% · Clinical + HRV | 10.51 ± 0.73 | -0.034 ± 0.065 | -0.12 [-0.22, -0.02] | no, worse |
| Random Forest · all · Clinical + HRV | 10.59 ± 0.68 | -0.047 ± 0.031 | -0.19 [-0.33, -0.05] | no, worse |
| XGBoost · RF selection · Clinical + HRV | 10.61 ± 0.77 | -0.058 ± 0.066 | -0.22 [-0.44, -0.01] | no, worse |
| Kernel ridge · all · Clinical + HRV | 10.62 ± 0.75 | -0.068 ± 0.106 | -0.23 [-0.37, -0.09] | no, worse |
| Linear regression · stepwise · Clinical + HRV | 10.89 ± 0.80 | -0.121 ± 0.115 | -0.50 [-0.75, -0.24] | no, worse |
| LightGBM · RF selection · Clinical + HRV | 10.93 ± 0.86 | -0.126 ± 0.105 | -0.54 [-0.86, -0.23] | no, worse |
| Decision tree · all · Clinical + HRV | 11.14 ± 0.97 | -0.165 ± 0.150 | -0.74 [-1.03, -0.46] | no, worse |

## Methods added after the teachers' feedback (FSIQ)
- Linear regression · stepwise · Clinical + HRV: MAE 10.89, gain -0.50 [-0.75, -0.24] (worse than the mean).
- Decision tree · all · Clinical + HRV: MAE 11.14, gain -0.74 [-1.03, -0.46] (worse than the mean).
- Kernel ridge · all · Clinical + HRV: MAE 10.62, gain -0.23 [-0.37, -0.09] (worse than the mean).
- Multiple kernel ridge · all · Clinical + HRV: MAE 10.38, gain +0.02 [-0.30, 0.32] (not demonstrated).
- Multiple kernel ridge weights (median over the folds): clinical 1.00, linear HRV 0.00, non-linear HRV 0.00.
- Variable ranking by a decision tree (impurity importance and SHAP): for FSIQ, variables above chance: none; agreement of the top 5 between folds 0.08 (stable above 0.15). For prematurity: nl|alpha2 (DFADay), lin|pcNN20 (TD24h); agreement 0.22 (stable above 0.28).
- Settings of each model (fixed or tuned, and how): section 3 of the notebook; settings chosen fold by fold: `results/en/settings_chosen_QIT.md`.

## Positive control, classification, hypotheses
- Positive control: HRV only → prematurity, AUC 0.79 [0.67, 0.93] (link demonstrated (CI > 0.5); acceptable discrimination; fair agreement).
- Classification FSIQ < 90 (40 children): best AUC 0.50 [0.39, 0.61] (Logistic · Clinical + HRV), balanced accuracy 0.50, kappa 0.00 (no link demonstrated (CI contains 0.5); poor discrimination; slight agreement).
- H1 · HRV at M0 (189 children): gain -0.13 [-0.22, -0.04] (not supported: worse than the mean).
- H1 · HRV at M6 (176 children): gain -0.07 [-0.32, 0.17] (not supported).
- H1 · HRV at M12 (177 children): gain -0.03 [-0.07, 0.01] (not supported).
- H1 · HRV at M18 (173 children): gain +0.01 [-0.14, 0.17] (not supported).
- H1 · HRV at M24 (182 children): gain -0.14 [-0.24, -0.05] (not supported: worse than the mean).
- H2 · change M0 → M12 (175 children): gain -0.05 [-0.19, 0.08] (not supported).
- H2 · change M0 → M24 (180 children): gain -0.10 [-0.26, 0.05] (not supported).
- With 7 hypotheses tested, an isolated supported hypothesis may be due to chance: to be confirmed.

## Limitations
- 189 children; no external cohort; IQ at age 3 depends on many factors not measured here (family environment, socio-economic status).
- Association is not causation: a predictive model does not say how to improve a child's IQ.

## Next steps
- Variables derived from the raw data (RR files, ECG signal): Group B's work.
- Confirm any hypothesis supported above on another cohort.
