# Predicting IQ at age 3 from neonatal HRV: summary
*Group A · Jade · generated on 05/11/2026 by `01_prediction_qit_en.ipynb` · 10 × 5 folds*

## Conclusion
**No model does better than predicting the mean score, for any target.** Yet HRV does recognise prematurity (AUC 0.79): the data and the method can see a link when there is one.

## Data and method
- 189 children of the AuBE cohort (recording at birth, complete WPPSI-III scores); 25 linear HRV indices, 45 non-linear, 6 clinical covariates.
- Models: Elastic Net, LightGBM, Random Forest, SVR, XGBoost; feature reduction within each fold; evaluation on the team's shared folds (10 × 5, `evaluate_model.py`).
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

## Positive control, classification, hypotheses
- Positive control: HRV only → prematurity, AUC 0.79 [0.67, 0.93].
- Classification FSIQ < 90 (40 children): best AUC 0.50 [0.39, 0.61] (Logistic · Clinical + HRV), balanced accuracy 0.51, kappa 0.02.
- H1 · HRV at M0 (189 children): gain -0.13 [-0.22, -0.04] (not supported).
- H1 · HRV at M6 (176 children): gain -0.07 [-0.32, 0.17] (not supported).
- H1 · HRV at M12 (177 children): gain -0.03 [-0.07, 0.01] (not supported).
- H1 · HRV at M18 (173 children): gain +0.01 [-0.14, 0.17] (not supported).
- H1 · HRV at M24 (182 children): gain -0.14 [-0.24, -0.05] (not supported).
- H2 · change M0 → M12 (175 children): gain -0.05 [-0.19, 0.08] (not supported).
- H2 · change M0 → M24 (180 children): gain -0.10 [-0.26, 0.05] (not supported).
- With 7 hypotheses tested, an isolated supported hypothesis may be due to chance: to be confirmed.

## Limitations
- About 190 children; no external cohort; IQ at age 3 depends on many factors not measured here (family environment, socio-economic status).
- Association is not causation: a predictive model does not say how to improve a child's IQ.

## Next steps
- Variables derived from the raw data (RR files, ECG signal): Group B's work.
- Confirm any hypothesis supported above on another cohort.
