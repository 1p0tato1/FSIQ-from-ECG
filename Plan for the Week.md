# Monday
## Main goal
Set up the repository, development environment, and load and inspect the three data files.
### Steps
- [ ] Initialize repository, create .gitignore excluding data/ and .venv (per README: never commit health data)
- [ ] Set up Python 3.14 virtual environment and install requirements.txt
- [ ] Each member creates a personal branch (e.g. groupA/firstname-topic); no direct push to main
- [ ] Pair 1 loads HRV_Lineaire_Full.csv (skiprows=1) and inspects columns, Passage values, missingness
- [ ] Pair 2 loads HRV_NonLineaire_Full.csv (skiprows=1) and inspects columns, Passage values, missingness
- [ ] Pair 3 loads Resultats WPPSI III.csv (header=1, sep="\t", fix comma decimals) and inspects IQ columns (QIT, QIV, QIP, NCL)
- [ ] Merge inspection: verify Monogramme + DDN keys align across files, confirm ~189 children at M0
---
# Tuesday
## Main goal
Clean, merge, and preprocess the data into a final modeling dataset following dataset.md.
### Steps
- [ ] Filter both HRV files to Passage == "M0" (birth recordings only)
- [ ] Drop WPPSI rows missing the target IQ for each analysis
- [ ] Merge linear + nonlinear HRV on Monogramme + DDN + Passage
- [ ] Merge HRV + WPPSI on Monogramme + DDN
- [ ] Drop fully-empty feature columns; impute remaining NaN with column mean
- [ ] Standardize features with StandardScaler fit on training data only
- [ ] Save cleaned dataset locally (not committed to repo)
- [ ] Document preprocessing pipeline to match dataset.md sections 5.2 and 7
---
# Wednesday
## Main goal
Implement regression models in method1.ipynb and generate first predictions.
### Steps
- [ ] Split 6 members into 3 pairs, each pair owns one model family:
- [ ] Group A-1: Elastic Net (tune α ∈ [0,1], λ), Bayesian Ridge or Gaussian Process
- [ ] Group A-2 B: Random Forest or GradientBoosting
- [ ] Group B: Work with raw data
- [ ] Each pair uncomments and configures their models
- [ ] Fit models on scaled training data, generate predictions on test set
- [ ] Compute MAE, R², RMSE per dataset.md section 9
- [ ] Create predicted vs actual and residual plots (Section 7 [[WS26-27/BIP Proyect/FSIQ-from-ECG/README|README]])
- [ ] Save trained models and scaler with joblib (Section 8 [[WS26-27/BIP Proyect/FSIQ-from-ECG/README|README]])
---
# Thursday
## Main goal
Run standardized comparison across all methods and all IQ targets using model_evaluation.py.
### Steps
- [ ] Update model_evaluation.py TARGET_COLUMNS to include QIT, QIV, QIP, NCL
- [ ] Register all pair models in METHOD_REGISTRY of model_evaluation.py
- [ ] Implement nested cross-validation (outer 5-fold performance, inner 5-fold tuning) to limit overfitting on n189, p100
- [ ] Run model_evaluation.py so every method is evaluated on every target with identical preprocessing and train/test split
- [ ] Review model_comparison.csv in results/
- [ ] Identify best performing method per IQ target
---
# Friday
## Main goal
Finalize results, compare methods, and prepare deliverables.
### Steps
- [ ] Consolidate comparison table across all targets and methods
- [ ] Interpret R² values (R² > 0 better than mean predictor, R² < 0 worse)
- [ ] Each pair presents their model results; group decides strongest approach
- [ ] Open pull requests to main, conduct code review, merge approved branches
- [ ] Write short summary of findings, limitations (small sample, high dimensionality), and next steps
- [ ] Verify no health data files are committed to the repository

