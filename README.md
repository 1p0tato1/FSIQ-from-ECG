# FSIQ-from-ECG

PRI 2026 (TSE): regression of Full-Scale IQ (WPPSI-III) in preschool children from neonatal ECG, R-R series and HRV features (AuBE cohort, CHU Saint-Étienne).
Research only, not a diagnostic tool.

## Setup
Python 3.14 (use 3.14.x so versions match).
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Put your data files in `data/` locally. They are children's health data and this repo is public:
**NEVER commit them, NEVER send them to an AI tool** (even anonymized).


## Git workflow
Work on your own branch (e.g. `groupA/firstname-topic`), open a PR to `main`. No direct push to `main`.

## Possible Regression Model Methods
1. Random forest
2. Elastic Net (tune α ∈ [0,1], λ)
3. Bayesian Ridge or Gaussian Process

## Project Structure 
The models are in each folder, the methods are still to be decided. In the root of the project there is a py file that evaluates all the methods the same way in order to compare them. 
```
.
├── data
│   ├── HRV_Lineaire_Full.csv
│   ├── HRV_NonLineaire_Full.csv
│   └── Résultats WPPSI III.csv
├── Group A
│   ├── Regression Method 1
│   │   └── method1.ipynb
│   └── Regression Method 2
│       └── method2.ipynb
├── Group B
│   └── groupb.ipynb
├── README.md
├── requirements.txt
└── model_evaluation.py
```
