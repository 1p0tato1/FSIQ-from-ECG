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
