| modèle | réglage | valeurs retenues (nombre de plis) |
|---|---|---|
| Elastic Net · toutes · Clinique seul | alpha | médiane 1.42 [0.162 ; 23.5] |
| Elastic Net · toutes · Clinique seul | l1_ratio | 0.1 (39/50), 0.3 (4/50), 0.7 (3/50), 0.5 (2/50), 0.9 (2/50) |
| Elastic Net · toutes · Clinique + HRV | alpha | médiane 22 [0.694 ; 57.1] |
| Elastic Net · toutes · Clinique + HRV | l1_ratio | 0.1 (33/50), 0.9 (8/50), 0.3 (4/50), 0.7 (3/50), 0.5 (2/50) |
| Random Forest · toutes · Clinique + HRV | n_estimators | 500 (25/50), 200 (25/50) |
| Random Forest · toutes · Clinique + HRV | min_samples_leaf | 20 (47/50), 10 (2/50), 5 (1/50) |
| Random Forest · toutes · Clinique + HRV | max_features | log2 (42/50), sqrt (8/50) |
| Random Forest · toutes · Clinique + HRV | max_depth | 3 (27/50), 7 (23/50) |
| LightGBM · sél. RF · Clinique + HRV | n_estimators | 300 (48/50), 100 (2/50) |
| LightGBM · sél. RF · Clinique + HRV | max_depth | 5 (37/50), 3 (12/50), -1 (1/50) |
| LightGBM · sél. RF · Clinique + HRV | reg_lambda | 10 (38/50), 0 (11/50), 1 (1/50) |
| LightGBM · sél. RF · Clinique + HRV | reg_alpha | 0 (38/50), 1 (11/50), 0.1 (1/50) |
| LightGBM · sél. RF · Clinique + HRV | min_child_samples | 20 (39/50), 10 (11/50) |
| LightGBM · sél. RF · Clinique + HRV | learning_rate | 0.01 (47/50), 0.1 (2/50), 0.05 (1/50) |
| XGBoost · sél. RF · Clinique + HRV | n_estimators | 100 (47/50), 300 (3/50) |
| XGBoost · sél. RF · Clinique + HRV | max_depth | 3 (23/50), 7 (22/50), 5 (5/50) |
| XGBoost · sél. RF · Clinique + HRV | reg_lambda | 10 (47/50), 1 (3/50) |
| XGBoost · sél. RF · Clinique + HRV | reg_alpha | 1 (32/50), 0.1 (18/50) |
| XGBoost · sél. RF · Clinique + HRV | learning_rate | 0.01 (46/50), 0.05 (3/50), 0.1 (1/50) |
| XGBoost · sél. RF · Clinique + HRV | subsample | 0.8 (36/50), 1 (14/50) |
| XGBoost · sél. RF · Clinique + HRV | min_child_weight | 1 (49/50), 5 (1/50) |
| SVR · PCA 90 % · Clinique + HRV | kernel | rbf (49/50), linear (1/50) |
| SVR · PCA 90 % · Clinique + HRV | gamma | auto (28/50), scale (21/50) |
| SVR · PCA 90 % · Clinique + HRV | epsilon | 1 (40/50), 0.1 (10/50) |
| SVR · PCA 90 % · Clinique + HRV | C | 1 (38/50), 0.1 (9/50), 10 (3/50) |
| SVR · PCA 90 % · Clinique + HRV | composantes PCA | 16 (31/50), 17 (19/50) |
| Régression linéaire · pas à pas · Clinique + HRV | variables | 2 variables [1 ; 6] ; les plus choisies : nl\|MP% (Symb24h) (19/50), nl\|alpha1 (DFA24h) (12/50), clin\|Réa à la naissance (11/50), clin\|Tabac (10/50), nl\|Katz (24h) (9/50) |
| Arbre de décision · toutes · Clinique + HRV | min_samples_leaf | 30 (23/50), 10 (15/50), 20 (8/50), 5 (4/50) |
| Arbre de décision · toutes · Clinique + HRV | max_depth | 2 (45/50), 3 (3/50), 4 (2/50) |
| Kernel ridge · toutes · Clinique + HRV | alpha | 100 (30/50), 1 (10/50), 10 (9/50), 0.1 (1/50) |
| Kernel ridge · toutes · Clinique + HRV | kernel | laplacian (23/50), rbf (18/50), linear (9/50) |
| Kernel ridge · toutes · Clinique + HRV | gamma | 0.001 (29/50), 0.05 (6/50), 0.005 (4/50), 0.01 (2/50) |
| Kernel ridge multi-noyaux · toutes · Clinique + HRV | alpha | 1 (32/50), 10 (18/50) |
| Kernel ridge multi-noyaux · toutes · Clinique + HRV | poids clinique | 1 (34/50), 0.5 (7/50), 0.75 (6/50), 0.25 (2/50), 0 (1/50) |
| Kernel ridge multi-noyaux · toutes · Clinique + HRV | poids HRV linéaire | 0 (40/50), 0.5 (4/50), 0.25 (4/50), 0.75 (1/50), 1 (1/50) |
| Kernel ridge multi-noyaux · toutes · Clinique + HRV | poids HRV non linéaire | 0 (43/50), 0.5 (4/50), 0.25 (3/50) |
