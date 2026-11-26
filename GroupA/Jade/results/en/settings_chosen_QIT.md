| model | setting | values chosen (number of folds) |
|---|---|---|
| Elastic Net · all · Clinical only | alpha | median 1.42 [0.162, 23.5] |
| Elastic Net · all · Clinical only | l1_ratio | 0.1 (39/50), 0.3 (4/50), 0.7 (3/50), 0.5 (2/50), 0.9 (2/50) |
| Elastic Net · all · Clinical + HRV | alpha | median 22 [0.694, 57.1] |
| Elastic Net · all · Clinical + HRV | l1_ratio | 0.1 (33/50), 0.9 (8/50), 0.3 (4/50), 0.7 (3/50), 0.5 (2/50) |
| Random Forest · all · Clinical + HRV | n_estimators | 500 (25/50), 200 (25/50) |
| Random Forest · all · Clinical + HRV | min_samples_leaf | 20 (47/50), 10 (2/50), 5 (1/50) |
| Random Forest · all · Clinical + HRV | max_features | log2 (42/50), sqrt (8/50) |
| Random Forest · all · Clinical + HRV | max_depth | 3 (27/50), 7 (23/50) |
| LightGBM · RF selection · Clinical + HRV | n_estimators | 300 (48/50), 100 (2/50) |
| LightGBM · RF selection · Clinical + HRV | max_depth | 5 (37/50), 3 (12/50), -1 (1/50) |
| LightGBM · RF selection · Clinical + HRV | reg_lambda | 10 (38/50), 0 (11/50), 1 (1/50) |
| LightGBM · RF selection · Clinical + HRV | reg_alpha | 0 (38/50), 1 (11/50), 0.1 (1/50) |
| LightGBM · RF selection · Clinical + HRV | min_child_samples | 20 (39/50), 10 (11/50) |
| LightGBM · RF selection · Clinical + HRV | learning_rate | 0.01 (47/50), 0.1 (2/50), 0.05 (1/50) |
| XGBoost · RF selection · Clinical + HRV | n_estimators | 100 (47/50), 300 (3/50) |
| XGBoost · RF selection · Clinical + HRV | max_depth | 3 (23/50), 7 (22/50), 5 (5/50) |
| XGBoost · RF selection · Clinical + HRV | reg_lambda | 10 (47/50), 1 (3/50) |
| XGBoost · RF selection · Clinical + HRV | reg_alpha | 1 (32/50), 0.1 (18/50) |
| XGBoost · RF selection · Clinical + HRV | learning_rate | 0.01 (46/50), 0.05 (3/50), 0.1 (1/50) |
| XGBoost · RF selection · Clinical + HRV | subsample | 0.8 (36/50), 1 (14/50) |
| XGBoost · RF selection · Clinical + HRV | min_child_weight | 1 (49/50), 5 (1/50) |
| SVR · PCA 90% · Clinical + HRV | kernel | rbf (49/50), linear (1/50) |
| SVR · PCA 90% · Clinical + HRV | gamma | auto (28/50), scale (21/50) |
| SVR · PCA 90% · Clinical + HRV | epsilon | 1 (40/50), 0.1 (10/50) |
| SVR · PCA 90% · Clinical + HRV | C | 1 (38/50), 0.1 (9/50), 10 (3/50) |
| SVR · PCA 90% · Clinical + HRV | PCA components | 16 (31/50), 17 (19/50) |
| Linear regression · stepwise · Clinical + HRV | variables | 2 variables [1, 6]; most chosen: nl\|MP% (Symb24h) (19/50), nl\|alpha1 (DFA24h) (12/50), clin\|Réa à la naissance (11/50), clin\|Tabac (10/50), nl\|Katz (24h) (9/50) |
| Decision tree · all · Clinical + HRV | min_samples_leaf | 30 (23/50), 10 (15/50), 20 (8/50), 5 (4/50) |
| Decision tree · all · Clinical + HRV | max_depth | 2 (45/50), 3 (3/50), 4 (2/50) |
| Kernel ridge · all · Clinical + HRV | alpha | 100 (30/50), 1 (10/50), 10 (9/50), 0.1 (1/50) |
| Kernel ridge · all · Clinical + HRV | kernel | laplacian (23/50), rbf (18/50), linear (9/50) |
| Kernel ridge · all · Clinical + HRV | gamma | 0.001 (29/50), 0.05 (6/50), 0.005 (4/50), 0.01 (2/50) |
| Multiple kernel ridge · all · Clinical + HRV | alpha | 1 (32/50), 10 (18/50) |
| Multiple kernel ridge · all · Clinical + HRV | weight clinical | 1 (34/50), 0.5 (7/50), 0.75 (6/50), 0.25 (2/50), 0 (1/50) |
| Multiple kernel ridge · all · Clinical + HRV | weight linear HRV | 0 (40/50), 0.5 (4/50), 0.25 (4/50), 0.75 (1/50), 1 (1/50) |
| Multiple kernel ridge · all · Clinical + HRV | weight non-linear HRV | 0 (43/50), 0.5 (4/50), 0.25 (3/50) |
