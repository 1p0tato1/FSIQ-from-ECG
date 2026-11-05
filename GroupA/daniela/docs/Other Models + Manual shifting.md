# 'Note composite_WPPSI'   (Full-Scale IQ)
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest      12.0618   0.8842  14.9736  -0.1101   0.0557
LightGBM          12.7959   0.8700  15.6817  -0.2167   0.0516
XGBoost           12.7685   0.7025  15.7256  -0.2229   0.0611
SVR               11.7495   0.8570  14.4163  -0.0284   0.0199

Best model by outer-CV MAE: SVR (MAE = 11.7495)
Best hyperparameters: {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'rbf'}
Refitted SVR on all 191 samples.
```
![[Pasted image 20261105113301.png]]

# 'NS Verb._WPPSI'         (Verbal IQ)  
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest       2.3701   0.1057   3.0170  -0.2132   0.0786
LightGBM           2.4881   0.1647   3.0568  -0.2456   0.1035
XGBoost            2.4433   0.1355   3.0793  -0.2682   0.1341
SVR                2.1829   0.1369   2.7966  -0.0387   0.0340

Best model by outer-CV MAE: SVR (MAE = 2.1829)
Best hyperparameters: {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'rbf'}
Refitted SVR on all 191 samples.
```
![[Pasted image 20261105113117.png]]
# 'NS Perf._WPPSI'         (Performance IQ)  
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest       1.8583   0.2030   2.4529  -0.2614   0.2762
LightGBM           1.9510   0.2351   2.4745  -0.2546   0.0863
XGBoost            1.9564   0.1559   2.4798  -0.2724   0.1070
SVR                1.7133   0.2181   2.2390  -0.0292   0.0392

Best model by outer-CV MAE: SVR (MAE = 1.7133)
Best hyperparameters: {'model__C': 0.1, 'model__gamma': 'auto', 'model__kernel': 'rbf'}
Refitted SVR on all 191 samples.
```
![[Pasted image 20261105112818.png]]
# 'Rang percentile_WPPSI'  (Percentile rank)
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest      23.9856   2.8306  28.4514  -0.1220   0.0474
LightGBM          26.1345   2.5442  31.0568  -0.3394   0.0769
XGBoost           25.8948   2.3776  31.1661  -0.3592   0.1989
SVR               24.2563   2.6994  28.7250  -0.1574   0.2588

Best model by outer-CV MAE: RandomForest (MAE = 23.9856)
Best hyperparameters: {'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 200}
Refitted RandomForest on all 189 samples.
```
![[Pasted image 20261105112707.png]]