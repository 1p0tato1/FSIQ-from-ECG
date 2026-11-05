# 'Note composite_WPPSI'   (Full-Scale IQ)
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest      12.0554   0.6258  14.9500  -0.1157   0.0573
LightGBM          12.7988   0.8919  15.8885  -0.2584   0.0367
XGBoost           12.6515   0.9840  15.6332  -0.2329   0.1826
SVR               12.2522   1.3408  15.0120  -0.1181   0.0818
```
![[Pasted image 20261105105414.png]]
# 'NS Verb._WPPSI'         (Verbal IQ)  
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest       2.4262   0.1970   3.0652  -0.2471   0.1057
LightGBM           2.5769   0.2183   3.1923  -0.3486   0.0762
XGBoost            2.6405   0.2920   3.2701  -0.4161   0.1600
SVR                2.1793   0.1959   2.7850  -0.0238   0.0203
```
![[Pasted image 20261105105702.png]]

# 'NS Perf._WPPSI'         (Performance IQ)  
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest       1.8333   0.2414   2.4786  -0.3253   0.2381
LightGBM           1.7940   0.1585   2.3854  -0.2556   0.2646
XGBoost            1.9102   0.2135   2.5497  -0.4037   0.2000
SVR                1.6766   0.2344   2.2356  -0.0650   0.0843
```
![[Pasted image 20261105111908.png]]
# 'Rang percentile_WPPSI'  (Percentile rank)
```shell
==============================================================
Nested CV summary (outer 5-fold × inner 4-fold GridSearch)
==============================================================
Model                 MAE     ±MAE     RMSE       R²      ±R²
RandomForest      24.4935   2.2939  28.9189  -0.1760   0.0446
LightGBM          26.2542   2.2968  31.2842  -0.3823   0.1280
XGBoost           25.1622   1.8077  30.0847  -0.2831   0.1239
SVR               24.7230   2.1707  29.2618  -0.2105   0.1347

Best model by outer-CV MAE: RandomForest (MAE = 24.4935)
Best hyperparameters: {'model__max_depth': 10, 'model__min_samples_split': 2, 'model__n_estimators': 200}
Refitted RandomForest on all 188 samples.
```
![[Pasted image 20261105112427.png]]