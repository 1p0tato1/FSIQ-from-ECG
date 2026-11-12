Mathematical methods in the model (pipeline description)

The model combines the following mathematical steps:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | **Parameters (input features)** | Raw ECG / HRV / clinical variables from the cleaned + merged dataset (`clean_data_shifted/` or `clean_data_m_shifted/`, `merged_dataset_features.csv`, tab-separated). Linear and nonlinear HRV indices are merged on `Monogramme` + `DDN` (+ `Passage`), then merged with WPPSI scores on `Monogramme` + `DDN`. Only M0 (birth) recordings are retained. |
| 2 | **Preprocessing** | Missing values in feature columns are imputed with the column mean (`SimpleImputer(strategy="mean")`); features are standardized to zero mean and unit variance (`StandardScaler`). All preprocessing/fitting is performed inside the cross-validation folds to prevent data leakage. |
| 3 | **Feature selection / dimensionality reduction** | Two-step selection on imputed features: (a) filter step — `SelectKBest` with ANOVA F-regression (`f_regression`), keeping the top-k features (k=20); (b) embedded step — `SelectFromModel` with a tree-based estimator (Random Forest Regressor) or Lasso at `threshold="median"`. The estimator-selected feature subset is used if non-empty; otherwise the SelectKBest subset is used. Result: ~30 features from ~59 raw features. |
| 4 | **ML model** | Regression models compared via nested cross-validation: Random Forest, LightGBM, XGBoost, SVR, Ridge, Lasso, ElasticNet, BayesianRidge. Each model is wrapped in a pipeline `SimpleImputer → StandardScaler → Model`. Hyperparameters are tuned with `GridSearchCV` (scoring = negative MAE, `n_jobs=-1`) on an inner 4-fold `KFold`; performance is estimated on an outer 5-fold `RepeatedKFold` (10 repeats). Selected by lowest outer-CV MAE; best model refit on all data (e.g., Random Forest `n_estimators=200, max_depth=None, min_samples_split=2`, or Ridge `alpha=100`). |
| 5 | **Output** | Continuous predicted value (WPPSI Full-Scale IQ / percentile rank). Evaluation metrics: MAE, RMSE, R² computed on outer-CV out-of-fold predictions. Diagnostics: Predicted vs. Actual scatter plot and residual plot. The trained pipeline is persisted with `joblib.dump(search_full.best_estimator_, f"regression_model_{TARGET_COLUMN}.pkl")`. |

Schematically:  
**Parameters → Preprocessing → Feature Selection → ML Regressor (nested CV tuned) → Output**

If your model uses a two-stage design (e.g., classifier first, then regression on the classifier output), describe it as:  
**Parameters → ML Classifier → Regression → Final Output**

---

## Mathematical Formulation of the Regression Models

Let $$X \in \mathbb{R}^{n \times p}$$be the standardized feature matrix, $$y \in \mathbb{R}^n$$ the target IQ scores, and $$\hat{y}_i = f(x_i)$$ the predicted value for sample $$i$$
### 1. Random Forest Regressor
An ensemble of \(B\) regression trees grown on bootstrap samples. Each tree \(T_b(x)\) is grown by recursively splitting the feature space to minimize within-node variance. The final prediction is the average over all trees:

$$
\hat{y}_{\text{RF}}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)
$$

### 2. LightGBM (Light Gradient Boosting Machine)
Gradient boosting that grows trees leaf-wise (best-first) rather than level-wise, using histogram-based split finding for efficiency. At iteration \(m\), a new tree \(h_m\) is fit to the negative gradient (pseudo-residuals) of the loss, and the model is updated as:

$$
F_m(x) = F_{m-1}(x) + \eta \, h_m(x), \qquad \hat{y} = F_M(x)
$$

where \($$eta$$) is the learning rate.

### 3. XGBoost (Extreme Gradient Boosting)
Gradient boosting with a second-order Taylor approximation of the loss and explicit regularization on tree complexity. The objective at iteration \(m\) is:

$$
\mathcal{L}^{(m)} = \sum_{i=1}^{n} \left[ g_i h_m(x_i) + \frac{1}{2} h_i h_m^2(x_i) \right] + \Omega(h_m)
$$

with \(g_i, h_i\) the first and second gradients, and \(\Omega\) a penalty on the number of leaves and leaf weights. Prediction:

$$
\hat{y}_{\text{XGB}}(x) = \sum_{m=1}^{M} \eta \, h_m(x)
$$

### 4. SVR (Support Vector Regression)
Finds a function \(f(x) = \langle w, \phi(x) \rangle + b\) that deviates from the true values by no more than \(\varepsilon\) (epsilon-insensitive loss), while minimizing the norm \(\|w\|^2\). The primal problem is:

$$
\min_{w,b,\xi,\xi^*} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)
$$
$$
\text{s.t. } y_i - \langle w,\phi(x_i)\rangle - b \le \varepsilon + \xi_i,\quad
\langle w,\phi(x_i)\rangle + b - y_i \le \varepsilon + \xi_i^*,\quad \xi,\xi^* \ge 0
$$

The dual is solved via kernels \(K(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle\) (e.g., RBF kernel), giving:

$$
\hat{y}_{\text{SVR}}(x) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(x_i, x) + b
$$

### 5. Ridge Regression
Linear regression with an L2 penalty on the coefficients:

$$
\hat{\beta}_{\text{Ridge}} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - x_i^\top\beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}
= (X^\top X + \lambda I)^{-1}X^\top y
$$

\(\lambda = \text{alpha}\) controls the strength of shrinkage.

### 6. Lasso (Least Absolute Shrinkage and Selection Operator)
Linear regression with an L1 penalty, which can drive coefficients exactly to zero:

$$
\hat{\beta}_{\text{Lasso}} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - x_i^\top\beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
$$

### 7. ElasticNet
A convex combination of L1 and L2 penalties:

$$
\hat{\beta}_{\text{ElasticNet}} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - x_i^\top\beta)^2 + \lambda \left[ \alpha \sum_{j=1}^{p} |\beta_j| + \frac{(1-\alpha)}{2} \sum_{j=1}^{p} \beta_j^2 \right] \right\}
$$

where \(\alpha = \text{l1\_ratio}\) balances the two penalties.

### 8. Bayesian Ridge Regression
Bayesian linear regression with a Gaussian prior on the weights \(w \sim \mathcal{N}(0, \alpha^{-1}I)\) and a Gamma prior on the noise precision \(\lambda\). The posterior over weights is Gaussian:

$$
p(w \mid X, y) = \mathcal{N}(w \mid \mu_N, \Sigma_N)
$$
$$
\Sigma_N = (\alpha I + \lambda X^\top X)^{-1}, \qquad \mu_N = \lambda \Sigma_N X^\top y
$$

Prediction for a new input \(x\):

$$
\hat{y} = x^\top \mu_N, \qquad \text{Var}(\hat{y}) = x^\top \Sigma_N x
$$

Hyperparameters \(\alpha\) and \(\lambda\) are estimated from the data by evidence maximization.
