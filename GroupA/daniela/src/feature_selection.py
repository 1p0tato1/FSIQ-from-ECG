#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-free feature selection used inside every CV fold.

The old notebooks selected features once on the full dataset and then ran
cross-validation, which leaks test information into the selection step.  The
:class:`FeatureSelector` below is a scikit-learn transformer, so it can live in
the model pipeline and is re-fitted on each training fold only:

1. ``SelectKBest`` with ANOVA ``f_regression`` keeps ``k`` features.
2. ``SelectFromModel`` with Lasso (or a Random Forest) prunes them further,
   using ``threshold="median"``.  If it selects nothing we keep the KBest set.
3. Optional SHAP refinement: a shallow ``DecisionTreeRegressor`` is explained
   with ``shap.TreeExplainer`` and the top ``shap_top_k`` features by mean
   ``|SHAP|`` are kept.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Filter + embedded (+ SHAP) feature selection as a pipeline step."""

    def __init__(self, k_best: int = 20, embedded: str = "lasso",
                 shap_refinement: bool = True, shap_top_k: int = 20,
                 shap_tree_max_depth: int = 5, random_state: int = 42):
        self.k_best = k_best
        self.embedded = embedded
        self.shap_refinement = shap_refinement
        self.shap_top_k = shap_top_k
        self.shap_tree_max_depth = shap_tree_max_depth
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_array(X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _remember_names(self, X) -> None:
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        elif not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = None

    def _embedded_estimator(self):
        name = str(self.embedded).lower()
        if name in ("none", "null", ""):
            return None
        if name in ("random_forest", "rf", "forest"):
            return RandomForestRegressor(
                n_estimators=200, random_state=self.random_state, n_jobs=1)
        # Default: Lasso, matching the regularized notebooks.
        return Lasso(alpha=0.1, random_state=self.random_state, max_iter=10000)

    # ------------------------------------------------------------------- fit
    def fit(self, X, y):
        self._remember_names(X)
        X_array = self._as_array(X)
        y_array = np.asarray(y, dtype=float).ravel()
        n_features = X_array.shape[1]

        k = max(1, min(int(self.k_best), n_features))
        self.kbest_ = SelectKBest(score_func=f_regression, k=k)
        self.kbest_.fit(X_array, y_array)
        kbest_mask = self.kbest_.get_support()
        kbest_indices = np.flatnonzero(kbest_mask)

        selected = kbest_indices
        estimator = self._embedded_estimator()
        self.embedded_selector_ = None
        if estimator is not None and len(kbest_indices) > 0:
            try:
                selector = SelectFromModel(estimator, threshold="median")
                selector.fit(X_array[:, kbest_indices], y_array)
                support = selector.get_support()
                self.embedded_selector_ = selector
                if support.sum() > 0:  # otherwise fall back to SelectKBest
                    selected = kbest_indices[support]
            except Exception:  # pragma: no cover - defensive, keep KBest
                selected = kbest_indices

        self.shap_importance_ = None
        if self.shap_refinement and len(selected) > 1:
            selected = self._refine_with_shap(X_array, y_array, selected)

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected] = True
        self.selected_indices_ = np.asarray(selected, dtype=int)
        if self.feature_names_in_ is not None:
            self.selected_features_ = [str(f) for f in self.feature_names_in_[selected]]
        else:
            self.selected_features_ = [f"x{i}" for i in selected]
        return self

    def _refine_with_shap(self, X_array: np.ndarray, y_array: np.ndarray,
                          selected: np.ndarray) -> np.ndarray:
        try:
            import shap

            tree = DecisionTreeRegressor(
                random_state=self.random_state,
                max_depth=self.shap_tree_max_depth,
            )
            tree.fit(X_array[:, selected], y_array)
            values = shap.TreeExplainer(tree).shap_values(X_array[:, selected])
            values = np.asarray(values)
            if values.ndim == 3:  # some shap versions return (n, p, 1)
                values = values[..., 0]
            importance = np.abs(values).mean(axis=0)
            self.shap_importance_ = importance
            keep = np.argsort(-importance, kind="stable")[: min(
                int(self.shap_top_k), len(selected))]
            return selected[np.sort(keep)]
        except Exception:  # shap is optional at prediction time
            return selected

    # --------------------------------------------------------------- transform
    def transform(self, X):
        X_array = self._as_array(X)
        return X_array[:, self.support_]

    def get_support(self, indices: bool = False):
        return self.selected_indices_ if indices else self.support_

    def get_selected_features(self) -> list[str]:
        return list(self.selected_features_)
