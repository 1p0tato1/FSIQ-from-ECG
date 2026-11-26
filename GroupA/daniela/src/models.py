#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers to load models saved by :mod:`src.train`.

The fitted pipelines contain ``src.feature_selection.FeatureSelector``.  Python
therefore needs ``GroupA/daniela`` on ``sys.path`` when unpickling.  Running from
``GroupA/daniela`` (the documented workflow) is enough; otherwise use
:func:`load_model`, which adds that directory automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

DANIELA_DIR = Path(__file__).resolve().parent.parent


def load_model(path: str | Path):
    """Load a joblib artifact produced by ``src.train``.

    ``path`` may point to ``model.pkl`` (a dict keyed by target) or to one of
    ``models/<target>.pkl``.
    """
    if str(DANIELA_DIR) not in sys.path:
        sys.path.insert(0, str(DANIELA_DIR))
    import joblib

    return joblib.load(path)


def load_target_model(path: str | Path, target: str):
    """Load one target's pipeline from a run's ``model.pkl``."""
    models = load_model(path)
    if isinstance(models, dict):
        return models[target]
    return models
