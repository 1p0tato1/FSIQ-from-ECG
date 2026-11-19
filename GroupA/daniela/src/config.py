#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration helpers for the GroupA/daniela pipeline.

The single source of configuration is ``experiments/config.yaml``.  Every entry
point (preprocessing, merge, features, train, notebooks) resolves paths through
the helpers in this module so that the pipeline behaves the same whether it is
started from the repository root, from ``GroupA/daniela`` or from a notebook.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

DANIELA_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = DANIELA_DIR / "experiments" / "config.yaml"

# Fallback if the marker-based search below cannot find the repository root.
DEFAULT_PROJECT_ROOT = DANIELA_DIR.parent.parent

DEFAULT_TARGETS = [
    "Note composite_WPPSI",
    "NS Verb._WPPSI",
    "NS Perf._WPPSI",
    "Rang percentile_WPPSI",
]

DEFAULT_ID_COLUMNS = [
    "Numéro",
    "Monogramme_WPPSI",
    "Monogramme_Linear",
    "Monogramme_NonLinear",
]

# Non-feature metadata.  The config lists the four canonical IDs; these extra
# prefixes catch the per-source copies (DDN_Linear, Date_NonLinear, ...).
ID_PREFIXES = ("Numéro", "Monogramme", "DDN", "Date", "Filename",
               "Recorder", "Passage")

# Files produced by the preprocessing step.
RESULTS_FILE = "Résultats_WPPSI_III.csv"
LINEAR_FILE = "HRV_Lineaire_Full.csv"
NONLINEAR_FILE = "HRV_NonLineaire_Full.csv"
CLEAN_FILES = [RESULTS_FILE, LINEAR_FILE, NONLINEAR_FILE]

SOURCE_DIRS = {
    "auto": "clean_data_auto",
    "manual": "clean_data_manual",
}


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root (the directory that owns ``data/``)."""
    start = Path(start or DANIELA_DIR).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "evaluate_model.py").exists():
            return candidate
    # The data directory may not exist yet in a fresh clone.
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir():
            return candidate
    return DEFAULT_PROJECT_ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` and resolve the project root.

    ``project.root`` is interpreted relative to the config file.  If the
    resolved directory does not contain ``data/`` we fall back to walking up
    from the config file, which keeps the schema's ``root: ".."`` working even
    when the project is moved.
    """
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise ImportError(
            "PyYAML is required to read experiments/config.yaml. "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    root_value = cfg.get("project", {}).get("root", "../..")
    root = Path(root_value)
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    if not (root / "data").is_dir():
        root = find_project_root(config_path.parent)
    cfg["_root"] = root
    cfg["_config_path"] = config_path
    cfg["_config_hash"] = sha256_file(config_path)
    return cfg


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"]) / cfg.get("project", {}).get("data_dir", "data")


def source_dir(cfg: dict[str, Any], source: str) -> Path:
    return data_dir(cfg) / source


def get_targets(cfg: dict[str, Any], target_col: str | None = None) -> list[str]:
    """Return the configured targets.

    Supported forms::

        features:
          targets: [A, B]
        features:
          target_col: A
    """
    if target_col:
        return [target_col]
    features = cfg.get("features", {}) or {}
    targets = features.get("targets")
    if targets is None:
        target = features.get("target_col")
        targets = [target] if target else list(DEFAULT_TARGETS)
    if isinstance(targets, str):
        targets = [targets]
    return [str(t) for t in targets]


def get_id_columns(cfg: dict[str, Any]) -> list[str]:
    features = cfg.get("features", {}) or {}
    ids = features.get("id_columns") or DEFAULT_ID_COLUMNS
    return [str(c) for c in ids]


def is_id_column(column: str, configured: list[str] | None = None) -> bool:
    """True for canonical IDs and their per-source metadata copies."""
    configured = configured or DEFAULT_ID_COLUMNS
    if column in configured:
        return True
    return any(column == prefix or column.startswith(prefix + "_")
               for prefix in ID_PREFIXES)


def slugify(value: str) -> str:
    """Filesystem-friendly slug used for artifact names."""
    keep = [ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(value)]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "target"


def run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return Path(cfg["_config_path"]).parent / "runs" / run_id


def get_run(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in cfg.get("runs", []) or []:
        if str(run.get("id")) == str(run_id):
            return run
    raise KeyError(
        f"Run '{run_id}' is not defined in {cfg['_config_path']}. "
        f"Known runs: {[r.get('id') for r in cfg.get('runs', [])]}"
    )


def git_commit(root: str | Path | None = None) -> str | None:
    """Return the current commit hash, or None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or DEFAULT_PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    """Small reproducibility fingerprint recorded in every run manifest."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "shap",
                    "lightgbm", "xgboost", "joblib", "matplotlib"):
        try:
            import importlib.metadata as metadata
            info[package] = metadata.version(package)
        except Exception:
            info[package] = None
    return info


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    """Make numpy / pathlib values JSON serialisable."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_runs(runs_dir: str | Path | None = None) -> Iterable[dict[str, Any]]:
    """Yield ``{run_id, dir, metrics, params, manifest}`` for existing runs."""
    runs_dir = Path(runs_dir or (DANIELA_DIR / "experiments" / "runs"))
    if not runs_dir.is_dir():
        return
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        yield {
            "run_id": path.name,
            "dir": path,
            "metrics": read_json(metrics_path),
            "params": read_json(path / "params.json") if (path / "params.json").exists() else {},
            "manifest": read_json(path / "manifest.json") if (path / "manifest.json").exists() else {},
        }
