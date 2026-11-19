#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate every ``experiments/runs/*/metrics.json`` into results/SUMMARY.md.

Usage (from ``GroupA/daniela``)::

    python -m src.summarize
    python -m src.summarize --out results/SUMMARY.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any

try:
    from .config import DANIELA_DIR, iter_runs
except ImportError:  # direct script execution
    from config import DANIELA_DIR, iter_runs  # type: ignore

METRIC_KEYS = ["MAE", "RMSE", "R2", "pearson", "spearman", "cv_mae_std"]


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    return f"{number:.{digits}f}"


def _best_model(metrics: dict[str, dict[str, dict[str, float]]],
                target: str) -> str | None:
    models = metrics.get(target, {})
    if not models:
        return None
    return min(models, key=lambda name: (
        models[name].get("cv_mae_mean", float("inf"))
        if models[name].get("cv_mae_mean") is not None else float("inf")
    ))


def build_summary(runs_dir: str | Path | None = None) -> str:
    runs = list(iter_runs(runs_dir))
    lines: list[str] = []
    lines.append("# Experiment summary")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    if not runs:
        lines.append("_No runs found yet. Run `python -m src.train` first._")
        return "\n".join(lines) + "\n"

    lines.append("Metrics are computed on out-of-fold predictions with feature "
                 "selection **inside** each CV fold. They are therefore more "
                 "conservative than the old notebook tables, which selected "
                 "features once on the full dataset.")
    lines.append("")
    lines.append("Negative R² means the model is worse than predicting the "
                 "training-fold mean — the HRV signal is weak for this cohort.")
    lines.append("")

    # ---------------------------------------------------------------- overview
    lines.append("## Runs")
    lines.append("")
    lines.append("| Run | Source | Children | Features | Best models | Commit |")
    lines.append("|---|---|---:|---:|---|---|")
    for run in runs:
        manifest = run["manifest"]
        best = manifest.get("best_models") or {}
        best_text = ", ".join(f"{t}: {m}" for t, m in best.items()) or "—"
        commit = (manifest.get("git_commit") or "—")[:8]
        lines.append(
            f"| `{run['run_id']}` | `{manifest.get('source', '—')}` | "
            f"{manifest.get('n_subjects', '—')} | {manifest.get('n_features', '—')} | "
            f"{best_text} | `{commit}` |"
        )
    lines.append("")

    targets: list[str] = []
    for run in runs:
        for target in run["metrics"]:
            if target not in targets:
                targets.append(target)

    # ------------------------------------------------- best model per target
    lines.append("## Best model per target")
    lines.append("")
    lines.append("Selection criterion: lowest mean outer-fold MAE (`cv_mae_mean`).")
    lines.append("")
    for target in targets:
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Run | Best model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for run in runs:
            metrics = run["metrics"].get(target, {})
            best = _best_model(run["metrics"], target)
            if not best:
                continue
            row = metrics[best]
            lines.append(
                f"| `{run['run_id']}` | {best} | {_fmt(row.get('MAE'))} | "
                f"{_fmt(row.get('RMSE'))} | {_fmt(row.get('R2'))} | "
                f"{_fmt(row.get('pearson'))} | {_fmt(row.get('spearman'))} | "
                f"{_fmt(row.get('cv_mae_std'))} |"
            )
        lines.append("")

    # ----------------------------------------------------- all model details
    lines.append("## All models")
    lines.append("")
    for target in targets:
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| Run | Model | MAE | RMSE | R² | Pearson | Spearman | CV MAE std | n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for run in runs:
            metrics = run["metrics"].get(target, {})
            for model in sorted(metrics, key=lambda name: (
                    metrics[name].get("cv_mae_mean", float("inf")))):
                row = metrics[model]
                lines.append(
                    f"| `{run['run_id']}` | {model} | {_fmt(row.get('MAE'))} | "
                    f"{_fmt(row.get('RMSE'))} | {_fmt(row.get('R2'))} | "
                    f"{_fmt(row.get('pearson'))} | {_fmt(row.get('spearman'))} | "
                    f"{_fmt(row.get('cv_mae_std'))} | {row.get('n', '—')} |"
                )
        lines.append("")

    # ---------------------------------------------------------- legacy context
    lines.append("## Legacy notebook reference")
    lines.append("")
    lines.append("The old result documents (now in `docs/reports/archive/legacy_results/`) "
                 "reported, for the auto-shift data with feature selection done "
                 "**before** CV:")
    lines.append("")
    lines.append("| Target | Best legacy model | Legacy MAE |")
    lines.append("|---|---|---:|")
    lines.append("| `Note composite_WPPSI` | BayesianRidge | 11.64 |")
    lines.append("| `NS Verb._WPPSI` | BayesianRidge | 2.16 |")
    lines.append("| `NS Perf._WPPSI` | BayesianRidge | 1.72 |")
    lines.append("| `Rang percentile_WPPSI` | BayesianRidge | 23.66 |")
    lines.append("")
    lines.append("Use these as a sanity floor, not as a target: the new numbers "
                 "cannot be directly compared because the leakage was removed.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(DANIELA_DIR / "experiments" / "runs"))
    ap.add_argument("--out", default=str(DANIELA_DIR / "results" / "SUMMARY.md"))
    args = ap.parse_args(argv)

    text = build_summary(args.runs_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
