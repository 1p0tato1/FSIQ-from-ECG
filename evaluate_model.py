#!/usr/bin/env python3
r"""
evaluate_model.py

One evaluation code for BOTH groups, so the results are comparable.
prepare_data.py chose the children and the folds; this file runs a model on
them and judges the result. The 3 targets QIV, QIP and QIT are predicted
together and evaluated one by one.

cohort.csv = the list of the children that are used, with their scores
(columns child_id, QIV, QIP, QIT).

1. run_cv(my_model, ...)  runs YOUR model on every fold and saves its predictions.
2. evaluate(...)          reads the predictions and tells you how good they are.

--- 1. run_cv -------------------------------------------------------------
Cross-validation: each round, one fold of children is hidden (test), the model
learns on the others (train) and predicts the hidden ones. run_cv is just that
loop, using the shared folds.csv. You only write the function that does one round:

    from evaluate_model import run_cv, load_labels
    labels = load_labels("data/cleaned/cohort.csv")   # table: child_id x (QIV, QIP, QIT)

    def my_model(train_ids, test_ids):                # lists of child ids
        X_train = ...                                 # your features of train_ids
        y_train = labels.loc[train_ids].to_numpy()    # shape (n_train, 3)
        X_test = ...                                  # your features of test_ids
        model = Ridge().fit(X_train, y_train)         # learn on train ONLY
        return model.predict(X_test)                  # shape (n_test, 3): QIV, QIP, QIT

    run_cv(my_model, "data/cleaned/folds.csv", "data/cleaned/cohort.csv", "predictions_B_ridge.csv")

The 3 columns are in the order of TARGETS = ["QIV", "QIP", "QIT"]. You can also
return a dict {"QIV": ..., "QIP": ..., "QIT": ...} (e.g. one model per target)
or a DataFrame with these columns. To predict only some targets:
run_cv(..., targets=["QIT"]).
Everything that learns (scaling, feature selection, tuning) goes INSIDE my_model,
on the training children only. Quick test: run_cv(..., repeats=[1]).

--- 2. evaluate -----------------------------------------------------------
    from evaluate_model import evaluate
    summary, comparison = evaluate("data/cleaned/folds.csv", "data/cleaned/cohort.csv",
                                   ["predictions_A_x.csv", "predictions_B_ridge.csv"])


or in the terminal (one line):
    python evaluate_model.py --folds data/cleaned/folds.csv --cohort data/cleaned/cohort.csv --predictions predictions_A_x.csv predictions_B_ridge.csv


It checks the files are valid and comparable (same children, same folds), then
reports, for EACH target (QIV, QIP, QIT):
  - MAE  : average error in IQ points (7.1 = off by about 7 points)
  - R2   : 0 = no better than predicting the average, negative = worse
  - RMSE, Pearson, Spearman
  - the mean baseline (predict the average IQ): the floor to beat
  - 95% intervals (bootstrap on children) and a paired comparison of the models
If an interval contains 0, the improvement is not established.

predictions file format: child_id, repeat, fold, then for each target
true_<T>, predicted_<T> (written by run_cv; one row per child per repeat of folds.csv).
Never change folds.csv and never use your own splits.
"""
import argparse
import itertools
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_print = print
TARGETS = ["QIV", "QIP", "QIT"]       


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def norm_id(x):
    """'302', '302-ZOUso-M0', 302.0 -> '302'."""
    m = re.search(r"\d+", str(x))
    return str(int(m.group())) if m else None


def _read(path):
    df = pd.read_csv(path, dtype={"child_id": str})
    df["child_id"] = df["child_id"].map(norm_id)
    return df


def load_labels(cohort_path, targets=None):
    """Table indexed by child_id with one column per target (default QIV, QIP, QIT),
    read from cohort.csv. Use labels.loc[list_of_ids].to_numpy() for the scores."""
    targets = list(targets or TARGETS)
    c = _read(cohort_path)
    miss = [t for t in targets if t not in c.columns]
    if miss:
        raise ValueError(f"{cohort_path} has no column {miss}. Run prepare_data.py again "
                         f"(the new version keeps QIV, QIP and QIT).")
    return c.set_index("child_id")[targets].astype(float)


def _as_matrix(out, targets, n):
    """Turn what the model returned into an array (n_test, n_targets)."""
    if isinstance(out, dict):
        miss = [t for t in targets if t not in out]
        if miss:
            raise ValueError(f"the dict returned by the model has no key {miss}")
        out = pd.DataFrame({t: np.asarray(out[t], float).ravel() for t in targets})
    if isinstance(out, pd.DataFrame):
        out = out[targets].to_numpy(float)
    a = np.asarray(out, float)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape != (n, len(targets)):
        raise ValueError(f"the model must return one prediction per test child and per "
                         f"target: expected shape ({n}, {len(targets)}) for {targets} "
                         f"(or a dict / DataFrame), got {a.shape}")
    return a


def _targets_of(cols):
    return [c[5:] for c in cols if c.startswith("true_") and "predicted_" + c[5:] in cols]


# ----------------------------------------------------------------------------
# cross-validation loop
# ----------------------------------------------------------------------------
def run_cv(fit_predict, folds_path, cohort_path, out_path, targets=None, repeats=None,
           verbose=True):
    """Run fit_predict on every (repeat, fold) of folds.csv and save predictions.

    fit_predict(train_ids, test_ids) -> predictions for test_ids (same order):
      an array (n_test, n_targets) with columns in the order of `targets`
      (default ["QIV", "QIP", "QIT"]), or a dict {target: array}, or a DataFrame.
    Both arguments are lists of child_id strings. The function must only use
    the training children to learn anything (scaling, selection, tuning...).
    `repeats`: optional list of repeat numbers to run (to test quickly with [1]).
    """
    targets = list(targets or TARGETS)
    folds = _read(folds_path)
    labels = load_labels(cohort_path, targets)
    unknown = sorted(set(folds["child_id"]) - set(labels.index), key=int)
    if unknown:
        raise ValueError(f"{len(unknown)} children of folds.csv are not in the cohort file, "
                         f"e.g. {unknown[:5]}. Use the folds.csv and cohort.csv of the same run.")
    rows = []
    for r in sorted(folds["repeat"].unique()):
        if repeats is not None and r not in repeats:
            continue
        fr = folds[folds["repeat"] == r]
        for k in sorted(fr["fold"].unique()):
            test = fr.loc[fr["fold"] == k, "child_id"].tolist()
            train = fr.loc[fr["fold"] != k, "child_id"].tolist()
            pred = _as_matrix(fit_predict(list(train), list(test)), targets, len(test))
            df = pd.DataFrame({"child_id": test, "repeat": r, "fold": k})
            for j, t in enumerate(targets):
                df[f"true_{t}"] = labels.loc[test, t].to_numpy()
                df[f"predicted_{t}"] = pred[:, j]
            rows.append(df)
            if verbose:
                print(f"  repeat {r} fold {k}: {len(train)} train / {len(test)} test",
                      flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_path, index=False)
    if verbose:
        print(f"Saved {len(out)} rows ({', '.join(targets)}) in {out_path}")
    return out


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def load_and_check(path, folds, cohort=None):
    """Read a predictions file, verify it, and return the data as matrices.

    Returns dict: children (list), repeats (list), targets (list), F (R,n) fold of
    each child in each repeat, and per target t: Y[t] (n,) true scores,
    P[t] (R,n) predictions, B[t] (R,n) mean-baseline predictions.
    """
    pred = _read(path)
    targets = _targets_of(pred.columns)
    if not targets:
        raise ValueError(f"[{path}]: no pair of columns true_<T> / predicted_<T> "
                         f"(e.g. true_QIT, predicted_QIT)")
    base = ["child_id", "repeat", "fold"]
    miss = [c for c in base if c not in pred.columns]
    if miss:
        raise ValueError(f"[{path}]: missing columns {miss}")
    vcols = [f"{p}_{t}" for t in targets for p in ("true", "predicted")]
    pred = pred[base + vcols].copy()
    if not np.isfinite(pred[vcols].to_numpy(float)).all():
        raise ValueError(f"[{path}]: NaN or infinite values in the predictions/true scores")
    if pred.duplicated(["child_id", "repeat"]).any():
        d = pred[pred.duplicated(["child_id", "repeat"], keep=False)]
        raise ValueError(f"[{path}]: {len(d)} rows are duplicated for the same "
                         f"(child, repeat), e.g. child {d.child_id.iloc[0]}")

    reps = sorted(pred["repeat"].unique())
    fr = folds[folds["repeat"].isin(reps)]
    m = pred.merge(fr, on=["child_id", "repeat"], how="outer",
                   suffixes=("", "_ref"), indicator=True)
    if (m["_merge"] == "right_only").any():
        x = m[m["_merge"] == "right_only"]
        raise ValueError(f"[{path}]: {len(x)} (child, repeat) of folds.csv have no "
                         f"prediction, e.g. child {x.child_id.iloc[0]} repeat {x.repeat.iloc[0]}. "
                         f"Every child must be predicted in every repeat you report.")
    if (m["_merge"] == "left_only").any():
        x = m[m["_merge"] == "left_only"]
        raise ValueError(f"[{path}]: {len(x)} predictions are for children not in "
                         f"folds.csv, e.g. child {x.child_id.iloc[0]}. Use the shared cohort.")
    if (m["fold"] != m["fold_ref"]).any():
        x = m[m["fold"] != m["fold_ref"]]
        raise ValueError(f"[{path}]: {len(x)} rows have a different fold than "
                         f"folds.csv (e.g. child {x.child_id.iloc[0]}). Splits must not be changed.")

    for t in targets:
        nun = pred.groupby("child_id")[f"true_{t}"].nunique()
        if (nun > 1).any():
            raise ValueError(f"[{path}]: true_{t} changes between repeats for child "
                             f"{nun[nun > 1].index[0]}")
        if cohort is not None:
            if t not in cohort.columns:
                raise ValueError(f"[{path}]: the cohort file has no column {t}")
            lab = pred.drop_duplicates("child_id").set_index("child_id")[f"true_{t}"]
            ref = cohort.set_index("child_id")[t].reindex(lab.index)
            bad = (np.abs(lab - ref) > 1e-6) | ref.isna()
            if bad.any():
                raise ValueError(f"[{path}]: true_{t} of child {lab.index[bad][0]} "
                                 f"differs from the cohort file")
        p = pred[f"predicted_{t}"]
        if p.min() < 20 or p.max() > 200:
            print(f"WARNING [{path}]: predicted_{t} goes from {p.min():.1f} to {p.max():.1f}; "
                  f"IQ scores are around 100. Is the output scaled/standardized by mistake?")

    children = sorted(pred["child_id"].unique(), key=lambda s: int(s))
    pos = {c: i for i, c in enumerate(children)}
    rpos = {r: i for i, r in enumerate(reps)}
    R, n = len(reps), len(children)
    ci = pred["child_id"].map(pos).to_numpy()
    ri = pred["repeat"].map(rpos).to_numpy()
    F = np.zeros((R, n), dtype=int)
    F[ri, ci] = pred["fold"].astype(int).to_numpy()
    out = {"children": children, "repeats": reps, "targets": targets, "F": F,
           "Y": {}, "P": {}, "B": {}}
    for t in targets:
        Y = np.zeros(n)
        Y[ci] = pred[f"true_{t}"].to_numpy(float)
        P = np.zeros((R, n))
        P[ri, ci] = pred[f"predicted_{t}"].to_numpy(float)
        # baseline: for each test fold, the mean score of the training children
        Bm = np.zeros((R, n))
        for i in range(R):
            for k in np.unique(F[i]):
                Bm[i, F[i] == k] = Y[F[i] != k].mean()
        out["Y"][t], out["P"][t], out["B"][t] = Y, P, Bm
    return out


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def _r2(Y, P):
    ss_tot = ((Y - Y.mean()) ** 2).sum()
    return 1.0 - ((P - Y) ** 2).sum(axis=-1) / ss_tot


def _corr(fn, y, p):
    if np.ptp(p) == 0:
        return np.nan                    # constant prediction (the baseline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(y, p)[0]


def per_repeat_metrics(Y, P, corr=True):
    """One row per repeat: metrics computed on the pooled out-of-fold predictions.
    corr=False for the mean baseline: its pooled correlation is meaningless
    (a constant per fold, and the leave-fold-out mean moves against the labels)."""
    err = P - Y
    return pd.DataFrame({
        "MAE": np.abs(err).mean(axis=1),
        "RMSE": np.sqrt((err ** 2).mean(axis=1)),
        "R2": _r2(Y, P),
        "Pearson": [_corr(stats.pearsonr, Y, p) if corr else np.nan for p in P],
        "Spearman": [_corr(stats.spearmanr, Y, p) if corr else np.nan for p in P],
    })


def bootstrap(Y, mats, n_boot, seed):
    """Resample CHILDREN with replacement (same draws for every model).

    mats: dict name -> (R,n) predictions. Returns dict name -> {"MAE": (B,), "R2": (B,)},
    each value being the mean over repeats for that bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    n = Y.size
    out = {k: {"MAE": np.empty(n_boot), "R2": np.empty(n_boot)} for k in mats}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        y = Y[idx]
        for k, P in mats.items():
            p = P[:, idx]
            out[k]["MAE"][b] = np.abs(p - y).mean()
            ss_tot = ((y - y.mean()) ** 2).sum()
            out[k]["R2"][b] = (1.0 - ((p - y) ** 2).sum(axis=1) / ss_tot).mean()
    return out


def _ci(x):
    return np.percentile(x, [2.5, 97.5])


# ----------------------------------------------------------------------------
# report for one target
# ----------------------------------------------------------------------------
def _report_target(target, Y, mats, names, n_boot, seed, print):
    """Metrics, bootstrap CIs and paired comparisons for ONE target.
    mats: dict model name -> (R,n) predictions, including 'mean_baseline'."""
    R = next(iter(mats.values())).shape[0]
    print(f"\n{'=' * 8} {target}: mean {Y.mean():.1f}, std {Y.std(ddof=1):.1f} {'=' * 8}")
    per_rep = {k: per_repeat_metrics(Y, P, corr=(k != "mean_baseline")) for k, P in mats.items()}
    print(f"Bootstrap on children ({n_boot} draws)...", flush=True)
    boot = bootstrap(Y, mats, n_boot, seed)

    summary = []
    for k in mats:
        pr = per_rep[k]
        row = {"model": k}
        for met in ["MAE", "RMSE", "R2", "Pearson", "Spearman"]:
            row[met] = pr[met].mean()
            row[met + "_std_repeats"] = pr[met].std(ddof=1) if R > 1 else np.nan
        row["MAE_ci_low"], row["MAE_ci_high"] = _ci(boot[k]["MAE"])
        row["R2_ci_low"], row["R2_ci_high"] = _ci(boot[k]["R2"])
        d = boot["mean_baseline"]["MAE"] - boot[k]["MAE"]          # improvement over the baseline
        row["MAE_gain_vs_baseline"] = per_rep["mean_baseline"]["MAE"].mean() - pr["MAE"].mean()
        row["gain_ci_low"], row["gain_ci_high"] = _ci(d)
        row["share_boot_gain_gt0"] = float((d > 0).mean())
        summary.append(row)
    summary = pd.DataFrame(summary)

    def fmt(m, s, nd=2):
        return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if np.isfinite(s) else "")
    show = pd.DataFrame({
        "model": summary["model"],
        "MAE (±std repeats)": [fmt(a, b) for a, b in zip(summary.MAE, summary.MAE_std_repeats)],
        "MAE 95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(summary.MAE_ci_low, summary.MAE_ci_high)],
        "RMSE": [fmt(a, b) for a, b in zip(summary.RMSE, summary.RMSE_std_repeats)],
        "R2": [fmt(a, b, 3) for a, b in zip(summary.R2, summary.R2_std_repeats)],
        "R2 95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(summary.R2_ci_low, summary.R2_ci_high)],
        "Pearson": [("-" if not np.isfinite(a) else f"{a:.3f}") for a in summary.Pearson],
        "Spearman": [("-" if not np.isfinite(a) else f"{a:.3f}") for a in summary.Spearman],
    })
    print(show.to_string(index=False))

    g = summary[summary.model != "mean_baseline"]
    print("\nGain over the mean baseline (positive = model has lower MAE):")
    print(pd.DataFrame({
        "model": g["model"],
        "MAE gain": [f"{v:+.2f}" for v in g.MAE_gain_vs_baseline],
        "95% CI": [f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(g.gain_ci_low, g.gain_ci_high)],
        "share of bootstrap draws with gain>0": [f"{v:.2f}" for v in g.share_boot_gain_gt0],
    }).to_string(index=False))
    print("A CI that contains 0 means: no demonstrated improvement over predicting the mean.")

    names_m = [k for k in mats if k != "mean_baseline"]
    comp = []
    if len(names_m) > 1:
        print("\nPaired comparison between models (same children, same folds):")
        for a, b in itertools.combinations(names_m, 2):
            d_boot = boot[a]["MAE"] - boot[b]["MAE"]          # negative: a is better
            d_rep = per_rep[a]["MAE"].to_numpy() - per_rep[b]["MAE"].to_numpy()
            try:
                p_w = stats.wilcoxon(d_rep).pvalue if R >= 6 else np.nan
            except ValueError:
                p_w = np.nan
            lo, hi = _ci(d_boot)
            comp.append({"model_a": a, "model_b": b, "MAE_a_minus_b": d_rep.mean(),
                         "ci_low": lo, "ci_high": hi, "wilcoxon_p_over_repeats": p_w})
            better = a if d_rep.mean() < 0 else b
            sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI contains 0 -> difference not established"
            print(f"{a} vs {b}: MAE({a}) - MAE({b}) = {d_rep.mean():+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]  ({better} lower; {sig}; "
                  f"Wilcoxon over repeats p={p_w:.3g})")
    return summary, (pd.DataFrame(comp) if comp else None), per_rep


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def evaluate(folds, cohort, predictions, names=None, targets=None, out_dir=None,
             n_boot=2000, seed=0, verbose=True):
    """Evaluate one or several predictions files. Usable in a script or a notebook.

        summary, comparison = evaluate("data/cleaned/folds.csv", "data/cleaned/cohort.csv",
                                       ["predictions_A_x.csv", "predictions_B_y.csv"])

    folds, cohort : paths of folds.csv and cohort.csv (cohort may be None)
    predictions   : list of predictions files
    names         : optional model names (default: file names)
    targets       : targets to report (default: all those present in every file)
    Returns (summary, comparison): DataFrames with a 'target' column (comparison is
    None with a single model). Prints the report unless verbose=False.
    Raises ValueError if the files are invalid or not comparable.
    """
    if isinstance(predictions, (str, Path)):
        predictions = [predictions]
    print = _print if verbose else (lambda *a, **k: None)
    names = names or [Path(p).stem.replace("predictions_", "") for p in predictions]
    if len(names) != len(predictions):
        raise ValueError("names must have as many entries as predictions")
    if len(set(names)) != len(names):
        raise ValueError("model names must be different")

    folds = _read(folds)
    cohort = _read(cohort) if cohort else None
    data = {nm: load_and_check(p, folds, cohort) for nm, p in zip(names, predictions)}

    # all models must be evaluated on exactly the same children, repeats and folds
    ref_name = names[0]
    ref = data[ref_name]
    for nm in names[1:]:
        d = data[nm]
        if d["children"] != ref["children"] or d["repeats"] != ref["repeats"] \
                or not np.array_equal(d["F"], ref["F"]):
            raise ValueError(f"'{nm}' and '{ref_name}' do not use the same children / "
                             f"repeats / folds, so they are NOT comparable.")
    common = [t for t in ref["targets"] if all(t in data[nm]["targets"] for nm in names)]
    if targets is not None:
        absent = [t for t in targets if t not in common]
        if absent:
            raise ValueError(f"target(s) {absent} are not in all the predictions files")
        common = list(targets)
    if not common:
        raise ValueError("the predictions files have no target in common")
    dropped = sorted({t for nm in names for t in data[nm]["targets"]} - set(common))
    if dropped and targets is None:
        print(f"NOTE: {dropped} is not in every file, so it is not evaluated.")
    n, R = len(ref["children"]), len(ref["repeats"])
    print(f"\n{n} children, {R} repeat(s) x {len(np.unique(ref['F']))} folds, "
          f"targets: {', '.join(common)}")

    summaries, comps, per_reps = [], [], []
    for t in common:
        mats = {"mean_baseline": ref["B"][t], **{nm: data[nm]["P"][t] for nm in names}}
        s, c, pr = _report_target(t, ref["Y"][t], mats, names, n_boot, seed, print)
        summaries.append(s.assign(target=t))
        if c is not None:
            comps.append(c.assign(target=t))
        for k, v in pr.items():
            per_reps.append(v.assign(model=k, target=t, repeat=ref["repeats"]))

    def first(df):                                  # put 'target' first
        return df[["target"] + [c for c in df.columns if c != "target"]]
    summary = first(pd.concat(summaries, ignore_index=True))
    comp = first(pd.concat(comps, ignore_index=True)) if comps else None

    if len(common) > 1:
        print(f"\n{'=' * 8} Overview (MAE and R2 per target) {'=' * 8}")
        print(summary.pivot(index="model", columns="target", values=["MAE", "R2"])
              [[(m, t) for m in ("MAE", "R2") for t in common]].round(3).to_string())

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "results.csv", index=False)
        first(pd.concat(per_reps, ignore_index=True)).to_csv(out / "per_repeat_metrics.csv", index=False)
        if comp is not None:
            comp.to_csv(out / "comparison.csv", index=False)
        print(f"\nSaved results.csv, per_repeat_metrics.csv"
              f"{', comparison.csv' if comp is not None else ''} in {out}/")
    return summary, comp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", required=True, help="folds.csv")
    ap.add_argument("--cohort", default=None, help="cohort.csv (checks the true scores)")
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None,
                    help="names of the models (default: file names)")
    ap.add_argument("--targets", nargs="+", default=None, help="e.g. QIT (default: all)")
    ap.add_argument("--out-dir", default=None, help="write csv results here")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    try:
        evaluate(a.folds, a.cohort, a.predictions, a.names, a.targets, a.out_dir,
                 a.n_boot, a.seed)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()