"""
Censoring sensitivity analysis and inverse probability of censoring
weighting (IPCW).

Chain length T correlates with problem difficulty, creating informative
censoring.  Error-free chains are right-censored at T, but harder problems
produce longer T — so late steps are enriched for hard problems, potentially
inflating the aging signal.

Three analyses:
  1. Censoring correlation test  (Spearman: n_steps vs event / difficulty)
  2. Sensitivity analysis         (re-fit Weibull at truncated horizons)
  3. IPCW                         (inverse-probability-of-censoring weighting)
"""

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import WeibullFitter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, ALPHA, RESULTS_DIR, DATA_DIR


def _load_data() -> pd.DataFrame:
    """Load survival_data.csv with standard preprocessing."""
    csv_path = DATA_DIR / "survival_data.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "survival_data.csv"
    df = pd.read_csv(csv_path)
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)
    df["n_steps"] = df["n_steps"].astype(int)
    return df


def _fit_weibull(durations: np.ndarray, events: np.ndarray,
                 weights: np.ndarray | None = None,
                 min_events: int = 3) -> dict | None:
    """Fit Weibull and return rho + lambda; None if insufficient data."""
    if events.sum() < min_events or len(durations) < 5:
        return None
    durations = np.maximum(durations.astype(float), 0.5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter()
            wf.fit(durations, events, weights=weights)
            summary = wf.summary
            col_low = [c for c in summary.columns if "lower" in c.lower()]
            col_high = [c for c in summary.columns if "upper" in c.lower()]
            rho_row = summary.loc["rho_"]
            rho_ci_low = float(rho_row[col_low[0]]) if col_low else float(wf.rho_) * 0.9
            rho_ci_high = float(rho_row[col_high[0]]) if col_high else float(wf.rho_) * 1.1
            return {
                "rho": float(wf.rho_),
                "lambda": float(wf.lambda_),
                "rho_ci_low": rho_ci_low,
                "rho_ci_high": rho_ci_high,
                "n": len(durations),
                "n_events": int(events.sum()),
            }
    except Exception:
        return None


def censoring_correlation_test(df: pd.DataFrame) -> dict:
    """
    For each dataset, compute:
      - Spearman(n_steps, event): does chain length predict whether error occurs?
      - Spearman(n_steps, 1 - final_correct): does chain length predict difficulty?

    If both are significant, informative censoring is confirmed.
    """
    results = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]

        rho_ev, p_ev = stats.spearmanr(sub["n_steps"], sub["event"])

        difficulty = 1 - sub["final_correct"].fillna(0).astype(int)
        rho_diff, p_diff = stats.spearmanr(sub["n_steps"], difficulty)

        rho_cens, p_cens = stats.spearmanr(sub["n_steps"], 1 - sub["event"])

        informative = (p_cens < ALPHA) and (abs(rho_cens) > 0.05)

        results[ds] = {
            "n_chains": len(sub),
            "n_censored": int((sub["event"] == 0).sum()),
            "censoring_rate": round(float((sub["event"] == 0).mean()), 4),
            "spearman_nsteps_event": {
                "rho": round(float(rho_ev), 4),
                "p": round(float(p_ev), 6),
                "significant": bool(p_ev < ALPHA),
            },
            "spearman_nsteps_difficulty": {
                "rho": round(float(rho_diff), 4),
                "p": round(float(p_diff), 6),
                "significant": bool(p_diff < ALPHA),
            },
            "spearman_nsteps_censored": {
                "rho": round(float(rho_cens), 4),
                "p": round(float(p_cens), 6),
                "significant": bool(p_cens < ALPHA),
            },
            "informative_censoring": informative,
            "interpretation": (
                f"Chain length and censoring are "
                f"{'correlated' if informative else 'uncorrelated'} "
                f"(rho={rho_cens:.3f}, p={p_cens:.4f})"
            ),
        }

    return results


TRUNCATION_HORIZONS = [5, 8, 10, 15, 20]


def censoring_sensitivity(df: pd.DataFrame) -> dict:
    """
    For each dataset x model cell, truncate chains at shorter horizons and
    re-fit Weibull.  Report how rho changes — stability implies censoring
    does not materially bias the estimate.
    """
    cell_results = []
    dataset_avg = {}

    for ds in sorted(df["dataset"].unique()):
        ds_rhos_by_horizon = {h: [] for h in TRUNCATION_HORIZONS + ["original"]}

        for model in sorted(df["model"].unique()):
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if len(sub) < 10 or sub["event"].sum() < 3:
                continue

            orig_dur = sub["duration"].values.astype(float)
            orig_evt = sub["event"].values.astype(int)
            orig_max = int(sub["n_steps"].max())

            orig_fit = _fit_weibull(orig_dur, orig_evt)
            if orig_fit is None:
                continue

            horizons_data = []

            for max_t in TRUNCATION_HORIZONS:
                if max_t >= orig_max:
                    continue
                trunc_dur = np.minimum(orig_dur, float(max_t))
                trunc_evt = np.where(
                    (orig_evt == 1) & (orig_dur <= max_t), 1, 0
                ).astype(int)

                fit = _fit_weibull(trunc_dur, trunc_evt)
                if fit is not None:
                    horizons_data.append({
                        "max_T": max_t,
                        "rho": round(fit["rho"], 4),
                        "lambda": round(fit["lambda"], 4),
                        "n_events": fit["n_events"],
                        "censoring_rate": round(1 - fit["n_events"] / fit["n"], 4),
                    })
                    ds_rhos_by_horizon[max_t].append(fit["rho"])

            horizons_data.append({
                "max_T": orig_max,
                "rho": round(orig_fit["rho"], 4),
                "lambda": round(orig_fit["lambda"], 4),
                "n_events": orig_fit["n_events"],
                "censoring_rate": round(1 - orig_fit["n_events"] / orig_fit["n"], 4),
            })
            ds_rhos_by_horizon["original"].append(orig_fit["rho"])

            if horizons_data:
                rho_orig = orig_fit["rho"]
                max_change_pct = max(
                    abs(h["rho"] - rho_orig) / rho_orig * 100
                    for h in horizons_data if h["max_T"] != orig_max
                ) if len(horizons_data) > 1 else 0.0
            else:
                max_change_pct = 0.0

            cell_results.append({
                "dataset": ds,
                "model": model,
                "original_rho": round(orig_fit["rho"], 4),
                "original_max_T": orig_max,
                "horizons": horizons_data,
                "max_rho_change_pct": round(max_change_pct, 2),
                "stable": max_change_pct < 15.0,
            })

        avg_data = {}
        for h_key in TRUNCATION_HORIZONS + ["original"]:
            vals = ds_rhos_by_horizon[h_key]
            if vals:
                avg_data[str(h_key)] = {
                    "mean_rho": round(float(np.mean(vals)), 4),
                    "std_rho": round(float(np.std(vals)), 4),
                    "n_cells": len(vals),
                }
        dataset_avg[ds] = avg_data

    all_stable = [c for c in cell_results if "stable" in c]
    n_stable = sum(1 for c in all_stable if c["stable"])

    return {
        "cells": cell_results,
        "dataset_averages": dataset_avg,
        "summary": {
            "n_cells_analyzed": len(cell_results),
            "n_stable": n_stable,
            "pct_stable": round(100 * n_stable / len(cell_results), 1) if cell_results else 0,
            "truncation_horizons": TRUNCATION_HORIZONS,
            "stability_threshold_pct": 15.0,
        },
    }


def ipcw_analysis(df: pd.DataFrame) -> dict:
    """
    For each dataset x model cell:
      1. Fit logistic regression: P(censored | log(n_steps), dataset difficulty)
      2. Compute censoring probability G(t) for each observation
      3. Weight each UNCENSORED observation by 1/G(t)
      4. Re-fit weighted Weibull
      5. Compare rho_standard vs rho_ipcw
    """
    cell_results = []

    problem_difficulty = (
        df.groupby(["dataset", "problem_id"])["final_correct"]
        .apply(lambda x: 1 - x.fillna(0).astype(int).mean())
        .reset_index()
        .rename(columns={"final_correct": "problem_difficulty"})
    )
    df_aug = df.merge(problem_difficulty, on=["dataset", "problem_id"], how="left")
    df_aug["problem_difficulty"] = df_aug["problem_difficulty"].fillna(0.5)

    for ds in sorted(df["dataset"].unique()):
        for model in sorted(df["model"].unique()):
            sub = df_aug[(df_aug["dataset"] == ds) & (df_aug["model"] == model)].copy()
            if len(sub) < 10 or sub["event"].sum() < 3:
                continue

            durations = sub["duration"].values.astype(float)
            events = sub["event"].values.astype(int)

            std_fit = _fit_weibull(durations, events)
            if std_fit is None:
                continue

            censored = (1 - events).astype(int)

            X = np.column_stack([
                np.log1p(sub["n_steps"].values),
                sub["problem_difficulty"].values,
            ])

            if censored.sum() < 2 or censored.sum() > len(censored) - 2:
                cell_results.append({
                    "dataset": ds,
                    "model": model,
                    "rho_standard": round(std_fit["rho"], 4),
                    "rho_ipcw": round(std_fit["rho"], 4),
                    "ipcw_note": "insufficient censoring variation",
                    "rho_change_pct": 0.0,
                    "censoring_model_auc": None,
                    "n_chains": len(sub),
                    "n_events": int(events.sum()),
                })
                continue

            try:
                from scipy.optimize import minimize

                def _logistic_nll(beta, X, y):
                    Xb = X @ beta
                    Xb = np.clip(Xb, -30, 30)
                    p = 1 / (1 + np.exp(-Xb))
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                beta0 = np.zeros(X_with_intercept.shape[1])
                opt = minimize(_logistic_nll, beta0, args=(X_with_intercept, censored),
                               method="L-BFGS-B", options={"maxiter": 1000})
                beta_hat = opt.x
                logits = X_with_intercept @ beta_hat
                logits = np.clip(logits, -30, 30)
                p_censored = 1 / (1 + np.exp(-logits))

                g_t = 1 - p_censored
                g_t = np.clip(g_t, 0.05, 1.0)

                weights = np.where(events == 1, 1.0 / g_t, 1.0)
                weights = weights * len(weights) / weights.sum()

                ipcw_fit = _fit_weibull(durations, events, weights=weights)

                if ipcw_fit is None:
                    ipcw_fit = std_fit

                try:
                    sorted_idx = np.argsort(-p_censored)
                    sorted_y = censored[sorted_idx]
                    n_pos = sorted_y.sum()
                    n_neg = len(sorted_y) - n_pos
                    if n_pos > 0 and n_neg > 0:
                        tpr_sum = np.cumsum(sorted_y)
                        fpr_sum = np.cumsum(1 - sorted_y)
                        auc = float(np.sum(sorted_y * fpr_sum) / (n_pos * n_neg))
                    else:
                        auc = None
                except Exception:
                    auc = None

                rho_change = (ipcw_fit["rho"] - std_fit["rho"]) / std_fit["rho"] * 100

                cell_results.append({
                    "dataset": ds,
                    "model": model,
                    "rho_standard": round(std_fit["rho"], 4),
                    "rho_ipcw": round(ipcw_fit["rho"], 4),
                    "rho_change_pct": round(rho_change, 2),
                    "lambda_standard": round(std_fit["lambda"], 4),
                    "lambda_ipcw": round(ipcw_fit["lambda"], 4),
                    "censoring_model_auc": round(float(auc), 4) if auc is not None else None,
                    "mean_weight": round(float(weights.mean()), 4),
                    "max_weight": round(float(weights.max()), 4),
                    "n_chains": len(sub),
                    "n_events": int(events.sum()),
                })

            except Exception as e:
                cell_results.append({
                    "dataset": ds,
                    "model": model,
                    "rho_standard": round(std_fit["rho"], 4),
                    "rho_ipcw": round(std_fit["rho"], 4),
                    "error": str(e),
                    "rho_change_pct": 0.0,
                    "n_chains": len(sub),
                    "n_events": int(events.sum()),
                })

    valid = [c for c in cell_results if "rho_standard" in c and "rho_ipcw" in c]
    if valid:
        changes = [abs(c["rho_change_pct"]) for c in valid]
        mean_change = float(np.mean(changes))
        max_change = float(np.max(changes))
        n_within_5pct = sum(1 for c in changes if c < 5.0)
        n_within_10pct = sum(1 for c in changes if c < 10.0)
    else:
        mean_change = max_change = 0.0
        n_within_5pct = n_within_10pct = 0

    return {
        "cells": cell_results,
        "summary": {
            "n_cells": len(valid),
            "mean_abs_rho_change_pct": round(mean_change, 2),
            "max_abs_rho_change_pct": round(max_change, 2),
            "n_within_5pct": n_within_5pct,
            "n_within_10pct": n_within_10pct,
            "interpretation": (
                f"IPCW adjustment changes rho by {mean_change:.1f}% on average "
                f"(max {max_change:.1f}%). {n_within_5pct}/{len(valid)} cells "
                f"within 5%, {n_within_10pct}/{len(valid)} within 10%."
            ),
        },
    }


def run_all() -> dict:
    """Run all censoring sensitivity analyses and save results."""
    print("Loading data...")
    df = _load_data()
    print(f"  {len(df)} chains, {df['event'].sum()} events, "
          f"{(df['event']==0).sum()} censored")

    print("\n1. Censoring correlation test...")
    corr = censoring_correlation_test(df)
    for ds, r in corr.items():
        print(f"  {ds}: rho(n_steps, censored) = {r['spearman_nsteps_censored']['rho']:.3f}, "
              f"p = {r['spearman_nsteps_censored']['p']:.4f} "
              f"-> {'INFORMATIVE' if r['informative_censoring'] else 'non-informative'}")

    print("\n2. Censoring sensitivity analysis (truncation)...")
    sensitivity = censoring_sensitivity(df)
    print(f"  {sensitivity['summary']['n_stable']}/{sensitivity['summary']['n_cells_analyzed']} "
          f"cells stable across truncation horizons")

    print("\n3. IPCW analysis...")
    ipcw = ipcw_analysis(df)
    print(f"  Mean |rho change|: {ipcw['summary']['mean_abs_rho_change_pct']:.1f}%")
    print(f"  {ipcw['summary']['n_within_5pct']}/{ipcw['summary']['n_cells']} cells within 5%")

    results = {
        "censoring_correlation": corr,
        "sensitivity_analysis": sensitivity,
        "ipcw": ipcw,
    }

    out_path = RESULTS_DIR / "censoring_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    run_all()
