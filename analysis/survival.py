"""
Core survival analysis: Kaplan-Meier, hazard rates, Cox PH, Simpson's Paradox.

All analyses use standard `lifelines` library for statistical rigor.
"""

import numpy as np
import pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, N_BOOTSTRAP, ALPHA


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """
    Compute Kaplan-Meier survival estimates with Greenwood confidence intervals.

    Args:
        durations: time-to-event (step index of first error, or chain length if censored)
        events: 1 = error occurred, 0 = right-censored (chain completed correctly)

    Returns:
        DataFrame with columns: step, at_risk, events, hazard, survival, ci_low, ci_high
    """
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, alpha=ALPHA)

    timeline = kmf.survival_function_.index.values
    survival = kmf.survival_function_.values.flatten()
    ci = kmf.confidence_interval_survival_function_

    rows = []
    for i, t in enumerate(timeline):
        rows.append({
            "step": t,
            "survival": survival[i],
            "ci_low": ci.iloc[i, 0],
            "ci_high": ci.iloc[i, 1],
        })

    return pd.DataFrame(rows)


def discrete_hazard_rate(durations: np.ndarray, events: np.ndarray,
                         max_step: int = None) -> pd.DataFrame:
    """
    Compute the discrete hazard rate h(t) = d(t) / n(t).

    d(t) = number of events at time t
    n(t) = number at risk at time t (alive and uncensored just before t)
    """
    if max_step is None:
        max_step = int(durations.max())

    rows = []
    for t in range(1, max_step + 1):
        at_risk = int(np.sum(durations >= t))
        n_events = int(np.sum((durations == t) & (events == 1)))
        n_censored = int(np.sum((durations == t) & (events == 0)))
        hazard = n_events / at_risk if at_risk > 0 else 0.0

        se = np.sqrt(hazard * (1 - hazard) / at_risk) if at_risk > 10 else np.nan

        rows.append({
            "step": t,
            "at_risk": at_risk,
            "events": n_events,
            "censored": n_censored,
            "hazard": hazard,
            "hazard_se": se,
        })

    return pd.DataFrame(rows)


def raw_error_rate(error_vectors: list[list[int]], max_step: int = None) -> pd.DataFrame:
    """
    Compute the raw (unconditional) error rate at each step.

    P(error at t) = count of chains with error at step t / count of chains with >= t steps

    This is the NAIVE metric that ignores survival conditioning.
    """
    if max_step is None:
        max_step = max(len(v) for v in error_vectors)

    rows = []
    for t in range(max_step):
        chains_at_step = [v for v in error_vectors if len(v) > t]
        n_total = len(chains_at_step)
        n_errors = sum(1 for v in chains_at_step if v[t] == 1)
        rate = n_errors / n_total if n_total > 0 else 0.0

        rows.append({
            "step": t + 1,
            "n_total": n_total,
            "n_errors": n_errors,
            "error_rate": rate,
        })

    return pd.DataFrame(rows)


def cox_ph_analysis(df: pd.DataFrame) -> dict:
    """
    Cox Proportional Hazards model to control for difficulty.

    Model: h(t | X) = h_0(t) * exp(beta_difficulty * difficulty + beta_model * model)

    Uses chain length as a proxy for difficulty (longer reference solutions = harder problems).
    """
    from lifelines import CoxPHFitter

    analysis_df = df[["duration", "event", "n_steps", "dataset", "model"]].copy()
    analysis_df["log_n_steps"] = np.log1p(analysis_df["n_steps"])

    model_dummies = pd.get_dummies(analysis_df["model"], prefix="model", drop_first=True)
    dataset_dummies = pd.get_dummies(analysis_df["dataset"], prefix="ds", drop_first=True)

    # pandas >= 2.0 returns bool dummies; lifelines needs numeric dtypes
    model_dummies = model_dummies.astype(np.float64)
    dataset_dummies = dataset_dummies.astype(np.float64)

    cox_df = pd.concat([
        analysis_df[["duration", "event", "log_n_steps"]],
        model_dummies,
        dataset_dummies,
    ], axis=1)

    cox_df = cox_df.apply(pd.to_numeric, errors="coerce").dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

    return {
        "summary": cph.summary.to_dict(),
        "concordance": cph.concordance_index_,
        "log_likelihood": cph.log_likelihood_,
        "aic": cph.AIC_partial_,
        "coefficients": cph.params_.to_dict(),
        "p_values": cph.summary["p"].to_dict(),
        "model_object": cph,
    }


def test_increasing_hazard(hazard_df: pd.DataFrame, min_at_risk: int = 20) -> dict:
    """
    Statistical test for whether the hazard rate is increasing (aging).

    Uses Mann-Kendall trend test on the hazard values, restricted to steps
    with sufficient sample size.
    """
    filtered = hazard_df[hazard_df["at_risk"] >= min_at_risk].copy()
    if len(filtered) < 4:
        return {"test": "mann_kendall", "error": "too few steps with sufficient data"}

    hazard_vals = filtered["hazard"].values
    steps = filtered["step"].values

    tau, p_value = stats.kendalltau(steps, hazard_vals)

    first_half = hazard_vals[:len(hazard_vals)//2]
    second_half = hazard_vals[len(hazard_vals)//2:]
    mean_early = np.mean(first_half)
    mean_late = np.mean(second_half)

    return {
        "test": "mann_kendall",
        "tau": round(tau, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < ALPHA,
        "direction": "increasing" if tau > 0 else "decreasing" if tau < 0 else "flat",
        "mean_early_hazard": round(mean_early, 4),
        "mean_late_hazard": round(mean_late, 4),
        "hazard_ratio_late_early": round(mean_late / mean_early, 2) if mean_early > 0 else None,
        "n_steps_used": len(filtered),
    }


def apply_fdr_correction(trend_results: list[dict], alpha: float = ALPHA) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction to Mann-Kendall trend test p-values."""
    pvals = []
    indices = []
    for i, t in enumerate(trend_results):
        if "p_value" in t and t["p_value"] is not None:
            pvals.append(t["p_value"])
            indices.append(i)

    if not pvals:
        return trend_results

    n = len(pvals)
    p_arr = np.array(pvals)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    bh_critical = np.arange(1, n + 1) / n * alpha
    corrected = np.empty(n)
    corrected[sorted_idx] = np.minimum.accumulate(
        (sorted_p * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    corrected = np.minimum(corrected, 1.0)

    for j, orig_idx in enumerate(indices):
        trend_results[orig_idx]["p_corrected_bh"] = round(float(corrected[j]), 6)
        trend_results[orig_idx]["significant_bh"] = bool(corrected[j] < alpha)

    return trend_results


def test_aging_clustering(trend_results: list[dict], dataset_names: list[str]) -> dict:
    """Fisher's exact test: are significant aging results clustered on one dataset?"""
    sig_by_ds = {ds: 0 for ds in dataset_names}
    nonsig_by_ds = {ds: 0 for ds in dataset_names}

    for t in trend_results:
        if "p_value" not in t:
            continue
        ds = t.get("dataset", "")
        is_sig_aging = (
            t.get("significant_bh", t.get("significant", False))
            and t.get("direction") == "increasing"
        )
        if is_sig_aging:
            sig_by_ds[ds] = sig_by_ds.get(ds, 0) + 1
        else:
            nonsig_by_ds[ds] = nonsig_by_ds.get(ds, 0) + 1

    observed = np.array([[sig_by_ds.get(ds, 0) for ds in dataset_names],
                         [nonsig_by_ds.get(ds, 0) for ds in dataset_names]])

    from scipy.stats import fisher_exact as _fe
    if observed.shape[1] == 2:
        _, p = _fe(observed)
    else:
        from scipy.stats import chi2_contingency
        _, p, _, _ = chi2_contingency(observed)

    return {
        "test": "fisher_exact_clustering",
        "contingency_table": observed.tolist(),
        "datasets": dataset_names,
        "p_value": round(float(p), 6),
        "significant": p < ALPHA,
        "interpretation": "Aging results are significantly clustered on specific dataset(s)"
                         if p < ALPHA else "No significant clustering",
    }


def bootstrap_hazard_ratio(durations: np.ndarray, events: np.ndarray,
                            n_boot: int = 2000, seed: int = RANDOM_SEED,
                            min_at_risk: int = 20) -> dict:
    """Bootstrap 95% CI for the late/early hazard ratio."""
    rng = np.random.default_rng(seed)
    n = len(durations)
    ratios = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        b_dur = durations[idx]
        b_evt = events[idx]

        max_t = int(b_dur.max())
        h_early, h_late = [], []
        mid = max_t // 2
        for t in range(1, max_t + 1):
            ar = int(np.sum(b_dur >= t))
            ne = int(np.sum((b_dur == t) & (b_evt == 1)))
            if ar < min_at_risk:
                break
            h = ne / ar
            if t <= mid:
                h_early.append(h)
            else:
                h_late.append(h)

        me = np.mean(h_early) if h_early else 0
        ml = np.mean(h_late) if h_late else 0
        if me > 0:
            ratios.append(ml / me)

    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None}

    ratios = np.array(ratios)
    return {
        "median": round(float(np.median(ratios)), 2),
        "ci_low": round(float(np.percentile(ratios, 2.5)), 2),
        "ci_high": round(float(np.percentile(ratios, 97.5)), 2),
        "n_valid_boots": len(ratios),
    }


def cox_ph_with_assumption_test(df: pd.DataFrame) -> dict:
    """Cox PH + Schoenfeld residual test + stratified Cox as robustness."""
    from lifelines import CoxPHFitter

    analysis_df = df[["duration", "event", "n_steps", "dataset", "model"]].copy()
    analysis_df["log_n_steps"] = np.log1p(analysis_df["n_steps"])

    model_dummies = pd.get_dummies(analysis_df["model"], prefix="model", drop_first=True)
    dataset_dummies = pd.get_dummies(analysis_df["dataset"], prefix="ds", drop_first=True)
    model_dummies = model_dummies.astype(np.float64)
    dataset_dummies = dataset_dummies.astype(np.float64)

    cox_df = pd.concat([
        analysis_df[["duration", "event", "log_n_steps"]],
        model_dummies,
        dataset_dummies,
    ], axis=1)
    cox_df = cox_df.apply(pd.to_numeric, errors="coerce").dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

    ph_test_results = {}
    try:
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cph.check_assumptions(cox_df, show_plots=False, p_value_threshold=0.01)
        ph_output = f.getvalue()
        ph_test_results["output"] = ph_output
        ph_test_results["passed"] = "proportional hazard" not in ph_output.lower() or "pass" in ph_output.lower()
    except Exception as e:
        ph_test_results["output"] = str(e)
        ph_test_results["passed"] = "pass" in str(e).lower()

    stratified_results = None
    try:
        strat_df = pd.concat([
            analysis_df[["duration", "event", "log_n_steps", "dataset"]],
            model_dummies,
        ], axis=1)
        strat_df = strat_df.apply(pd.to_numeric, errors="coerce").dropna()
        strat_df["dataset"] = analysis_df["dataset"].values[:len(strat_df)]

        cph_strat = CoxPHFitter()
        cph_strat.fit(strat_df, duration_col="duration", event_col="event",
                      strata=["dataset"], show_progress=False)
        stratified_results = {
            "coefficients": cph_strat.params_.to_dict(),
            "p_values": cph_strat.summary["p"].to_dict(),
            "concordance": cph_strat.concordance_index_,
        }
    except Exception as e:
        stratified_results = {"error": str(e)}

    return {
        "standard": {
            "summary": cph.summary.to_dict(),
            "concordance": cph.concordance_index_,
            "log_likelihood": cph.log_likelihood_,
            "aic": cph.AIC_partial_,
            "coefficients": cph.params_.to_dict(),
            "p_values": cph.summary["p"].to_dict(),
            "model_object": cph,
        },
        "ph_assumption_test": ph_test_results,
        "stratified_by_dataset": stratified_results,
    }


def compute_half_life(km_df: pd.DataFrame) -> float:
    """Compute the median survival time (half-life) from Kaplan-Meier estimates."""
    below_50 = km_df[km_df["survival"] <= 0.5]
    if len(below_50) == 0:
        return float("inf")
    return float(below_50.iloc[0]["step"])


def weibull_per_cell(df: pd.DataFrame, min_chains: int = 10) -> dict:
    """
    Fit Weibull model to each dataset×model pair using WeibullFitter.

    Returns dict with:
      - cells: list of per-cell results (rho, CI, z-test, interpretation)
      - dataset_summary: per-dataset pooled rho
      - binomial_test: test for H0: rho <= 1 in all cells
    """
    import warnings
    from lifelines import WeibullFitter

    cells = []
    for ds in sorted(df["dataset"].unique()):
        for model in sorted(df["model"].unique()):
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if len(sub) < min_chains or sub["event"].sum() < 3:
                continue

            durations = sub["duration"].values.astype(float)
            events = sub["event"].values.astype(float)
            durations = np.maximum(durations, 0.5)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf = WeibullFitter()
                    wf.fit(durations, events)

                rho = float(wf.rho_)
                lambda_ = float(wf.lambda_)

                summary = wf.summary
                rho_row = summary.loc["rho_"]
                col_low = [c for c in summary.columns if "lower" in c.lower()]
                col_high = [c for c in summary.columns if "upper" in c.lower()]
                rho_ci_low = float(rho_row[col_low[0]]) if col_low else rho * 0.9
                rho_ci_high = float(rho_row[col_high[0]]) if col_high else rho * 1.1

                se = (rho_ci_high - rho_ci_low) / (2 * 1.96)
                z = (rho - 1.0) / se if se > 0 else 0.0
                p = 2 * (1 - stats.norm.cdf(abs(z)))

                if rho_ci_low > 1.0:
                    interp = "AGING"
                elif rho_ci_high < 1.0:
                    interp = "LINDY"
                else:
                    interp = "CONSTANT"

                cells.append({
                    "dataset": ds,
                    "model": model,
                    "rho": round(rho, 4),
                    "rho_ci_low": round(rho_ci_low, 4),
                    "rho_ci_high": round(rho_ci_high, 4),
                    "lambda": round(lambda_, 4),
                    "z_test": round(z, 2),
                    "p_value": round(p, 6),
                    "interpretation": interp,
                    "n_chains": len(sub),
                    "n_events": int(sub["event"].sum()),
                })
            except Exception as e:
                cells.append({
                    "dataset": ds, "model": model,
                    "error": str(e), "n_chains": len(sub),
                })

    ds_summary = {}
    for ds in sorted(df["dataset"].unique()):
        ds_cells = [c for c in cells if c.get("dataset") == ds and "rho" in c]
        if ds_cells:
            rhos = [c["rho"] for c in ds_cells]
            ds_summary[ds] = {
                "mean_rho": round(float(np.mean(rhos)), 4),
                "min_rho": round(float(np.min(rhos)), 4),
                "max_rho": round(float(np.max(rhos)), 4),
                "n_aging": sum(1 for c in ds_cells if c["interpretation"] == "AGING"),
                "n_total": len(ds_cells),
            }

    valid = [c for c in cells if "rho" in c]
    n_aging = sum(1 for c in valid if c["rho"] > 1 and c.get("rho_ci_low", 0) > 1)
    n_total = len(valid)
    binom_p = float(stats.binomtest(n_aging, n_total, 0.5, alternative="greater").pvalue) if n_total > 0 else 1.0

    return {
        "cells": cells,
        "dataset_summary": ds_summary,
        "binomial_test": {
            "n_aging_sig": n_aging,
            "n_total": n_total,
            "p_value": round(binom_p, 10),
            "interpretation": f"{n_aging}/{n_total} cells show significant aging (rho>1, CI excludes 1)",
        },
    }


def weibull_gof_test(df: pd.DataFrame, min_chains: int = 10) -> dict:
    """
    Weibull goodness-of-fit: compare Weibull vs LogNormal vs LogLogistic
    using AIC/BIC. Also compute log-log linearity R² for each cell.
    """
    import warnings
    from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter

    cells = []
    for ds in sorted(df["dataset"].unique()):
        for model in sorted(df["model"].unique()):
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if len(sub) < min_chains or sub["event"].sum() < 3:
                continue

            durations = np.maximum(sub["duration"].values.astype(float), 0.5)
            events = sub["event"].values.astype(float)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    wf = WeibullFitter()
                    wf.fit(durations, events)
                    aic_weibull = wf.AIC_
                    bic_weibull = wf.BIC_

                    lnf = LogNormalFitter()
                    lnf.fit(durations, events)
                    aic_lognormal = lnf.AIC_
                    bic_lognormal = lnf.BIC_

                    llf = LogLogisticFitter()
                    llf.fit(durations, events)
                    aic_loglogistic = llf.AIC_
                    bic_loglogistic = llf.BIC_

                aics = {"Weibull": aic_weibull, "LogNormal": aic_lognormal,
                        "LogLogistic": aic_loglogistic}
                best = min(aics, key=aics.get)

                from lifelines import KaplanMeierFitter
                kmf = KaplanMeierFitter()
                kmf.fit(durations, events)
                km_t = kmf.survival_function_.index.values[1:]
                km_s = kmf.survival_function_.values.flatten()[1:]
                valid_mask = (km_s > 0) & (km_s < 1) & (km_t > 0)
                r2_loglog = np.nan
                if valid_mask.sum() >= 3:
                    log_t = np.log(km_t[valid_mask])
                    log_neg_log_s = np.log(-np.log(km_s[valid_mask]))
                    if len(log_t) >= 3:
                        slope, intercept = np.polyfit(log_t, log_neg_log_s, 1)
                        ss_res = np.sum((log_neg_log_s - (slope * log_t + intercept)) ** 2)
                        ss_tot = np.sum((log_neg_log_s - np.mean(log_neg_log_s)) ** 2)
                        r2_loglog = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                cells.append({
                    "dataset": ds, "model": model,
                    "aic_weibull": round(aic_weibull, 1),
                    "aic_lognormal": round(aic_lognormal, 1),
                    "aic_loglogistic": round(aic_loglogistic, 1),
                    "bic_weibull": round(bic_weibull, 1),
                    "bic_lognormal": round(bic_lognormal, 1),
                    "bic_loglogistic": round(bic_loglogistic, 1),
                    "best_aic": best,
                    "weibull_is_best": best == "Weibull",
                    "weibull_delta_aic": round(aic_weibull - min(aics.values()), 1),
                    "loglog_r2": round(float(r2_loglog), 4) if not np.isnan(r2_loglog) else None,
                })
            except Exception as e:
                cells.append({"dataset": ds, "model": model, "error": str(e)})

    valid = [c for c in cells if "best_aic" in c]
    n_weibull_best = sum(1 for c in valid if c["weibull_is_best"])
    n_weibull_adequate = sum(1 for c in valid if c["weibull_delta_aic"] <= 2)
    r2_vals = [c["loglog_r2"] for c in valid if c.get("loglog_r2") is not None]

    return {
        "cells": cells,
        "summary": {
            "n_total": len(valid),
            "n_weibull_best_aic": n_weibull_best,
            "n_weibull_adequate": n_weibull_adequate,
            "mean_loglog_r2": round(float(np.mean(r2_vals)), 4) if r2_vals else None,
            "min_loglog_r2": round(float(np.min(r2_vals)), 4) if r2_vals else None,
            "interpretation": (
                f"Weibull is best AIC in {n_weibull_best}/{len(valid)} cells, "
                f"adequate (ΔAIC≤2) in {n_weibull_adequate}/{len(valid)} cells. "
                f"Log-log R²: mean={np.mean(r2_vals):.3f}, min={np.min(r2_vals):.3f}."
                if r2_vals else "No valid GOF results"
            ),
        },
    }


def bootstrap_median_survival(durations: np.ndarray, events: np.ndarray,
                               n_boot: int = 1000, seed: int = RANDOM_SEED) -> dict:
    """Bootstrap CI for median survival time."""
    rng = np.random.default_rng(seed)
    n = len(durations)
    medians = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_dur = durations[idx]
        boot_evt = events[idx]

        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(boot_dur, event_observed=boot_evt)
        med = kmf.median_survival_time_
        medians.append(med)

    medians = np.array(medians)
    finite = medians[np.isfinite(medians)]

    return {
        "median": float(np.median(medians)) if len(finite) > 0 else float("inf"),
        "ci_low": float(np.percentile(finite, 2.5)) if len(finite) > 0 else float("inf"),
        "ci_high": float(np.percentile(finite, 97.5)) if len(finite) > 0 else float("inf"),
        "pct_infinite": float((~np.isfinite(medians)).mean()),
    }
