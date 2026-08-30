"""
Shared frailty modelling for unobserved problem-level heterogeneity.

The frailty confound: harder problems produce longer chains, so late steps
are enriched for hard problems, creating APPARENT aging even with constant
true hazard.

This module implements three complementary approaches and reports which
matters:

  Approach C (primary): Variance-components / ICC decomposition
    - For each problem_id with >=3 models, fit a within-problem Weibull
    - Decompose variance into between-problem vs within-problem
    - ICC quantifies how much problem identity matters
    - Within-problem Weibull directly controls for problem difficulty

  Approach A (secondary): CoxPH with problem-level fixed effects
    - Approximate frailty via penalized CoxPH with problem dummies
    - Extract frailty variance from coefficient dispersion

  EM-based Gamma frailty (Approach B fallback):
    - Fit Weibull + Gamma(1/theta, 1/theta) frailty via EM
    - Reports theta (frailty variance) per dataset
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize
from lifelines import WeibullFitter, CoxPHFitter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_SEED, ALPHA, RESULTS_DIR, DATA_DIR


def _load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "survival_data.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "survival_data.csv"
    df = pd.read_csv(csv_path)
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)
    df["n_steps"] = df["n_steps"].astype(int)
    return df


def _fit_weibull_safe(durations, events, min_events=3):
    """Fit Weibull; return (rho, lambda) or None."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=float)
    if events.sum() < min_events or len(durations) < 5:
        return None
    durations = np.maximum(durations, 0.5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter()
            wf.fit(durations, events)
            rho = float(wf.rho_)
            if rho > 50 or rho < 0.01:
                return None
            return {"rho": rho, "lambda": float(wf.lambda_)}
    except Exception:
        return None


def variance_components_analysis(df: pd.DataFrame,
                                  min_models_per_problem: int = 3) -> dict:
    """
    For each dataset:
      1. Identify problems with >= min_models_per_problem model runs
      2. Fit per-problem Weibull using runs from different models
      3. Decompose: Var(rho) = Var(between-problem) + Var(within-problem)
      4. Compute ICC
      5. Report within-problem rho (frailty-controlled)
    """
    results_by_ds = {}

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]

        prob_model_counts = ds_df.groupby("problem_id")["model"].nunique()
        eligible_problems = prob_model_counts[
            prob_model_counts >= min_models_per_problem
        ].index.tolist()

        if len(eligible_problems) < 5:
            results_by_ds[ds] = {
                "error": f"Only {len(eligible_problems)} problems with "
                         f">={min_models_per_problem} models; skipping",
                "n_eligible_problems": len(eligible_problems),
            }
            continue

        problem_rhos = {}
        problem_level_data = []

        for pid in eligible_problems:
            prob_df = ds_df[ds_df["problem_id"] == pid]
            fit = _fit_weibull_safe(
                prob_df["duration"].values, prob_df["event"].values,
                min_events=2,
            )
            if fit is not None:
                problem_rhos[pid] = fit["rho"]
                problem_level_data.append({
                    "problem_id": pid,
                    "rho": fit["rho"],
                    "lambda": fit["lambda"],
                    "n_runs": len(prob_df),
                    "n_events": int(prob_df["event"].sum()),
                    "mean_n_steps": float(prob_df["n_steps"].mean()),
                })

        if len(problem_level_data) < 5:
            results_by_ds[ds] = {
                "error": "Too few valid per-problem Weibull fits",
                "n_eligible_problems": len(eligible_problems),
                "n_valid_fits": len(problem_level_data),
            }
            continue

        rho_values = np.array([d["rho"] for d in problem_level_data])

        # ICC via one-way ANOVA approach
        # ICC(1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
        groups = []
        for pid in eligible_problems:
            prob_df = ds_df[ds_df["problem_id"] == pid]
            if prob_df["event"].sum() >= 1:
                groups.append(prob_df["duration"].values.astype(float))

        if len(groups) >= 5:
            all_vals = np.concatenate(groups)
            grand_mean = all_vals.mean()
            n_groups = len(groups)
            group_sizes = np.array([len(g) for g in groups])
            k_bar = group_sizes.mean()

            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2 for g in groups
            )
            ss_within = sum(
                np.sum((g - g.mean()) ** 2) for g in groups
            )

            df_between = n_groups - 1
            df_within = len(all_vals) - n_groups

            ms_between = ss_between / df_between if df_between > 0 else 0
            ms_within = ss_within / df_within if df_within > 0 else 1e-10

            icc = (
                (ms_between - ms_within)
                / (ms_between + (k_bar - 1) * ms_within)
            )
            icc = float(np.clip(icc, 0, 1))

            f_stat = ms_between / ms_within if ms_within > 0 else 0
            p_icc = 1 - stats.f.cdf(f_stat, df_between, df_within)
        else:
            icc = None
            p_icc = None
            f_stat = None

        var_between = float(np.var(rho_values, ddof=1))
        mean_rho = float(np.mean(rho_values))
        median_rho = float(np.median(rho_values))

        pooled_fit = _fit_weibull_safe(
            ds_df["duration"].values, ds_df["event"].values
        )
        pooled_rho = pooled_fit["rho"] if pooled_fit else None

        frailty_controlled_rho = mean_rho

        # theta = var_between / mean_rho^2 (coefficient of variation squared)
        theta = var_between / (mean_rho ** 2) if mean_rho > 0 else 0

        results_by_ds[ds] = {
            "n_eligible_problems": len(eligible_problems),
            "n_valid_fits": len(problem_level_data),
            "pooled_rho": round(pooled_rho, 4) if pooled_rho else None,
            "within_problem_mean_rho": round(mean_rho, 4),
            "within_problem_median_rho": round(median_rho, 4),
            "var_between_problem_rho": round(var_between, 4),
            "frailty_variance_theta": round(theta, 4),
            "theta_interpretation": (
                "negligible" if theta < 0.1 else
                "low" if theta < 0.5 else
                "moderate" if theta < 1.0 else
                "high"
            ),
            "icc_duration": round(icc, 4) if icc is not None else None,
            "icc_f_stat": round(float(f_stat), 2) if f_stat is not None else None,
            "icc_p_value": round(float(p_icc), 6) if p_icc is not None else None,
            "icc_significant": bool(p_icc < ALPHA) if p_icc is not None else None,
            "aging_after_frailty": mean_rho > 1.0,
            "aging_after_frailty_strong": (
                mean_rho > 1.0 and
                (median_rho > 1.0) and
                (stats.ttest_1samp(rho_values, 1.0).pvalue / 2 < ALPHA
                 if len(rho_values) >= 5 else False)
            ),
            "ttest_rho_gt_1": {
                "t_stat": round(float(stats.ttest_1samp(rho_values, 1.0).statistic), 3),
                "p_value_one_sided": round(
                    float(stats.ttest_1samp(rho_values, 1.0).pvalue / 2), 6
                ),
            } if len(rho_values) >= 5 else None,
        }

    return results_by_ds


def coxph_problem_effects(df: pd.DataFrame, max_problems: int = 200) -> dict:
    """
    For each dataset, fit CoxPH with problem_id dummies.
    The dispersion of problem coefficients proxies frailty variance.
    Uses penalized CoxPH (L2) when problem count is large.
    """
    results_by_ds = {}

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds].copy()

        prob_counts = ds_df.groupby("problem_id").size()
        valid_pids = prob_counts[prob_counts >= 3].index
        ds_df = ds_df[ds_df["problem_id"].isin(valid_pids)].copy()

        if len(ds_df) < 20 or ds_df["event"].sum() < 10:
            results_by_ds[ds] = {"error": "insufficient data"}
            continue

        unique_pids = ds_df["problem_id"].unique()
        if len(unique_pids) > max_problems:
            rng = np.random.default_rng(RANDOM_SEED)
            sampled_pids = rng.choice(unique_pids, size=max_problems, replace=False)
            ds_df = ds_df[ds_df["problem_id"].isin(sampled_pids)].copy()

        prob_dummies = pd.get_dummies(
            ds_df["problem_id"], prefix="pid", drop_first=True
        ).astype(np.float64)

        cox_df = pd.concat([
            ds_df[["duration", "event"]].reset_index(drop=True),
            pd.DataFrame({"log_n_steps": np.log1p(ds_df["n_steps"].values)}),
            prob_dummies.reset_index(drop=True),
        ], axis=1)

        cox_df = cox_df.apply(pd.to_numeric, errors="coerce").dropna()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(
                    cox_df, duration_col="duration", event_col="event",
                    show_progress=False,
                )

            pid_cols = [c for c in cph.params_.index if c.startswith("pid_")]
            if pid_cols:
                pid_coefs = cph.params_[pid_cols].values
                frailty_var = float(np.var(pid_coefs))
                mean_coef = float(np.mean(pid_coefs))
            else:
                frailty_var = 0.0
                mean_coef = 0.0

            log_nsteps_coef = float(cph.params_.get("log_n_steps", 0))
            log_nsteps_p = float(
                cph.summary.loc["log_n_steps", "p"]
            ) if "log_n_steps" in cph.summary.index else None

            results_by_ds[ds] = {
                "n_chains": len(cox_df),
                "n_problems": len(ds_df["problem_id"].unique()),
                "concordance": round(cph.concordance_index_, 4),
                "frailty_variance_proxy": round(frailty_var, 4),
                "mean_problem_coef": round(mean_coef, 4),
                "log_nsteps_coef": round(log_nsteps_coef, 4),
                "log_nsteps_p": round(log_nsteps_p, 6) if log_nsteps_p is not None else None,
                "aic": round(cph.AIC_partial_, 1),
            }

        except Exception as e:
            results_by_ds[ds] = {"error": str(e)}

    return results_by_ds


def _weibull_loglik_with_frailty(params, durations, events, z_i):
    """
    Log-likelihood of Weibull with known frailty z_i.
    h(t|z) = z * rho/lambda * (t/lambda)^(rho-1)
    S(t|z) = exp(-z * (t/lambda)^rho)
    """
    log_rho, log_lambda = params
    rho = np.exp(log_rho)
    lam = np.exp(log_lambda)

    t = np.maximum(durations, 0.01)

    cum_haz = z_i * (t / lam) ** rho

    ll = 0.0
    ll += np.sum(events * (np.log(z_i + 1e-30) + np.log(rho + 1e-30) -
                           np.log(lam + 1e-30) + (rho - 1) * np.log(t / lam + 1e-30)))
    ll -= np.sum(cum_haz)

    return -ll


def em_gamma_frailty(df: pd.DataFrame, max_iter: int = 50, tol: float = 1e-4) -> dict:
    """
    EM algorithm for Weibull + Gamma(1/theta, 1/theta) frailty.

    For each dataset:
      E-step: compute posterior mean of z_i given current Weibull params and theta
      M-step: update Weibull params given z_i; update theta from z_i variance
    """
    results_by_ds = {}

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]

        problems = ds_df.groupby("problem_id").agg({
            "duration": list,
            "event": list,
            "n_steps": "first",
        }).reset_index()

        problems = problems[problems["duration"].apply(len) >= 2]
        if len(problems) < 10:
            results_by_ds[ds] = {"error": "too few problems with >=2 obs"}
            continue

        all_dur = ds_df["duration"].values.astype(float)
        all_evt = ds_df["event"].values.astype(int)
        init_fit = _fit_weibull_safe(all_dur, all_evt)
        if init_fit is None:
            results_by_ds[ds] = {"error": "initial Weibull fit failed"}
            continue

        rho = init_fit["rho"]
        lam = init_fit["lambda"]
        theta = 0.5

        converged = False
        for iteration in range(max_iter):
            # E-step: posterior mean of z_i
            # z_i | data ~ Gamma(a_i, b_i)
            # a_i = 1/theta + d_i, b_i = 1/theta + sum_j (t_ij / lambda)^rho
            z_vals = []
            for _, row in problems.iterrows():
                durs = np.maximum(np.array(row["duration"], dtype=float), 0.01)
                evts = np.array(row["event"], dtype=int)
                d_i = evts.sum()
                cum_haz_sum = np.sum((durs / lam) ** rho)

                a_i = 1.0 / (theta + 1e-10) + d_i
                b_i = 1.0 / (theta + 1e-10) + cum_haz_sum

                z_i = a_i / (b_i + 1e-10)
                z_vals.append(max(z_i, 0.01))

            z_arr = np.array(z_vals)

            # M-step: update rho, lambda, theta
            theta_new = float(np.var(z_arr, ddof=1))
            theta_new = max(theta_new, 1e-6)

            z_obs = []
            for idx, (_, row) in enumerate(problems.iterrows()):
                z_obs.extend([z_vals[idx]] * len(row["duration"]))
            z_obs = np.array(z_obs)

            dur_all = []
            evt_all = []
            for _, row in problems.iterrows():
                dur_all.extend(row["duration"])
                evt_all.extend(row["event"])
            dur_all = np.maximum(np.array(dur_all, dtype=float), 0.01)
            evt_all = np.array(evt_all, dtype=int)

            try:
                res = optimize.minimize(
                    _weibull_loglik_with_frailty,
                    x0=[np.log(rho), np.log(lam)],
                    args=(dur_all, evt_all, z_obs),
                    method="Nelder-Mead",
                    options={"maxiter": 500},
                )
                rho_new = np.exp(res.x[0])
                lam_new = np.exp(res.x[1])
            except Exception:
                rho_new, lam_new = rho, lam

            delta = abs(rho_new - rho) + abs(theta_new - theta)
            rho = rho_new
            lam = lam_new
            theta = theta_new

            if delta < tol:
                converged = True
                break

        results_by_ds[ds] = {
            "rho_marginal": round(init_fit["rho"], 4),
            "rho_frailty": round(rho, 4),
            "lambda_frailty": round(lam, 4),
            "theta": round(theta, 4),
            "theta_interpretation": (
                "negligible" if theta < 0.1 else
                "low" if theta < 0.5 else
                "moderate" if theta < 1.0 else
                "high"
            ),
            "converged": converged,
            "n_iterations": iteration + 1,
            "n_problems": len(problems),
            "rho_change_pct": round(
                (rho - init_fit["rho"]) / init_fit["rho"] * 100, 2
            ),
            "aging_after_frailty": rho > 1.0,
        }

    return results_by_ds


def per_cell_frailty_rho(df: pd.DataFrame,
                          min_models_per_problem: int = 3) -> dict:
    """
    For each dataset x model cell, compute a frailty-adjusted rho using
    the within-problem approach:
      1. For each problem with >= min_models_per_problem runs,
         extract the model-specific observation
      2. Fit Weibull within each problem (across models)
      3. The frailty-adjusted rho = average per-problem rho

    Also compute a simpler "matched" approach:
      - For each problem, compute problem-level rho from all models
      - Then weight each cell's contribution by 1/problem_rho
        (downweight easy problems, upweight hard ones)
    """
    cell_results = {}

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]

        prob_model_counts = ds_df.groupby("problem_id")["model"].nunique()
        eligible_pids = prob_model_counts[
            prob_model_counts >= min_models_per_problem
        ].index.tolist()

        if len(eligible_pids) < 5:
            eligible_pids = ds_df["problem_id"].unique().tolist()

        problem_rhos = {}
        for pid in eligible_pids:
            pdata = ds_df[ds_df["problem_id"] == pid]
            fit = _fit_weibull_safe(pdata["duration"].values, pdata["event"].values,
                                    min_events=2)
            if fit is not None:
                problem_rhos[pid] = fit["rho"]

        for model in sorted(df["model"].unique()):
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            if len(sub) < 10 or sub["event"].sum() < 3:
                continue

            std_fit = _fit_weibull_safe(sub["duration"].values, sub["event"].values)
            if std_fit is None:
                continue

            model_pids = set(sub["problem_id"].unique()) & set(problem_rhos.keys())
            if len(model_pids) >= 5:
                adj_rhos = [problem_rhos[pid] for pid in model_pids]
                frailty_rho = float(np.mean(adj_rhos))
            else:
                all_prob_rhos = list(problem_rhos.values())
                frailty_rho = float(np.mean(all_prob_rhos)) if all_prob_rhos else std_fit["rho"]

            cell_results[(ds, model)] = {
                "rho_standard": round(std_fit["rho"], 4),
                "rho_frailty": round(frailty_rho, 4),
                "rho_change_pct": round(
                    (frailty_rho - std_fit["rho"]) / std_fit["rho"] * 100, 2
                ),
                "n_problems_used": len(model_pids) if len(model_pids) >= 5 else len(problem_rhos),
            }

    return cell_results


def run_all() -> dict:
    """Run all frailty analyses and save results."""
    print("Loading data...")
    df = _load_data()
    print(f"  {len(df)} chains, {len(df['problem_id'].unique())} unique problems")

    print("\n1. Variance-components / ICC analysis (Approach C - primary)...")
    vc = variance_components_analysis(df)
    for ds, r in vc.items():
        if "error" not in r:
            print(f"  {ds}: within-problem rho = {r['within_problem_mean_rho']:.3f}, "
                  f"ICC = {r['icc_duration']}, theta = {r['frailty_variance_theta']:.3f} "
                  f"({r['theta_interpretation']})")
        else:
            print(f"  {ds}: {r['error']}")

    print("\n2. CoxPH with problem effects (Approach A - secondary)...")
    cox = coxph_problem_effects(df)
    for ds, r in cox.items():
        if "error" not in r:
            print(f"  {ds}: frailty_var_proxy = {r['frailty_variance_proxy']:.4f}, "
                  f"concordance = {r['concordance']:.3f}")
        else:
            print(f"  {ds}: {r['error']}")

    print("\n3. EM Gamma frailty (Approach B - validation)...")
    em = em_gamma_frailty(df)
    for ds, r in em.items():
        if "error" not in r:
            print(f"  {ds}: rho_marginal = {r['rho_marginal']:.3f}, "
                  f"rho_frailty = {r['rho_frailty']:.3f}, "
                  f"theta = {r['theta']:.4f} ({r['theta_interpretation']}), "
                  f"converged = {r['converged']}")
        else:
            print(f"  {ds}: {r['error']}")

    print("\n4. Per-cell frailty-adjusted rho...")
    cell_frailty = per_cell_frailty_rho(df)
    n_aging = sum(1 for v in cell_frailty.values() if v["rho_frailty"] > 1.0)
    print(f"  {n_aging}/{len(cell_frailty)} cells retain rho > 1 after frailty adjustment")

    results = {
        "variance_components": vc,
        "coxph_problem_effects": cox,
        "em_gamma_frailty": em,
        "per_cell_frailty": {
            f"{k[0]}_{k[1]}": v for k, v in cell_frailty.items()
        },
        "summary": {
            "n_cells_total": len(cell_frailty),
            "n_aging_after_frailty": n_aging,
            "pct_aging_after_frailty": round(100 * n_aging / len(cell_frailty), 1)
            if cell_frailty else 0,
            "dataset_ranking_after_frailty": _dataset_ranking(vc),
        },
    }

    out_path = RESULTS_DIR / "shared_frailty_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return results


def _dataset_ranking(vc: dict) -> list:
    """Rank datasets by frailty-adjusted rho (descending)."""
    ranking = []
    for ds, r in vc.items():
        if "error" not in r and r.get("within_problem_mean_rho") is not None:
            ranking.append({
                "dataset": ds,
                "rho_frailty": r["within_problem_mean_rho"],
            })
    ranking.sort(key=lambda x: x["rho_frailty"], reverse=True)
    return ranking


if __name__ == "__main__":
    run_all()
