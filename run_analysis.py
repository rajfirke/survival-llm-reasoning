"""
Main analysis script for survival analysis of step-level reasoning errors.

Loads all existing data, computes survival statistics,
generates all paper figures, and prints a summary report.

Usage:
    python run_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASETS, MODELS, FIGURES_DIR, RESULTS_DIR
from analysis.load_data import load_all_survival_data, compute_step_level_hazard
from analysis.survival import (
    kaplan_meier, discrete_hazard_rate, raw_error_rate,
    cox_ph_analysis, cox_ph_with_assumption_test,
    test_increasing_hazard, compute_half_life,
    apply_fdr_correction, test_aging_clustering,
    bootstrap_hazard_ratio, weibull_per_cell, weibull_gof_test,
)
from analysis.figures import (
    figure1_km_curves, figure2_hazard_rates, figure3_simpsons_paradox,
    figure4_half_life_comparison, figure5_cox_forest_plot,
    figure7_weibull_heatmap,
)


def main():
    print("=" * 70)
    print("Survival Analysis of Step-Level Error Hazard in LLM Chains")
    print("=" * 70)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_all_survival_data()
    df.to_csv(RESULTS_DIR / "survival_data.csv", index=False)

    # ── Step 2: Kaplan-Meier curves ────────────────────────────────────
    print("\n[2/6] Computing Kaplan-Meier survival curves...")
    km_data = {}
    for ds_key in DATASETS:
        km_data[ds_key] = {}
        for model_key in MODELS:
            sub = df[(df["dataset"] == ds_key) & (df["model"] == model_key)]
            if len(sub) < 10:
                continue
            km_df = kaplan_meier(sub["duration"].values, sub["event"].values)
            km_data[ds_key][model_key] = km_df

    print("  Generating Figure 1: KM curves...")
    figure1_km_curves(km_data)

    # ── Step 3: Hazard rates ───────────────────────────────────────────
    print("\n[3/6] Computing hazard rates...")
    hazard_data = {}
    hazard_tests = []
    for ds_key in DATASETS:
        hazard_data[ds_key] = {}
        for model_key in MODELS:
            sub = df[(df["dataset"] == ds_key) & (df["model"] == model_key)]
            if len(sub) < 10:
                continue
            h_df = discrete_hazard_rate(sub["duration"].values, sub["event"].values)
            hazard_data[ds_key][model_key] = h_df

            trend = test_increasing_hazard(h_df)
            trend["dataset"] = ds_key
            trend["model"] = model_key
            hazard_tests.append(trend)

            ds_label = DATASETS[ds_key]["label"]
            model_label = MODELS[model_key]["label"]
            direction = trend.get("direction", "?")
            tau = trend.get("tau", "?")
            p = trend.get("p_value", "?")
            print(f"  {ds_label:15s} x {model_label:15s}: "
                  f"trend={direction}, tau={tau}, p={p}")

    print("  Generating Figure 2: Hazard rates...")
    figure2_hazard_rates(hazard_data)

    # Apply BH-FDR correction
    print("\n  Applying BH-FDR correction...")
    hazard_tests = apply_fdr_correction(hazard_tests)
    sig_bh = [t for t in hazard_tests if t.get("significant_bh") and t.get("direction") == "increasing"]
    print(f"  Significant aging after BH: {len(sig_bh)}/{len(hazard_tests)}")
    for t in sig_bh:
        print(f"    {t['dataset']} x {t['model']}: p_raw={t['p_value']}, p_BH={t['p_corrected_bh']}")

    # Test clustering of aging results
    print("\n  Testing aging clustering (Fisher's exact)...")
    clustering = test_aging_clustering(hazard_tests, list(DATASETS.keys()))
    print(f"  Clustering p-value: {clustering['p_value']} ({'significant' if clustering['significant'] else 'not significant'})")

    # Bootstrap CIs on hazard ratios for significant pairs
    print("\n  Bootstrap CIs on h_late/h_early...")
    for t in hazard_tests:
        if t.get("significant_bh") and t.get("direction") == "increasing":
            sub = df[(df["dataset"] == t["dataset"]) & (df["model"] == t["model"])]
            ci = bootstrap_hazard_ratio(sub["duration"].values, sub["event"].values)
            t["hazard_ratio_ci"] = ci
            print(f"    {t['dataset']} x {t['model']}: {ci['median']}x [{ci['ci_low']}, {ci['ci_high']}]")

    with open(RESULTS_DIR / "hazard_trend_tests.json", "w") as f:
        json.dump(hazard_tests, f, indent=2, default=str)
    with open(RESULTS_DIR / "aging_clustering_test.json", "w") as f:
        json.dump(clustering, f, indent=2, default=str)

    # ── Step 3b: Weibull analysis on all 35 cells ──────────────────────
    print("\n[3b/7] Running Weibull analysis on all dataset x model cells...")
    weibull_results = weibull_per_cell(df)
    n_wb_aging = weibull_results["binomial_test"]["n_aging_sig"]
    n_wb_total = weibull_results["binomial_test"]["n_total"]
    binom_p = weibull_results["binomial_test"]["p_value"]
    print(f"  Weibull aging (rho>1, CI excludes 1): {n_wb_aging}/{n_wb_total} (binomial p={binom_p})")
    for ds, s in weibull_results["dataset_summary"].items():
        ds_label = DATASETS[ds]["label"]
        print(f"    {ds_label:15s}: mean_rho={s['mean_rho']:.3f}, aging={s['n_aging']}/{s['n_total']}")

    with open(RESULTS_DIR / "weibull_all_cells.json", "w") as f:
        json.dump(weibull_results, f, indent=2, default=str)

    print("  Generating Figure 7: Weibull heatmap...")
    figure7_weibull_heatmap(weibull_results)

    print("\n  Running Weibull goodness-of-fit tests (vs LogNormal, LogLogistic)...")
    gof_results = weibull_gof_test(df)
    gof_summary = gof_results["summary"]
    print(f"  Weibull best AIC: {gof_summary['n_weibull_best_aic']}/{gof_summary['n_total']}")
    print(f"  Weibull adequate (dAIC<=2): {gof_summary['n_weibull_adequate']}/{gof_summary['n_total']}")
    print(f"  Log-log R2: mean={gof_summary.get('mean_loglog_r2', 'N/A')}, min={gof_summary.get('min_loglog_r2', 'N/A')}")
    with open(RESULTS_DIR / "weibull_gof.json", "w") as f:
        json.dump(gof_results, f, indent=2, default=str)

    # ── Step 4: Simpson's Paradox ──────────────────────────────────────
    print("\n[4/6] Computing Simpson's Paradox overlay...")
    error_rate_data = {}
    for ds_key in DATASETS:
        error_rate_data[ds_key] = {}
        for model_key in MODELS:
            sub = df[(df["dataset"] == ds_key) & (df["model"] == model_key)]
            if len(sub) < 10:
                continue
            evecs = sub["error_vector"].tolist()
            e_df = raw_error_rate(evecs)
            error_rate_data[ds_key][model_key] = e_df

    best_simpson = None
    best_score = -1
    for ds_key in hazard_data:
        for model_key in hazard_data[ds_key]:
            h_df = hazard_data[ds_key][model_key]
            e_df = error_rate_data.get(ds_key, {}).get(model_key)
            if e_df is None:
                continue
            h_filtered = h_df[h_df["at_risk"] >= 20]
            if len(h_filtered) < 3:
                continue
            h_vals = h_filtered["hazard"].values
            e_merged = e_df[e_df["step"].isin(h_filtered["step"])]
            if len(e_merged) < 3:
                continue
            e_vals = e_merged["error_rate"].values
            h_trend = np.polyfit(range(len(h_vals)), h_vals, 1)[0]
            e_trend = np.polyfit(range(len(e_vals)), e_vals, 1)[0]
            score = h_trend - e_trend
            if score > best_score:
                best_score = score
                best_simpson = (ds_key, model_key)

    if best_simpson:
        ds_key, model_key = best_simpson
        print(f"  Best Simpson's Paradox: {DATASETS[ds_key]['label']} x {MODELS[model_key]['label']}")
        figure3_simpsons_paradox(hazard_data, error_rate_data, ds_key, model_key)
    else:
        print("  WARNING: No clear Simpson's Paradox found")

    # ── Step 5: Half-lives & Cox PH ────────────────────────────────────
    print("\n[5/6] Computing half-lives and Cox PH...")
    half_life_rows = []
    for ds_key in km_data:
        for model_key in km_data[ds_key]:
            km_df = km_data[ds_key][model_key]
            hl = compute_half_life(km_df)
            half_life_rows.append({
                "dataset": ds_key,
                "model": model_key,
                "half_life": hl,
                "dataset_label": DATASETS[ds_key]["label"],
                "model_label": MODELS[model_key]["label"],
            })

    half_lives = pd.DataFrame(half_life_rows)
    print("\n  Half-lives:")
    for _, row in half_lives.iterrows():
        hl_str = "inf" if np.isinf(row["half_life"]) else f"{row['half_life']:.0f}"
        print(f"    {row['dataset_label']:15s} x {row['model_label']:15s}: {hl_str} steps")

    figure4_half_life_comparison(half_lives)
    half_lives.to_csv(RESULTS_DIR / "half_lives.csv", index=False)

    print("\n  Running Cox PH model (with PH assumption test)...")
    try:
        cox_full = cox_ph_with_assumption_test(df)
        cox = cox_full["standard"]
        print(f"  Concordance: {cox['concordance']:.3f}")
        print(f"  Coefficients:")
        for k, v in cox["coefficients"].items():
            p = cox["p_values"].get(k, "?")
            print(f"    {k}: b={v:.3f}, HR={np.exp(v):.2f}, p={p:.4f}")

        ph_test = cox_full["ph_assumption_test"]
        print(f"\n  PH Assumption Test:")
        print(f"    Passed: {ph_test.get('passed', '?')}")
        if ph_test.get("output"):
            for line in ph_test["output"].strip().split("\n")[:5]:
                print(f"    {line}")

        strat = cox_full["stratified_by_dataset"]
        if strat and "concordance" in strat:
            print(f"\n  Stratified Cox (by dataset) concordance: {strat['concordance']:.3f}")

        figure5_cox_forest_plot(cox)

        cox_save = {k: v for k, v in cox.items() if k != "model_object"}
        cox_save["ph_assumption_test"] = ph_test
        cox_save["stratified_by_dataset"] = strat
        with open(RESULTS_DIR / "cox_ph_results.json", "w") as f:
            json.dump(cox_save, f, indent=2, default=str)
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 6: Downstream validation (verification simulation) ──────
    print("\n[6/7] Running verification simulation...")
    try:
        from analysis.verification_simulation import (
            run_verification_simulation, figure6_verification_sim,
            run_verification_train_test,
        )
        sim_results = run_verification_simulation(df, "proofwriter")
        if sim_results:
            for model, sim in sim_results.items():
                budgets = sim["budget"]
                mid = len(budgets) // 2
                if mid < len(budgets):
                    delta = sim["late_weighted"][mid] - sim["uniform"][mid]
                    print(f"  {model}: at K={budgets[mid]}, "
                          f"uniform={sim['uniform'][mid]:.2f}, "
                          f"late={sim['late_weighted'][mid]:.2f}, "
                          f"advantage={delta:+.2f}")

            with open(RESULTS_DIR / "verification_simulation.json", "w") as f:
                json.dump(sim_results, f, indent=2, default=str)
            figure6_verification_sim(sim_results)
        else:
            print("  No erroneous chains found for simulation")

        print("\n  Running train/test verification (per-model, held-out)...")
        tt_results = run_verification_train_test(df)
        print(f"  Train/test results (K=1, per-model 100 splits):")
        for ds, r in tt_results.items():
            ds_label = DATASETS.get(ds, {}).get("label", ds)
            print(f"    {ds_label:15s}: uniform={r['uniform_mean']:.3f}+/-{r['uniform_std']:.3f}, "
                  f"late={r['late_weighted_mean']:.3f}+/-{r['late_weighted_std']:.3f}, "
                  f"hazard-prop={r['hazard_prop_mean']:.3f}+/-{r['hazard_prop_std']:.3f}, "
                  f"d_late={r['advantage_late']:+.3f}, d_haz={r['advantage_hazprop']:+.3f}")
        with open(RESULTS_DIR / "verification_train_test.json", "w") as f:
            json.dump(tt_results, f, indent=2, default=str)

    except Exception as e:
        print(f"  Verification simulation failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 7: Summary report ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    print("\n-- Hazard Trend Tests --")
    trend_df = pd.DataFrame(hazard_tests)
    if "direction" in trend_df.columns:
        increasing = trend_df[
            (trend_df["direction"] == "increasing") & (trend_df["significant"] == True)
        ]
        print(f"  Significantly increasing hazard: {len(increasing)} / {len(trend_df)}")
        for _, row in increasing.iterrows():
            print(f"    {row['dataset']} x {row['model']}: tau={row['tau']}, p={row['p_value']}")

    print("\n-- Key Finding --")
    n_aging = len(increasing) if "direction" in trend_df.columns else 0
    n_total = len(trend_df)
    if n_aging > n_total * 0.3:
        print(f"  STRONG SIGNAL: {n_aging}/{n_total} dataset x model pairs show aging")
        print("  Near-universal aging thesis is supported.")
    elif n_aging > 0:
        print(f"  PARTIAL SIGNAL: {n_aging}/{n_total} pairs show aging")
        print("  Task-dependent and model-dependent aging patterns.")
    else:
        print("  WEAK SIGNAL: No significant aging detected")
        print("  Consider Lindy effect or constant hazard framing.")

    print(f"\n-- Figures saved to {FIGURES_DIR} --")
    for p in sorted(FIGURES_DIR.glob("fig*")):
        print(f"  {p.name}")

    print(f"\n-- Results saved to {RESULTS_DIR} --")
    for p in sorted(RESULTS_DIR.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
