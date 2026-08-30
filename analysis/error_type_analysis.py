"""
Error-type categorization and positional analysis.

Analyzes error patterns from the data to categorize errors and check
whether error types vary by step position.
"""

import ast
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, DATA_DIR, DATASETS, MODELS, RANDOM_SEED


def load_survival_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "survival_data.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "survival_data.csv"
    df = pd.read_csv(csv_path)
    df["error_vector"] = df["error_vector"].apply(ast.literal_eval)
    return df


def analyze_error_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze error patterns from the data. Since we don't have manual error type
    labels, we use structural properties of errors as proxy categories:

    1. Terminal errors: error at the very last step (conclusion/final answer)
    2. Cascade errors: error followed by more errors (propagating)
    3. Isolated errors: single error surrounded by correct steps (transient)
    4. Early errors: error in first third of chain
    5. Late errors: error in last third of chain
    """
    results = {}

    for ds in sorted(df["dataset"].unique()):
        ds_sub = df[(df["dataset"] == ds) & (df["event"] == 1)]
        if len(ds_sub) == 0:
            continue

        terminal_errors = 0
        cascade_errors = 0
        isolated_errors = 0
        early_first_errors = 0
        mid_first_errors = 0
        late_first_errors = 0

        error_density_by_position = {"early": [], "mid": [], "late": []}
        multi_error_patterns = {"cascade_then_recover": 0, "isolated_then_cascade": 0,
                                "pure_cascade": 0, "pure_isolated": 0}

        for _, row in ds_sub.iterrows():
            ev = row["error_vector"]
            n = len(ev)
            if n == 0:
                continue

            first_err_idx = next((i for i, e in enumerate(ev) if e == 1), None)
            if first_err_idx is None:
                continue

            relative_pos = first_err_idx / n

            if relative_pos < 1/3:
                early_first_errors += 1
            elif relative_pos < 2/3:
                mid_first_errors += 1
            else:
                late_first_errors += 1

            if ev[-1] == 1:
                terminal_errors += 1

            total_errors = sum(ev)
            if total_errors == 1:
                isolated_errors += 1
            elif total_errors > 1:
                has_cascade = any(ev[i] == 1 and i + 1 < n and ev[i + 1] == 1 for i in range(n - 1))
                has_recovery = any(ev[i] == 1 and i + 1 < n and ev[i + 1] == 0 for i in range(n - 1))

                if has_cascade and has_recovery:
                    multi_error_patterns["cascade_then_recover"] += 1
                elif has_cascade and not has_recovery:
                    multi_error_patterns["pure_cascade"] += 1
                elif has_recovery and not has_cascade:
                    multi_error_patterns["pure_isolated"] += 1

                if has_cascade:
                    cascade_errors += 1

            for i, e in enumerate(ev):
                rel = i / n
                if rel < 1/3:
                    error_density_by_position["early"].append(e)
                elif rel < 2/3:
                    error_density_by_position["mid"].append(e)
                else:
                    error_density_by_position["late"].append(e)

        n_total = len(ds_sub)
        early_density = np.mean(error_density_by_position["early"]) if error_density_by_position["early"] else 0
        mid_density = np.mean(error_density_by_position["mid"]) if error_density_by_position["mid"] else 0
        late_density = np.mean(error_density_by_position["late"]) if error_density_by_position["late"] else 0

        results[ds] = {
            "n_chains_with_errors": n_total,
            "first_error_position": {
                "early_third": early_first_errors,
                "mid_third": mid_first_errors,
                "late_third": late_first_errors,
                "early_pct": round(100 * early_first_errors / n_total, 1),
                "mid_pct": round(100 * mid_first_errors / n_total, 1),
                "late_pct": round(100 * late_first_errors / n_total, 1),
            },
            "error_types": {
                "terminal_errors": terminal_errors,
                "terminal_pct": round(100 * terminal_errors / n_total, 1),
                "cascade_errors": cascade_errors,
                "cascade_pct": round(100 * cascade_errors / n_total, 1),
                "isolated_errors": isolated_errors,
                "isolated_pct": round(100 * isolated_errors / n_total, 1),
            },
            "error_density_by_position": {
                "early": round(float(early_density), 4),
                "mid": round(float(mid_density), 4),
                "late": round(float(late_density), 4),
                "gradient": round(float(late_density - early_density), 4),
                "ratio_late_early": round(float(late_density / early_density), 2) if early_density > 0 else None,
            },
            "multi_error_patterns": multi_error_patterns,
        }

        if early_density > 0 and late_density > 0:
            early_arr = np.array(error_density_by_position["early"])
            late_arr = np.array(error_density_by_position["late"])
            try:
                t_stat, p_val = stats.ttest_ind(late_arr, early_arr, equal_var=False)
                results[ds]["density_test"] = {
                    "t_statistic": round(float(t_stat), 3),
                    "p_value": round(float(p_val), 6),
                    "significant": bool(p_val < 0.05),
                    "direction": "late > early" if t_stat > 0 else "early > late",
                }
            except Exception:
                pass

    return results


def analyze_error_type_vs_fan_in(results: dict) -> dict:
    """Check if error types support the fan-in hypothesis."""
    assessment = {}
    for ds, r in results.items():
        pos = r["first_error_position"]
        types = r["error_types"]
        density = r["error_density_by_position"]

        supports_fan_in = (
            pos["late_pct"] > pos["early_pct"] and
            density["late"] > density["early"]
        )

        assessment[ds] = {
            "supports_fan_in": supports_fan_in,
            "late_vs_early_first_error": f"{pos['late_pct']:.1f}% vs {pos['early_pct']:.1f}%",
            "density_gradient": density["gradient"],
            "cascade_rate": types["cascade_pct"],
            "interpretation": (
                f"{'Supports' if supports_fan_in else 'Does not support'} fan-in hypothesis: "
                f"errors are {'more' if supports_fan_in else 'not more'} concentrated at late positions "
                f"(density gradient: {density['gradient']:.4f}). "
                f"Cascade rate: {types['cascade_pct']:.1f}% — "
                f"{'consistent' if types['cascade_pct'] > 20 else 'not consistent'} with error propagation."
            ),
        }

    return assessment


def generate_latex_table(results: dict) -> str:
    """Generate a LaTeX table for the paper or supplementary."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Early\%} & \textbf{Late\%} & "
        r"\textbf{Terminal\%} & \textbf{Cascade\%} & \textbf{Isolated\%} \\",
        r"\midrule",
    ]

    for ds in ["proofwriter", "mathqa", "gsm8k", "bbh_tracking", "mmlu_pro"]:
        if ds not in results:
            continue
        r = results[ds]
        label = DATASETS.get(ds, {}).get("label", ds)
        pos = r["first_error_position"]
        types = r["error_types"]
        lines.append(
            f"  {label} & {pos['early_pct']:.0f} & {pos['late_pct']:.0f} & "
            f"{types['terminal_pct']:.0f} & {types['cascade_pct']:.0f} & "
            f"{types['isolated_pct']:.0f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Error pattern analysis. Early/Late\%: first errors in first/last third "
        r"of chain. Terminal: error at final step. Cascade: consecutive errors. "
        r"Isolated: single non-propagating errors. Late-concentrated errors and "
        r"high cascade rates support the compositional fan-in hypothesis.}",
        r"\label{tab:error_types}",
        r"\end{table}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("Loading survival data...")
    df = load_survival_data()

    print("\nAnalyzing error patterns...")
    results = analyze_error_patterns(df)

    for ds, r in results.items():
        label = DATASETS.get(ds, {}).get("label", ds)
        pos = r["first_error_position"]
        types = r["error_types"]
        density = r["error_density_by_position"]
        print(f"\n  {label}:")
        print(f"    First error position: early={pos['early_pct']:.1f}%, "
              f"mid={pos['mid_pct']:.1f}%, late={pos['late_pct']:.1f}%")
        print(f"    Error types: terminal={types['terminal_pct']:.1f}%, "
              f"cascade={types['cascade_pct']:.1f}%, isolated={types['isolated_pct']:.1f}%")
        print(f"    Error density: early={density['early']:.4f}, "
              f"mid={density['mid']:.4f}, late={density['late']:.4f}")

    print("\nFan-in assessment:")
    fan_in = analyze_error_type_vs_fan_in(results)
    for ds, a in fan_in.items():
        label = DATASETS.get(ds, {}).get("label", ds)
        print(f"  {label}: {'SUPPORTS' if a['supports_fan_in'] else 'DOES NOT SUPPORT'} fan-in")

    full_results = {"error_patterns": results, "fan_in_assessment": fan_in}

    with open(RESULTS_DIR / "error_type_analysis.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'error_type_analysis.json'}")

    table_tex = generate_latex_table(results)
    print("\nLaTeX table:\n")
    print(table_tex)
