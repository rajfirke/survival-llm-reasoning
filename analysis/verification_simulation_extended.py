"""
Extended verification experiment at multiple budget levels (K=1,2,3,5)
with paired significance tests.

Extends the existing verification_simulation.py to:
1. Run at K=1, 2, 3, 5 (not just K=1)
2. Add paired Wilcoxon signed-rank test: hazard-proportional vs late-weighted
3. Report p-values and effect sizes for all comparisons
4. Identify the crossover point where hazard-proportional matches late-weighted
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
from analysis.survival import discrete_hazard_rate


def load_survival_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "survival_data.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "survival_data.csv"
    df = pd.read_csv(csv_path)
    df["error_vector"] = df["error_vector"].apply(ast.literal_eval)
    return df


def run_extended_verification(df: pd.DataFrame, n_splits: int = 100,
                               train_frac: float = 0.7, seed: int = RANDOM_SEED) -> dict:
    """
    Extended verification experiment at K=1,2,3,5 with paired significance tests.
    """
    rng = np.random.default_rng(seed)
    K_values = [1, 2, 3, 5]
    all_results = {}

    for ds in sorted(df["dataset"].unique()):
        ds_sub = df[df["dataset"] == ds]
        ds_label = DATASETS.get(ds, {}).get("label", ds)
        print(f"  Processing {ds_label}...")

        k_results = {}
        for K in K_values:
            per_split_uniform = []
            per_split_late = []
            per_split_hazprop = []

            for model in ds_sub["model"].unique():
                model_sub = ds_sub[ds_sub["model"] == model]
                if len(model_sub) < 20:
                    continue

                for _ in range(n_splits):
                    indices = np.arange(len(model_sub))
                    rng.shuffle(indices)
                    split = int(len(indices) * train_frac)
                    train = model_sub.iloc[indices[:split]]
                    test = model_sub.iloc[indices[split:]]

                    h_df = discrete_hazard_rate(train["duration"].values, train["event"].values)
                    h_profile = h_df["hazard"].values

                    test_evecs = test["error_vector"].tolist()
                    erroneous = [ev for ev in test_evecs if sum(ev) > 0]
                    if len(erroneous) < 3:
                        continue

                    u_det, l_det, hp_det = 0, 0, 0
                    for ev in erroneous:
                        T = len(ev)
                        k = min(K, T)

                        u_checked = rng.choice(T, size=k, replace=False)
                        if any(ev[c] == 1 for c in u_checked):
                            u_det += 1

                        late_steps = list(range(max(0, T - k), T))
                        if any(ev[s] == 1 for s in late_steps):
                            l_det += 1

                        h = h_profile[:T] if len(h_profile) >= T else np.ones(T) / T
                        h_sum = h.sum()
                        p = h / h_sum if h_sum > 0 else np.ones(T) / T
                        try:
                            hp_checked = rng.choice(T, size=k, replace=False, p=p)
                            if any(ev[c] == 1 for c in hp_checked):
                                hp_det += 1
                        except ValueError:
                            hp_checked = rng.choice(T, size=k, replace=False)
                            if any(ev[c] == 1 for c in hp_checked):
                                hp_det += 1

                    n = len(erroneous)
                    per_split_uniform.append(u_det / n)
                    per_split_late.append(l_det / n)
                    per_split_hazprop.append(hp_det / n)

            if not per_split_uniform:
                continue

            u_arr = np.array(per_split_uniform)
            l_arr = np.array(per_split_late)
            hp_arr = np.array(per_split_hazprop)

            try:
                stat_hp_vs_u, p_hp_vs_u = stats.wilcoxon(hp_arr, u_arr, alternative="greater")
            except ValueError:
                stat_hp_vs_u, p_hp_vs_u = 0.0, 1.0

            try:
                stat_l_vs_u, p_l_vs_u = stats.wilcoxon(l_arr, u_arr, alternative="greater")
            except ValueError:
                stat_l_vs_u, p_l_vs_u = 0.0, 1.0

            try:
                stat_hp_vs_l, p_hp_vs_l = stats.wilcoxon(hp_arr, l_arr, alternative="greater")
            except ValueError:
                stat_hp_vs_l, p_hp_vs_l = 0.0, 1.0

            effect_hp_u = float(np.mean(hp_arr - u_arr) / (np.std(hp_arr - u_arr) + 1e-10))
            effect_l_u = float(np.mean(l_arr - u_arr) / (np.std(l_arr - u_arr) + 1e-10))
            effect_hp_l = float(np.mean(hp_arr - l_arr) / (np.std(hp_arr - l_arr) + 1e-10))

            k_results[K] = {
                "uniform_mean": round(float(u_arr.mean()), 4),
                "uniform_std": round(float(u_arr.std()), 4),
                "late_weighted_mean": round(float(l_arr.mean()), 4),
                "late_weighted_std": round(float(l_arr.std()), 4),
                "hazard_prop_mean": round(float(hp_arr.mean()), 4),
                "hazard_prop_std": round(float(hp_arr.std()), 4),
                "best_strategy": "late" if l_arr.mean() > hp_arr.mean() else "hazard_prop",
                "tests": {
                    "hazprop_vs_uniform": {
                        "wilcoxon_stat": round(float(stat_hp_vs_u), 2),
                        "p_value": round(float(p_hp_vs_u), 6),
                        "significant": bool(p_hp_vs_u < 0.05),
                        "effect_size_d": round(effect_hp_u, 3),
                    },
                    "late_vs_uniform": {
                        "wilcoxon_stat": round(float(stat_l_vs_u), 2),
                        "p_value": round(float(p_l_vs_u), 6),
                        "significant": bool(p_l_vs_u < 0.05),
                        "effect_size_d": round(effect_l_u, 3),
                    },
                    "hazprop_vs_late": {
                        "wilcoxon_stat": round(float(stat_hp_vs_l), 2),
                        "p_value": round(float(p_hp_vs_l), 6),
                        "significant": bool(p_hp_vs_l < 0.05),
                        "effect_size_d": round(effect_hp_l, 3),
                    },
                },
                "n_splits": len(per_split_uniform),
            }

        crossover_k = None
        for K in sorted(k_results.keys()):
            if k_results[K]["hazard_prop_mean"] >= k_results[K]["late_weighted_mean"]:
                crossover_k = K
                break

        all_results[ds] = {
            "k_results": k_results,
            "crossover_k": crossover_k,
            "interpretation": (
                f"Hazard-proportional matches or exceeds late-weighted at K={crossover_k}"
                if crossover_k else "Late-weighted dominates at all K values tested"
            ),
        }

    return all_results


def generate_latex_table(results: dict) -> str:
    """Generate extended Table 6 with all K values."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "ccc" * 4 + "}",
        r"\toprule",
    ]

    k_headers = []
    for K in [1, 2, 3, 5]:
        k_headers.extend([f"\\textbf{{Unif.}}", f"\\textbf{{Late}}", f"\\textbf{{Haz-P}}"])

    lines.append(r" & \multicolumn{3}{c}{$K{=}1$} & \multicolumn{3}{c}{$K{=}2$} & "
                 r"\multicolumn{3}{c}{$K{=}3$} & \multicolumn{3}{c}{$K{=}5$} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}")
    lines.append(r"\textbf{Dataset} & " + " & ".join(k_headers) + r" \\")
    lines.append(r"\midrule")

    for ds in ["proofwriter", "bbh_tracking", "mmlu_pro", "gsm8k", "mathqa"]:
        if ds not in results:
            continue
        label = DATASETS.get(ds, {}).get("label", ds)
        kr = results[ds]["k_results"]
        cells = [label]
        for K in [1, 2, 3, 5]:
            if K in kr:
                r = kr[K]
                u = f".{int(r['uniform_mean']*100):02d}" if r['uniform_mean'] < 1 else "1.0"
                l = f".{int(r['late_weighted_mean']*100):02d}" if r['late_weighted_mean'] < 1 else "1.0"
                hp = f".{int(r['hazard_prop_mean']*100):02d}" if r['hazard_prop_mean'] < 1 else "1.0"
                best = r["best_strategy"]
                if best == "late":
                    l = r"\textbf{" + l + "}"
                else:
                    hp = r"\textbf{" + hp + "}"
                cells.extend([u, l, hp])
            else:
                cells.extend(["--", "--", "--"])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Extended held-out error detection rate at $K=1,2,3,5$. "
        r"Bold: best strategy per dataset per $K$. On strongly aging tasks ($\rho > 1.5$), "
        r"late-weighted dominates at low $K$. At higher $K$, hazard-proportional becomes competitive "
        r"as it distributes verification effort across the full hazard profile.}",
        r"\label{tab:verification_extended}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("Loading survival data...")
    df = load_survival_data()

    print("\nRunning extended verification experiment (K=1,2,3,5)...")
    results = run_extended_verification(df, n_splits=100)

    print("\nResults Summary:")
    for ds, r in results.items():
        label = DATASETS.get(ds, {}).get("label", ds)
        print(f"\n  {label} (crossover K={r['crossover_k']}):")
        for K, kr in r["k_results"].items():
            print(f"    K={K}: uniform={kr['uniform_mean']:.3f}, "
                  f"late={kr['late_weighted_mean']:.3f}, "
                  f"hazprop={kr['hazard_prop_mean']:.3f} "
                  f"[best: {kr['best_strategy']}]")
            tests = kr["tests"]
            if tests["hazprop_vs_late"]["significant"]:
                print(f"      ** hazprop > late (p={tests['hazprop_vs_late']['p_value']:.4f})")

    serializable = {}
    for ds, r in results.items():
        serializable[ds] = {
            "k_results": r["k_results"],
            "crossover_k": r["crossover_k"],
            "interpretation": r["interpretation"],
        }

    with open(RESULTS_DIR / "verification_extended.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'verification_extended.json'}")

    table_tex = generate_latex_table(results)
    with open(RESULTS_DIR / "table6_extended.tex", "w") as f:
        f.write(table_tex)
    print(f"Saved: {RESULTS_DIR / 'table6_extended.tex'}")
