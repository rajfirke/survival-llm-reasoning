"""
Downstream validation: does survival-aware verification allocation beat uniform?

Simulates a verification budget (check K of T steps per chain) and compares:
  1. Uniform: check K random steps
  2. Late-weighted: check the last K steps (survival-aware)
  3. Hazard-proportional: check steps proportional to estimated h(t)

Metric: error detection rate = fraction of erroneous chains where at least
one checked step contains an error.

Uses existing ProofWriter error_vectors — no new model inference needed.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, RESULTS_DIR, MODELS
from analysis.load_data import load_all_survival_data
from analysis.survival import discrete_hazard_rate


def simulate_verification(error_vectors: list[list[int]],
                          hazard_profile: np.ndarray = None,
                          n_repeats: int = 500,
                          seed: int = 42) -> dict:
    """
    Simulate verification under three strategies for budget K = 1..max_steps-1.

    Returns dict with detection rates per strategy per budget.
    """
    rng = np.random.default_rng(seed)

    erroneous = [ev for ev in error_vectors if sum(ev) > 0]
    if not erroneous:
        return {}

    max_steps = max(len(ev) for ev in erroneous)
    results = {"budget": [], "uniform": [], "late_weighted": [], "hazard_proportional": []}

    for K in range(1, max_steps):
        detected_uniform = 0
        detected_late = 0
        detected_hazard = 0
        total = 0

        for ev in erroneous:
            T = len(ev)
            if K >= T:
                detected_uniform += 1
                detected_late += 1
                detected_hazard += 1
                total += 1
                continue

            total += 1

            uniform_detections = 0
            for _ in range(n_repeats):
                checked = rng.choice(T, size=K, replace=False)
                if any(ev[c] == 1 for c in checked):
                    uniform_detections += 1
            detected_uniform += uniform_detections / n_repeats

            late_steps = list(range(T - K, T))
            if any(ev[s] == 1 for s in late_steps):
                detected_late += 1

            if hazard_profile is not None and len(hazard_profile) >= T:
                h = hazard_profile[:T]
                h_norm = h / h.sum() if h.sum() > 0 else np.ones(T) / T
                try:
                    checked_h = rng.choice(T, size=K, replace=False, p=h_norm)
                    if any(ev[c] == 1 for c in checked_h):
                        detected_hazard += 1
                except ValueError:
                    detected_hazard += detected_uniform / total if total > 0 else 0
            else:
                detected_hazard += detected_uniform / total if total > 0 else 0

        if total > 0:
            results["budget"].append(K)
            results["uniform"].append(round(detected_uniform / total, 4))
            results["late_weighted"].append(round(detected_late / total, 4))
            results["hazard_proportional"].append(round(detected_hazard / total, 4))

    return results


def run_verification_simulation(df: pd.DataFrame, dataset: str = "proofwriter") -> dict:
    """Run the full verification simulation on a specific dataset."""
    sub = df[df["dataset"] == dataset]
    all_results = {}

    for model in sub["model"].unique():
        model_sub = sub[sub["model"] == model]
        error_vectors = model_sub["error_vector"].tolist()

        h_df = discrete_hazard_rate(
            model_sub["duration"].values, model_sub["event"].values
        )
        h_profile = h_df["hazard"].values

        sim = simulate_verification(error_vectors, h_profile)
        if sim:
            all_results[model] = sim

    return all_results


def figure6_verification_sim(sim_results: dict, save: bool = True):
    """Plot verification simulation: detection rate vs budget for 3 strategies."""
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.3)

    n_models = len(sim_results)
    if n_models == 0:
        return None

    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(5 * min(n_models, 4), 4.5),
                              sharey=True)
    if min(n_models, 4) == 1:
        axes = [axes]

    model_keys = list(sim_results.keys())[:4]
    for ax, model_key in zip(axes, model_keys):
        sim = sim_results[model_key]
        budgets = sim["budget"]

        ax.plot(budgets, sim["uniform"], label="Uniform", color="#888888",
                linewidth=2, linestyle="--")
        ax.plot(budgets, sim["late_weighted"], label="Late-weighted",
                color="#c0392b", linewidth=2.5)
        ax.plot(budgets, sim["hazard_proportional"], label="Hazard-prop.",
                color="#2980b9", linewidth=2, linestyle="-.")

        model_label = MODELS.get(model_key, {}).get("label", model_key)
        ax.set_title(model_label, fontweight="bold")
        ax.set_xlabel("Verification Budget $K$")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Error Detection Rate")
    fig.suptitle("Verification Allocation: Survival-Aware vs. Uniform (ProofWriter)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig6_verification.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def run_verification_train_test(df: pd.DataFrame, datasets: list[str] = None,
                                n_splits: int = 100, train_frac: float = 0.7,
                                seed: int = 42) -> dict:
    """
    Per-model train/test verification with hazard-proportional allocation.
    Splits within each dataset x model pair to avoid cross-model variance.
    """
    from analysis.survival import discrete_hazard_rate

    if datasets is None:
        datasets = df["dataset"].unique().tolist()

    rng = np.random.default_rng(seed)
    all_results = {}

    for ds in datasets:
        ds_sub = df[df["dataset"] == ds]
        ds_uniform = []
        ds_late = []
        ds_hazprop = []

        for model in ds_sub["model"].unique():
            model_sub = ds_sub[ds_sub["model"] == model]
            erroneous_mask = model_sub["error_vector"].apply(lambda ev: sum(ev) > 0)
            if erroneous_mask.sum() < 5:
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
                if not erroneous:
                    continue

                K = 1
                u_det, l_det, hp_det = 0, 0, 0

                for ev in erroneous:
                    T = len(ev)
                    if T <= K:
                        u_det += 1; l_det += 1; hp_det += 1
                        continue

                    if any(ev[c] == 1 for c in rng.choice(T, size=K, replace=False)):
                        u_det += 1

                    if any(ev[s] == 1 for s in range(T - K, T)):
                        l_det += 1

                    h = h_profile[:T] if len(h_profile) >= T else np.ones(T) / T
                    h_sum = h.sum()
                    if h_sum > 0:
                        p = h / h_sum
                    else:
                        p = np.ones(T) / T
                    try:
                        checked = rng.choice(T, size=K, replace=False, p=p)
                        if any(ev[c] == 1 for c in checked):
                            hp_det += 1
                    except ValueError:
                        if any(ev[c] == 1 for c in rng.choice(T, size=K, replace=False)):
                            hp_det += 1

                n = len(erroneous)
                ds_uniform.append(u_det / n)
                ds_late.append(l_det / n)
                ds_hazprop.append(hp_det / n)

        if ds_uniform:
            all_results[ds] = {
                "uniform_mean": round(float(np.mean(ds_uniform)), 4),
                "uniform_std": round(float(np.std(ds_uniform)), 4),
                "late_weighted_mean": round(float(np.mean(ds_late)), 4),
                "late_weighted_std": round(float(np.std(ds_late)), 4),
                "hazard_prop_mean": round(float(np.mean(ds_hazprop)), 4),
                "hazard_prop_std": round(float(np.std(ds_hazprop)), 4),
                "advantage_late": round(float(np.mean(ds_late) - np.mean(ds_uniform)), 4),
                "advantage_hazprop": round(float(np.mean(ds_hazprop) - np.mean(ds_uniform)), 4),
                "n_splits_total": len(ds_uniform),
            }

    return all_results


if __name__ == "__main__":
    df = load_all_survival_data()
    print("Running verification simulation on ProofWriter...")
    results = run_verification_simulation(df, "proofwriter")

    for model, sim in results.items():
        budgets = sim["budget"]
        mid = len(budgets) // 2
        if mid < len(budgets):
            print(f"  {model}: at K={budgets[mid]}, "
                  f"uniform={sim['uniform'][mid]:.2f}, "
                  f"late={sim['late_weighted'][mid]:.2f}, "
                  f"delta={sim['late_weighted'][mid] - sim['uniform'][mid]:+.2f}")

    with open(RESULTS_DIR / "verification_simulation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    figure6_verification_sim(results)
    print("Done. Figure saved to figures/fig6_verification.pdf")
