"""
Publication-quality figures for the Survival Analysis paper.

Figure 1: Kaplan-Meier survival curves (THE main figure)
Figure 2: Hazard rate plots showing aging vs Lindy
Figure 3: Simpson's Paradox — raw error rate vs hazard rate
Figure 4: Cox PH forest plot (controlling for difficulty)
Figure 5: Half-life comparison across models/datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, DATASETS, MODELS

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def figure1_km_curves(km_data: dict, save: bool = True):
    """
    Kaplan-Meier survival curves: one panel per dataset, lines per model.

    Uses a 2-row x 3-column grid (3 top + 2 bottom, last cell hidden)
    for print readability instead of cramming all panels in one row.
    """
    datasets = list(km_data.keys())
    n_ds = len(datasets)

    n_cols = 3
    n_rows = 2
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                                  sharey=True)
    axes = axes_grid.flatten()

    for idx in range(n_ds, n_rows * n_cols):
        axes[idx].set_visible(False)

    for idx, ds_key in enumerate(datasets):
        ax = axes[idx]
        ds_cfg = DATASETS[ds_key]
        for model_key, km_df in km_data[ds_key].items():
            model_cfg = MODELS[model_key]
            ax.step(km_df["step"], km_df["survival"],
                    where="post", label=model_cfg["label"],
                    color=model_cfg["color"], linewidth=2.5)
            ax.fill_between(km_df["step"], km_df["ci_low"], km_df["ci_high"],
                            alpha=0.12, step="post", color=model_cfg["color"])

        ax.set_xlabel("Reasoning Step $t$")
        ax.set_title(ds_cfg["label"], fontweight="bold", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.legend(loc="lower left")

    axes[0].set_ylabel("Survival Probability $S(t)$")
    if n_ds > n_cols:
        axes[n_cols].set_ylabel("Survival Probability $S(t)$")

    fig.suptitle("Kaplan-Meier Survival Curves of Reasoning Chains",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig1_km_curves.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def _smooth_hazard(values, window=5, polyorder=2):
    """Apply Savitzky-Golay smoothing to hazard values if enough points exist."""
    n = len(values)
    if n < window:
        return values
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return values
    smoothed = savgol_filter(values, window_length=w, polyorder=min(polyorder, w - 1))
    return np.maximum(smoothed, 0.0)


def figure2_hazard_rates(hazard_data: dict, min_at_risk: int = 20, save: bool = True):
    """
    Hazard rate h(t) vs step: reveals aging (increasing) vs Lindy (decreasing).

    Uses a 2-row x 3-column grid and Savitzky-Golay smoothing to reduce noise.
    Raw data is shown as small translucent markers behind the smoothed line.
    """
    datasets = list(hazard_data.keys())
    n_ds = len(datasets)

    n_cols = 3
    n_rows = 2
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                                  sharey=False)
    axes = axes_grid.flatten()

    for idx in range(n_ds, n_rows * n_cols):
        axes[idx].set_visible(False)

    for idx, ds_key in enumerate(datasets):
        ax = axes[idx]
        ds_cfg = DATASETS[ds_key]
        for model_key, h_df in hazard_data[ds_key].items():
            model_cfg = MODELS[model_key]
            filtered = h_df[h_df["at_risk"] >= min_at_risk]
            if len(filtered) == 0:
                continue

            steps = filtered["step"].values
            raw_h = filtered["hazard"].values

            ax.scatter(steps, raw_h, s=20, alpha=0.35,
                       color=model_cfg["color"], zorder=2)

            smooth_h = _smooth_hazard(raw_h)
            ax.plot(steps, smooth_h,
                    marker="o", markersize=5, label=model_cfg["label"],
                    color=model_cfg["color"], linewidth=2.0, zorder=3)

            if "hazard_se" in filtered.columns:
                se = filtered["hazard_se"].fillna(0)
                ax.fill_between(steps,
                                smooth_h - 1.96 * se.values,
                                smooth_h + 1.96 * se.values,
                                alpha=0.10, color=model_cfg["color"])

        ax.set_xlabel("Reasoning Step $t$")
        ax.set_title(ds_cfg["label"], fontweight="bold", fontsize=14)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.legend()

    axes[0].set_ylabel("Hazard Rate $h(t)$")
    if n_ds > n_cols:
        axes[n_cols].set_ylabel("Hazard Rate $h(t)$")

    fig.suptitle("Hazard Rate: Conditional Probability of Error",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig2_hazard_rates.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def figure3_simpsons_paradox(hazard_data: dict, error_rate_data: dict,
                              dataset: str, model: str, save: bool = True):
    """
    The Simpson's Paradox figure: raw error rate DECREASES while hazard INCREASES.

    This is the key conceptual insight of the paper. The dataset/model pair
    is selected automatically by run_analysis.py to show the most dramatic
    divergence (largest positive hazard slope minus error-rate slope).
    """
    h_df = hazard_data[dataset][model]
    e_df = error_rate_data[dataset][model]
    min_risk = 20

    h_filtered = h_df[h_df["at_risk"] >= min_risk]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    color_hazard = "#c0392b"
    color_error = "#2980b9"

    h_steps = h_filtered["step"].values
    h_vals = h_filtered["hazard"].values
    ax1.plot(h_steps, h_vals,
             marker="o", markersize=7, color=color_hazard,
             linewidth=2.5, label="Hazard $h(t)$ (conditional)", zorder=3)
    ax1.set_xlabel("Reasoning Step $t$")
    ax1.set_ylabel("Hazard Rate $h(t)$", color=color_hazard, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=color_hazard)

    ax2 = ax1.twinx()
    e_filtered = e_df[e_df["step"].isin(h_filtered["step"])]
    e_steps = e_filtered["step"].values
    e_vals = e_filtered["error_rate"].values
    ax2.plot(e_steps, e_vals,
             marker="s", markersize=7, color=color_error,
             linewidth=2.5, linestyle="--", label="Raw Error Rate $P(E_t)$", zorder=3)
    ax2.set_ylabel("Raw Error Rate", color=color_error, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=color_error)

    if len(h_steps) >= 3:
        h_slope = np.polyfit(range(len(h_vals)), h_vals, 1)
        h_trend = np.polyval(h_slope, range(len(h_vals)))
        ax1.plot(h_steps, h_trend, color=color_hazard, alpha=0.4,
                 linewidth=1.5, linestyle=":", zorder=2)

    if len(e_steps) >= 3:
        e_slope = np.polyfit(range(len(e_vals)), e_vals, 1)
        e_trend = np.polyval(e_slope, range(len(e_vals)))
        ax2.plot(e_steps, e_trend, color=color_error, alpha=0.4,
                 linewidth=1.5, linestyle=":", zorder=2)

    mid_idx = len(h_steps) // 2
    ax1.annotate(
        "Divergence:\nhazard rises while\nerror rate falls",
        xy=(h_steps[mid_idx], h_vals[mid_idx]),
        xytext=(h_steps[-1] * 0.75, h_vals.min() + (h_vals.max() - h_vals.min()) * 0.15),
        fontsize=10, fontstyle="italic", color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    ds_label = DATASETS[dataset]["label"]
    model_label = MODELS[model]["label"]
    ax1.set_title(f"Simpson's Paradox of Survival — {ds_label} ({model_label})",
                  fontweight="bold", fontsize=14)

    plt.tight_layout()

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig3_simpsons_paradox.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def figure4_half_life_comparison(half_lives: pd.DataFrame, save: bool = True):
    """
    Bar chart comparing median survival time (half-life) across models and datasets.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    datasets = half_lives["dataset"].unique()
    models = half_lives["model"].unique()
    x = np.arange(len(datasets))
    width = 0.25

    for i, model_key in enumerate(models):
        model_data = half_lives[half_lives["model"] == model_key]
        vals = []
        for ds in datasets:
            row = model_data[model_data["dataset"] == ds]
            val = row["half_life"].values[0] if len(row) > 0 else 0
            if np.isinf(val):
                val = model_data["half_life"].replace(np.inf, np.nan).max() * 1.2
            vals.append(val)

        model_cfg = MODELS.get(model_key, {"label": model_key, "color": "#888"})
        bars = ax.bar(x + i * width, vals, width,
                      label=model_cfg["label"], color=model_cfg["color"],
                      alpha=0.85, edgecolor="black", linewidth=0.5)

        for bar, v, ds in zip(bars, vals, datasets):
            orig = half_lives[(half_lives["model"] == model_key) & (half_lives["dataset"] == ds)]
            if len(orig) > 0 and np.isinf(orig["half_life"].values[0]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        "inf", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASETS[ds]["label"] for ds in datasets], rotation=20, ha="right")
    ax.set_ylabel("Median Survival (steps)")
    ax.set_title("Half-Life of Reasoning Chains", fontweight="bold", fontsize=15)
    ax.legend(framealpha=0.9, edgecolor="#cccccc")
    plt.tight_layout()

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig4_half_life.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def figure7_weibull_heatmap(weibull_results: dict, save: bool = True):
    """
    Heatmap of Weibull shape parameter rho across datasets × models.
    Red = aging (rho > 1), blue = Lindy (rho < 1), white = constant (rho = 1).
    """
    cells = [c for c in weibull_results["cells"] if "rho" in c]
    if not cells:
        return None

    ds_order = [k for k in DATASETS.keys() if any(c["dataset"] == k for c in cells)]
    model_order = [k for k in MODELS.keys() if any(c["model"] == k for c in cells)]

    matrix = np.full((len(ds_order), len(model_order)), np.nan)
    sig_matrix = np.full((len(ds_order), len(model_order)), False)

    for c in cells:
        if c["dataset"] in ds_order and c["model"] in model_order:
            i = ds_order.index(c["dataset"])
            j = model_order.index(c["model"])
            matrix[i, j] = c["rho"]
            sig_matrix[i, j] = c.get("rho_ci_low", 0) > 1

    fig, ax = plt.subplots(figsize=(10, 5))

    vmin = min(0.7, np.nanmin(matrix) - 0.1)
    vmax = max(3.0, np.nanmax(matrix) + 0.1)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            star = "*" if sig_matrix[i, j] else ""
            text_color = "white" if val > 2.2 or val < 0.8 else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([MODELS[m]["label"] for m in model_order],
                       rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels([DATASETS[d]["label"] for d in ds_order], fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Weibull Shape $\\rho$ ($>$1 = Aging)", fontsize=12)
    cbar.ax.axhline(y=1.0, color="black", linewidth=2)

    n_aging = int(np.sum(sig_matrix))
    n_total = int(np.sum(~np.isnan(matrix)))
    ax.set_title(f"Weibull Shape Parameter: {n_aging}/{n_total} Cells Show Significant Aging",
                 fontweight="bold", fontsize=14)

    plt.tight_layout()
    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig7_weibull_heatmap.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig


def figure5_cox_forest_plot(cox_results: dict, save: bool = True):
    """Forest plot of Cox PH coefficients."""
    summary = cox_results["summary"]
    coefs = pd.DataFrame(summary)

    names = list(coefs.index)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(names) * 0.7 + 1.5)))

    exp_coefs = [np.exp(coefs.loc[n, "coef"]) for n in names]
    ci_low = [np.exp(coefs.loc[n, "coef lower 95%"]) for n in names]
    ci_high = [np.exp(coefs.loc[n, "coef upper 95%"]) for n in names]

    y_pos = np.arange(len(names))

    ax.errorbar(exp_coefs, y_pos,
                xerr=[np.array(exp_coefs) - np.array(ci_low),
                      np.array(ci_high) - np.array(exp_coefs)],
                fmt="o", color="#1d3557", capsize=5, markersize=8,
                linewidth=1.5, capthick=1.2)

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)

    display_names = []
    for n in names:
        n = n.replace("model_", "").replace("ds_", "").replace("log_n_steps", "Difficulty (log steps)")
        display_names.append(n)
    ax.set_yticklabels(display_names)

    ax.set_xlabel("Hazard Ratio (exp($\\beta$))")
    ax.set_title("Cox Proportional Hazards: Covariate Effects",
                 fontweight="bold", fontsize=15)
    plt.tight_layout()

    if save:
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig5_cox_forest.{ext}",
                        bbox_inches="tight", dpi=300)
    return fig
