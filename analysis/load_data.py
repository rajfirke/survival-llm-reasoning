"""
Load pre-processed step-level survival data from CSV.

Each row represents one reasoning chain with:
  - duration: step at which first error occurs (or chain length if censored)
  - event: 1 if error occurred, 0 if chain completed correctly (right-censored)
"""

import ast
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, DATASETS, MODELS


def load_all_survival_data() -> pd.DataFrame:
    """Load the pre-processed survival dataset."""
    csv_path = DATA_DIR / "survival_data.csv"
    df = pd.read_csv(csv_path)
    df["error_vector"] = df["error_vector"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)
    df["n_steps"] = df["n_steps"].astype(int)
    print(f"Loaded {len(df)} chains: "
          f"{df['dataset'].nunique()} datasets, {df['model'].nunique()} models")
    print(f"Events (errors): {df['event'].sum()} / {len(df)} "
          f"({df['event'].mean():.1%})")
    return df


def compute_step_level_hazard(df: pd.DataFrame, dataset: str, model: str) -> pd.DataFrame:
    """
    Compute the discrete hazard rate h(t) = P(error at t | survived to t).
    """
    sub = df[(df["dataset"] == dataset) & (df["model"] == model)].copy()
    max_steps = sub["n_steps"].max()

    rows = []
    for t in range(1, max_steps + 1):
        at_risk = len(sub[(sub["duration"] >= t)])
        events = len(sub[(sub["event"] == 1) & (sub["duration"] == t)])
        hazard = events / at_risk if at_risk > 0 else 0.0

        rows.append({
            "step": t,
            "at_risk": at_risk,
            "events": events,
            "hazard": round(hazard, 4),
            "dataset": dataset,
            "model": model,
        })

    result = pd.DataFrame(rows)
    cum_survival = np.cumprod(1 - result["hazard"].values)
    result["survival"] = np.round(cum_survival, 4)
    return result


if __name__ == "__main__":
    df = load_all_survival_data()
    print("\nSample:")
    print(df.head(10).to_string(index=False))

    print("\n\nSummary by dataset x model:")
    summary = df.groupby(["dataset", "model"]).agg(
        n_chains=("event", "count"),
        n_events=("event", "sum"),
        event_rate=("event", "mean"),
        median_duration=("duration", "median"),
    ).round(3)
    print(summary.to_string())
