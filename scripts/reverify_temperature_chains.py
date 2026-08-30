#!/usr/bin/env python3
"""
Re-verify temperature ablation chains using the REAL ProofWriter verification
pipeline (verify_chain), NOT the broken heuristic fallback.

DOES NOT MODIFY the original checkpoint files. Writes re-verified results
to a new file: results/temperature_reverified.json

Usage:
    python scripts/reverify_temperature_chains.py
"""

import json
import sys
import re
import warnings
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_CHECKPOINT_DIR = _RESULTS_DIR / "temperature_checkpoints"

# The UNIFIED_PIPELINE directory must be provided as an environment variable
# or placed as a sibling to this repository.
import os
_PIPELINE_DIR = Path(os.environ.get(
    "UNIFIED_PIPELINE_DIR",
    str(_PROJECT_ROOT.parent / "UNIFIED_PIPELINE"),
))

sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import proofwriter_pipeline as pw


def load_proofwriter_problems(n_problems: int, seed: int = 42) -> list:
    """Load the SAME 200 problems used by the temperature ablation script."""
    problems = pw.load_proofwriter(n_problems, min_depth=3, max_depth=5)
    rng = np.random.default_rng(seed)
    if len(problems) > n_problems:
        indices = rng.choice(len(problems), size=n_problems, replace=False)
        indices.sort()
        problems = [problems[i] for i in indices]
    return problems


def extract_model_answer(response: str) -> str:
    """Extract True/False/Unknown from a model response."""
    lower = response.lower()
    last_500 = lower[-500:] if len(lower) > 500 else lower

    for pattern in [
        r'\*\*(?:final\s+)?answer[:\s]*\*?\*?\s*(true|false|unknown)',
        r'(?:final\s+)?answer[:\s]+\*?\*?\s*(true|false|unknown)',
        r'the\s+statement.*?is\s+\*?\*?(true|false|unknown)',
        r'(?:conclusion|verdict|result)[:\s]+\*?\*?\s*(true|false|unknown)',
    ]:
        m = re.search(pattern, last_500, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    for word in ["True", "False", "Unknown"]:
        if word.lower() in last_500:
            return word

    return ""


def reverify_one_chain(chain: dict, problem: dict) -> dict:
    """
    Re-verify a single chain using pw.verify_chain().

    Returns a NEW dict with corrected error_vector, first_error, n_errors.
    Does NOT modify the input chain dict.
    """
    steps = chain.get("steps", [])
    response = chain.get("response", "")

    if not steps:
        return {
            "error_vector": [],
            "first_error": None,
            "n_errors": 0,
            "final_correct": False,
            "verify_method": "no_steps",
        }

    theory = problem.get("theory", "")
    question = problem.get("question", "")
    answer = problem.get("answer", "")
    all_proofs = problem.get("all_proofs", "")
    model_answer = extract_model_answer(response)

    try:
        error_vector, metadata = pw.verify_chain(
            steps=steps,
            theory_text=theory,
            question_text=question,
            answer=answer,
            model_answer=model_answer,
            all_proofs=all_proofs,
        )
    except Exception as e:
        return {
            "error_vector": [0] * len(steps),
            "first_error": None,
            "n_errors": 0,
            "final_correct": False,
            "verify_method": f"verify_chain_error: {str(e)[:100]}",
        }

    first_error = None
    for i, e in enumerate(error_vector):
        if e == 1:
            first_error = i + 1
            break

    final_correct = False
    if model_answer and answer:
        final_correct = model_answer.strip().lower() == answer.strip().lower()

    return {
        "error_vector": error_vector,
        "first_error": first_error,
        "n_errors": sum(error_vector),
        "final_correct": final_correct,
        "model_answer": model_answer,
        "verify_method": "verify_chain",
        "metadata": {
            k: v for k, v in metadata.items()
            if isinstance(v, (int, float, str, bool))
        } if isinstance(metadata, dict) else {},
    }


def fit_weibull(reverified: list, label: str = "") -> dict:
    """Fit Weibull to re-verified chains."""
    from lifelines import WeibullFitter

    valid = [r for r in reverified if r["n_steps"] > 0]
    if len(valid) < 10:
        return {"label": label, "n": len(valid), "error": "too few chains"}

    durations = []
    events = []
    for r in valid:
        fe = r["first_error"]
        ns = r["n_steps"]
        if fe is not None and fe > 0:
            durations.append(float(fe))
            events.append(1.0)
        else:
            durations.append(float(ns))
            events.append(0.0)

    durations = np.maximum(np.array(durations), 0.5)
    events = np.array(events)

    n_events = int(events.sum())
    if n_events < 3:
        return {
            "label": label, "n": len(valid), "n_events": n_events,
            "error": "too few events",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = WeibullFitter()
            wf.fit(durations, events)

        rho = float(wf.rho_)
        lam = float(wf.lambda_)
        summary = wf.summary
        rho_row = summary.loc["rho_"]
        cl = [c for c in summary.columns if "lower" in c.lower()]
        ch = [c for c in summary.columns if "upper" in c.lower()]
        ci_low = float(rho_row[cl[0]]) if cl else rho * 0.9
        ci_high = float(rho_row[ch[0]]) if ch else rho * 1.1

        if ci_low > 1.0:
            status = "AGING"
        elif ci_high < 1.0:
            status = "LINDY"
        else:
            status = "CONSTANT (CI includes 1)"

        return {
            "label": label,
            "n": len(valid),
            "n_events": n_events,
            "censored": len(valid) - n_events,
            "event_rate": round(n_events / len(valid), 3),
            "rho": round(rho, 4),
            "rho_ci_low": round(ci_low, 4),
            "rho_ci_high": round(ci_high, 4),
            "lambda": round(lam, 4),
            "status": status,
        }
    except Exception as e:
        return {"label": label, "n": len(valid), "error": str(e)}


def main():
    print("=" * 70)
    print("  RE-VERIFY TEMPERATURE ABLATION CHAINS")
    print("  Using real ProofWriter verify_chain() pipeline")
    print("  Original checkpoints will NOT be modified")
    print("=" * 70)
    print()

    # Load the same 200 ProofWriter problems
    print("Loading ProofWriter problems (same 200 as ablation)...")
    problems = load_proofwriter_problems(200)
    print(f"  Loaded {len(problems)} problems")
    print(f"  Sample: theory={problems[0]['theory'][:60]}...")
    print(f"          question={problems[0]['question']}")
    print(f"          answer={problems[0]['answer']}")
    print()

    # Build problem lookup by index
    problem_map = {}
    for i, p in enumerate(problems):
        problem_map[f"p{i}"] = p

    # Find checkpoint files
    checkpoints = sorted(_CHECKPOINT_DIR.glob("temp_*_proofwriter_checkpoint.json"))
    if not checkpoints:
        print("ERROR: No checkpoint files found in", _CHECKPOINT_DIR)
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint files:")
    for cp in checkpoints:
        with open(cp) as f:
            n = len(json.load(f))
        print(f"  {cp.name}: {n} chains")
    print()

    all_results = {}

    for cp_path in checkpoints:
        # Parse temperature from filename: temp_0.3_proofwriter_checkpoint.json
        temp_str = cp_path.name.split("_")[1]
        temperature = float(temp_str)
        label = f"ProofWriter T={temperature}"

        print(f"--- {label} ---")

        with open(cp_path) as f:
            chains = json.load(f)
        print(f"  Loaded {len(chains)} chains")

        reverified_chains = []
        errors_found = 0
        verify_errors = 0

        for i, chain in enumerate(chains):
            pid = chain.get("problem_id", f"p{i // 5}")
            problem = problem_map.get(pid)

            if problem is None:
                verify_errors += 1
                reverified_chains.append({
                    **{k: chain[k] for k in ["problem_id", "chain_idx", "n_steps", "temperature"]},
                    "error_vector": chain.get("error_vector", []),
                    "first_error": chain.get("first_error"),
                    "n_errors": chain.get("n_errors", 0),
                    "final_correct": chain.get("final_correct", False),
                    "verify_method": "problem_not_found",
                })
                continue

            result = reverify_one_chain(chain, problem)

            reverified_chains.append({
                "problem_id": pid,
                "chain_idx": chain.get("chain_idx", 0),
                "n_steps": chain.get("n_steps", 0),
                "temperature": temperature,
                **result,
            })

            if result["n_errors"] > 0:
                errors_found += 1

        print(f"  Re-verified: {len(reverified_chains)} chains")
        print(f"  Chains with errors: {errors_found} ({errors_found/len(reverified_chains):.1%})")
        print(f"  Verification failures: {verify_errors}")

        # Error position analysis
        fe_positions = [r["first_error"] for r in reverified_chains if r["first_error"] is not None]
        if fe_positions:
            n_steps_err = [r["n_steps"] for r in reverified_chains if r["first_error"] is not None]
            relative = [fe / ns for fe, ns in zip(fe_positions, n_steps_err) if ns > 0]
            early = sum(1 for r in relative if r <= 0.33)
            mid = sum(1 for r in relative if 0.33 < r <= 0.66)
            late = sum(1 for r in relative if r > 0.66)
            print(f"  First error position: early={early}, mid={mid}, late={late}")
            print(f"  Mean first error step: {np.mean(fe_positions):.1f}")

        # Fit Weibull
        weibull = fit_weibull(reverified_chains, label)
        print(f"  Weibull: ", end="")
        if "rho" in weibull:
            print(f"rho={weibull['rho']:.3f} [{weibull['rho_ci_low']:.3f}, {weibull['rho_ci_high']:.3f}] "
                  f"=> {weibull['status']}")
        else:
            print(f"ERROR: {weibull.get('error', 'unknown')}")

        all_results[f"T={temperature}"] = {
            "temperature": temperature,
            "n_chains": len(reverified_chains),
            "n_errors": errors_found,
            "error_rate": round(errors_found / len(reverified_chains), 3),
            "weibull": weibull,
            "chains": reverified_chains,
        }
        print()

    # Add baseline comparison from main paper
    print("--- Baseline (T=0.0, from main paper) ---")
    print("  Qwen2.5-14B × ProofWriter: rho=2.697 [2.456, 2.939] => AGING")
    print("  (500 chains, 255 events)")
    print()

    # Summary comparison
    print("=" * 70)
    print("  SUMMARY: Does aging persist under sampling?")
    print("=" * 70)
    print()
    print(f"  {'Condition':<25} {'rho':>8} {'95% CI':>22} {'Status':>12} {'Events':>8}")
    print(f"  {'-'*75}")
    print(f"  {'T=0.0 (baseline)':25} {'2.697':>8} {'[2.456, 2.939]':>22} {'AGING':>12} {'255':>8}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        w = r["weibull"]
        if "rho" in w:
            print(f"  {key:<25} {w['rho']:8.3f} "
                  f"{'[' + str(w['rho_ci_low']) + ', ' + str(w['rho_ci_high']) + ']':>22} "
                  f"{w['status']:>12} {w['n_events']:>8}")
        else:
            print(f"  {key:<25} {'--':>8} {'--':>22} {w.get('error','?')[:12]:>12} {'--':>8}")

    print()

    # Save results (WITHOUT modifying checkpoints)
    output = {
        "experiment": "temperature_reverification",
        "dataset": "proofwriter",
        "model": "qwen2.5:14b",
        "baseline_rho": 2.697,
        "baseline_ci": [2.456, 2.939],
        "conditions": {
            k: {kk: vv for kk, vv in v.items() if kk != "chains"}
            for k, v in all_results.items()
        },
        "chain_details": {
            k: v["chains"] for k, v in all_results.items()
        },
    }

    output_path = _RESULTS_DIR / "temperature_reverified.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {output_path}")
    print(f"  Original checkpoints: UNTOUCHED")
    print()


if __name__ == "__main__":
    main()
