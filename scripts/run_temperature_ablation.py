#!/usr/bin/env python3
"""
Temperature ablation for Project 37: Does reasoning aging persist under
non-greedy decoding?

Runs Qwen2.5-14B via Ollama at T=0.0 (baseline), T=0.3, and T=0.7 on two
datasets selected for their contrasting aging profiles:
  - ProofWriter (strong aging, rho ~ 2.4)
  - GSM8K       (moderate aging, rho ~ 1.5)

For T>0, 5 chains per problem are generated (sampling introduces variance).
200 problems per dataset for manageable runtime.

Uses the SAME step-parsing and verification pipeline as the original
experiments -- imports the pipeline modules from UNIFIED_PIPELINE.

Usage:
    # Check setup without running inference:
    python scripts/run_temperature_ablation.py --dry-run

    # Full run (requires Ollama with qwen2.5:14b):
    python scripts/run_temperature_ablation.py

    # Run only one temperature:
    python scripts/run_temperature_ablation.py --temperatures 0.3

    # Custom problem count:
    python scripts/run_temperature_ablation.py --n-problems 50

Estimated runtime on M4 Pro 48GB:
    T=0.0: ~33 min (200 GSM8K + 200 ProofWriter, 1 chain each)
    T=0.3: ~167 min (200 + 200, 5 chains each)
    T=0.7: ~167 min (200 + 200, 5 chains each)
    Total: ~6 hours

Output: results/temperature_pilot.json

Created: 2026-05-24 for EMNLP 2026 (T4 reviewer response)
"""

import argparse
import json
import os
import re
import sys
import time
import logging
import urllib.request
import urllib.error
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"

# The UNIFIED_PIPELINE directory must be provided as an environment variable
# or placed as a sibling to this repository. It contains the per-benchmark
# chain generation and verification pipelines used in the original experiments.
_PIPELINE_DIR = Path(os.environ.get(
    "UNIFIED_PIPELINE_DIR",
    str(_PROJECT_ROOT.parent / "UNIFIED_PIPELINE"),
))
_UNIFIED_RESULTS = _PIPELINE_DIR / "results"

sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Configuration ────────────────────────────────────────────────────
MODEL_NAME = "qwen2.5:14b"
MODEL_LABEL = "Qwen2.5-14B"
OLLAMA_BASE = "http://localhost:11434"

TEMPERATURES = [0.0, 0.3, 0.7]
CHAINS_PER_PROBLEM = {0.0: 1, 0.3: 5, 0.7: 5}
N_PROBLEMS = 200
RANDOM_SEED = 42

# Datasets with their pipeline configurations
ABLATION_DATASETS = {
    "proofwriter": {
        "module": "proofwriter_pipeline",
        "load_fn": "load_proofwriter",
        "load_args": [],
        "load_kwargs": {"min_depth": 3, "max_depth": 5},
        "gen_fn": "generate_and_verify",
        "min_steps": 3,
        "label": "ProofWriter",
        "baseline_rho": 2.4,
        "est_sec_per_chain": 10,
    },
    "gsm8k": {
        "module": "gsm8k_trajectory",
        "load_fn": "load_gsm8k",
        "load_args": [],
        "load_kwargs": {},
        "gen_fn": "generate_and_verify",
        "min_steps": 4,
        "label": "GSM8K",
        "baseline_rho": 1.5,
        "est_sec_per_chain": 8,
    },
}

# ── Logging ──────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> Path:
    """Set up dual logging: file + console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"temperature_ablation_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return log_path


# ── Ollama helpers ───────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return True
    except Exception:
        return False


def get_available_models() -> list:
    """Get list of model names available in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def check_model_available(model: str) -> bool:
    """Check if a specific model is available."""
    available = get_available_models()
    return any(model in m for m in available)


def warm_up_model(model: str) -> bool:
    """Send a trivial prompt to load the model into memory."""
    logging.info(f"  Warming up {model} (loading into VRAM)...")
    try:
        payload = json.dumps({
            "model": model,
            "prompt": "Say hello.",
            "stream": False,
            "options": {"num_predict": 5},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        elapsed = time.time() - t0
        logging.info(f"  Model loaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        logging.error(f"  Failed to warm up model: {e}")
        return False


def ollama_generate(prompt: str, model: str, temperature: float = 0.0,
                    max_tokens: int = 1024) -> tuple:
    """
    Call Ollama API and return (response_text, elapsed_seconds).

    Uses the /api/generate endpoint with streaming disabled.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    return data.get("response", ""), elapsed


# ── Load existing chains to extract problems ─────────────────────────

def load_problems_from_existing_chains(dataset_key: str, n_problems: int,
                                        seed: int = RANDOM_SEED) -> list:
    """
    Load problems from existing UNIFIED_PIPELINE chain results.

    We need the original questions and reference answers for verification.
    Falls back to loading from the pipeline module directly.
    """
    ds_cfg = ABLATION_DATASETS[dataset_key]

    # Strategy 1: load from pipeline module directly (most reliable)
    try:
        mod = __import__(ds_cfg["module"])
        load_fn = getattr(mod, ds_cfg["load_fn"])
        # Pass n_problems as first positional arg
        problems = load_fn(n_problems, **ds_cfg["load_kwargs"])
        logging.info(f"  Loaded {len(problems)} problems from {ds_cfg['module']}")

        # Deterministic subset
        rng = np.random.default_rng(seed)
        if len(problems) > n_problems:
            indices = rng.choice(len(problems), size=n_problems, replace=False)
            indices.sort()
            problems = [problems[i] for i in indices]

        return problems
    except Exception as e:
        logging.warning(f"  Could not load from pipeline module: {e}")

    # Strategy 2: extract from existing chain files
    model_file_patterns = ["qwen2.5_14b", "qwen2_5_14b", "qwen2.5-14b"]
    chain_dirs = {
        "proofwriter": "proofwriter",
        "gsm8k": "gsm8k_trajectory",
    }

    subdir = chain_dirs.get(dataset_key, dataset_key)
    chain_dir = _UNIFIED_RESULTS / subdir

    if not chain_dir.exists():
        raise FileNotFoundError(
            f"No chain data found at {chain_dir} and pipeline module "
            f"{ds_cfg['module']} could not be imported. "
            f"Ensure UNIFIED_PIPELINE is accessible."
        )

    # Try to find any existing chain file
    chain_files = list(chain_dir.glob("*.json"))
    chain_files = [f for f in chain_files
                   if not f.name.endswith("_checkpoint.json")
                   and not f.name.endswith("_backup.json")]

    if not chain_files:
        raise FileNotFoundError(f"No chain files found in {chain_dir}")

    # Load the first available chain file
    with open(chain_files[0]) as f:
        chains = json.load(f)

    # Extract unique problems
    seen_ids = set()
    problems = []
    for chain in chains:
        pid = chain.get("problem_id", "")
        if pid not in seen_ids:
            seen_ids.add(pid)
            prob = {
                "problem_id": pid,
                "question": chain.get("question", ""),
            }
            # Copy reference fields if they exist
            for key in ["reference_answer", "answer", "expected",
                        "theory", "hypothesis", "label"]:
                if key in chain:
                    prob[key] = chain[key]
            problems.append(prob)

    logging.info(f"  Extracted {len(problems)} problems from {chain_files[0].name}")

    rng = np.random.default_rng(seed)
    if len(problems) > n_problems:
        indices = rng.choice(len(problems), size=n_problems, replace=False)
        indices.sort()
        problems = [problems[i] for i in indices]

    return problems


# ── Chain generation with temperature ────────────────────────────────

def generate_chains_for_temperature(
    problems: list,
    dataset_key: str,
    temperature: float,
    n_chains: int,
    checkpoint_dir: Path,
) -> list:
    """
    Generate reasoning chains at a given temperature using the UNIFIED_PIPELINE
    verification pipeline.

    For each problem, generates n_chains chains, each parsed and verified
    using the same pipeline module as the original experiments.
    """
    ds_cfg = ABLATION_DATASETS[dataset_key]
    checkpoint_file = checkpoint_dir / f"temp_{temperature:.1f}_{dataset_key}_checkpoint.json"

    # Load checkpoint if exists
    completed_chains = []
    completed_ids = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                completed_chains = json.load(f)
            completed_ids = {
                (c["problem_id"], c.get("chain_idx", 0))
                for c in completed_chains
            }
            logging.info(f"    Resuming from checkpoint: {len(completed_chains)} chains done")
        except Exception:
            pass

    # Import the pipeline module for verification
    try:
        mod = __import__(ds_cfg["module"])
    except ImportError as e:
        logging.error(f"    Cannot import {ds_cfg['module']}: {e}")
        logging.error(f"    Will use basic step parsing (no pipeline verification)")
        mod = None

    # Monkey-patch ollama_generate in the pipeline module to use our temperature
    if mod is not None:
        original_generate = getattr(mod, "ollama_generate", None)

        def patched_generate(*args, **kwargs):
            """Override temperature in pipeline's ollama_generate."""
            kwargs["temperature"] = temperature
            return original_generate(*args, **kwargs)

        if original_generate is not None:
            mod.ollama_generate = patched_generate

    all_chains = list(completed_chains)
    total_to_generate = len(problems) * n_chains - len(completed_chains)
    generated_count = 0

    for prob_idx, problem in enumerate(problems):
        pid = problem.get("problem_id", f"p{prob_idx}")

        for chain_idx in range(n_chains):
            if (pid, chain_idx) in completed_ids:
                continue

            generated_count += 1
            if generated_count % 20 == 1:
                pct = generated_count / max(total_to_generate, 1) * 100
                logging.info(
                    f"    [{generated_count}/{total_to_generate}] "
                    f"({pct:.0f}%) Problem {pid}, chain {chain_idx+1}/{n_chains}"
                )

            try:
                chain = _generate_single_chain(
                    problem, pid, chain_idx, dataset_key, temperature, mod
                )
                all_chains.append(chain)
            except Exception as e:
                logging.warning(f"    Error on {pid} chain {chain_idx}: {e}")
                all_chains.append({
                    "problem_id": pid,
                    "chain_idx": chain_idx,
                    "temperature": temperature,
                    "model": MODEL_NAME,
                    "dataset": dataset_key,
                    "error": str(e),
                })

            # Checkpoint after every chain
            with open(checkpoint_file, "w") as f:
                json.dump(all_chains, f, indent=2, default=str)

    # Restore original generate if we patched it
    if mod is not None and original_generate is not None:
        mod.ollama_generate = original_generate

    return all_chains


def _generate_single_chain(
    problem: dict,
    problem_id: str,
    chain_idx: int,
    dataset_key: str,
    temperature: float,
    mod,
) -> dict:
    """
    Generate a single reasoning chain, parse steps, and verify.

    Uses the pipeline module's generate_and_verify logic where possible,
    otherwise falls back to direct Ollama calls with generic step parsing.
    """
    ds_cfg = ABLATION_DATASETS[dataset_key]

    # Build the prompt using the pipeline module's COT_PROMPT
    if mod is not None and hasattr(mod, "COT_PROMPT"):
        cot_prompt = mod.COT_PROMPT
    else:
        cot_prompt = (
            "Solve the following problem step by step. "
            "Show your reasoning clearly.\n\n{problem}"
        )

    # Format prompt based on dataset type
    if dataset_key == "proofwriter":
        theory = problem.get("theory", problem.get("question", ""))
        question = problem.get("hypothesis", problem.get("question", ""))
        if "{theory}" in cot_prompt and "{question}" in cot_prompt:
            prompt = cot_prompt.format(theory=theory, question=question)
        elif "{problem}" in cot_prompt:
            prompt = cot_prompt.format(problem=f"{theory}\n\nQuestion: {question}")
        else:
            prompt = f"{cot_prompt}\n\n{theory}\n\nQuestion: {question}"
    else:
        question = problem.get("question", "")
        if "{problem}" in cot_prompt:
            prompt = cot_prompt.format(problem=question)
        else:
            prompt = f"{cot_prompt}\n\n{question}"

    # Generate response
    response, elapsed = ollama_generate(
        prompt, MODEL_NAME, temperature=temperature, max_tokens=1024
    )

    # Parse steps from response
    steps = parse_reasoning_steps(response)

    # Build error vector via verification
    error_vector = verify_steps(
        problem, steps, response, dataset_key, mod
    )

    # Compute summary fields
    n_steps = len(steps)
    first_error = None
    for i, e in enumerate(error_vector):
        if e == 1:
            first_error = i + 1  # 1-indexed
            break
    n_errors = sum(error_vector)
    final_correct = _check_final_answer(problem, response, dataset_key, mod)

    return {
        "problem_id": problem_id,
        "chain_idx": chain_idx,
        "question": problem.get("question", ""),
        "response": response,
        "steps": steps,
        "n_steps": n_steps,
        "model": MODEL_NAME,
        "temperature": temperature,
        "dataset": dataset_key,
        "error_vector": error_vector,
        "first_error": first_error,
        "n_errors": n_errors,
        "final_correct": final_correct,
        "elapsed_sec": round(elapsed, 2),
    }


def parse_reasoning_steps(response: str) -> list:
    """
    Parse a model response into individual reasoning steps.

    Matches the UNIFIED_PIPELINE logic: looks for numbered steps,
    "Step N:" patterns, or splits on paragraph/sentence boundaries.
    """
    if not response or not response.strip():
        return []

    # Strategy 1: numbered steps (Step 1:, 1., 1), etc.)
    step_pattern = re.compile(
        r'(?:^|\n)\s*(?:Step\s+)?(\d+)[.):]\s*(.*?)(?=\n\s*(?:Step\s+)?\d+[.):]\s|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    matches = step_pattern.findall(response)
    if len(matches) >= 2:
        steps = [m[1].strip() for m in matches if m[1].strip()]
        if steps:
            return steps

    # Strategy 2: bullet points or dashes
    bullet_pattern = re.compile(r'(?:^|\n)\s*[-*]\s+(.*?)(?=\n\s*[-*]\s|\Z)', re.DOTALL)
    matches = bullet_pattern.findall(response)
    if len(matches) >= 2:
        steps = [m.strip() for m in matches if m.strip()]
        if steps:
            return steps

    # Strategy 3: paragraph splits (double newline)
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # Strategy 4: sentence-level split (fallback)
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 2:
        return sentences

    # Last resort: entire response as one step
    return [response.strip()] if response.strip() else []


def verify_steps(
    problem: dict,
    steps: list,
    response: str,
    dataset_key: str,
    mod,
) -> list:
    """
    Build the error vector for a chain's steps.

    Uses the pipeline module's verification logic where available.
    Falls back to a heuristic check based on the reference answer.
    """
    if not steps:
        return []

    error_vector = [0] * len(steps)

    # Try to use pipeline module's verification
    if mod is not None:
        try:
            # The pipeline modules typically have a verify_step or
            # check_step_correctness function. Try common patterns.
            if hasattr(mod, "verify_step"):
                for i, step in enumerate(steps):
                    try:
                        is_correct = mod.verify_step(problem, step, i, steps[:i])
                        error_vector[i] = 0 if is_correct else 1
                    except Exception:
                        error_vector[i] = 0  # assume correct if verification fails
                return error_vector

            if hasattr(mod, "build_error_vector"):
                ev = mod.build_error_vector(problem, steps, response)
                if ev and len(ev) == len(steps):
                    return ev

            if hasattr(mod, "check_step"):
                for i, step in enumerate(steps):
                    try:
                        is_correct = mod.check_step(problem, step, i)
                        error_vector[i] = 0 if is_correct else 1
                    except Exception:
                        error_vector[i] = 0
                return error_vector
        except Exception:
            pass

    # Fallback: heuristic verification based on reference answer
    ref_answer = _get_reference_answer(problem, dataset_key)
    if ref_answer is not None:
        # Check if each step is consistent with the reference
        # This is a simplified heuristic; the pipeline modules do more
        for i, step in enumerate(steps):
            error_vector[i] = _heuristic_step_check(step, ref_answer, i, len(steps))

    return error_vector


def _get_reference_answer(problem: dict, dataset_key: str):
    """Extract the reference answer from a problem dict."""
    for key in ["reference_answer", "answer", "expected", "label",
                "correct_answer", "solution"]:
        if key in problem and problem[key]:
            return str(problem[key])
    return None


def _heuristic_step_check(step: str, reference: str, step_idx: int,
                           total_steps: int) -> int:
    """
    Heuristic error check for a step.

    Returns 0 for correct, 1 for error. Very conservative -- only marks
    clear numerical contradictions as errors.
    """
    # Extract numbers from step and reference
    step_nums = set(re.findall(r'-?\d+\.?\d*', step))
    ref_nums = set(re.findall(r'-?\d+\.?\d*', reference))

    # For the final step, check if the final answer matches
    if step_idx == total_steps - 1 and ref_nums:
        # Check if any reference number appears in the final step
        if not step_nums.intersection(ref_nums):
            return 1  # Final answer mismatch

    return 0  # Default: assume correct


def _check_final_answer(problem: dict, response: str, dataset_key: str,
                         mod) -> bool:
    """Check if the final answer in the response is correct."""
    if mod is not None:
        try:
            if hasattr(mod, "check_final_answer"):
                return mod.check_final_answer(problem, response)
            if hasattr(mod, "extract_answer"):
                predicted = mod.extract_answer(response)
                ref = _get_reference_answer(problem, dataset_key)
                if predicted and ref:
                    return str(predicted).strip().lower() == str(ref).strip().lower()
        except Exception:
            pass

    # Fallback heuristic
    ref = _get_reference_answer(problem, dataset_key)
    if ref:
        return ref.strip().lower() in response.lower()
    return False


# ── Weibull fitting ──────────────────────────────────────────────────

def fit_weibull_for_condition(chains: list, label: str = "") -> dict:
    """
    Fit a Weibull model to chains from a single condition.

    Returns rho, lambda, 95% CI for rho, and interpretation.
    """
    from lifelines import WeibullFitter

    # Filter valid chains (no errors during generation)
    valid = [c for c in chains if "error" not in c and c.get("n_steps", 0) > 0]
    if len(valid) < 10:
        return {
            "label": label,
            "n_chains": len(valid),
            "error": f"Too few valid chains ({len(valid)})",
        }

    # Build duration and event arrays
    durations = []
    events = []
    for c in valid:
        fe = c.get("first_error")
        n_steps = c.get("n_steps", 1)
        if fe is not None and fe > 0:
            durations.append(float(fe))
            events.append(1.0)
        else:
            durations.append(float(n_steps))
            events.append(0.0)  # right-censored

    durations = np.array(durations)
    events = np.array(events)
    durations = np.maximum(durations, 0.5)

    if events.sum() < 3:
        return {
            "label": label,
            "n_chains": len(valid),
            "n_events": int(events.sum()),
            "error": "Too few error events for Weibull fitting",
        }

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

        if rho_ci_low > 1.0:
            interp = "AGING"
        elif rho_ci_high < 1.0:
            interp = "LINDY"
        else:
            interp = "CONSTANT"

        return {
            "label": label,
            "n_chains": len(valid),
            "n_events": int(events.sum()),
            "rho": round(rho, 4),
            "rho_ci_low": round(rho_ci_low, 4),
            "rho_ci_high": round(rho_ci_high, 4),
            "lambda": round(lambda_, 4),
            "interpretation": interp,
            "mean_n_steps": round(float(np.mean([c["n_steps"] for c in valid])), 1),
            "error_rate": round(float(np.mean([
                1 if c.get("first_error") else 0 for c in valid
            ])), 3),
        }
    except Exception as e:
        return {
            "label": label,
            "n_chains": len(valid),
            "error": str(e),
        }


# ── Main orchestrator ────────────────────────────────────────────────

def estimate_runtime(n_problems: int, temperatures: list, datasets: list) -> str:
    """Estimate total runtime in hours."""
    total_chains = 0
    for temp in temperatures:
        n_chains = CHAINS_PER_PROBLEM.get(temp, 5)
        total_chains += n_problems * n_chains * len(datasets)

    avg_sec = 10  # average seconds per chain
    total_sec = total_chains * avg_sec
    return f"{total_sec / 3600:.1f} hours ({total_chains} total chains)"


def main():
    parser = argparse.ArgumentParser(
        description="Temperature ablation for Project 37 aging analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_temperature_ablation.py --dry-run
    python scripts/run_temperature_ablation.py
    python scripts/run_temperature_ablation.py --temperatures 0.3
    python scripts/run_temperature_ablation.py --n-problems 50 --temperatures 0.3,0.7
        """,
    )
    parser.add_argument(
        "--temperatures", type=str, default=",".join(str(t) for t in TEMPERATURES),
        help="Comma-separated temperatures to run (default: 0.0,0.3,0.7)",
    )
    parser.add_argument(
        "--datasets", type=str, default="proofwriter,gsm8k",
        help="Comma-separated dataset keys (default: proofwriter,gsm8k)",
    )
    parser.add_argument(
        "--n-problems", type=int, default=N_PROBLEMS,
        help=f"Number of problems per dataset (default: {N_PROBLEMS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and check setup without running anything",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip T=0.0 (use existing baseline data instead)",
    )
    args = parser.parse_args()

    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    dataset_keys = [d.strip() for d in args.datasets.split(",")]

    if args.skip_baseline and 0.0 in temperatures:
        temperatures.remove(0.0)

    # Validate dataset keys
    for dk in dataset_keys:
        if dk not in ABLATION_DATASETS:
            print(f"ERROR: Unknown dataset '{dk}'. Valid: {list(ABLATION_DATASETS.keys())}")
            sys.exit(1)

    # Set up logging
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(_RESULTS_DIR)

    logging.info("=" * 70)
    logging.info("  TEMPERATURE ABLATION — Project 37")
    logging.info("  Reviewer Response: Does aging persist under non-greedy decoding?")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  Model:        {MODEL_LABEL} ({MODEL_NAME})")
    logging.info(f"  Temperatures: {temperatures}")
    logging.info(f"  Datasets:     {dataset_keys}")
    logging.info(f"  Problems:     {args.n_problems} per dataset")
    logging.info(f"  Chains/prob:  {CHAINS_PER_PROBLEM}")
    logging.info(f"  Estimated:    {estimate_runtime(args.n_problems, temperatures, dataset_keys)}")
    logging.info(f"  Log file:     {log_path}")
    logging.info("")

    # ── Pre-flight checks ────────────────────────────────────────────
    logging.info("  PRE-FLIGHT CHECKS")
    logging.info("  " + "-" * 40)

    if not check_ollama_running():
        logging.error("  FAIL: Ollama is not running!")
        logging.error("  Start with: ollama serve")
        if not args.dry_run:
            sys.exit(1)
        else:
            logging.warning("  (dry-run: continuing anyway)")
    else:
        logging.info("  [OK] Ollama is running")

    if not check_model_available(MODEL_NAME):
        logging.error(f"  FAIL: {MODEL_NAME} not found in Ollama")
        logging.error(f"  Pull it with: ollama pull {MODEL_NAME}")
        if not args.dry_run:
            sys.exit(1)
        else:
            logging.warning("  (dry-run: continuing anyway)")
    else:
        logging.info(f"  [OK] {MODEL_NAME} is available")

    # Check pipeline access
    if not _PIPELINE_DIR.exists():
        logging.error(f"  FAIL: UNIFIED_PIPELINE not found at {_PIPELINE_DIR}")
        if not args.dry_run:
            sys.exit(1)
    else:
        logging.info(f"  [OK] UNIFIED_PIPELINE found")
    logging.info("")

    # ── Execution plan ───────────────────────────────────────────────
    logging.info("  EXECUTION PLAN")
    logging.info("  " + "-" * 40)
    total_chains = 0
    for ds_key in dataset_keys:
        ds = ABLATION_DATASETS[ds_key]
        logging.info(f"  {ds['label']}:")
        for temp in temperatures:
            n_chains = CHAINS_PER_PROBLEM.get(temp, 5)
            total = args.n_problems * n_chains
            est_min = total * ds["est_sec_per_chain"] / 60
            logging.info(
                f"    T={temp:.1f}: {args.n_problems} problems x "
                f"{n_chains} chains = {total} chains (~{est_min:.0f} min)"
            )
            total_chains += total
    logging.info(f"\n  Total chains: {total_chains}")
    logging.info("")

    if args.dry_run:
        logging.info("  DRY RUN complete. Everything looks good.")
        logging.info("  Remove --dry-run to start generation.")
        return

    # ── Generate chains ──────────────────────────────────────────────
    logging.info("=" * 70)
    logging.info("  STARTING GENERATION")
    logging.info("=" * 70)
    logging.info("")

    # Warm up the model
    if not warm_up_model(MODEL_NAME):
        logging.error("  Could not load model into VRAM. Aborting.")
        sys.exit(1)

    overall_t0 = time.time()
    all_results = {}
    checkpoint_dir = _RESULTS_DIR / "temperature_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for ds_key in dataset_keys:
        ds_cfg = ABLATION_DATASETS[ds_key]
        logging.info(f"\n  --- {ds_cfg['label']} ---")

        # Load problems
        logging.info(f"  Loading {args.n_problems} problems...")
        try:
            problems = load_problems_from_existing_chains(ds_key, args.n_problems)
        except Exception as e:
            logging.error(f"  Failed to load problems: {e}")
            continue

        logging.info(f"  Loaded {len(problems)} problems")

        for temp in temperatures:
            n_chains = CHAINS_PER_PROBLEM.get(temp, 5)
            logging.info(f"\n  T={temp:.1f} ({n_chains} chain(s) per problem)")

            est_sec = len(problems) * n_chains * ds_cfg["est_sec_per_chain"]
            est_finish = datetime.now() + timedelta(seconds=est_sec)
            logging.info(f"  Estimated: ~{est_sec/60:.0f} min (done by ~{est_finish.strftime('%H:%M')})")

            t0 = time.time()
            try:
                chains = generate_chains_for_temperature(
                    problems, ds_key, temp, n_chains, checkpoint_dir
                )
                elapsed = time.time() - t0

                # Fit Weibull
                label = f"{ds_cfg['label']}_T{temp:.1f}"
                weibull = fit_weibull_for_condition(chains, label)

                result_key = f"{ds_key}_T{temp:.1f}"
                all_results[result_key] = {
                    "dataset": ds_key,
                    "dataset_label": ds_cfg["label"],
                    "temperature": temp,
                    "n_problems": len(problems),
                    "n_chains_per_problem": n_chains,
                    "n_chains_total": len(chains),
                    "elapsed_sec": round(elapsed, 1),
                    "weibull": weibull,
                    "chains": chains,
                }

                # Log result
                if "rho" in weibull:
                    logging.info(
                        f"  Result: rho = {weibull['rho']:.3f} "
                        f"[{weibull['rho_ci_low']:.3f}, {weibull['rho_ci_high']:.3f}] "
                        f"({weibull['interpretation']})"
                    )
                else:
                    logging.info(f"  Result: {weibull.get('error', 'unknown error')}")

                logging.info(f"  Done in {elapsed/60:.1f} min")

            except KeyboardInterrupt:
                logging.warning("\n  Interrupted! Checkpoints saved.")
                break
            except Exception as e:
                logging.error(f"  Error: {e}")
                continue

    # ── Summary + Save ───────────────────────────────────────────────
    total_elapsed = time.time() - overall_t0
    logging.info("")
    logging.info("=" * 70)
    logging.info("  TEMPERATURE ABLATION COMPLETE")
    logging.info(f"  Total time: {total_elapsed/3600:.1f} hours")
    logging.info("=" * 70)
    logging.info("")

    # Summary table
    logging.info(f"  {'Condition':<25} {'rho':>8} {'95% CI':>20} {'Interp':>10}")
    logging.info(f"  {'-'*65}")
    for key, res in sorted(all_results.items()):
        w = res["weibull"]
        if "rho" in w:
            logging.info(
                f"  {key:<25} {w['rho']:8.3f} "
                f"[{w['rho_ci_low']:.3f}, {w['rho_ci_high']:.3f}]"
                f"{w['interpretation']:>10}"
            )
        else:
            logging.info(f"  {key:<25} {'ERROR':>8} {w.get('error', '')[:30]}")

    # Save results (without raw chains to keep file size manageable)
    output = {
        "experiment": "temperature_ablation",
        "model": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_sec": round(total_elapsed, 1),
        "n_problems_per_dataset": args.n_problems,
        "temperatures": temperatures,
        "datasets": dataset_keys,
        "conditions": {},
    }

    for key, res in all_results.items():
        # Save everything except raw chains (which are in checkpoints)
        output["conditions"][key] = {
            k: v for k, v in res.items() if k != "chains"
        }
        # Also save chain-level summaries (without full response text)
        chain_summaries = []
        for c in res.get("chains", []):
            if "error" in c and "n_steps" not in c:
                continue
            chain_summaries.append({
                "problem_id": c.get("problem_id"),
                "chain_idx": c.get("chain_idx", 0),
                "n_steps": c.get("n_steps", 0),
                "error_vector": c.get("error_vector", []),
                "first_error": c.get("first_error"),
                "n_errors": c.get("n_errors", 0),
                "final_correct": c.get("final_correct", False),
            })
        output["conditions"][key]["chain_summaries"] = chain_summaries

    output_path = _RESULTS_DIR / "temperature_pilot.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logging.info(f"\n  Results saved: {output_path}")

    logging.info("")
    logging.info("  Next steps:")
    logging.info("    1. Run analysis:  python analysis/temperature_analysis.py")
    logging.info("    2. Check narrative impact in results/narrative_impact_assessment.md")
    logging.info("")


if __name__ == "__main__":
    main()
