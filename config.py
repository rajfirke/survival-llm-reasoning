"""
Configuration for survival analysis of step-level reasoning errors in LLM chains.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "gsm8k": {
        "path_key": "gsm8k_fixed",
        "label": "GSM8K",
        "category": "mathematical",
        "color": "#2a9d8f",
    },
    "mathqa": {
        "path_key": "mathqa_fixed",
        "label": "MathQA",
        "category": "mathematical",
        "color": "#264653",
    },
    "bbh_tracking": {
        "path_key": "bbh_tracking",
        "label": "BBH Tracking",
        "category": "multi_step_logic",
        "color": "#e76f51",
    },
    "proofwriter": {
        "path_key": "proofwriter",
        "label": "ProofWriter",
        "category": "logical_deduction",
        "color": "#f4a261",
    },
    "mmlu_pro": {
        "path_key": "mmlu_pro",
        "label": "MMLU-Pro",
        "category": "multi_domain",
        "color": "#6a4c93",
    },
}

MODELS = {
    "llama3.2_3b": {"label": "Llama-3.2-3B", "size": 3, "color": "#e63946"},
    "qwen2.5_7b": {"label": "Qwen2.5-7B", "size": 7, "color": "#457b9d"},
    "gemma2_9b": {"label": "Gemma2-9B", "size": 9, "color": "#2d6a4f"},
    "qwen2.5_14b": {"label": "Qwen2.5-14B", "size": 14, "color": "#1d3557"},
    "gemini_2_5_flash": {"label": "Gemini-2.5-Flash", "size": 0, "color": "#f77f00"},
    "claude_haiku_4_5": {"label": "Claude-Haiku-4.5", "size": 0, "color": "#7209b7"},
    "llama_3_3_70b": {"label": "Llama-3.3-70B", "size": 70, "color": "#d62828"},
}

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000
ALPHA = 0.05
