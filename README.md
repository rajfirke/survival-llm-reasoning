# When Does Reasoning Age? Survival Analysis of Step-Level Error Hazard in LLM Chains

**EMNLP 2026 Main Conference**  
Raj Firke (Red Hat) · Rajeswari Kannan (Pimpri Chinchwad College of Engineering)

[Paper](https://aclanthology.org/TBD) | [EMNLP 2026](https://2026.emnlp.org/)

## Overview

This repository contains the code, data, and pre-computed results for our paper analyzing
step-level error dynamics in LLM chain-of-thought reasoning through the lens of survival analysis.

**Key finding:** The conditional probability of reasoning error increases with step depth
(Weibull ρ > 1) in 34 of 35 dataset–model cells across 18,969 chains, five benchmarks,
and seven models — a near-universal "aging" effect invisible to raw error rates due to
survivorship bias.

## Repository Structure

```
analysis/       Core survival analysis modules (KM, Weibull, IPCW, frailty, Cox)
scripts/        Experiment scripts (verifier audit, distribution comparison, temp ablation)
data/           Chain-level dataset with step-level correctness labels (18,969 chains)
results/        Pre-computed analysis outputs matching all paper tables and appendices
```

## Setup

```bash
git clone https://github.com/rajfirke/survival-llm-reasoning
cd survival-llm-reasoning
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, lifelines≥0.28, pandas≥2.0, numpy≥1.24, scipy≥1.12,
matplotlib≥3.8, seaborn≥0.13, statsmodels≥0.14, pymannkendall≥1.4

## Reproducing Results

### Main analysis (all 35 dataset–model cells)
```bash
python run_analysis.py
```
Outputs to `results/`. Matches Tables 1–4 and Appendices A–F in the paper.

### Temperature ablation (Appendix H)
```bash
# Requires Qwen2.5-14B access and ProofWriter data
python scripts/run_temperature_ablation.py --temperatures 0.3 0.7 --n_chains 1000
```

### Verifier audit (§4.4 / Appendix D)
```bash
python scripts/audit_verifier.py
```

### Distribution comparison (Appendix B)
```bash
python scripts/compare_distributions.py
```

## Data

`data/survival_data.csv` contains 18,969 reasoning chains with the following columns:

| Column | Description |
|--------|-------------|
| `chain_id` | Unique chain identifier |
| `dataset` | Benchmark (GSM8K, MathQA, BBH, ProofWriter, MMLU-Pro) |
| `model` | Model name |
| `step` | Step index (1-indexed) |
| `correct` | Binary correctness label (1 = correct, 0 = error) |
| `is_censored` | 1 if chain completed without error (right-censored) |
| `event_time` | Step of first error (or chain length if censored) |

Labels are programmatic (not LLM-as-judge): math steps verified against reference
trajectories, logic steps against derivable facts, state-tracking against ground-truth states,
MMLU-Pro via answer-letter tracking.

## Pre-computed Results

All JSON/CSV files in `results/` are ready to use without re-running analysis. Each file
maps directly to a paper table or appendix section (see `results/` structure above).

## Citing

```bibtex
@inproceedings{firke2026reasoning,
  title     = {When Does Reasoning Age? Survival Analysis of Step-Level Error Hazard in {LLM} Chains},
  author    = {Firke, Raj and Kannan, Rajeswari},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
}
```

## License

Code: MIT License. Data (`data/`): CC BY 4.0.
