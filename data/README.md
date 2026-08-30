# Data: Survival Analysis of Step-Level Error Hazard in LLM Chains

Supplementary data for the paper "When Does Reasoning Age? Survival Analysis of Step-Level Error Hazard in LLM Chains."

## Files

- `survival_data.csv`: Pre-processed step-level survival observations for 18,969 reasoning chains.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| problem_id | string | Unique problem identifier from the source benchmark |
| dataset | string | Benchmark name (gsm8k, mathqa, bbh_tracking, proofwriter, mmlu_pro) |
| model | string | Model identifier |
| n_steps | int | Total number of reasoning steps in the chain |
| duration | int | Step at which first error occurs, or chain length if censored |
| event | int | 1 = error occurred (event), 0 = chain completed correctly (right-censored) |
| first_error_step | int/null | 0-indexed step of first error, or null if no error |
| n_errors | int | Total number of erroneous steps in the chain |
| final_correct | bool | Whether the final answer was correct |
| error_vector | list[int] | Binary vector of per-step correctness (1=error, 0=correct) |

## Benchmarks

| Dataset | Chains | Avg Steps | Type |
|---------|--------|-----------|------|
| GSM8K | 5,412 | 5.9 | Mathematical reasoning |
| MathQA | 4,729 | 8.4 | Mathematical reasoning |
| BBH Tracking | 3,342 | 6.0 | State tracking |
| ProofWriter | 3,185 | 7.8 | Logical deduction |
| MMLU-Pro | 2,301 | 8.2 | Multi-domain QA |

## Models

Seven models from four architecture families: Llama-3.2-3B, Qwen2.5-7B, Gemma2-9B, Qwen2.5-14B, Gemini-2.5-Flash, Claude-Haiku-4.5, Llama-3.3-70B. All inference uses greedy decoding (temperature 0).

## Verification

All step-level labels are programmatic (not LLM-as-judge):
- **Math** (GSM8K, MathQA): Numerical comparison against reference solution trajectories
- **Logic** (ProofWriter): Derivation validity against provably derivable facts
- **State tracking** (BBH): Comparison against ground-truth state tables
- **Multi-domain** (MMLU-Pro): Answer-letter tracking against gold answers
