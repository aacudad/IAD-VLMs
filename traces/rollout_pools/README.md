# Rollout pools behind the KCR loop (thesis §5.7, Figure 5.2)

Two pools, one per backbone, each over the same 10,236 training images (6,000 SFT + 4,236 GRPO) at k = 8.

| | `qwen_phase0_10k/` | `llava_phase0_10k/` |
|---|---:|---:|
| policy | Qwen run-2 GRPO ckpt-530 | LLaVA SFT+GRPO ckpt-530 |
| sampler | `scripts/03_rollout_star/phase0_rollout_kscoring.py` (HF, temp 0.7, top-p 0.9) | `phase0_rollout_llava_vllm.py` (vLLM, temp 1.0, top-p 0.9, top-k 50) |
| `rollouts_raw.jsonl.gz` | 10,236 items x 8 rollouts, per-rollout reward fields | same |
| `good_traces.json.gz` (kept) | 8,872 | 9,179 |
| `needs_correction.json` + `gemini_corrected.jsonl` | 1,364 | 1,057 |
| `needs_rewrite_combined.json` + `gemini_rewritten.jsonl` | 2,406 | 2,513 |
| `good_traces_faithful.json` (judge-confirmed) | 8,660 | 9,045 |
| `judge_report.jsonl.gz` | 10,236 judge verdicts | 10,236 |
| `difficulty.json` | per-item difficulty | same |

Kept + corrected partition each pool exactly (8,872 + 1,364 and 9,179 + 1,057 = 10,236). The rewrite set is a
subset of the kept set. `gunzip` the three `.gz` files before use. Bucketing script `phase0_bucket.py`,
Arm builders `build_abc_datasets.py` (Qwen) and `build_llava_arms_original.py` (LLaVA).
