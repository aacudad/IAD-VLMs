# LLaVA-OneVision-7B-SI, KCR corpus, corrected build (the thesis LLaVA KCR row)

Corpus: `traces/llava_kcr/sft_llava_C_original_train.json` (6,000 items, all inside the 6,000-image SFT split,
balanced on the `<answer>` verdict: 3,000 / 3,000). Built by `scripts/03_rollout_star/build_llava_arms_original.py`
from the LLaVA rollout pool `traces/rollout_pools/llava_phase0_10k/`. Same SFT recipe as every LLaVA arm
(frozen vision tower, projector + LM trained, 4 epochs, lr 1e-5, effective batch 32, ZeRO-3 offload).

Two evaluation paths per checkpoint. `*_trainprompt_vllm.json` is the vLLM path (greedy, 1024 tokens,
system turn "Please answer by yes or no"), `*_trainprompt.json` is the HuggingFace path on the same weights.
**The thesis reports the vLLM files.** Strict scoring: an output without a parsable `<answer>` counts as wrong.

| ckpt (epoch) | DS-MVTec vLLM | VisA vLLM | DS-MVTec HF | VisA HF |
|---|---:|---:|---:|---:|
| 188 (1) | 84.35 | 71.36 | 84.18 | 70.97 |
| 376 (2), **selected on DS-MVTec** | **87.32** | **72.65** (13 unparsed) | 87.36 | 73.27 |
| 564 (3) | 86.96 | 73.57 | 86.28 | 73.61 |
| 748 (4) | 86.60 | 74.29 | 87.01 | 73.42 |

The first (leaky) build of this corpus is `results/sft_llava_ov_7b_frozen_llava_iter1_C/` (88.45 / 74.25 at
epoch 4). It drew 2,484 of its 6,000 images from the GRPO split and was balanced on the folder name (45.0 %
anomalous). Thesis Appendix M documents the difference. Every DS-MVTec number on this backbone carries the
LLaVA-OneVision-Data contamination caveat (`results/contamination_llava_ov_data/`), VisA does not.

## 2026-09-07: held-out Real-IAD

`checkpoint-376/eval_realiad4k_full_trainprompt_vllm.json`: the corrected-corpus KCR checkpoint on the 4,236-image
held-out Real-IAD split (`traces/.../new_sft_c1_train.json` images, disjoint from SFT and GRPO), vLLM path,
training prompt. Strict: BA 84.03, accuracy 83.97, precision 93.21, recall 73.46, TNR 94.59, F1 82.16.
Thesis Appendix L, Table L.1, and one sentence in §6.9. Script: `scripts/04_eval/evaluate_vllm_llava_heldout.py`.
