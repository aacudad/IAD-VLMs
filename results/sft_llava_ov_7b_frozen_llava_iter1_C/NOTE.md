# LLaVA-OneVision-7B-SI + native KCR corpus (the cross-architecture result)

**Keep-Correct-Revise (KCR), called Arm C in the code and in every path below.** This run is the
KCR loop run *natively on LLaVA*: the rollouts come from a LLaVA policy, the teacher corrects and
revises LLaVA's own failures, and the resulting 6,000-trace corpus fine-tunes the LLaVA base model.
Nothing in it is Qwen-derived. The method is defined once in the [`README.md`](../../README.md),
section "The method: Keep-Correct-Revise (KCR)".

- Backbone: `lmms-lab/llava-onevision-qwen2-7b-si`, the same backbone IAD-R1 uses.
- Corpus: 6,000 records, 50/50 anomaly/normal, LlamaFactory key `iad_sft_llava_iter1_C`.
- SFT recipe: SigLIP vision tower **frozen**, projector trained, 4 epochs, same schedule as the
  Qwen Arm-C run. Checkpoints 188 / 376 / 564 / 748 = epochs 1 / 2 / 3 / 4.
- Evaluation: identical harness and identical subsets as every other row in this repo
  (DS-MVTec n=1670, VisA n=2141), inference served through **vLLM**, hence the `_vllm` tail on the
  filenames. The prompt mode is still `_trainprompt`. The `_vllm` tail only records the serving path.

## Balanced accuracy per epoch

Recompute any cell with [`results/compute_ba.py`](../compute_ba.py) from the `tp/tn/fp/fn` block.

| Epoch | Checkpoint | DS-MVTec BA | tp | tn | fp | fn | n | VisA BA | tp | tn | fp | fn | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | checkpoint-188 | 85.29 | 1111 | 355 | 89 | 115 | 1670 | 72.32 | 1023 | 554 | 388 | 169 | 2134 |
| 2 | checkpoint-376 | 85.60 | 1033 | 386 | 58 | 193 | 1670 | 70.10 | 939 | 583 | 361 | 258 | 2141 |
| 3 | checkpoint-564 | 86.45 | 1029 | 395 | 49 | 197 | 1670 | 73.56 | 894 | 683 | 260 | 303 | 2140 |
| **4** | **checkpoint-748** | **88.45** | 1045 | 407 | 37 | 181 | 1670 | **74.25** | 942 | 659 | 285 | 255 | 2141 |

Epoch 4 is the best epoch on both benchmarks at once, so it is the released checkpoint
([`aacudad/AnomalyThink-LLaVA-OneVision-7B-KCR`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-KCR)).

A few VisA rows have n = 2134 or 2140 instead of 2141. Those are samples where the served model
returned no parseable verdict, so they are dropped rather than scored as wrong. The effect on BA is
below 0.05 pp and it does not change any ordering.

## Why this run matters

On this backbone the KCR corpus (**88.45 / 74.25**) is above SFT-then-GRPO on the same backbone
(**87.66 / 72.58**, [`../grpo_llava_ov_from_ep1/checkpoint-530/`](../grpo_llava_ov_from_ep1/checkpoint-530/)).
Corpus curation beats reinforcement learning here, on LLaVA-OneVision-7B-SI, which is IAD-R1's own
headline backbone. That is the same ordering we already see on Qwen2.5-VL, so it is a property of the
method and not of one architecture.

It is also above our IAD-R1 reference row (**81.92 / 71.34**,
[`../iad_r1_qwen_recanon/`](../iad_r1_qwen_recanon/)). Read that comparison carefully: it is
same-harness but **not** same-backbone. Our IAD-R1 row is their released **Qwen2.5-VL-7B**
checkpoint re-evaluated here. We did not re-evaluate their LLaVA-OneVision checkpoint on the full
subsets, so "we beat IAD-R1 on its own backbone" is not a claim this folder supports. What it
supports is "our pipeline on their backbone scores 88.45 / 74.25 under the same harness that scores
their released model 81.92 / 71.34".

## Contamination caveat, DS-MVTec only

`lmms-lab/LLaVA-OneVision-Data`, config `vision_flan(filtered)`, contains **426 rows whose id
matches `%MVTecAD%`**. VisA matches **0**. So every DS-MVTec number for a LLaVA-OneVision-derived
model in this repo, including IAD-R1's released checkpoint, carries a pretraining-exposure asterisk.
The VisA column does not. Read the VisA gain as the clean one. Verified live through the Hugging
Face datasets-server filter.

## What is not in this folder

- Model weights. They are on Hugging Face, not here.
- Epoch checkpoints 188 / 376 / 564 of the released model. Only the eval JSONs, `trainer_state.json`,
  `config.json` and `generation_config.json` per checkpoint are shipped, matching the layout of
  [`../sft_qwen25vl_7b_abc_C_full_patched/`](../sft_qwen25vl_7b_abc_C_full_patched/).
