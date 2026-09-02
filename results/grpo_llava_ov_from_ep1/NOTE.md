# LLaVA-OneVision-7B-SI, SFT then GRPO (the RL arm of the cross-architecture replication)

The Qwen GRPO recipe, moved to LLaVA-OneVision without changing the objective. It is the control
that the KCR corpus result is measured against: same backbone, same harness, same subsets, RL
instead of corpus curation.

- Init: LLaVA 6K SFT **epoch 1** = [`../sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/`](../sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/) (85.91 / 68.26).
- Reward: accuracy + format (consistency), unweighted sum, exactly as in the Qwen runs.
- G = 4, 32 rollouts per step, LR 1e-6, temperature 1.0, **β = 0** (KL monitored, not in the loss),
  ZeRO-3 with CPU offload.
- The **vision tower is trained during GRPO** (frozen during SFT). That matches what IAD-R1 does.
- checkpoint-530 = epoch 1. Evaluation served through vLLM, hence the `_vllm` tail. The prompt mode
  is still `_trainprompt`.

## Balanced accuracy

| Benchmark | BA | tp | tn | fp | fn | n | Eval JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| DS-MVTec | **87.66** | 1078 | 388 | 56 | 148 | 1670 | [`checkpoint-530/eval_dsmvtec_full_trainprompt_vllm.json`](checkpoint-530/eval_dsmvtec_full_trainprompt_vllm.json) |
| VisA | **72.58** | 958 | 611 | 330 | 236 | 2135 | [`checkpoint-530/eval_visa_full_trainprompt_vllm.json`](checkpoint-530/eval_visa_full_trainprompt_vllm.json) |

Against its own init that is **+1.75 DS-MVTec / +4.32 VisA**. GRPO clearly works on this backbone.
It just does not reach the KCR corpus, which gets **88.45 / 74.25** from supervised fine-tuning
alone ([`../sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/`](../sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/)).

Released as [`aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT-GRPO`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT-GRPO).

## Contamination caveat, DS-MVTec only

The LLaVA-OneVision training mixture (`lmms-lab/LLaVA-OneVision-Data`, config
`vision_flan(filtered)`) contains 426 rows whose id matches `%MVTecAD%`. VisA matches 0. Every
DS-MVTec number for a LLaVA-derived model here therefore carries a pretraining-exposure asterisk.
VisA does not.

## Two files deliberately left out of this folder

1. **`eval_dsmvtec_BROKEN_rope.json`** (2.3 MB in the source tree). A failed evaluation, not a
   result. The checkpoint was saved under transformers 5.0, which writes the RoPE settings into a
   `rope_parameters` block. Loading it under transformers 4.57 or vLLM silently falls back to a
   `rope_theta` that is off by 100x, and the model then produces fluent text but answers "no" to
   nearly everything (BA 50.70, TP 17, FN 1204). It looks exactly like training collapse and it is
   not. Anyone re-evaluating a transformers-5.0 checkpoint on an older stack should check
   `rope_theta` in the loaded config first. The 2.3 MB of all-"no" predictions add nothing that this
   paragraph does not.
2. **The epoch-2 restart** (`outputs/grpo_llava_ov_from_ep1_ep2/checkpoint-530/`, 87.86 / 72.10). A
   wash relative to epoch 1: +0.20 DS-MVTec, −0.48 VisA. It also is not a clean epoch 2, because the
   run was killed by server maintenance and restarted from ckpt-530 with a fresh Adam state and a
   re-seeded dataloader. Epoch 1 is kept as the LLaVA GRPO champion and the epoch-2 JSONs are left
   in the source tree.
