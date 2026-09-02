# LLaVA-OneVision-7B-SI on the 6K Gemini corpus (the LLaVA SFT baseline)

The plain SFT arm of the cross-architecture replication. Same 6,000 AnomalyThink traces that
produced the Qwen headline SFT model, same frozen-tower recipe, different backbone.

- Backbone: `lmms-lab/llava-onevision-qwen2-7b-si`.
- Corpus: [`traces/anomalythink_6k/`](../../traces/anomalythink_6k/), 6,000 Gemini-2.5-Flash traces.
- Recipe: SigLIP vision tower frozen, projector trained, 4 epochs. Checkpoints 188 / 376 / 564 / 748
  are epochs 1 / 2 / 3 / 4.
- Harness: DS-MVTec n=1670, VisA n=2141, `_trainprompt` mode.

## Balanced accuracy per epoch

| Epoch | Checkpoint | DS-MVTec BA | tp | tn | fp | fn | n | VisA BA | tp | tn | fp | fn | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **checkpoint-188** | **85.91** | 991 | 404 | 40 | 235 | 1670 | **68.26** | 749 | 698 | 246 | 448 | 2141 |
| 2 | checkpoint-376 | 81.95 | 1090 | 333 | 111 | 136 | 1670 | 64.35 | 1020 | 410 | 533 | 177 | 2140 |
| 3 | checkpoint-564 | 83.13 | 1130 | 329 | 115 | 96 | 1670 | 66.94 | 1023 | 457 | 487 | 174 | 2141 |
| 4 | checkpoint-748 | 82.78 | 1124 | 328 | 116 | 102 | 1670 | 67.44 | 1025 | 465 | 479 | 172 | 2141 |

Epoch 1 is the best epoch on both benchmarks and is the released checkpoint
([`aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT)).
It is also the init for the GRPO run in [`../grpo_llava_ov_from_ep1/`](../grpo_llava_ov_from_ep1/).

Epochs 2 to 4 trade recall for precision and lose balanced accuracy on both benchmarks. False
positives on VisA climb from 246 to 533 between epoch 1 and epoch 2. The model keeps learning to say
"yes", which is the same over-calling pattern the Qwen runs show past epoch 3.

## Where this sits in the LLaVA line

| Model on LLaVA-OneVision-7B-SI | DS-MVTec | VisA |
|---|---:|---:|
| Base, fair yes/no prompt | 75.66 | 53.80 |
| **This run, 6K Gemini SFT, epoch 1** | **85.91** | **68.26** |
| SFT then GRPO, checkpoint-530 | 87.66 | 72.58 |
| KCR corpus SFT, epoch 4 | **88.45** | **74.25** |

## Contamination caveat, DS-MVTec only

The LLaVA-OneVision training mixture contains 426 rows whose id matches `%MVTecAD%` (VisA: 0), so
every DS-MVTec cell for a LLaVA-derived model carries a pretraining-exposure asterisk. VisA does not.

## One filename to be aware of

`checkpoint-748/eval_dsmvtec_full_trainprompt.json` in this folder was produced on the HF inference
path. Its sibling in [`../sft_llava_ov_7b_frozen_llava_iter1_C/`](../sft_llava_ov_7b_frozen_llava_iter1_C/)
carries a `_vllm` tail because that run was served through vLLM. The prompt is identical in both
cases. On a probe the two serving paths agreed on 99 percent of samples, at 0.71 s per sample for
vLLM against about 9 s for the HF path.
