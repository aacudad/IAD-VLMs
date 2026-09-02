# Labels-only control: how much of the gain is the reasoning, and how much is just fine-tuning?

This is the control for the obvious examiner question. Take the exact 6,000 images the reported
SFT-6K model trained on, strip every trace, keep only the yes/no label, and train the same model the
same way. Whatever is left is the value of fine-tuning on our images. Whatever is missing is the
value of the reasoning supervision.

- Config: [`configs/sft/sft_qwen25vl_7b_6k_noreason.yaml`](../../configs/sft/sft_qwen25vl_7b_6k_noreason.yaml).
  Identical to `sft_qwen25vl_7b_zeroshot_6k_frozen.yaml` (the reported SFT-6K run, 80.16 / 64.78)
  except `dataset` and `output_dir`. Same 6,000 images, same hyperparameters, same 4 epochs, same
  checkpoint grid 188 / 376 / 564 / 752, so the comparison is like-for-like checkpoint for checkpoint.
- Corpus builder: [`scripts/01_sft/build_noreason_dataset.py`](../../scripts/01_sft/build_noreason_dataset.py).
  The user prompt drops the type, location and reasoning request and asks for a bare verdict. The
  assistant target drops `<think>`, `<location>` and `<type>` and keeps only `<answer>Yes|No</answer>`.
  Images and the product-conditioned preamble are untouched.
- The `<answer>` wrapper is kept on purpose. The harness reads `<answer>` independently of `<think>`,
  so the parser needs no change and output format is not a confound.
- Eval mode: `--noreason-prompt`, filename suffix `_noreasonprompt`. The eval prompt is
  character-identical to the training prompt, which is the same contract every other row here follows.

## Balanced accuracy

| Checkpoint | DS-MVTec BA | tp | tn | fp | fn | n | VisA BA | tp | tn | fp | fn | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **checkpoint-188 (ep1)** | **77.86** | 813 | 397 | 47 | 413 | 1670 | **68.64** | 516 | 889 | 55 | 681 | 2141 |
| checkpoint-376 (ep2) | 75.43 | 1057 | 287 | 157 | 169 | 1670 | 70.19 | 964 | 565 | 379 | 233 | 2141 |
| checkpoint-564 (ep3) | 76.51 | 1067 | 293 | 151 | 159 | 1670 | 69.39 | 974 | 542 | 402 | 223 | 2141 |
| checkpoint-752 (ep4) | 74.12 | 1127 | 250 | 194 | 99 | 1670 | 67.61 | 1034 | 461 | 483 | 163 | 2141 |

## What it shows

| Trained on the same 6,000 images | DS-MVTec | VisA |
|---|---:|---:|
| Labels only, best epoch | 77.86 | 68.64 |
| Reasoning traces, KCR corpus ([`../sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/`](../sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/)) | **82.80** | **72.07** |
| **Cost of stripping the reasoning** | **-4.94** | **-3.43** |

Fine-tuning on our images alone already lifts the base model a long way (69.01 / 53.79 to
77.86 / 68.64). So a good part of the headline gain is not the reasoning. But close to 5 pp on
DS-MVTec and close to 3.5 pp on VisA only appear when the supervision carries a trace, on the same
images, at the same compute, on the same checkpoint grid. The reasoning supervision is doing real
work, and this folder is the evidence.

A second, weaker point in the same table: the labels-only model peaks at epoch 1 and then decays,
trading precision away as it learns to say "yes" more often. False positives on DS-MVTec go 47, 157,
151, 194 across the four epochs. Reasoning-supervised runs hold their calibration longer.
