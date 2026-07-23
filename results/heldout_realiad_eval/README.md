# Held-out Real-IAD — the generalisation test

Balanced accuracy on the **held-out Real-IAD split** (n=4,236; the third split, disjoint from
the SFT and rollout pools — see `docs/heldout_4k_rollout.md`). Run via `--realiad-4k`, each model
in its **native prompt mode** (IAD-R1 = grpoprompt, the rest = trainprompt).

| Model | BA | recall | specificity | n |
|---|--:|--:|--:|--:|
| SFT+GRPO ckpt-530 | **80.87** | 68.3 | 93.4 | 4236 |
| Qwen3-VL-8B SFT (ckpt-376) | **83.81** | 73.6 | 94.0 | 4236 |
| IAD-R1 | **79.58** | 70.0 | 89.1 | 4236 |
| Arm-C (final SFT, ckpt-376) | **79.32** | 65.9 | 92.7 | 4236 |
| base Qwen3-VL-8B | **56.38** | 13.2 | 99.6 | 4236 |
| base Qwen2.5-VL-7B | **50.00** | 0.0 | 100.0 | 4236 |

**Reading:** both base models collapse to roughly chance (50.0 / 56.4) on Real-IAD they have never
seen, while every fine-tuned model lands at ~79-84. That is the generalisation claim, measured on a
split none of them trained on. (SFT+GRPO's 80.87 was already in the repo; the rest are new.)
