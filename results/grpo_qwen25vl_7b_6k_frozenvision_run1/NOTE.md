# GRPO with the vision encoder frozen (thesis Appendix N, reference run)

Same recipe as `grpo_qwen25vl_7b_6k_frozen_ep3_full_run2` (SFT+GRPO), started from the same SFT ckpt-564, two epochs,
checkpoints every 265 steps, launcher `scripts/02_grpo/run_grpo_7b_frozen_vision.sh` with `FREEZE_VISION_TOWER=1`.
The flag freezes the 385 vision-tower tensors and wraps the tower forward in no_grad; the merger (5 tensors) sits inside
that forward and receives no gradient either, so only the language model trains. Verified on the saved weights
(ckpt-795: all 385 tower and 5 merger tensors identical to the SFT init; the SFT+GRPO ckpt-530 differs on 322 of 385).

Evals by `scripts/04_eval/watch_and_eval_grpo_frozenvision.sh`, GRPO prompt (same as the SFT+GRPO row), strict scoring:

| checkpoint | DS-MVTec | VisA |
|---|---|---|
| 265 | 80.57 | 67.25 |
| 530 | 82.15 | 68.95 |
| 795 | 82.08 | 69.81 |
| 1060 | 81.94 | 69.71 |

SFT init 80.16 / 64.78, SFT+GRPO ckpt-530 82.73 / 70.39. Final KL 0.116. Files: `checkpoint-*/eval_{dsmvtec,visa}_full_trainprompt.json`
(the suffix is the watcher's, the prompt mode inside is `grpoprompt`), `train.log`. Not referenced from the thesis chapters.
