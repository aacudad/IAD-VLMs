# GRPO from the 6K SFT checkpoint with the vision tower frozen (ablation, in progress on 2026-09-06)

Every other GRPO run in this repository trains the vision tower (verified from the weight deltas of run 2).
This run freezes it (`FREEZE_VISION_TOWER=1`, env-guarded block in
`scripts/02_grpo/stage_rl/trainer/sc_grpo_trainer.py`, launcher `scripts/02_grpo/run_grpo_7b_frozen_vision.sh`).
Same data, reward, G = 4, beta 0, lr 1e-6, 2 epochs, model-only saves at 265 / 530 / 795 / 1060.
`train_partial_20260906.log` is the log up to the time of this commit. Evaluations land in
`checkpoint-*/eval_*.json` when the watcher (`scripts/04_eval/watch_and_eval_grpo_frozenvision.sh`) finishes.
