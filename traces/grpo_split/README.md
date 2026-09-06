# The GRPO training file

`grpo_train.json` (4,236 prompts, 2,118 anomalous / 2,118 normal by `gt_label`, 23 products, C1 only) is the
file every GRPO run in this repository trained on (`--dataset_name`, see `scripts/02_grpo/run_grpo_7b_resume_run2.sh`).
Fields: `image_id, image_path, product, is_anomaly, question, answer, gt_label`, where `answer` is the
reference trace used only for the format reward's gold verdict.

It is not the same file as `traces/anomalythink_15k/grpo_train.json`. That one holds the same 4,236 images
with regenerated traces and the SFT-style question and is the GRPO partition of the AnomalyThink release.
The thesis dataset counts (Chapter 3, Appendix A and I) hold for both. Rollout pass rates quoted in the thesis
(84.6 % per rollout, 90.8 % any-of-8) are over the 6,000 SFT images of `traces/rollout_pools/qwen_phase0_10k/`.
