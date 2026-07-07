# checkpoint-954 — eval-aligned GRPO on the Arm-C init (NOT a thesis result)

**This is future work and is deliberately NOT reported in the thesis.** It is shipped here
only as supporting evidence for the future-work discussion. The thesis headline model remains
the **Arm-C SFT** checkpoint (DS-MVTec **82.80** / VisA **72.07**, avg 77.44). See
[`THESIS_HEADLINE_DECISION.md`](../../../THESIS_HEADLINE_DECISION.md).

## What this run is

GRPO applied **on top of the strong Arm-C SFT init** (Arm-C ckpt-376), with the RL training
prompt **aligned to the evaluation prompt** (`sft_sys`: system "Please answer by yes or no" +
Analyze), KL penalty β = 0.1, ~3 epochs. Reward = accuracy + consistency (unweighted, max 3.0).

- Training script: [`scripts/03_rollout_star/run_grpo_sftprompt_3ep.sh`](../../../scripts/03_rollout_star/run_grpo_sftprompt_3ep.sh)
- Per-checkpoint sweep + KL/reward analysis: [`docs/grpo_sftprompt_runs.md`](../../../docs/grpo_sftprompt_runs.md), [`docs/STATUS_HANDOFF.md`](../../../docs/STATUS_HANDOFF.md) §2
- Reward/KL/BA curve: [`../reward_kl_ba_curve.png`](../reward_kl_ba_curve.png)

## Balanced accuracy (checkpoint-954, best average of this run)

Recompute either cell with [`results/compute_ba.py`](../../compute_ba.py) from the
`tp/tn/fp/fn` block of each JSON below.

| Benchmark | BA | tp | tn | fp | fn | n | Eval JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| DS-MVTec | **82.95** | 1004 | 373 | 71 | 222 | 1670 | [`eval_dsmvtec_full_trainprompt.json`](eval_dsmvtec_full_trainprompt.json) |
| VisA | **72.62** | 719 | 804 | 140 | 478 | 2141 | [`eval_visa_full_trainprompt.json`](eval_visa_full_trainprompt.json) |

Average **77.78**, versus the Arm-C init's 77.44. It *edges* the ceiling on average (best DS
82.95 and best VisA of the run 73.70 each slightly above Arm-C), but the trajectory bounces
across epoch 3 rather than climbing monotonically, so the gain is small and not yet clearly
stable. That is exactly why it is held as future work and not written up as "GRPO beat Arm-C".

## Weights

The checkpoint-954 weights are **not** published on Hugging Face. The released `*-SFT-GRPO`
model ([`aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO`](https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO))
is a **different** run (GRPO on the SFT-6K init, ckpt-530, 82.73 / 70.39). The eval JSONs in this
folder are the reproducible artifact for the 82.95 / 72.62 numbers.
