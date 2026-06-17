# GRPO KL-Penalty Drift Test (0.5 epoch on Arm-C)

**Date:** 2026-06-17 · **GPUs:** CUDA 1,2 · **Init:** Arm-C SFT `checkpoint-376`
(82.80 DS-MVTec / 72.07 VisA) · **Script:** `scripts/03_rollout_star/run_grpo_kl_halfep.sh`

## Question

GRPO on top of the strong Arm-C SFT init (`grpo_qwen25vl_7b_abc_C_grpo`, β=0, 4 epochs)
did **not** improve over SFT — VisA drifted down over epochs. Does turning on a **KL
penalty** (β>0, anchoring to the SFT init) arrest that drift?

## Setup (identical to abc_C except the KL penalty)

- Same dataset (`grpo_train.json`, 4,236 prompts), same prompt, same G=4, same LR (1e-6).
- Reward = **accuracy + format** (dropped the inert `reasoning` term — its :5100 judge is
  down so it returned a flat ~0.5 that GRPO's per-group advantage normalisation cancels).
- Embed server :5200 (Nomic type-similarity) UP for `accuracy_reward`.
- **β = 0.1** (prior runs used β=0). `--max_steps 265` (= 0.5 epoch; 4,236/EBS 8 = 530/ep).
- 5 checkpoints saved (53/106/159/212/265) for a drift curve; each eval'd DS→cuda1, VisA→cuda2.

## Results

| step | KL=0.1 DS | KL=0.1 VisA | no-KL (abc_C) DS | no-KL VisA |
|---|---|---|---|---|
| init (Arm-C 376) | 82.80 | 72.07 | 82.80 | 72.07 |
| 53 | 80.97 | 69.92 | – | – |
| 106 | 80.76 | 69.14 | – | – |
| 159 | 81.75 | 70.49 | – | – |
| 212 | 81.87 | 70.24 | – | – |
| **265 (0.5ep)** | **81.90** | **70.01** | 80.61 | 72.68 |
| 530 (1ep) | – | – | 80.32 | 70.34 |
| 1060 (2ep) | – | – | 81.20 | 69.17 |
| 1855 | – | – | 78.81 | 67.58 |
| 2120 (4ep) | – | – | 80.38 | 67.86 |

## Findings

1. **KL=0.1 is stable over 0.5 epoch** — no within-run drift; DS rises 80.97→81.90,
   VisA holds ~70. So the literal "does it drift?" answer is **no**.
2. **Neither run beats the SFT init (82.80/72.07)** — GRPO on the strong Arm-C init
   yields no gain, KL or not. Consistent with over-optimization / near-data-ceiling.
3. **Caveat — the test is short.** The no-KL run's real damage is a long-horizon VisA
   decline (72.68 → 67.86 across epochs 1–4). At matched 0.5 epoch the two are comparable
   (KL 81.90/70.01 vs no-KL 80.61/72.68). So this 0.5-epoch run shows KL is *short-term
   stable* but is **too short to prove KL arrests the multi-epoch collapse** — that needs
   a 2–4 epoch KL run.

## Artifacts (not committed; weights excluded)

`outputs/grpo_abc_C_kl0.1_halfep/`: `train.log`, `ba_table.txt`, `checkpoint-{53,106,159,212,265}/`
(+ per-checkpoint `eval_{dsmvtec,visa}_full_trainprompt.json`), `done.flag`.
