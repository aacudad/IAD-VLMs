# Prompt-aligned GRPO on the Arm-C initialisation, beta = 0.1, 3 epochs (complete, 15 checkpoints)

Init: Arm-C ckpt-376 (82.80 / 72.07). Training prompt = the evaluation prompt with the system turn
(`--prompt_style sft_sys`), beta 0.1, lr 1e-6, G = 4, 1,590 steps, save every 106.
Launcher `scripts/02_grpo/run_grpo_sftprompt_3ep.sh`.

| ckpt | DS-MVTec | VisA |
|---|---:|---:|
| 106 | 80.18 | 69.51 |
| 212 | 81.52 | 70.22 |
| 318 | 81.83 | 72.25 |
| 424 | 81.49 | 73.01 |
| 530 | 82.88 | 71.34 |
| 636 | 81.41 | 73.70 |
| 742 | 82.35 | 72.17 |
| 848 | 80.75 | 72.73 |
| 954 | 82.95 | 72.62 |
| 1060 | 81.59 | 72.25 |
| 1166 | 81.50 | 72.23 |
| 1272 | 80.64 | 72.08 |
| 1378 | 80.26 | 72.34 |
| 1484 | 80.92 | 72.01 |
| 1590 | 81.25 | 71.51 |

Two of fifteen checkpoints are above the initialisation on DS-MVTec (530, 954) and eleven on VisA. The
best, ckpt-954, is +0.15 / +0.55 over the initialisation and the run ends below it. Single seed. The
thesis (§6.8 and §8.4) records this as inconclusive and leaves a prompt-aligned GRPO stage to future work.
The production GRPO-on-C run the thesis tables report is `results/grpo_qwen25vl_7b_abc_C_grpo/` (beta 0,
GRPO prompt), where every checkpoint is below the initialisation on DS-MVTec.
