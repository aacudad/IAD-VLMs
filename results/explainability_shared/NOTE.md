# Shared-set explainability judge (2026-09-06, 22:14)

Every row is scored on the same images: 90 DS-MVTec (6 per product, 15 products) and 47 VisA (4 per product, pcb1 has 3), drawn with seed 42 from the intersection of correctly detected anomalies of all nine rows (`shared_ids.json`). Same judge (Gemini-3-Flash, reference-guided, five axes 0 to 2, median of three samples) and same script family as `results/explainability_multi/` (`scripts/04_eval/explainability_judge_shared.py`, `--image-list`). Nine processes ran in parallel with the Vertex service account. Because the set is the intersection, it holds the defects every model detects, which are the easier ones; absolute scores are therefore higher than on the own-100 sets and the two protocols answer different questions.

| model | DS grounding | DS faith | DS evid | DS coh | DS conc | **DS /10** | DS loc | DS type | VisA grounding | VisA faith | VisA evid | VisA coh | VisA conc | **VisA /10** | VisA loc | VisA type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen SFT 6K | 1.82 | 1.51 | 1.99 | 1.92 | 1.92 | **9.17** | 0.77 | 0.56 | 1.72 | 1.36 | 2.00 | 1.87 | 1.74 | **8.70** | 0.38 | 0.55 |
| Qwen SFT+GRPO | 1.87 | 1.61 | 2.00 | 1.93 | 1.92 | **9.33** | 0.82 | 0.56 | 1.81 | 1.60 | 1.98 | 1.98 | 1.87 | **9.23** | 0.64 | 0.54 |
| Qwen KCR | 1.77 | 1.49 | 1.98 | 1.97 | 1.89 | **9.09** | 0.78 | 0.53 | 1.79 | 1.47 | 2.00 | 1.94 | 1.91 | **9.11** | 0.55 | 0.56 |
| Qwen3-VL-8B (Qwen KCR corpus) | 1.91 | 1.62 | 2.00 | 1.93 | 1.94 | **9.41** | 0.82 | 0.56 | 1.91 | 1.72 | 2.00 | 1.96 | 1.94 | **9.53** | 0.62 | 0.54 |
| LLaVA SFT | 1.88 | 1.68 | 1.99 | 1.93 | 1.88 | **9.36** | 0.78 | 0.58 | 1.68 | 1.53 | 1.89 | 1.81 | 1.66 | **8.57** | 0.49 | 0.56 |
| LLaVA SFT+GRPO | 1.88 | 1.77 | 1.99 | 1.92 | 1.91 | **9.47** | 0.81 | 0.55 | 1.89 | 1.74 | 1.98 | 1.91 | 1.89 | **9.43** | 0.70 | 0.53 |
| LLaVA KCR (corrected) | 1.93 | 1.77 | 2.00 | 1.97 | 1.94 | **9.61** | 0.86 | 0.56 | 1.87 | 1.70 | 2.00 | 1.94 | 1.83 | **9.34** | 0.60 | 0.55 |
| IAD-R1, our prompt | 1.13 | 0.84 | 1.63 | 1.56 | 0.97 | **6.13** | 0.74 | 0.55 | 1.47 | 1.06 | 1.74 | 1.79 | 0.89 | **6.96** | 0.68 | 0.55 |
| IAD-R1, own prompt | 0.84 | 0.62 | 0.97 | 0.93 | 0.99 | **4.36** | 0.34 | 0.30 | 1.26 | 1.04 | 1.68 | 1.64 | 0.87 | **6.49** | 0.62 | 0.53 |

## Paired differences against Qwen KCR on the same images (mean, 95 % CI, paired t p, Wilcoxon p)

| model | DS-MVTec | VisA |
|---|---|---|
| Qwen SFT 6K | +0.08 [-0.26, +0.42], p = 0.653 / 0.848 | -0.40 [-1.04, +0.23], p = 0.219 / 0.245 |
| Qwen SFT+GRPO | +0.24 [-0.05, +0.54], p = 0.109 / 0.121 | +0.13 [-0.39, +0.65], p = 0.632 / 0.567 |
| Qwen3-VL-8B (Qwen KCR corpus) | +0.32 [-0.03, +0.68], p = 0.078 / 0.116 | +0.43 [-0.04, +0.89], p = 0.079 / 0.078 |
| LLaVA SFT | +0.27 [-0.10, +0.63], p = 0.153 / 0.140 | -0.53 [-1.33, +0.27], p = 0.199 / 0.338 |
| LLaVA SFT+GRPO | +0.38 [+0.02, +0.73], p = 0.039 / 0.061 | +0.32 [-0.18, +0.82], p = 0.220 / 0.242 |
| LLaVA KCR (corrected) | +0.52 [+0.19, +0.85], p = 0.003 / 0.003 | +0.23 [-0.28, +0.75], p = 0.376 / 0.334 |
| IAD-R1, our prompt | -2.96 [-3.50, -2.41], p = 0.000 / 0.000 | -2.15 [-2.83, -1.47], p = 0.000 / 0.000 |
| IAD-R1, own prompt | -4.73 [-5.49, -3.98], p = 0.000 / 0.000 | -2.62 [-3.25, -1.98], p = 0.000 / 0.000 |

Among our seven rows (21 pairs per benchmark), paired t below 0.05: DS-MVTec 4 of 21, VisA 7 of 21 (`paired_tests.json`). The pattern: on VisA the two SFT-only rows (Qwen SFT 8.70, LLaVA SFT 8.57) sit below every row that went through GRPO or the KCR corpus (9.1 to 9.5), and on DS-MVTec the LLaVA KCR row is highest (9.61, +0.52 over Qwen KCR, p = 0.003). The gap to IAD-R1 is 2.2 to 3.0 points under our prompt on the identical images, p < 1e-6 on both benchmarks. Files: `raw_results_<model>.json`, `raw_results_all.json`, `summary_<model>.json`, `summary_shared.json`, `paired_tests.json`, `run_<model>.log`.