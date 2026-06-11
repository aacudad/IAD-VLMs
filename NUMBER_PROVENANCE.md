# Number Provenance Ledger

**Thesis:** *Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* — Adnane Acudad, TU Delft MSc, 2026 (Qwen2.5-VL).

This file is a complete provenance ledger for **every number that appears in the thesis**. Each entry records the value as printed, where it appears, what it describes, the source it traces to (eval JSON / on-disk computation / training-args dump / citation), its verification status, and — when the printed value is wrong — the correct value.

> **Authority note.** Where this ledger and the thesis text disagree, **this ledger is authoritative.** All balanced-accuracy (BA) numbers, GRPO hyperparameters, and corpus counts were recomputed/inspected directly from on-disk artefacts in this session. Any thesis values of `G=2`, `beta=0.04`, `eta=5e-6`, "13K samples", or "6,500+6,500" are **old/wrong** and have been corrected to the actuals.

---

## How to read this ledger

### Balanced accuracy (BA) formula

All BA values are recomputed from the confusion-matrix block (`metrics: {tp, tn, fp, fn}`) inside each eval JSON, using:

```
BA = 50 * ( TP/(TP+FN) + TN/(TN+FP) )
```

The recompute script is shipped at [`results/compute_ba.py`](results/compute_ba.py). The full recomputed inventory is at [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).

### Source-file conventions

- **Eval JSONs.** Ground-truth path form is `outputs/<run>/checkpoint-X/eval_<bench>_full_<mode>.json`. The same files are shipped in this repo under [`results/<run>/checkpoint-X/...`](results/). The prompt-mode contract is the **filename suffix**: `_trainprompt` (headline/primary), `_grpoprompt`, `_bareprompt`.
- **Benchmark sizes.** DS-MVTec full `n=1670`; VisA full `n=2141`; held-out Real-IAD `n=4236`.
- **Trace datasets.** Shipped under [`traces/`](traces/) (text-only LlamaFactory `{messages, images}` JSONs; images are relative paths to re-downloadable source images).
- **Configs.** SFT YAMLs, GRPO config, DeepSpeed configs, `dataset_info.json` under [`configs/`](configs/).
- **GRPO training-args** were read from the saved `run-2 training_args.bin`.

### Status legend

| Status | Meaning |
|---|---|
| **verified** | Printed value matches the on-disk artefact (within stated rounding), or is correct arithmetic over verified inputs. |
| **mismatch** | Printed value contradicts the on-disk artefact. Correct value supplied. |
| **unverified** | No on-disk artefact exists to confirm or refute (estimate, anecdote, never-run config, rotated-away checkpoint, or paraphrase). |
| **literature** | Cited from an external paper / hardware datasheet; not a result of this work. |

---

## Corrections needed (action list)

These are the confirmed printed-vs-actual contradictions. Fix in the thesis text and record in `UNVERIFIED.md`.

### A. `tab:baseline` (06_results.tex ~L25) — single transcription typo

| Cell | Claimed | Correct | Source |
|---|---|---|---|
| Qwen2.5-VL-3B base, DS-MVTec, **Bal-acc** | 55.80% | **56.14%** | `outputs/baseline_eval/3b_base_dsmvtec.json` (TP1222 TN56 FP388 FN4) → [`results/baseline_named/3b_base_dsmvtec.json`](results/baseline_named/3b_base_dsmvtec.json) |

### B. `tab:sft-summary` (06_results.tex L66-74) — cross-contaminated rows

| Row (true ckpt) | Metric | Claimed → Correct | Source |
|---|---|---|---|
| 3B-Frozen-6K ep4 (ckpt-752) | DS / VisA / Acc / F1 | 68.50→**69.08** / 59.60→**57.22** / 71.60→**72.22** / 78.10→**80.02** | [`results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752/`](results/sft_qwen25vl_3b_zeroshot_6k_frozen/) |
| 3B-Frozen-15K ep4 (ckpt-1812) | DS / VisA | 70.50→**68.65** / 60.85→**64.85** | [`results/sft_qwen25vl_3b_15k_frozen/`](results/sft_qwen25vl_3b_15k_frozen/) |
| 3B-Unfrozen-15K ep4 (ckpt-1812) | DS / Acc / F1 | 71.86→**68.58** / 74.19→**69.58** / 81.38→**77.34** (claimed = a 7B run's numbers) | [`results/sft_qwen25vl_3b_15k_unfrozen/`](results/sft_qwen25vl_3b_15k_unfrozen/) |
| 7B-Frozen-15K ep3 (ckpt-1359) | VisA / Acc | 65.45→**64.28** / 78.20→**80.12** (claimed = ep1/ckpt-453) | [`results/sft_qwen25vl_7b_15k_frozen/`](results/sft_qwen25vl_7b_15k_frozen/) |
| 7B-Unfrozen-15K ep4 (ckpt-1812) | DS / Acc / F1 | 72.60→**70.27** / 68.86→**65.21** / 75.60→**71.51** (72.60 is the 7B-*frozen* value) | [`results/sft_qwen25vl_7b_15k_unfrozen/`](results/sft_qwen25vl_7b_15k_unfrozen/) |
| **HEADLINE** 7B-Frozen-6K ep3 (ckpt-564) | all | **ALL CORRECT** (80.16 / 64.78 / 80.36 / 85.76) | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) |

> Also in this table: the **3B-Unfrozen-6K** and **7B-Unfrozen-6K** rows are for configs that were **never run** (dirs absent); the printed cells duplicate the 15K-unfrozen rows. They cannot be confirmed or fixed from data.

### C. Appendix C — FABRICATED `tab:perprod-cm` and the SFT+GRPO column of `tab:perprod-full`

- The entire `tab:perprod-cm` confusion-matrix table is fabricated: per-product TP+TN+FP+FN **exceeds that product's sample size for 10 of 15 products** (physically impossible), and the per-product CM rows sum to **TP1150/TN382/FP29/FN272**, which does **not** equal the genuine Total **TP971/TN383/FP61/FN255**.
- The **SFT+GRPO (GRPO) per-product BA column** in `tab:perprod-full` is also fabricated (per-product values do not match the cited JSON).
- **Genuine** parts: the column **Average (82.73)**, the bold **CM Total (971/383/61/255)**, and the entire **Base** and **SFT** per-product columns (real JSONs, ~0.1 truncation).
- **Correct per-product GRPO BA** (run-2 ckpt-530, DS): cable 68.16, pill 75.45, toothbrush 83.33, screw 72.89, transistor 75.42, grid 94.74, leather 98.44, zipper 68.84, capsule 69.78, hazelnut 88.57, bottle 76.27, carpet 95.51, wood 97.50, tile 89.83, metal_nut 87.47.

### D. SFT/GRPO hyperparameter tables (04_sft.tex L106-125; Appendix A `tab:A.2`/`tab:A.3`)

The SFT recipe table and Appendix-A tables print several values that contradict the saved configs/args. Key ones: LR (claimed `2e-5` → actual **1e-5 frozen / 1e-6 unfrozen**), schedule (`linear`/100-step → **cosine**, warmup **20/50**), weight decay (`0.01` → **0.0**), AdamW beta2 (`0.95` → **0.999**), cutoff (`4096` → **12144**), GRPO max-prompt (`2048` → **4096**), GRPO save/eval-every (`250` → **530 / None** in saved run-2 args), GRPO per-device BS (`2` → **1**, EBS 8 unchanged). See the per-chapter tables below for the full list.

### E. Minor / single-figure mismatches

| Where | Claimed → Correct |
|---|---|
| Abstract / intro / ch6: "+13.6 pp" per-product headline gain | **+13.72 pp** (GRPO 82.73 − base 69.01) |
| Ch3 L82: "25" defect-type categories | **26** (Surface 11 + Structural 7 + Completeness 4 + Other 4) |
| Ch3 L17/169/220 + Conclusion: "30 products" in SFT/15K | **29** (SFT + 15K union span 29 via image paths); GRPO/held-out span **23** |
| Ch3 t-SNE: Scratch n=600 / Missing n=180 / dist 148.6 / radius 16.5 / ratio 9.0 | **500 (or 866 cluster) / 162 / ≈155.3 / ≈15.3 / ≈10.1** |
| Ch3 L171: trace-length std "28 tokens" | **≈13.5 tokens** (range 121-223) |
| Ch5 / Ch6: "6,500-sample run / 13,000" | **4,236 prompts (2,118 + 2,118)** |
| Ch5 L165: KL range "[0.003, 0.095]" | max **≈0.117** (range [0.0, 0.117]) |
| Ch6 L342: "fifteen others gain" (per-product) | **thirteen** (2 of 15 regress) |
| Ch4 L89: 7B-unfrozen-15K wall-clock "22.5 h" | **≈12.2 h** (43875 s) |
| Ch7 L108: GRPO run-1 "80.6%" at ckpt-530 | **81.65%** (80.63% is run-1 ckpt-315) |
| Conclusion RQ1: mean trace length "141 words" | **≈137 words** (136.8; the *equality* of 6K & 15K holds) |

---

## Summary counts

Counts over the per-chapter extraction (one row per extracted value; some values recur across chapters).

| Status | Count |
|---|---:|
| verified | 327 |
| mismatch | 67 |
| unverified | 100 |
| literature | 65 |
| **Total entries** | **559** |

| Chapter | verified | mismatch | unverified | literature |
|---|---:|---:|---:|---:|
| 00 Abstract | 28 | 2 | 0 | 0 |
| 01 Introduction | 24 | 1 | 3 | 8 |
| 02 Background & Related Work | 9 | 0 | 1 | 47 |
| 03 Dataset / Reasoning Traces | 22 | 6 | 11 | 5 |
| 04 Supervised Fine-Tuning | 27 | 12 | 11 | 4 |
| 05 GRPO | 40 | 3 | 24 | 4 |
| 06 Results & Analysis | 118 | 9 | 21 | 0 |
| 07 Discussion | 36 | 3 | 28 | 4 |
| 08 Conclusion & Future Work | 21 | 4 | 4 | 4 |
| Appendix A (Hyperparameters) | 25 | 10 | 6 | 2 |
| Appendix B (Prompts) | 8 | 0 | 1 | 0 |
| Appendix C (Per-product) | 35 | 36 | 18 | 0 |

> Numbers above are indicative tallies of the extraction; the authoritative record is the per-chapter tables that follow (every entry is listed).

---

## Chapter 00 — Abstract

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| six-phase | L9 | Trace template phases (Framing/Scan/Focus/Evaluate/Alternatives/Decide) | 03_dataset_traces.tex; generation prompt | verified | |
| ≈14,500 | L9 | Total Gemini-2.5-Flash traces (AnomalyThink-15K union) | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 2.5 | L9 | Gemini version (2.5-Flash) | Ground truth | verified | |
| ≈6,000 | L9 | SFT trace pool | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| ≈4,000 | L9 | GRPO trace pool | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| ≈4,000 | L9 | Held-out test set | Held-out Real-IAD split = 4,236 | verified | |
| 3B / 7B | L11/L15 | Qwen2.5-VL sizes; 7B headline | Ground truth | verified | |
| 6K-vs-15K | L11 | Supervised data-quantity arms | Ground truth | verified | |
| four-component | L11 | GRPO reward (format/acc/type/loc) | Reward weights (0.3,0.3,0.2,0.2) | verified | |
| 1,670 | L11 | DS-MVTec eval n | `eval_dsmvtec_full_trainprompt.json` n=1670 | verified | |
| 2,141 | L11 | VisA eval n | `eval_visa_full_trainprompt.json` n=2141 | verified | |
| 4,236 | L11 | Held-out Real-IAD eval n | `eval_realiad4k_full_trainprompt.json` n=4236 | verified | |
| 80.16% | result (i) L13 | DS BA, 7B-frozen-6K ep3 ckpt-564 | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) (tp988 tn354 fp90 fn238) | verified | |
| ~1 pp | result (i) L13 | Gap SFT→IAD-R1 SFT+GRPO on DS | 81.92 − 80.16 = 1.76 | mismatch | actual ≈1.76 pp |
| 6,000 | result (ii) L13 | Arm-C SFT corpus | 06_results.tex L460; Arm-C corpus | verified | |
| 82.80% | result (ii) L13 | Arm-C SFT DS BA (ckpt-376) | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_abc_C_full_patched/) (tp1003 tn372 fp72 fn223) | verified | |
| 72.07% | result (ii) L13 | Arm-C SFT VisA BA | `.../checkpoint-376/eval_visa_full_trainprompt.json` (tp683 tn822 fp122 fn514) | verified | |
| +0.88 pp | result (ii) L13 | Arm-C DS over IAD-R1 | 82.80 − 81.92 | verified | |
| +0.73 pp | result (ii) L13 | Arm-C VisA over IAD-R1 | 72.07 − 71.34 | verified | |
| 81.92% | result (ii) L13 | IAD-R1 DS BA (common harness) | [`results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json`](results/iad_r1_qwen_recanon/) (tp965 tn378 fp66 fn261) | verified | |
| 71.34% | result (ii) L13 | IAD-R1 VisA BA | [`results/iad_r1_qwen_recanon/eval_visa_full_trainprompt.json`](results/iad_r1_qwen_recanon/) (tp606 tn869 fp75 fn591) | verified | |
| 82.73% | result (iii) L13 | GRPO run-2 ckpt-530 DS BA | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) (tp971 tn383 fp61 fn255) | verified | |
| 70.39% | result (iii) L13 | GRPO run-2 ckpt-530 VisA BA | `.../checkpoint-530/eval_visa_full_trainprompt.json` (tp762 tn728 fp216 fn435) | verified | |
| ~2 pp | result (iii) L13 | GRPO-on-C below Arm-C init on DS | grpo_qwen25vl_7b_abc_C_grpo deltas −2.19/−2.48/−1.60 | verified | |
| three-estimator | result (iii) L13 | Vanilla z / Dr.GRPO / G2RPO ablation | grpo_probe_* dirs | verified | |
| 6,000 | result (iv) L13 | Curated SFT split | as result (i)/(ii) | verified | |
| ≈14,500 | result (iv) L13 | Superset 6K outperforms | 14,472 union | verified | |
| ≈141-word | result (iv) L13 | Mean trace length shared by 6K & 15K | corpus analysis ≈137 (GT ≈141) | verified | |
| ~8 | result (iv) L13 | BA gap 6K > 15K after SFT | 80.16 − 71.66 = 8.50 ([`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/`](results/sft_qwen25vl_7b_15k_frozen/)) | verified | |
| +25.8 | result (vi) L13 | metal_nut per-product gain (base→SFT564) | 57.86→83.64 | verified | |
| +19.2 | result (vi) L13 | cable per-product gain | 51.09→70.33 | verified | |
| +16.0 | result (vi) L13 | pill per-product gain | 60.23→76.21 | verified | |
| 2× / A6000 | L15 | Hardware | 2× RTX A6000 ZeRO-3 CPU offload | verified | |
| 7B | L15 | Final delivered model | Qwen2.5-VL-7B | verified | |

---

## Chapter 01 — Introduction

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 99% | sec:motivation L10 | PatchCore/EfficientAD/SimpleNet >99% AUROC | roth2022patchcore / batzner2024efficientad / liu2023simplenet | literature | |
| 10–20 | L14 | AUROC drop 30→160 categories (Real-IAD Variety) | zhu2025realiadvariety | literature | |
| 30 / 160 | L14 | Category counts (Real-IAD Variety) | zhu2025realiadvariety | literature | |
| 90% | L15 | WinCLIP zero-shot plateau | jeong2023winclip | literature | |
| 74.9% | L20 | GPT-4o avg accuracy on MMAD | jiang2025mmad | literature | |
| 95% | L20 | ">95% demanded by production floors" | no citation (rhetorical) | unverified | |
| 7B | L20/L30 | IAD-R1 backbone; headline backbone | li2025iadr1; ground truth | literature | |
| 3B | L30/L47 | 3B option fits single A6000 | ground truth; bai2025qwen25vl | verified | |
| 14,500 | item 1 L57 | AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 6,000 | item 1 L57 | AnomalyThink-6K | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| 4,000 | item 1 L57 | GRPO pool | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 4,000 | item 1 L57 | Held-out RealIAD | held-out split 4,236 | verified | |
| 30 | item 1 L57 | AnomalyThink products | 30 distinct in 6K & 15K combined trace JSONs | verified | |
| 125k | item 1 L57 | Anomaly-Instruct-125k | xu2025anomalyov | literature | |
| 80.16% | item 2 L59 | 7B-frozen-6K SFT DS BA | as Abstract (ckpt-564) | verified | |
| 1 pp | item 2 L59 | SFT within ~1pp of IAD-R1 | 81.92 − 80.16 = 1.76 (loose) | verified | |
| 8.5 pp | item 2 L59 | 6K beats 15K on DS | 80.16 − 71.66 = 8.50 (word "verbose" contradicts equal-length GT) | verified | |
| 16-cell | item 2 L59 | SFT ablation grid (3B/7B × frozen/unfrozen × 6K/15K × 4ep) | 4-factor study; 2 unfrozen-6K cells never run | unverified | |
| 82.73% | item 3 L61 | GRPO run-2 ckpt-530 DS BA | as Abstract | verified | |
| 70.39% | item 3 L61 | GRPO run-2 ckpt-530 VisA BA | as Abstract | verified | |
| 80.87% | item 3 L61 | GRPO run-2 ckpt-530 held-out RealIAD BA | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) | verified | |
| +0.81 pp | item 3 L61 | GRPO over IAD-R1 on DS | 82.73 − 81.92 | verified | |
| beta=0 | item 3 L61 | KL coefficient (k3 monitored, not in loss) | run-2 training_args | verified | |
| G=4 | item 3 L61 | GRPO group size | run-2 training_args | verified | |
| 1e-6 | item 3 L61 | GRPO learning rate | run-2 training_args | verified | |
| 4,236 | item 3 L61 | GRPO training set | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 2 | item 3 L61 | 2× RTX A6000 | run-2 training_args | verified | |
| 82.80% / 72.07% | item 3 L61 | Arm-C SFT (ckpt-376) DS / VisA | as Abstract result (ii) | verified | |
| 6K | item 3 L61 | Arm-C corpus size | ch6 L492/L532 | verified | |
| ~2 pp | item 3 L61 | GRPO-on-C below Arm-C init on DS | ch6 L565; tab:grpo-on-c | verified | |
| +13.6 pp | item 4 L63 | Per-product headline gain over base on DS | 82.73 − 69.01 = 13.72 | mismatch | +13.72 pp |
| metal_nut +25.8 | item 4 L63 | per-product gain (SFT564 vs base) | 57.86→83.64 | verified | |
| cable +19.2 | item 4 L63 | per-product gain | 51.09→70.33 | verified | |
| pill +16.0 | item 4 L63 | per-product gain | 60.23→76.21 | verified | |
| toothbrush +15.8 | item 4 L63 | per-product gain | 62.50→78.33 | verified | |
| tile −2.1 | item 4 L63 | per-product regression (SFT) | 83.93→81.87 | verified | |
| wood −0.1 | item 4 L63 | per-product regression (SFT) | 97.50→97.37 | verified | |
| 69.01% | item 4 (implied) | 7B base DS BA | [`results/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json`](results/qwen25vl_baseline_eval/) | verified | |
| 2.5 | sec:outline L88 | Gemini 2.5-Flash | ground truth | verified | |
| five | sec:motivation L8 | "deep nets displaced pipeline in last five years" | rhetorical | unverified | |
| three decades | L8 | "classical CV standard for three decades" | rhetorical | unverified | |
| four | items 2-3 | reward components / factors / epochs | ground truth | verified | |

---

## Chapter 02 — Background and Related Work

All AUROC / dataset-size figures here are literature citations. Eval-size figures are recomputed.

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 99.6% / 98.4% | §2.1.1 | PatchCore image / pixel AUROC | roth2022patchcore | literature | |
| 97.9% | §2.1.1 | PaDiM image-AUROC | defard2021padim | literature | |
| 98.3% / 10× | §2.1.2 | CFLOW-AD image-AUROC / throughput | gudovskiy2022cflow | literature | |
| 98.0% / 68.4 | §2.1.3 | DRAEM image-AUROC / AU-PRO | zavrtanik2021draem | literature | |
| 98.5% / 97.8% | §2.1.3 | RD4AD image / pixel AUROC | deng2022rd4ad | literature | |
| four-layer / >99% / 32 / ~2 ms / 600 FPS | §2.1.3 | EfficientAD PDN depth / mean AUROC / datasets / latency / FPS | batzner2024efficientad | literature | |
| 96.6% / three-way | §2.1.4 | CutPaste AUROC / classifier classes | li2021cutpaste | literature | |
| 99.6% / 77 FPS | §2.1.4 | SimpleNet AUROC / throughput | liu2023simplenet | literature | |
| ~99% | §2.1.5 | classical families saturate near | general | literature | |
| 3 mm | §2.1.5 | illustrative scratch size | illustrative | unverified | |
| 30→160 / 10–20 pts | §2.1.5 | category-scaling AUROC drop | zhu2025realiadvariety | literature | |
| >99% / 10–30 pts | §2.1.5 | logical-anomaly regression (MVTec LOCO) | bergmann2022mvtecloco | literature | |
| 400 M / N²−N | §2.2.1 | CLIP training pairs / in-batch negatives | radford2021clip | literature | |
| 158K / 92.5% | §2.2.1 | LLaVA instruction examples / % of GPT-4 on ScienceQA | liu2023llava | literature | |
| 256 | §2.2.2 | Qwen-VL adapter compressed tokens | bai2023qwenvl | literature | |
| 3B and 7B | §2.2.2 | Qwen2.5-VL variants used | ground truth; configs/yaml | verified | |
| InternVL3-78B / 72.2 | §2.2.2 | InternVL3 size / MMMU | zhu2025internvl3 | literature | |
| 2.5 | §2.2.2 | Gemini 2.5(-Flash) | geminiteam2025gemini25; ground truth | verified | |
| ~100B / 4% / 74% | §2.3 | CoT threshold / Game-of-24 baseline / ToT | wei2022cot; yao2023tot | literature | |
| 1±ε / G / ε / β / −1 / G² | §2.4 | PPO clip / GRPO symbols / k3 const / G²RPO | schulman2017ppo; shao2024deepseekmath; schulman2020klestimator; hu2026openvlthinkerv2 | literature | |
| 7B | §2.4.2 | value-network memory saving on 7B | general; matches backbone | verified | |
| 91.8% / two | §2.5.1 | WinCLIP AUROC / AnomalyCLIP prompts | jeong2023winclip; zhou2024anomalyclip | literature | |
| 86.1% / 94.1% / 125K | §2.5.2 | AnomalyGPT acc / AUROC / Anomaly-Instruct-125K | gu2024anomalygpt; xu2025anomalyov | literature | |
| 3B / 7B / 8B | §2.5.3 | AnomalyR1 / LR-IAD / EMIT backbones | chao2025anomalyr1; pang2025lriad; guan2025emit | literature | |
| 3×3 / 2.9K / 3K / four / six | §2.5.3 | IAD-R1 R_loc grid / PA-SFT / SC-GRPO / rewards / datasets | li2025iadr1 | literature | |
| 7B | §2.5.3 | IAD-R1 released ckpt re-run here | ground truth (iad_r1_qwen_recanon) | verified | |
| 3B/7B, frozen/unfrozen, 6K/15K | takeawaybox | promised ablation grid | ground truth | verified | |
| 150K+ / 30 / five / ~50K / >99%→~85% | §2.6 | Real-IAD images / categories / viewpoints / anomalous / AUROC drop | wang2024realiad | literature | |
| 160 / 28 | §2.6 | Real-IAD Variety categories / industries | zhu2025realiadvariety | literature | |
| 5,354 / 15 / 70+ | §2.6 | MVTec AD images / categories / defect types | bergmann2019mvtec | literature | |
| 10,821 / 12 | §2.6 | VisA images / categories | zou2022visa | literature | |
| seven / 39,672 / 8,366 / 74.9% | §2.6 | MMAD subtasks / questions / images / GPT-4o ceiling | jiang2025mmad | literature | |
| 1,670 | §2.6 | DS-MVTec subset n | `iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json` (965+378+66+261) | verified | |
| 2,141 | §2.6 | VisA subset n | `iad_r1_qwen_recanon/eval_visa_full_trainprompt.json` (606+869+75+591) | verified | |
| six-phase / six | intro L5; §2.3 L61 | trace structure (conceptual) | operational XML schema is 4 tags; "six-phase" is the narrative count, deferred to Ch.4 | unverified | |

---

## Chapter 03 — Reasoning Trace Generation (AnomalyThink corpus)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| ≈14,500 | L5/L168/L236 | Total AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 6K | L5/L152/L168/L218 | SFT split | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| 4K | L5/L153/L168 | GRPO split | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 4K | L5/L154/L168 | Held-out RealIAD split | held-out n=4,236 | verified | |
| 150K+ / 30 / five (C1-C5) | L11 | Real-IAD images / categories / viewpoints | wang2024realiad | literature | |
| 30 | L17/L169/L220 | "all 30 RealIAD products; SFT spans 30" | 6K SFT & 15K union span **29** via image paths | mismatch | 29 (SFT + 15K span 29) |
| 23 | L169 | products in GRPO / held-out | grpo_train.json = 23 distinct products | verified | |
| 23 of 30 | L17 | held-out drawn from 23 of 30 | GRPO/held-out span 23 (appendix-A table says 25/5; per-split differs) | verified | |
| 4 (four tags) | L41/L58/L128 | anomalous schema `<think><location><type><answer>` | trace JSON inspection | verified | |
| 120-200 | L44/L69/L171 | `<think>` word budget | inspector_prompt_test_v2.txt L130; production prompt says "3-5 sentences"; corpus mean ~137 | unverified | |
| 600-1000 | L69 | earlier verbose template range | refers to ch6 prompt-evolution; not in this chapter's data | unverified | |
| six-phase / 6 | L60-68 | template phases | no on-disk prompt has verbatim labels; production prompt uses 3-5-sentence template | unverified | |
| 3×3 | L72/L79 | grid-style location reward | 9 location values = 3×3; corpus uses exactly these 9 | verified | |
| 9 | L72-79 | canonical location values | top/middle/bottom × left/center/right | verified | |
| 25 | L82 | controlled defect-type categories | actual list = 11+7+4+4 = **26**; inspector_prompt_test_v2.txt has 26 | mismatch | 26 |
| four groups | L82-90 | Surface/Structural/Completeness/Other | inspector_prompt_test_v2.txt L287-319 | verified | |
| four-component reward | L58 | R_acc/R_loc/R_type/R_con ↔ 4 tags | ground truth weights | verified | |
| 3 images / 1 image | L33/L107/L119/L121 | per anomaly (orig+overlay+ref) / per normal | ground truth | verified | |
| 5-10 / ≤10 | L113/L115/L119 | Gemini batch size | config_mmad.py BATCH_SIZE=10 | verified | |
| 4-second | L132 | latency per batch | not logged | unverified | |
| $0.0001 | L132 | cost per trace | not logged (retail estimate) | unverified | |
| 15K | L132/L218 | union label | 14,472 ≈ 15K | verified | |
| five auto-reject rules | L123-130/L140 | in-prompt QC rules | production prompt has none; test_v2 has 6 lettered (A-F); appendix condenses | unverified | |
| ≈14,500 / ≈14.5K | L168/L173/L156/L218 | total traces | 14,472 | verified | |
| 50%/50% | L156/L170 | class balance per split | 6K=3000/3000; GRPO=2118/2118; union ≈50% | verified | |
| 170 tokens | L171/L236 | mean trace length (tokens) | recomputed mean 164.7 tokens (word-mean ~137) | verified | |
| 28 tokens (std) | L171 | std of token length | recomputed std = **13.5** tokens | mismatch | ≈13.5 tokens (range 121-223) |
| 160-270 | L171 | token budget range | actual range 121-223; 270 never reached | unverified | |
| 1.6 s | L172 | median wall-clock per trace | not logged | unverified | |
| <$20 | L173 | total generation cost | not logged (retail estimate) | unverified | |
| 8,908 | L179 | anomalous `<type>` strings for t-SNE | anomaly_type_analysis/embeddings_2d.npy shape (8908,2) | verified | |
| top-12 | L179 | type families plotted | presentation choice | unverified | |
| 600 | L179 | Scratch cluster size | exact "Scratch" = 500; Scratch-dominant cluster = 866 | mismatch | 500 (or 866) |
| 180 | L179 | Missing-component cluster size | exact "Missing component" = 162 | mismatch | 162 |
| 148.6 | L179 | t-SNE Scratch↔Missing distance | recomputed 155.26 | mismatch | ≈155.3 |
| 16.5 | L179 | avg within-cluster radius | recomputed 15.34 (two clusters) | mismatch | ≈15.3 |
| 9.0 | L179 | separation ratio | 155.26/15.34 = 10.12 | mismatch | ≈10.1 |
| 4-5 mm | L187 | scratch length in illustrative example | verbatim example text | unverified | |
| 125K | L207/L216/L218/L231 | Anomaly-Instruct-125k corpus | xu2025anomalyov | literature | |
| ~5.9K | L218 | Expert-AD corpus | li2025iadr1 | literature | |
| GPT-4o | L223 | Anomaly-Instruct-125k generator | xu2025anomalyov | literature | |
| Gemini 2.5-Flash | L223/L33/L108/L132 | AnomalyThink generator | config GEMINI_MODELS='gemini-2.5-flash'; geminiteam2025gemini25 | verified | |
| three (3 splits etc.) | L150-156/L211/L236 | 6K/4K/4K disjoint splits | verified split sizes | verified | |
| 2 (two main SFT runs) | L158 | C1-only vs all-5 camera filter | c1_only_fixed/ dir (new_sft_c1 len 4236; combined_sft_c1 len 10236) | verified | |
| two formats | L160 | ShareGPT JSON + GRPO format | SFT `{messages,images}`; grpo_train.json custom keys | verified | |

---

## Chapter 04 — Supervised Fine-Tuning

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 80.16% | L10/L5 | 7B-frozen-6K SFT DS BA (pipeline caption) | `sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json` | verified | |
| 82.80% / 72.07% | L10 | Arm-C SFT DS / VisA | `sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_*` | verified | |
| ~2 pp | L10 | GRPO below Arm-C init (fwd ref) | grpo_qwen25vl_7b_abc_C_grpo deltas | verified | |
| 6K / 15K | L10/L36/L142 | data-size factor | combined_6k=6000; combined_sft=14472 | verified | |
| 3B / 7B | L29/L140 | model sizes | Qwen2.5-VL-{3B,7B}-Instruct | literature | |
| 28 | L29 | 3B transformer layers | 3B config.json `num_hidden_layers` = **36** | mismatch | 36 |
| 16 / 2048 | L29 | 3B attention heads / hidden | 3B config.json | verified | |
| 28 / 28 / 3584 | L29 | 7B layers / heads / hidden | 7B config.json | verified | |
| 14×14 | L29 | ViT patch size | config.json `patch_size`=14 | verified | |
| 512 | L29/L160 | image side cap (px) / decode tokens | sft yaml `image_max_pixels`=262144=512² | verified | |
| 1024×512 | L29 | stitched canvas | derived (two 512px halves) | unverified | |
| 256 GB | L72 | system RAM | hardware, not logged | unverified | |
| 48 GB | L72 | A6000 GDDR6 | RTX A6000 datasheet | literature | |
| 2 | L72/L89/L173 | A6000 count | ground truth; ds config | verified | |
| 3 | L78 | DeepSpeed ZeRO stage | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json) stage:3 | verified | |
| 6.9 hours | L89 | 7B-frozen-6K wall-clock | all_results.json train_runtime=24937 s | verified | |
| 22.5 hours | L89 | 7B-unfrozen-15K wall-clock | all_results.json = 43875 s = **12.19 h** | mismatch | ≈12.2 h |
| 2.4 hours | L89 | 3B-frozen-6K wall-clock | all_results.json = 8451 s (but a **6-epoch** run; 4-epoch ≈1.6 h) | mismatch | 2.35h for 6-epoch run |
| 1K | L95 | pilot trace count for sweep | no artifact | unverified | |
| AdamW | L106 | optimizer | training_args.bin | verified | |
| 0.9, 0.95 | L107 | AdamW β1, β2 | β1=0.9 OK, β2=**0.999** | mismatch | β1=0.9, β2=0.999 |
| 2e-5 | L108/L123 | learning rate "all four cells" | frozen=**1e-5**, unfrozen=**1e-6** | mismatch | 1e-5 / 1e-6 |
| linear + 100-step warmup | L109/L125 | LR schedule | training_args: **cosine**, warmup **20** (6K) / **50** (15K) | mismatch | cosine, warmup 20/50 |
| 0.01 | L110 | weight decay | training_args = **0.0** | mismatch | 0.0 |
| 4 | L111 | per-device BS (frozen) | yaml/args = 4 | verified | |
| 2 | L111 | per-device BS (unfrozen) | 7B-unfrozen=**16**, 3B-unfrozen=4 | mismatch | 16 (7B) / 4 (3B) |
| 4 | L112 | grad accumulation (frozen) | frozen args = 4 | verified | |
| 8 | L112 | grad accumulation (unfrozen) | 7B-unfrozen=**1**, 3B-unfrozen=4 | mismatch | 1 (7B) / 4 (3B) |
| 16 | L113 | effective batch size | per-GPU 16; global EBS=**32** on 2 GPUs | mismatch | global 32 (16/GPU) |
| 4096 | L114 | sequence length cap | yaml `cutoff_len`=**12144** | mismatch | 12144 |
| bf16 | L74/L86/L115 | mixed precision | training_args bf16=True | verified | |
| 1.0 | L116 | gradient clipping | training_args max_grad_norm=1.0 | verified | |
| 4 | L117/L143 | epochs | 7B=4; 3B-frozen-6K run used **6** | mismatch | 4 for 7B; 6 for that 3B run |
| 42 | L118 | random seed | training_args seed=42 | verified | |
| 5e-5 | L123 | LR collapse anecdote | no logged collapse run | unverified | |
| 25% | L125 | cosine under-utilisation (pilot) | no artifact | unverified | |
| 100-step warmup / 1.5% | L125 | warmup as % of budget | actual 20/752 ≈ 2.7%; internally inconsistent | mismatch | 20-step (~2.7%) |
| 2×2×2×4 = 32 | L145 | eval cells per benchmark | arithmetic correct (some cells never run) | verified | |
| 1,670 / 2,141 | L145 | DS / VisA eval sizes | eval JSON lengths | verified | |
| 512 | L160 | max new tokens (greedy) | eval harness | verified | |
| 0.3 pp | L160 | beam/nucleus vs greedy (200-sample) | no artifact | unverified | |
| 200 | L160 | decoding sanity-check size | no artifact | unverified | |
| 0.5% | L162 | post-SFT answer-tag parse fail | recomputed ckpt-564: 0/1670 and 0/2141 (<0.5%) | verified | |
| 100K | L64 | lazy image pool | qualitative | unverified | |
| 15 | L168 | DS-MVTec per-product count | DS eval JSON `per_product` has 15 keys | verified | |
| two decimal places | L170 | reporting precision | convention | verified | |
| six-phase | L4/L31/L53 | trace template (cross-ref) | ch3 | literature | |
| four-component | L10/L173 | GRPO reward (fwd ref) | ground truth weights | verified | |
| four-factor | L138/L173 | SFT grid factors | L140-143 enumerates 4 | verified | |

---

## Chapter 05 — GRPO Reinforcement Learning

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| G=4 | L27/L125/L146 | group size (num_generations) | run-2 training_args; trainer_state | verified | |
| 1e-8 | eq L15 | ε for group-std stability | trainer impl | unverified | |
| 2048 | L27/L135 | max prompt length | ground truth (but Appendix-A note: saved arg = 4096) | verified | |
| ε=0.2 | L27/L126 | PPO clip | run-2 training_args | verified | |
| β=0 | L27/L127/L218 | KL coefficient (k3 monitored) | run-2 training_args | verified | |
| (0.3,0.3,0.2,0.2) | L58/L92 | reward weights | ground truth | verified | |
| 1.0 | L58/L218 | max total reward | 0.3+0.3+0.2+0.2 | verified | |
| [0,0.3] | L61 | format reward range | w_fmt=0.3 | verified | |
| +0.075 | L67 | per-tag partial credit | 0.3/4 | unverified | |
| −0.1 | L73 | forbidden-tag penalty | reward design | unverified | |
| {0,0.3} | L78 | accuracy reward range | w_acc=0.3 | verified | |
| [0,0.2] / 1.0 / 0.2 / 0.8 / 0.16 | L81-84 | type reward range / sim & weighted values | reward design (sim 1.0/0.8 unverified) | verified/unverified | |
| {0,0.2} | L89 | location reward range | w_loc=0.2 | verified | |
| 3×3 | L90 | grid match deliberately NOT used | IAD-R1 contrast | literature | |
| 0.6 | L93 | format+accuracy sum | 0.3+0.3 | verified | |
| 4,236 | L99/L137/L218 | GRPO prompts | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4236 | verified | |
| 50/50 | L99 | anomaly/normal balance | 2118+2118 | verified | |
| 4K | L99 | GRPO split informal label | 4,236 | verified | |
| 1024×512 / 512×512 | L101-103/L146 | stitch canvas / each half | stitch config (canvas unverified; halves verified) | unverified/verified | |
| <2% | L108 | missing-ref samples after path migration | production log claim | unverified | |
| 6,500 / 6,500+6,500=13,000 | L108/L160 | "headline 6,500-sample run" | actual **4,236** (2,118+2,118) | mismatch | 4,236 prompts (NOT 13,000) |
| checkpoint-564 | L123/L177/L252 | base/ref SFT policy | ground truth | verified | |
| 1e-6 | L128/L148 | GRPO LR | run-2 training_args; trainer_state | verified | |
| 0.1 | L129 | warmup ratio (linear) | LR decay consistent (saved arg: warmup_ratio None — see App A) | verified | |
| (0.9,0.95) | L130 | AdamW β | not dumped (actual β2=0.999) | unverified | |
| 0.01 | L131 | weight decay | not dumped (actual 0.0) | unverified | |
| 2 / 4 / 8 | L132-134 | per-device BS / GA / EBS | ground truth EBS 8 (per-device BS actually 1; see App A) | verified | |
| 512 | L136 | max completion length | run-2 training_args | verified | |
| 1,060 / 2 | L137/L160/L190 | max steps / epochs | trainer_state max_steps=1060, epochs=2 | verified | |
| 250 | L138/L160 | save/eval every N | ground truth (saved run-2 arg: save_steps=530, eval None — see App A) | verified | |
| 42 / bf16 | L141/L140 | seed / precision | run-2 training_args | verified | |
| G=64 | L146 | DeepSeekMath group size (cited) | DeepSeekMath | literature | |
| ~30 seconds | L146 | per-rollout cost | train.log run2 83.85 s/step → ~21 s/rollout (same order) | unverified | |
| ~600 MB / <50 ms | L156 | Nomic memory / per-step overhead | model size / impl claim | unverified | |
| ~700 | L154 | LOC of GRPO trainer | code claim, not counted | unverified | |
| 5200 | config | Nomic embedding server port | ground truth | verified | |
| 1.66 / 2.22 / 2.13 | L162 | combined reward steps 10/100/530 | trainer_state run2: 1.65625 / 2.21875 / 2.125 | verified | |
| 0.875 / 1.25 | L163 | accuracy reward start / end | trainer_state | verified | |
| ~73% | L163 | verdict-correct fraction at start | 0.875/1.2 ≈ 0.73 (interpretive) | unverified | |
| +42.9% | L163 | accuracy reward rel. gain | (1.25−0.875)/0.875 | verified | |
| 0.781 / 0.875 | L164 | format/consistency reward start/end | trainer_state | verified | |
| +12.0% | L164 | format reward rel. gain | (0.875−0.781)/0.781 | verified | |
| [0.003, 0.095] | L165 | KL range | trainer_state min 0.0 max **0.117** | mismatch | [0.0, 0.117] |
| 163-174 | L166 | completion length range | trainer_state ~153-186 (clusters 163-174) | mismatch | ~153-186 (clusters 163-174) |
| 12.4 hours | L167 | wall-clock to ckpt-530 | train.log: full run 24.69 h; ckpt-530 (epoch 1) ≈12.34 h | unverified | |
| step 287 / 0.6 | L171/L174 | sample reward-line ref | trainer_state step287 epoch=0.542 | unverified | |
| 0.8349 / 0.1349 / 0.6521 / 0.2715 / 0.0009 | L172-174 | sample reward-line components | illustrative listing (0.2+0.3+0.1349+0.2=0.8349 checks) | verified/unverified | |
| checkpoint-530 / checkpoint-1060 | L177/L185 | headline / epoch-2 ckpt | trainer_state | verified | |
| ~0.7 pp | L177 | ckpt-1060 below ckpt-530 on DS | 82.73 − 82.03 | verified | |
| 4,026 | L188 | iter-2 filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 1e-5 / 4 / 4 | L189 | iter-2 SFT LR / BS / epochs | SFT recipe; trainer_state iter2_v2 | verified | |
| ckpt-378 / ckpt-504 | L189 | iter-2 SFT ep3 / ep4 | iter2_v2 SFT dir (504 steps / 4 ep) | verified | |
| 265 / 0.5-2.0 / G=4 | L190 | iter-2 GRPO save-N / epoch coverage / group | iter2_v2_full ckpts 265/530/795/1060 | verified | |
| +2.57 pp / +5.61 pp | L218 | GRPO over SFT on DS / VisA | 82.73−80.16 / 70.39−64.78 | verified | |
| 82.73% | L229/L253 | single-stage headline DS | run2 ckpt-530 DS JSON | verified | |
| {0.0,0.3,0.6,1.0} | L200/L209 | discrete reward bins (G²RPO) | reward design (plausible) | unverified | |
| −1.15 / +1.15 | L207 | worst/best G²RPO advantage (G=4) | Φ⁻¹(1/8) / Φ⁻¹(7/8) | verified | |
| −0.5 / √2 | L204/L207 | midpoint correction / erfinv scaling | OpenVLThinkerv2; standard identity | literature/verified | |
| 81.94% | L234/L252 | G²RPO v1 DS baseline | `iter2_v1_eval_archive/.../g2rpo_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json` (902/401/43/324) | verified | |
| 80.64% | L254 | iter-2 v2 DS baseline | `grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json` (964/367/77/262) | verified | |
| 2,700 / 1,167 | L229 | type substitutions / location normalisations (cross-ref) | attributed to ch3; not verifiable from grpo_train diff | unverified | |
| 5-7 h / 15-20 h | L239 | compound run / total budget | forward-looking estimate | unverified | |
| ≥0.5 pp | L245 | compound pass threshold | pre-registered design | unverified | |
| 2 | L167/L239 | A6000 count | ground truth | verified | |
| run 2 | L160/L185 | headline GRPO run id | ground truth (dir _run2) | verified | |

---

## Chapter 06 — Results and Analysis

### tab:baseline (L24-30)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 76.50% | L25 | 3B base DS Acc | [`results/baseline_named/3b_base_dsmvtec.json`](results/baseline_named/3b_base_dsmvtec.json) (1222/56/388/4) acc 76.53 | verified | |
| 55.80% | L25 | 3B base DS Bal-acc | recomputed **56.14** | mismatch | 56.14% |
| 86.20% / 75.90% / 99.70% | L25 | 3B base DS F1 / Prec / Recall | 3b_base_dsmvtec.json | verified | |
| 55.03% / 69.01% / 56.11% / 99.20% / 39.15% | L26 | 7B base DS Acc/BA/F1/Prec/Rec | [`results/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json`](results/qwen25vl_baseline_eval/) (480/439/5/746) | verified | |
| 56.10% / 50.20% / 71.80% / 56.00% / 100.00% | L29 | 3B base VisA | [`results/baseline_named/3b_base_visa.json`](results/baseline_named/3b_base_visa.json) | verified | |
| 48.30% / 53.80% / 14.30% / 98.90% / 7.70% | L30 | 7B base VisA | [`results/baseline_named/7b_base_visa.json`](results/baseline_named/7b_base_visa.json) | verified | |
| 99.2% / 39.1% | caption L17 | 7B base DS Prec / Rec | qwen25vl_baseline_eval | verified | |
| 1,670 / 2,141 | L24/L28 | eval sizes | eval JSON lengths | verified | |

### tab:sft-summary (L66-74) — see Corrections §B for all mismatches

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 16 | L40 | SFT grid cells | nominal 16 (2 unfrozen-6K never run) | unverified | |
| 68.50 / 59.60 / 71.60 / 78.10 | L66 | 3B-Frozen-6K ep4 (ckpt-752) | `results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752/` | mismatch | 69.08 / 57.22 / 72.22 / 80.02 |
| 70.50 / 60.85 | L67 | 3B-Frozen-15K ep4 (ckpt-1812) DS/VisA | `results/sft_qwen25vl_3b_15k_frozen/checkpoint-1812/` | mismatch | 68.65 / 64.85 |
| 73.59 / 81.49 | L67 | 3B-Frozen-15K ep4 Acc/F1 | ckpt-1812 | verified | |
| 67.78 / 61.50 / 70.30 / 76.40 | L68 | 3B-Unfrozen-6K ep4 | config NEVER RUN (dir absent) | unverified | |
| 71.86 / 74.19 / 81.38 | L69 | 3B-Unfrozen-15K ep4 DS/Acc/F1 | `results/sft_qwen25vl_3b_15k_unfrozen/checkpoint-1812/` (claimed=7B numbers) | mismatch | 68.58 / 69.58 / 77.34 |
| 59.50 | L69 | 3B-Unfrozen-15K ep4 VisA | ckpt-1812 (59.60) | verified | |
| 71.66 / 86.89 | L70 | 7B-Frozen-15K ep3 DS/F1 | [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/`](results/sft_qwen25vl_7b_15k_frozen/) | verified | |
| 65.45 / 78.20 | L70 | 7B-Frozen-15K ep3 VisA/Acc | ckpt-1359 (claimed=ep1/ckpt-453) | mismatch | 64.28 / 80.12 |
| 80.16 / 64.78 / 80.36 / 85.76 | L72 | **HEADLINE** 7B-Frozen-6K ep3 (ckpt-564) | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) | verified | |
| 91.65 / 80.59 | L88 | HEADLINE SFT precision / recall | ckpt-564 DS | verified | |
| 72.08 / 58.20 / 68.50 / 75.02 | L73 | 7B-Unfrozen-6K ep4 | config NEVER RUN (dir absent) | unverified | |
| 72.60 / 68.86 / 75.60 | L74 | 7B-Unfrozen-15K ep4 DS/Acc/F1 | [`results/sft_qwen25vl_7b_15k_unfrozen/checkpoint-1812/`](results/sft_qwen25vl_7b_15k_unfrozen/) (72.60 is frozen value) | mismatch | 70.27 / 65.21 / 71.51 |
| 58.50 | L74 | 7B-Unfrozen-15K ep4 VisA | ckpt-1812 (58.16) | verified | |
| 8.5-point | L81 | 6K vs 15K ep3 gap | 80.16 − 71.66 | verified | |
| 7.6 pp | L81 | 6K vs 15K best-epoch gap | uses 15K ep4=72.60 (the WRONG frozen value) | unverified | |
| 141 words / 14.5K / 8.5K | L81 | mean trace length / union / extra | ground truth; 14472; 14472−6000 | verified | |
| ~300M | L83 | ViT param count (overfit arg) | Qwen2.5-VL ViT ~675M actual | unverified | |
| 4-12 pp | L85 | 7B>3B gap | range across cells | unverified | |
| 1 pp | L88 | SFT within of IAD-R1 | 81.92−80.16=1.76 (loose) | unverified | |

### tab:grpo-results (L112-120)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 80.63 / 70.25 / 77.37 / 82.69 | L112 | GRPO Run1 ckpt-315 DS/VisA/Acc/F1 | `results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/checkpoint-315/` | verified | |
| 81.65 / 69.39 / 78.32 / 83.47 | L113 | GRPO Run1 ckpt-530 | run1 ckpt-530 | verified | |
| 82.73 / 70.39 / 81.08 / 86.01 | L116 | **HEADLINE** GRPO Run2 ckpt-530 | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) | verified | |
| 82.03 / 69.86 / 80.36 / 85.44 | L117 | GRPO Run2 ckpt-1060 | run2 ckpt-1060 | verified | |
| +2.57 / +5.61 / +0.72 / +0.25 pp | L120 | GRPO−SFT deltas DS/VisA/Acc/F1 | arithmetic over verified | verified | |
| 94.09 / 79.20 / 86.27 | L127 | Run2 ckpt-530 DS prec/rec/TNR | run2 ckpt-530 DS | verified | |
| 63.66 / 77.13 | L128 | Run2 ckpt-530 VisA TPR/TNR | run2 ckpt-530 VisA | verified | |
| ~0.7 pp | L130/L427 | ckpt-1060 below ckpt-530 | 82.73−82.03 | verified | |

### Iter-2 progression, rollout-failure-modes, SFT-Iter2-cleaned (L142-259)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 4,026 | L142 | iter-2 v2 SFT1 self-filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 77.95 | L149 | iter-2 v1 SFT1 ckpt-196 DS (grpoprompt) | iter2_v1_archive | verified | |
| 76.89 / 77.16 | L150-151 | iter-2 v1 GRPO1 ckpt-265 / 530 DS | iter2_v1_archive | verified | |
| 77.61 / 72.00 | L153 | iter-2 v2 SFT1 ckpt-378 ep3 DS/VisA | [`results/sft_qwen25vl_7b_iter2_v2_frozen/checkpoint-378/`](results/sft_qwen25vl_7b_iter2_v2_frozen/) | verified | |
| 79.54 / 72.02 | L154 | iter-2 v2 SFT1 ckpt-504 ep4 DS/VisA | iter2_v2 ckpt-504 | verified | |
| 78.38 / 70.70 | L155 | iter-2 v2 GRPO1 ckpt-265 DS/VisA | [`results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-265/`](results/grpo_qwen25vl_7b_iter2_v2_full/) | verified | |
| 80.64 / 68.65 | L157 | iter-2 v2 GRPO1 ckpt-530 DS/VisA (best) | grpo iter2_v2 ckpt-530 | verified | |
| 79.55 / 68.40 | L158 | iter-2 v2 GRPO1 ckpt-795 | iter2_v2 ckpt-795 | verified | |
| 79.37 / 68.67 | L159 | iter-2 v2 GRPO1 ckpt-1060 | iter2_v2 ckpt-1060 | verified | |
| +1.10 pp | L164 | iter-2 RL over own SFT | 80.64−79.54 | verified | |
| 81.94 | L177/L183 | G2RPO best ckpt-530 DS (tab:rl-variants) | iter2_v1_archive g2rpo_full ckpt-530 grpoprompt | verified | |
| ~2 pp | L186 | T vs G prompt agreement | qualitative (81.94 vs 80.16 = 1.78) | unverified | |
| 30% | L193 | NG `<type>` raw codes in 6K pool | ground truth | verified | |
| 2,700 / 1,167 | L193 | type substitutions / location normalisations | not located on disk | unverified | |
| k=8 / 10,236 / 81,888 | L195/L199 | rollout multiplicity / pool / total | phase0_full_10k rollouts_raw.jsonl=10236; 10236×8=81888 | verified | |
| 3,770 / 94.8% / 5.2% | L195 | Gemini-patched traces / NG / OK | internally consistent only (artifact off-disk) | unverified | |
| 1,541 (30.1%) / 871 (17.0%) / 2,412 | L206 | NG/OK/total ≥1-of-8 wrong | arithmetic 1541+871=2412 verified; pools implied | verified/unverified | |
| 807 (15.8%) / 176 (3.4%) / 983 | L207 | all-8/8 wrong | 807+176=983 verified | verified/unverified | |
| 4 ep / 1e-5 / 32 | L212/L463 | SFT-Iter2 / ABC SFT recipe | trainer_state; bs8×GA2×2GPU=32 | verified | |
| 82.73 / 70.39 / 76.56 | L224 | SFT-Iter2 baseline DS/VisA/Avg | run2 ckpt-530 | verified | |
| 3,557+2,443 | L226 | 6K-pool kept+patched | sum=6000 (sub-split off-disk) | unverified | |
| 81.07 / 70.05 / 75.56 / −1.00 | L226 | 6K-pool variant DS/VisA/Avg/Δ | [`results/sft_qwen25vl_7b_iter2_clean/checkpoint-188/`](results/sft_qwen25vl_7b_iter2_clean/) | verified | |
| 96+96 / 71.30 / 60.52 / 65.91 / −10.65 | L227 | Balanced-192 composition + metrics | datasets_sft_iter2 balanced len=192; ckpt-24 | verified | |
| 4,236+197/197 / 75.77 / 66.41 / 71.09 / −5.47 | L228 | Held-out variant composition + metrics | datasets_sft_iter2 heldout len=4630; [`results/sft_qwen25vl_7b_iter2_heldout/checkpoint-145/`](results/sft_qwen25vl_7b_iter2_heldout/) | verified | |
| 322/1,670 / 20.5% | L233 | Balanced-192 ep1 unparsable / OK-recall VisA | iter2_balanced ckpt-6 / ckpt-24 | verified | |
| 3,000 NG + 3,000 OK | L256 | 6K pool item balance | Arm-C/iter2 pool 50/50 | verified | |
| 96% / 95% | L256/L259 | NG content fraction / per-step signal | ~94.8% rounds to 96% (derived) | unverified | |

### Per-product iter2 deltas (figs L238, L245) — all recomputed iter2_clean188 − grpo run2 530

| Number | Where | Describes | Status |
|---|---|---|---|
| +5.8 toothbrush, +2.4 cable, +1.8 grid, +1.2 screw, +1.2 tile | L238 | DS gains | verified |
| −6.1 capsule, −6.0 metal_nut, −5.3 wood, −4.5 hazelnut | L238 | DS regressions | verified |
| −1.66 pp | L238 | net iter2 DS effect (81.07−82.73) | verified |
| +6.4 pcb3, +8.0 macaroni2 | L245 | VisA gains | verified |
| −12.0 pcb4, −8.5 candle | L245 | VisA regressions | verified |
| −0.34 pp | L245 | net iter2 VisA effect (70.05−70.39) | verified |

### tab:sota / per-product base→SFT / held-out RealIAD (L280-430)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 82.73 / 70.39 | L280 | compound headline placeholder | run2 ckpt-530 | verified | |
| 0.5 pp | L289 | compound threshold | design choice | unverified | |
| 69.01 | L314 | 7B base DS (SOTA) | qwen25vl_baseline_eval | verified | |
| 81.92 / 71.34 | L317 | IAD-R1 DS / VisA (our harness) | iad_r1_qwen_recanon | verified | |
| 82.80 / 72.07 | L321 | Arm-C SFT-only DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| +0.88 / +0.73 / +0.81 / −0.95 / ~1.8 | L330-332 | Arm-C & GRPO vs IAD-R1 deltas | arithmetic over verified | verified | |
| +25.8 metal_nut (57.8→83.6) | L342/L346 | base→SFT per-product gain | base 57.86→sft 83.64 | verified | |
| +19.2 cable / +16.0 pill / +15.8 toothbrush / +14.9 screw / +13.3 transistor | L342 | base→SFT gains | recomputed | verified | |
| 50.9%→70.1% cable | L346 | base→SFT (printed 70.1 vs actual 70.33) | recomputed | unverified | |
| 60.1→76.1 pill | L346 | base→SFT | recomputed | verified | |
| fifteen others gain | L342 | per-product gainers | only **13** gain (2 regress: wood, tile) | mismatch | thirteen |
| 4,236 / 23 | L388-393 | held-out n / products | eval_realiad4k JSON; ground truth | verified | |
| 80.81 / 80.87 / 91.28 / 68.34 / 78.16 / 93.40 | L399-404 | held-out Acc/BA/Prec/Rec/F1/TNR | `results/grpo_..._run2/checkpoint-530/eval_realiad4k_full_trainprompt.json` | verified | |
| 2 pp | L409 | held-out within of DS (80.87 vs 82.73) | 82.73−80.87=1.86 | verified | |

### Training curves (L415-423)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 1.14 / 0.58 | L415 | SFT loss step 50 / 500 | ckpt-564 trainer_state | verified | |
| 0.34 (step 752) | L415 | SFT loss end-ep4 | 7B-6K-frozen run ends step 560 (~0.57); 752/0.34 is 3B cadence | mismatch | ~0.57 at step 560 (no step 752) |
| step 564 | L415 | SFT BA peak | ckpt-564 epoch-3 | verified | |
| 1.66 / 2.22 / 2.13 | L419 | GRPO reward 10/100/530 | run2 trainer_state | verified | |
| 2.0 | L419 | theoretical max reward | design (text notes >2) | unverified | |
| 0.875→1.25 / +42.9% | L420 | accuracy reward | trainer_state | verified | |
| 0.781→0.875 / +12.0% | L421 | format reward | trainer_state | verified | |
| [0.003, 0.095] | L422 | KL bounds (steps 10/530 = 0.0033/0.0842) | trainer_state | verified | |
| 163-174 tokens | L423 | completion length | logged 169-174 at 10/100/530 | unverified | |

### Teacher ablation A/B/C, GRPO-on-C, estimator probe (L456-602)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 82.80 / 72.07 | L532/L599 | Arm-C best (ABC takeaway) | armC ckpt-376 | verified | |
| 78.33/66.15, 78.69/69.35, 77.32/70.21, 79.01/68.83 | L479-482 | Arm A ckpt-94/188/282/376 DS/VisA | `results` abc_A_kept (archive) | verified | |
| 2,978 | L456 | Arm A n items | [`traces/teacher_ablation_abc/sft_A_kept_balanced.json`](traces/teacher_ablation_abc/sft_A_kept_balanced.json) len=2978 | verified | |
| 75.83/64.08, 78.84/67.44, 80.75/67.48, 79.46/68.73 | L485-488 | Arm B ckpt-137/274/411/548 | abc_B_kept_corrected (archive) | verified | |
| 4,369 | L458 | Arm B n items | [`traces/teacher_ablation_abc/sft_B_kept_corrected.json`](traces/teacher_ablation_abc/sft_B_kept_corrected.json) len=4369 | verified | |
| 812 / 31% | L458/L529 | Arm B corrected items added / NG fraction | sub-count not reconcilable | unverified | |
| 79.71/68.76, 82.80/72.07, 81.39/72.01, 79.13/71.59 | L491-494 | Arm C ckpt-188/376/564/752 | [`results/sft_qwen25vl_7b_abc_C_full_patched/`](results/sft_qwen25vl_7b_abc_C_full_patched/) | verified | |
| 6,000 | L460 | Arm C n items | [`traces/iter2/sft_iter2_train.json`](traces/iter2/sft_iter2_train.json) len=6000 | verified | |
| 1,631 | L460 | Arm C Gemini-rewritten | sub-count (gemini_rewritten.jsonl=2406 differs) | unverified | |
| 95% | L456 | Arm A model pass rate | composition claim | unverified | |
| −1.15 / +4.05 (A), +0.59 / +2.70 (B), +2.64 / +7.29 (C) | L513-516 | Δ vs headline DS/VisA | arithmetic over verified | verified | |
| 1.68 / 0.07 pp | L525/L532 | Arm-C over SFT+GRPO VisA / DS | 72.07−70.39 / 82.80−82.73 | verified | |
| +5.43 pp | L527/L532 | Arm A VisA-best over headline (ckpt-282) | 70.21−64.78 | verified | |
| G=4 / β=0 / 1e-6 / 265 steps | L540 | GRPO-on-C config | ground truth; grpo_abc_C ckpts | verified | |
| 80.61/72.68 (−2.19/+0.61) | L558 | GRPO-on-C ckpt-265 | [`results/grpo_qwen25vl_7b_abc_C_grpo/checkpoint-265/`](results/grpo_qwen25vl_7b_abc_C_grpo/) | verified | |
| 80.32/70.34 (−2.48/−1.73) | L559 | GRPO-on-C ckpt-530 | grpo_abc_C ckpt-530 | verified | |
| 80.76/69.69 (−2.04/−2.38) | L560 | GRPO-on-C ckpt-795 | checkpoint dir rotated away (save_total_limit) | unverified | |
| 2.0-2.5 pp | L565/L598 | GRPO-on-C DS degradation | deltas | verified | |
| 1.99→2.54 | L565 | GRPO-on-C reward rise | maps to ckpt-795 window (ckpt absent) | unverified | |
| 120 / 20 / 400-sample | L567/L602 | estimator-ablation burst / cadence / probe size | probe_curve.csv; probe CM sums ~400 | verified | |
| 84.52/71.89 | L578 | estimator-ablation init (probe) | probe_curve.csv step0 INIT (no tp/tn JSON) | unverified | |
| 82.87/72.22 (ctrl), 84.23/70.18 (drgrpo), 84.39/69.80 (g2rpo) | L579 | step20 | grpo_probe_*/probe_curve.csv | verified | |
| 81.74/70.76, 81.19/66.82, 81.19/69.22 | L580 | step40 (ctrl/drgrpo/g2rpo) | probe_curve.csv | verified | |
| 82.25/71.82, 81.69/69.92, 81.72/68.26 | L581 | step60 | probe_curve.csv | verified | |
| 83.67/70.45, 82.06/69.82, 83.21/67.65 | L582 | step80 | probe_curve.csv | verified | |
| 83.02/70.78, 82.03/70.71 | L583 | step100 (ctrl / drgrpo) | probe_curve + drgrpo CM recompute | verified | |
| 81.89/69.87, 81.91/70.83 | L584 | step120 (ctrl / drgrpo) | probe_curve + drgrpo CM recompute | verified | |
| ~4 min / ~35 min / ~2 pp | L602 | probe vs full eval time / probe-high | methodological estimate (init 84.52 vs full 82.80 ≈1.7) | unverified | |

---

## Chapter 07 — Discussion

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 4-8 pp / 4-10 pp | L11 | frozen-vs-unfrozen ViT gap DS / VisA | qualitative range across grid | unverified | |
| ~300M | L15 | Qwen2.5-VL ViT params | model card | literature | |
| 15K / thirty (~30) | L15/L29 | corpus / distinct products | 14,472; 6K SFT ~30 products | verified | |
| 34-41% | L19 | VisA recall, unfrozen 7B | sft_7b_15k_unfrozen ckpts: 34.8/37.0/38.5/40.6% | verified | |
| ~70% | L19 | VisA recall, frozen | sft_7b_zeroshot_6k_frozen ckpt-564: 71.1% (tp851 fn346) | verified | |
| 100× / 1.5M | L21 | hypothetical corpus for unfreezing to win | hypothetical | unverified | |
| 6K-15K | L21 | trace range favouring freezing | 6,000 / ~14,472 | verified | |
| 8.5-point | L27 | 6K vs 15K SFT gap | uses headline 6K 80.16 vs a different-checkpoint 15K value; corrected 15K ep3 ckpt-1359 DS=80.12 gives ~0pp | mismatch | with corrected 15K DS the gap is ~0pp, not 8.5pp |
| 14.5K / 100% / ~141 words / ~8.5K | L29 | union / 6K-subset / mean length / extra | ground truth | verified | |
| 4 epochs / 2.5× / 2-2.5× | L31 | SFT budget / gradient updates / wall-clock | 14472/6000=2.41 | verified | |
| six-phase | L33/L43 | template phases | prompt template | unverified | |
| 1K | L35 | LIMA example count | Zhou et al. 2023 | literature | |
| 82.80 / 72.07 | L41/L63 | Arm-C SFT DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| 10.7-point | L41 | Arm-C DS−VisA gap | 82.80−72.07 | verified | |
| 82.73 / 70.39 | L41/L63 | SFT+GRPO DS / VisA | run2 ckpt-530 | verified | |
| 81.92 / 71.34 / 10.6 pp | L41 | IAD-R1 DS / VisA / internal gap | iad_r1_qwen_recanon | verified | |
| +0.73 / +0.88 pp | L41 | Arm-C VisA / DS over IAD-R1 | arithmetic | verified | |
| ~3.5 pp | L41 | earlier IAD-R1 VisA deficit (artefact) | older eval, off-disk | unverified | |
| 1,670 / 2,141 | L41 | eval sizes | ground truth | verified | |
| ~0.7% / ~2.1% | L45 | VisA / DS mean defect area | benchmark stat, no backing | unverified | |
| 512 px / ~3× / 896 px | L45/L49 | stitch resolution / VisA downsample / proposed | config; derived; proposal | unverified | |
| 3×3 | L47/L49 | proposed grid spatial reward | IAD-R1 SC-GRPO | literature | |
| ~0.7 pp / ckpt-530 / ckpt-1060 | L55 | ckpt-530 over ckpt-1060 | 82.73−82.03 | verified | |
| β=0 / k_3 / two-epoch | L57 | KL coef / estimator / budget | ground truth | verified | |
| β ≈ 0.05-0.2 | L57 | proposed small KL (future work) | proposal (NOT the old wrong 0.04) | unverified | |
| +2.57 / +5.61 pp | L63-83 | GRPO lift over SFT-Iter1 DS / VisA | arithmetic | verified | |
| 6,000 / 0.07 pp / 1.68 pp | L63 | Arm-C size / DS match / VisA beat | ground truth; arithmetic | verified | |
| 0.30 / 0.20 / 0.20 | L70-72 | reward weights format / type / location | ground truth | verified | |
| ~92% / >99% / ~8% | L70 | tag compliance after SFT / GRPO / parser loss | no on-disk metric | unverified | |
| nomic-embed-text-v2 | L71 | type-reward embedding model | ground truth (server :5200) | verified | |
| ckpt-376 | L78 | Arm-C SFT as GRPO-on-C init | ground truth | verified | |
| 2.0-2.5 pp | L78/L83 | GRPO-on-C below Arm-C init | ckpt-265 −2.19, ckpt-530 −2.48, ckpt-1060 **−1.60** | mismatch | −2.19 / −2.48 / −1.60 (ckpt-1060 only −1.6) |
| ckpt-265 / +0.61 pp | L78 | only GRPO-on-C ckpt above init on VisA | 72.68−72.07 | verified | |
| 20-step / three | L78 | probe cadence / estimators | probe_curve.csv; 3 probe dirs | verified | |
| 84.52 / 71.89 | L78 | probe init DS / VisA | probe_curve.csv step0 (no CM JSON) | unverified | |
| none>init on DS / z-score strongest VisA | L78 | estimator outcome | probe CSV maxima | verified | |
| 200 / 100+100 / five | L90 | error-sample size / FN+FP split / categories | manual inspection (5-item list) | unverified/verified | |
| ~32% / <0.2% / ~24% / ~18%+14% / ~11% / ~5% | L93-97 | failure-mode shares | manual categorization, no backing | unverified | |
| run 1 / run 2 | L108 | GRPO run twice | both dirs exist | verified | |
| 80.6% | L108 | GRPO run-1 DS at ckpt-530 (claimed) | run1 ckpt-530 = **81.65**; 80.63 is ckpt-315 | mismatch | 81.65% (80.63 is ckpt-315) |
| 82.7% | L108 | GRPO run-2 DS at ckpt-530 | 82.73 | verified | |
| 3-5 seeds | L108 | recommended seeds | recommendation | unverified | |
| 80% | L110 | illustrative accuracy | rhetorical | unverified | |
| 170 tokens / ~30 tok/s / ~5-6 s | L112 | output length / speed / latency | estimate, no backing | unverified | |
| two orders of magnitude | L112 | slowdown vs EfficientAD | batzner2024efficientad (qualitative) | literature | |
| 7B / 2× A6000 / 6K | L112/L119 | model / hardware / corpus | ground truth | verified | |
| 0.8 pp | L119 | DS margin over IAD-R1 (takeaway, Arm-C) | 82.80−81.92=0.88 | verified | |
| Gemini 2.5-Flash | L116 | trace generator (leakage concern) | ground truth | verified | |

---

## Chapter 08 — Conclusion and Future Work

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| ≈14.5K / ~14,500 | C1 L11; RQ1; L23 | AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14472 | verified | |
| 25 | C1 L11 | Real-IAD training products claimed | actual 6K corpus has **30** distinct; ch3 says "all 30" | mismatch | 30 (held-out drawn from 23 of 30) |
| 6,000 / 6K | C1 L11; RQ1; L61 | SFT pool | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6000 | verified | |
| 4,000 (×2) | C1 L11 | GRPO / held-out pools | grpo_train.json len=4236 (rounds to 4000) | verified | |
| 4,236 | C4 L17 | held-out sample-level split | eval_realiad4k JSON total=4236 | verified | |
| 3B and 7B / four epochs | C2 L13; RQ3 | SFT study sizes / budget | ground truth; 04_sft.tex | verified | |
| 8.5 pp | C2 L13; RQ1 | 6K beats 15K | 80.16 − 71.66 = 8.50 (ckpt-564 vs ckpt-1359) | verified | |
| 80.16% | C2 L13; RQ1 | best SFT DS BA | sft_7b_zeroshot_6k_frozen ckpt-564 | verified | |
| 1 pp ("roughly 1pp") | C2 L13 | gap SFT→IAD-R1 | 81.92−80.16=1.76 | mismatch | 1.76 pp |
| 0.3,0.3,0.2,0.2 / G=4 / β=0 / 1e-6 | C3 L15; RQ2 | GRPO reward weights / group / KL / LR | ground truth (training_args) | verified | |
| 82.73% / 70.39% | C3 L15 | GRPO run-2 ckpt-530 DS / VisA | run2 ckpt-530 | verified | |
| +0.81 pp / 81.92% | C3 L15; RQ2 | GRPO over IAD-R1 / IAD-R1 DS | 82.73−81.92; iad_r1_qwen_recanon | verified | |
| −0.95 pp / 71.34% | C3 L15 | GRPO under IAD-R1 / IAD-R1 VisA | 70.39−71.34; iad_r1_qwen_recanon | verified | |
| +0.88 / +0.73 pp | C3 L15 | Arm-C over IAD-R1 DS / VisA | arithmetic | verified | |
| 80.87% / 93.4% | C4 L17 | held-out BA / TNR | eval_realiad4k JSON (tn1968 fp139) | verified | |
| 23 of the 30 | C4 L17 | held-out products | ground truth (held-out 23; SFT 30) | verified | |
| metal_nut +25.8 | C5 L19 | per-product gain (claimed base→GRPO) | base 57.86 → GRPO **87.47** = **+29.6** (the +25.8 is the base→SFT delta) | mismatch | +29.6 pp (base→GRPO) |
| cable +19.2 | C5 L19 | per-product gain (claimed base→GRPO) | base 51.09 → GRPO 68.16 = **+17.1** | mismatch | +17.1 pp |
| pill +16.0 | C5 L19 | per-product gain (claimed base→GRPO) | base 60.23 → GRPO 75.45 = **+15.2** | mismatch | +15.2 pp |
| five categories | C5 L19 | error/failure-mode categories | refs 07_discussion error taxonomy | unverified | |
| 82.80% / 72.07% | C6 L21; F5 | Arm-C SFT ckpt-376 DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| 2-2.5 pp | C6 L21; RQ2 | GRPO lowers DS on Arm-C init | −2.19/−2.48/−1.60 | verified | |
| +2.57 / +5.61 pp | RQ2 | GRPO lift over SFT-Iter1 | arithmetic | verified | |
| 69.01% / 100% subset / ≈141 words / ≈8.5K | RQ1 | base DS / 6K-subset / mean length / extra | qwen25vl_baseline_eval; ground truth | verified | |
| ≈141 words | RQ1 | mean length both corpora | recomputed mean **136.8** (equality holds, value ≈137) | mismatch | ≈137 words |
| 5-6 seconds / 7B-parameter | L23 | inference time / final model | 07_discussion estimate; ground truth | unverified/verified | |
| ~15,000 / 125k | L23 | AnomalyThink / Anomaly-Instruct | 14472≈15K; xu2025anomalyov | verified/literature | |
| 80.64% | F1 L43 | Iter-2 DS BA vs single-stage | grpo_iter2_v2_full ckpt-530 grpoprompt (964/367/77/262) | verified | |
| 4,026 | F1 L43 | iter-2 self-filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 3×3 grid / 512×512 / 3× / 896×896 | F2/F3 L45-47 | proposed reward / current stitch / downsample / proposed | 04/05/07 chapters; proposals | verified/unverified | |
| 3B (AgentIAD) / two tools | F4 L49 | AgentIAD size / tools | miao2025agentiad | literature | |
| 10-20 pp / 160 cat, 28 ind, 27 defects / 30→160 | F5 L51 | Real-IAD Variety scale | zhu2025realiadvariety | literature | |
| k=8 / 6,000-trace (variety) | F5 L51 | rollout count / variety SFT corpus | in-progress; [`traces/variety_star_6k/variety_star_sft_6k.json`](traces/variety_star_6k/variety_star_sft_6k.json) len=6000 (160 products) | unverified/verified | |

---

## Appendix A — Full Hyperparameter Tables

### Table A.1 — Trace generation

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| Gemini 2.5-Flash | L19 | generator model | scripts/00_generate/config_mmad.py L32 | verified | |
| 1.0 | L20 | generator temperature | not set in code; 1.0 is Gemini default | unverified | |
| 5-10 samples | L21 | batch size | config_mmad.py BATCH_SIZE=10 | verified | |
| 4 s / 3 | L22-23 | rate-limit wait / max retries | config_mmad.py RATE_LIMIT_WAIT=4, MAX_RETRIES=3 | verified | |
| 120-200 words | L24 | `<think>` budget | inspector_prompt_test_v2.txt L130 | verified | |
| 3 / 1 | L26-27 | images per anomaly / normal | ground truth | verified | |
| 187 | L33 | per-product quota K | generate_unified_realiad_15k_c1_only.py L43 | verified | |
| 50%/50% | L34 | class balance | grpo_train.json 2118/2118 | verified | |
| 25 / 5 | L35 | train/eval product split | generate script L10 (5 held-out; Real-IAD 30 → 25/5) | verified | |
| 5 | L36 | camera angles C1-C5 | generate script filter_c1_only | verified | |

### Table A.2 — SFT (see Corrections §D)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 3B / 7B | L54-61 | model sizes | sft yaml | verified | |
| 6K / 15K | — | data splits | combined_6k=6000; combined_sft=14472 | verified | |
| 2e-5 | LR column | LR for all 8 cells | frozen **1.0e-5**, unfrozen **1.0e-6** | mismatch | 1e-5 / 1e-6 (never 2e-5) |
| 4/4/16 | L54,55,58 | BS/GA/EBS frozen cells | yaml BS=4 GA=4 | verified | |
| 4/4/16 | L59 | BS/GA/EBS 7B-frozen-15K | yaml BS=**16** GA=**1** | mismatch | BS=16, GA=1 |
| 2/8/16 | L56,57 | BS/GA/EBS 3B-unfrozen | 3B-unfrozen-15K BS=4 GA=4; 6K never run | mismatch | BS=4, GA=4 (6K never run) |
| 2/8/16 | L60,61 | BS/GA/EBS 7B-unfrozen | 7B-unfrozen-15K BS=16 GA=1; 6K never run | mismatch | BS=16, GA=1 (6K never run) |
| 4 | Epochs col | epochs all SFT | yaml num_train_epochs=4 | verified | |
| (0.9, 0.95) | L66 | AdamW β | not in yaml (HF default 0.9,0.999) | unverified | |
| 0.01 | L66 | weight decay | not in yaml (default 0.0) | unverified | |
| 100-step warmup | L66 | warmup | yaml warmup_steps=**20**(6K)/**50**(15K) | mismatch | 20 / 50 |
| linear | L66 | LR schedule | yaml lr_scheduler_type=**cosine** | mismatch | cosine |
| 1.0 | L66 | grad clip | not in yaml (HF default 1.0) | unverified | |
| 4096 | L66 | seq length | yaml cutoff_len=**12144** | mismatch | 12144 |
| 42 | L66 | seed | not explicit (LlamaFactory default 42) | unverified | |
| 3 (ZeRO-3) | L66 | DeepSpeed stage | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json) | verified | |
| 2 / 48 GB | L66/L103 | A6000 count / VRAM | run scripts; RTX A6000 datasheet | verified/literature | |

### Table A.3 — GRPO (see Corrections §D)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 7B-frozen-6K SFT ckpt-564 | L80 | base/ref policy | run_grpo script MODEL_NAME_OR_PATH | verified | |
| 4 | L81 | group size G | run-2 training_args num_generations=4 | verified | |
| 0.2 / 0 | L82-83 | clip ε / KL β | run-2 training_args | verified | |
| 1e-6 | L85 | learning rate | run-2 training_args | verified | |
| 0.1 | L86 | LR warmup ratio | saved arg warmup_ratio=**None** (schedule linear verified) | mismatch | linear verified; warmup_ratio None |
| (0.9, 0.95) | L87 | AdamW β | β1=0.9 OK, β2=**0.999** | mismatch | (0.9, 0.999) |
| 0.01 | L88 | weight decay | run-2 args = **0.0** | mismatch | 0.0 |
| 2 | L89 | per-device BS | run-2 args = **1** (EBS 8 via GA4×2GPU) | mismatch | 1 |
| 4 / 8 | L90-91 | grad accum / EBS | run-2 args GA=4; EBS=1×4×2=8 | verified | |
| 2048 | L92 | max prompt length | run-2 args = **4096** | mismatch | 4096 |
| 512 | L93 | max completion length | run-2 args=512 | verified | |
| 4,236 / 2,118 / 2,118 | L94 | samples / anomaly / normal | grpo_train.json | verified | |
| 1,060 | L95 | max training steps | args max_steps=−1 (derived from epochs=2 over 4236); 1060 is the derived total | unverified | |
| 250 | L96 | save every | run-2 args save_steps=**530** | mismatch | 530 |
| 250 | L97 | eval every | run-2 args eval_steps=**None** (post-hoc eval) | mismatch | None |
| (0.3,0.3,0.2,0.2) | L98 | reward weights | ground truth (custom combiner; TRL arg None) | verified | |
| nomic-embed-text-v2-moe | L99 | type-embedding model | gemini_judge_server_v2.py L142; server :5200 | verified | |
| 1024×512 | L100 | stitch size (LEFT=ref RIGHT=test) | production run-2 used `--single_img 1` (NO stitch); 1024×512 is the unused 1-shot path | mismatch | N/A to run-2 (single image, no stitch) |
| 42 | L102 | seed | run-2 args seed=42 | verified | |
| 2 | L103 | A6000 count | run script CUDA_VISIBLE_DEVICES=1,2; nproc=2 | verified | |
| 480000 | (run-2 script only) | max_pixels | run_grpo_7b_resume_run2.sh `--max_pixels 480000` (appendix omits) | verified | |

---

## Appendix B — Prompts

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| two | L5 | prompts reproduced (Gemini + inference) | two `\section` blocks | verified | |
| 2.5 | L5/L8 | Gemini 2.5-Flash | ground truth; scripts/00_generate | verified | |
| SIX (6) | L14/L25-36 | six-phase inspection template | test_v2 encodes A-F flow; explicit "SIX-PHASE" labelling is paraphrase | unverified | |
| 120-200 | L42/L48/L74 | `<think>` word budget | inspector_prompt_test_v2.txt L130; corpus mean 136.9 (~55.6% strictly inside) | verified | |
| 9 | L51-55 | location grid cells (3×3) | inspector_prompt_test_v2.txt L258-273 | verified | |
| 11 / 7 / 4 / 4 | L58-63 | Surface / Structural / Completeness / Other defect types | inspector_prompt_test_v2.txt L287-319 | verified | |
| 26 | L57-63 (derived) | total controlled vocabulary | 11+7+4+4 | verified | |
| 5 | L65-71 | auto-reject rules (condensed from 6) | ground truth; test_v2 L404-426 (A-F) | verified | |
| 3 | L17-23 | images inspector may see (anomaly) | ground truth; test_v2 L16-22 | verified | |

---

## Appendix C — Per-product tables (see Corrections §C)

> **`tab:perprod-full` Base and SFT columns are GENUINE** (real JSONs, ~0.1 truncation). The **SFT+GRPO (GRPO) column is FABRICATED**, and the entire **`tab:perprod-cm` is FABRICATED** (CM rows physically impossible). Only the column **Average (82.73)** and the bold **CM Total (971/383/61/255)** are genuine.

### Genuine averages & headline

| Number | Where | Describes | Source | Status |
|---|---|---|---|---|
| 82.73 | L34/L67 | SFT+GRPO Average / CM Total BA | `results/grpo_..._run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json` (971/383/61/255) | verified |
| 70.49 | L34 | Base avg per-product BA | mean of 15 genuine base cells (~70.58, truncated) | verified |
| 80.05 | L34 | SFT avg per-product BA | genuine ckpt-564 avg 80.13 (truncated; overall 80.16) | verified |
| +9.56 / +2.68 | L34 | avg Δ SFT / Δ GRPO | mean of printed delta cells (9.55 / 2.69) | verified |
| 530 / 2 / 7B / 6K / 3 | caption L10 | ckpt / run / size / split / epoch | ground truth | verified |

### Genuine Base & SFT per-product cells (DS-MVTec)

| Product | Base BA | SFT BA | Δ SFT | Source (recomputed) | Status |
|---|---|---|---|---|---|
| metal_nut | 57.82 | 83.62 | +25.80 | base 57.86 / ckpt-564 83.64 | verified |
| cable | 50.93 | 70.13 | +19.20 | base 51.09 / 70.33 | verified |
| pill | 60.12 | 76.12 | +16.00 | base 60.23 / 76.21 | verified |
| toothbrush | 62.43 | 78.23 | +15.80 | base 62.50 / 78.33 | verified |
| screw | 55.88 | 70.78 | +14.90 | base 55.88 / 70.79 | verified |
| transistor | 66.95 | 80.25 | +13.30 | base 67.08 / 80.42 | verified |
| grid | 83.27 | 92.07 | +8.80 | base 83.33 / 92.11 | verified |
| leather | 90.65 | 98.25 | +7.60 | base ~90.5 / 98.44 | verified |
| zipper | 62.53 | 70.03 | +7.50 | base 62.61 / 70.10 | verified |
| capsule | 57.72 | 64.02 | +6.30 | base 57.80 / 64.06 | verified |
| hazelnut | 60.57 | 66.47 | +5.90 | base 60.71 / 66.61 | verified |
| bottle | 76.73 | 80.33 | +3.60 | base 76.87 / 80.48 | verified |
| carpet | 90.34 | 91.14 | +0.80 | base 90.45 / 91.27 | verified |
| wood | 97.50 | 97.40 | −0.10 | base 97.50 / 97.37 | verified |
| tile | 83.92 | 81.82 | −2.10 | base 83.93 / 81.87 | verified |

### FABRICATED GRPO column + CM rows — claimed vs genuine

Every "GRPO BA" cell in `tab:perprod-full` (L18-32) and every CM row in `tab:perprod-cm` (L51-65) is fabricated. The Δ GRPO cells (L18-32) are arithmetic over the fabricated GRPO cells → **unverified**. Genuine values (run-2 ckpt-530):

| Product | Claimed GRPO BA | Genuine GRPO BA | Claimed CM (TP/TN/FP/FN) | Genuine CM (TP/TN/FP/FN, n) |
|---|---|---|---|---|
| metal_nut | 87.40 | **87.47** | 87/19/1/23 (sum130>92) | 62/19/3/8 (n=92) |
| cable | 75.50 | **68.16** | 95/26/1/30 (sum152>150) | 35/57/1/57 (n=150) |
| pill | 79.20 | **75.45** | 78/35/5/18 | 120/3/2/12 (n=137) |
| toothbrush | 82.00 | **83.33** | 24/16/0/8 (sum48>42) | 25/10/2/5 (n=42) |
| screw | 74.55 | **72.89** | 80/36/4/27 | 69/36/5/50 (n=160) |
| transistor | 83.10 | **75.42** | 50/25/0/10 | 31/44/16/9 (n=100) |
| grid | 92.95 | **94.74** | 70/28/1/6 (sum105>76) | 51/19/0/6 (n=76) |
| leather | 98.40 | **98.44** | 92/32/0/3 (sum127>124) | 92/31/1/0 (n=124) |
| zipper | 74.10 | **68.84** | 96/28/0/31 (sum155>151) | 56/29/3/63 (n=151) |
| capsule | 67.65 | **69.78** | 86/28/5/37 (sum156>132) | 100/11/12/9 (n=132) |
| hazelnut | 73.10 | **88.57** | 50/28/0/18 | 68/32/8/2 (n=110) |
| bottle | 81.05 | **76.27** | 83/27/4/20 (sum134>83) | 52/14/6/11 (n=83) |
| carpet | 91.55 | **95.51** | 91/23/0/16 (sum130>117) | 81/28/0/8 (n=117) |
| wood | 97.55 | **97.50** | 79/17/0/4 (sum100>79) | 57/19/0/3 (n=79) |
| tile | 82.90 | **89.83** | 89/14/8/21 (sum132>117) | 72/31/2/12 (n=117) |
| **CM Total (bold)** | — | — | fabricated rows sum 1150/382/29/272 | **GENUINE 971/383/61/255** |

---

*End of ledger. Recompute any BA with `python results/compute_ba.py <eval_json>`.*
