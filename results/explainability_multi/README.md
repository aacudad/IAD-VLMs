# Explainability judge — five reasoning axes, six models

Extension of the thesis §6.8 explainability evaluation from 2 models to **six**, so the full
pipeline progression (base → finetuned) is visible on **both backbones**.

## The five axes (each scored 0 / 1 / 2 by the judge)

| Axis | What it asks |
|---|---|
| **Visual grounding** | Do the claims point at specific, actually-visible regions/cues (vs generic statements that would fit any image)? |
| **Defect faithfulness** | Does the anomaly the trace describes correspond to the **real** defect in the red mask region — no hallucinated or different defect? |
| **Evidence before conclusion** | Does it describe the visual cues *first* and then conclude, rather than asserting the defect and back-filling? |
| **Coherence** | Are the steps internally consistent and do they actually support the conclusion? |
| **Conciseness** | Compact and on-point, no padded/repetitive template narration. (Length is never a positive signal.) |

`overall` = sum of the five axes (**0–10**). Two further metrics are **mechanical**, not judge-scored:
`loc_met` (predicted location hits the GT mask on a 3×3 grid, 0/1) and `type_sim` (Nomic embedding
similarity between predicted and GT defect type, 0–1).

## Method

- Each model is scored **independently on its own correctly-detected anomalies** (`gt=yes & pred=yes`),
  so weak base models are not penalised by a shrinking cross-model intersection.
- **n = 100** product-diverse traces per model per benchmark.
- Judge: **Gemini-3-Flash**, shown the original image + a **red GT-mask overlay** + the GT defect type +
  the model's trace. It is told the verdict is already correct, so it scores *reasoning quality only*.
- **Median of 3 judge samples** per trace (damps the run-to-run fluctuation).
- Identical methodology for every model → the table is internally consistent.

Sanity check vs the thesis §6.8 (which used a matched 2-model design): Arm-C **9.05** here vs 9.14 there,
IAD-R1 **4.50** vs 4.29 — i.e. consistent within noise.

## Files

| File | What |
|---|---|
| `explainability_5axes_heatmap.png` | models × 5 axes, one panel per benchmark (every cell annotated) |
| `explainability_5axes_bars.png` | grouped bars: 5 axes on x, models as series, per benchmark |
| `explainability_5axes_table.md` | the same numbers as a table |
| `summary.json` | aggregated means per model × benchmark (all 5 axes + loc/type + n) |
| **`raw_results.json`** | **raw per-sample judgments** — 1100 records, one per judged trace (model, bench, image_id, product, gt_defect, all 5 axis scores, overall, loc_met, type_sim) |

Regenerate: `scripts/04_eval/explainability_judge_multi.py` (scoring) →
`scripts/05_figures/plot_explainability_axes.py` (figures).

## The two IAD-R1 rows (read this before quoting its score)

IAD-R1 appears **twice**, because the number depends entirely on which prompt it is given:

| | DS-MVTec | VisA | what it measures |
|---|--:|--:|---|
| **IAD-R1 (native prompt)** | 4.50 | 6.32 | explanation **reliability** under its own GRPO prompt, which never asks for reasoning — ~40% of its answers are a bare "Yes" and score 0 |
| **IAD-R1 (prompt-matched)** | **6.14** | **7.04** | explanation **quality** when asked the same way as our models — then 100% of its answers contain a trace |

Both are honest; they answer different questions. Quoting only 4.50 understates IAD-R1. Either way it
stays clearly below Arm-C (9.05 / 8.52).

## Reading the result

- **Finetuning is what creates explainability.** Qwen3-VL base **1.79 → 9.23** (DS-MVTec) after SFT on the
  curated traces; the same holds on Qwen2.5. Both base models score **0.00 on loc and type** — they may
  describe, but they do not localise or name the defect in a structured way.
- **The AnomalyThink-trained models cluster high** (Arm-C 9.05/8.52, SFT+GRPO 9.24/8.62, Qwen3-SFT
  9.23/8.77) and clearly above **IAD-R1 (4.50/6.32)**.
- **Two nuances the axis breakdown exposes** that the overall score hides:
  - base Qwen2.5's weak point is specifically **evidence-before-conclusion (1.12** vs ~1.99 finetuned) —
    it concludes first and justifies after.
  - Qwen3-VL base scores ≈0 on everything **except conciseness (1.05 / 0.86)** — that axis rewards short
    output, and a model that barely reasons is trivially short. Do not read it as a strength.

⚠️ **base Qwen2.5-VL has no VisA traces** on disk, so it is DS-MVTec only (shown as `n/a`, never as 0).
