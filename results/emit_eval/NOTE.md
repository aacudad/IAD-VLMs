# EMIT baseline — single-image, no-RAG, binary anomaly detection (NOT in the thesis yet)

External baseline **EMIT** (Guan et al., *EMIT: Enhancing MLLMs for Industrial Anomaly Detection via
Difficulty-Aware GRPO*, arXiv:2507.21619) evaluated on **our** protocol so it is comparable to our
models. **This is not written into the thesis yet** — it is recorded here as evidence for a possible
external-baselines table.

## Results (our balanced accuracy, recomputed from tp/tn/fp/fn)

`BA = 0.5 * (TP/(TP+FN) + TN/(TN+FP))`

| Benchmark | BA | recall | specificity | tp | tn | fp | fn | n | unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS-MVTec | **77.42** | 82.54 | 72.30 | 1012 | 321 | 123 | 214 | 1670 | 0 |
| VisA | **76.37** | 69.26 | 83.47 | 829 | 788 | 156 | 368 | 2141 | 0 |
| **average** | **76.90** | | | | | | | | |

Source JSONs (with per-sample outputs): `emit_dsmvtec_binary.json`, `emit_visa_binary.json`.

## For context (all balanced accuracy)

| model | backbone | DS-MVTec | VisA | avg |
|---|---|---:|---:|---:|
| Arm-C (our headline SFT) | Qwen2.5-VL-7B | 82.80 | 72.07 | 77.44 |
| SFT-6K (traces) | Qwen2.5-VL-7B | 80.16 | 64.78 | 72.47 |
| **EMIT (single-img, no-RAG)** | **InternVL3-8B** | 77.42 | 76.37 | 76.90 |
| no-reasoning SFT (ablation, ckpt-564) | Qwen2.5-VL-7B | 76.51 | 69.39 | 72.95 |
| base Qwen2.5-VL-7B | Qwen2.5-VL-7B | 69.01 | 53.79 | 61.40 |

## Exactly what was and was NOT done (read before citing)

- **What EMIT natively expects:** a NORMAL reference image (one-shot) **+** a RAG "domain knowledge"
  text block, answering MMAD's multiple-choice questions. Its paper reports ~71.44 avg over the 7 MMAD
  MCQ subtasks in that native setting.
- **What we ran here (harder than native):** **single query image only, `rag=""` (NO RAG)**, on MMAD's
  binary "Anomaly Detection" question (`Is there any defect? A. Yes / B. No`), scored as balanced
  accuracy. EMIT supports single-image inference natively (its loader has `is_one_shot=False`), so unlike
  JUDO this is NOT an amputated setting — but it is harder than EMIT's headline one-shot+RAG setting and
  MUST be labelled that way. `unparseable=0`, recall 69-83% and specificity 72-83% => a genuine,
  non-degenerate baseline.
- **Off-backbone:** EMIT is **InternVL3-8B**, not our Qwen2.5-VL-7B — different architecture and 8B vs 7B.

## Contamination: NOT a concern in this no-RAG form

EMIT's four training **tasks** come from **Vision + Real-IAD + MPDD** — not from MVTec/VisA test images.
Per the paper (Data Collection), MVTec-AD / VisA / GoodsAD were touched only to build the RAG domain
knowledge: *"instead of directly using these subdatasets ... we randomly select five images of normal
objects for each object type and generate descriptive text ... to prevent data leakage."* So EMIT saw
only ~5 **normal** images per object type (text-only, for RAG), no defect images and no test images.
Since we strip RAG, we remove EMIT's only point of contact with the test domain — so there is **no
meaningful test-set leakage** in this no-RAG evaluation.

Reproduce with `scripts/04_eval/eval_emit_binary.py` (see `scripts/04_eval/EMIT_SETUP.md`).
