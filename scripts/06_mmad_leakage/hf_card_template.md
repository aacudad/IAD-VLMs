---
license: cc-by-nc-sa-4.0
task_categories:
- image-text-to-text
- visual-question-answering
language:
- en
tags:
- industrial-anomaly-detection
- reasoning
- chain-of-thought
- vision-language-model
- benchmark-leakage
- mmad
pretty_name: AnomalyThink-MMAD
configs:
- config_name: all_8293
  default: true
  data_files: mmad_all_8293.json
- config_name: run1_unbalanced_train_1600
  data_files: splits/run1_unbalanced_train_1600.json
- config_name: run1_unbalanced_test_6693
  data_files: splits/run1_unbalanced_test_6693.json
- config_name: run2_balanced_train_1600
  data_files: splits/run2_balanced_train_1600.json
- config_name: run2_balanced_test_6693
  data_files: splits/run2_balanced_test_6693.json
- config_name: one_per_product_and_defect_144
  data_files: splits/one_per_product_and_defect_144.json
- config_name: one_per_product_39
  data_files: splits/one_per_product_39.json
- config_name: traces_with_provenance
  data_files: traces_8293.jsonl
---

# AnomalyThink-MMAD: reasoning traces on the MMAD benchmark, for a leakage experiment

**{{DATASET_ID}}** holds one structured reasoning trace for 8,293 of the 8,366 images of the
**MMAD** benchmark (DS-MVTec, VisA, GoodsAD, MVTec-LOCO), written with the same teacher recipe as
the [AnomalyThink](https://huggingface.co/datasets/aacudad/AnomalyThink) corpus of the MSc thesis
*Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* (TU Delft, 2026).

**Why it exists.** Several published VLM anomaly detectors put MMAD images into their training
data (OmniAD v1: "one example per category from MMAD" for both SFT and GRPO; AnomalyR1: 600 images
from the four MMAD source datasets) and then report MMAD scores. This dataset makes that practice
measurable. Training a Qwen2.5-VL-7B on 1,600 of these traces and testing on the other 6,693
images of the same products gives the tables in `RESULTS.md`. It is a side experiment, not part
of the thesis, and no MMAD image was used to train any thesis model.

**This is not a benchmark and not a recommended training set.** Anyone who trains on it and
reports MMAD numbers is doing exactly what the experiment quantifies.

## Important: images are NOT included
Every example references an MMAD image by relative path (`MMAD/DS-MVTec/bottle/image/broken_large/000.png`),
which is the key of the image in MMAD's `mmad.json`. Obtain MMAD from its official source
(<https://huggingface.co/datasets/jiang-cc/MMAD>, CC BY-NC-SA 4.0) and place it so the paths
resolve. This dataset inherits MMAD's non-commercial licence.

## How the traces were written
- Teacher: `gemini-3.6-flash` (Vertex AI), thinking level MINIMAL, JSON output, batches of 10.
- System prompt: `inspector_prompt_test_v2` of the thesis (six-phase inspection, XML tags,
  hard rules, auto-reject rules), user prompt of the v4 Real-IAD generator.
- Inputs per anomalous image: the image, the same image with the MMAD ground-truth mask as a red
  overlay, and a normal reference image of the same product taken from MMAD's `similar_templates`.
  Normal images: the image only.
- Hints: the MMAD multiple-choice answers for defect type, location, appearance and effect were
  given to the teacher as internal labels and are stored in `hints` of `traces_8293.jsonl`.
- Output: `<think>...</think>` then, for anomalies, `<location>` on a 3x3 grid, `<type>` from the
  thesis defect vocabulary, and `<answer>Yes</answer>`, otherwise `<answer>No</answer>`.
  Mean think length 113 words (anomalous) and 127 words (normal).
- Excluded: 69 images without a verdict question in MMAD and 4 MVTec-LOCO anomalies without a mask.

## Files
| file | rows | content |
|---|---|---|
| `mmad_all_8293.json` | 8,293 | ShareGPT (LLaMA-Factory) records, all traces |
| `traces_8293.jsonl` | 8,293 | one record per image with `hints`, `dataset`, `product`, `is_anomaly`, teacher, and the split membership of both runs |
| `splits/run1_unbalanced_train_1600.json` / `_test_6693.json` | 1,600 / 6,693 | 19.3 % of every (dataset, product, label) stratum, MMAD's own label ratio |
| `splits/run2_balanced_train_1600.json` / `_test_6693.json` | 1,600 / 6,693 | 800 anomalous + 800 normal, at most 60 % of a product's images of either label, 6,521 test images shared with run 1 |
| `splits/one_per_product_and_defect_144.json` | 144 | one image per (dataset, product, defect folder), the OmniAD "one example per category" reading |
| `splits/one_per_product_39.json` | 39 | one image per product, the other reading |
| `splits/split_keys.json` | | the key lists of every split |
| `RESULTS.md` | | the experiment: recipe, epoch tables, sensitivity/specificity, memorisation checks |

## Headline of the experiment (strict balanced accuracy on unseen images of the same products)
| Subset | Qwen2.5-VL-7B base | 1,600 MMAD traces, balanced, best epoch | thesis SFT on 6,000 Real-IAD traces (no MMAD) |
|---|---|---|---|
| DS-MVTec | 69.7 | 79.2 | 80.2 |
| VisA | 53.8 | 67.2 | 64.8 |
| GoodsAD | 51.0 | 61.1 | n/a |
| MVTec-LOCO | 50.5 | 54.7 | n/a |

Scoring the same checkpoint on its own training images gives only 1 to 2 points more than on the
held-out images, so the gain is familiarity with the benchmark's products, not memorisation.

## Citation
Cite the thesis and MMAD. Code: <https://github.com/aacudad/IAD-VLMs> (`scripts/06_mmad_leakage`).
