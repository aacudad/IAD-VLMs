# Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection

> TU Delft MSc thesis (2026) — Adnane Acudad. This repository is the reproducible code-and-data companion to the thesis. It is meant to be the **front door** for the next student who inherits this work.

This project teaches a vision-language model (VLM) not just to say *whether* an
industrial part is defective, but to **explain why** in natural language. We start
from a base VLM, fine-tune it on Gemini-2.5-Flash reasoning traces, and then improve
the *corpus* rather than the optimiser. Across the board this lifts balanced accuracy
on held-out anomaly benchmarks (DS-MVTec, VisA, Real-IAD) while producing
human-readable inspection reasoning. All headline numbers in this README were
**recomputed from the raw evaluation JSONs** shipped in `results/`, see
[`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) and the provenance
docs linked below.

### The method: Keep-Correct-Revise (KCR)

**KCR** is the corpus-curation loop that produces every headline model here. Roll out
k = 8 answers from the current policy on the training pool, then bucket each item:

1. **Keep** the traces the policy already gets right. They are on-policy and they are correct, so they need no teacher at all.
2. **Correct** the ones it gets wrong. A teacher (Gemini) rewrites the trace from the gold verdict.
3. **Revise** the ones that are right but weakly grounded. The verdict is correct, the reasoning is not really looking at the image, so the teacher revises it.

Then fine-tune the **base** model on the result. Not the policy that produced the
rollouts, the base model. That is the whole method. It needs no reward function, no
KL term and no rollout budget at training time.

> **Naming.** The paper calls this **KCR**. This repository calls it **Arm C**, because
> it is the third arm of a three-arm teacher-distillation ablation (A = keep only,
> B = keep + correct, C = keep + correct + revise = KCR). Every filename, dataset key
> and results directory uses `abc_C` / `arm_C` / `iter2`. They are the same thing.
> "Arm C" below is used only where one arm of the ablation is meant.

---

## 1. Headline results

**Balanced accuracy (BA)** on the two primary benchmarks, defined as
`BA = 0.5 * (TP/(TP+FN) + TN/(TN+FP))` and computed straight from the `metrics`
block (`tp, tn, fp, fn`) of each eval JSON. Recompute any cell with
[`results/compute_ba.py`](results/compute_ba.py); the full inventory of every
checkpoint we ever evaluated is in [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).

#### Main line, Qwen2.5-VL-7B

| Stage | Model / checkpoint | DS-MVTec BA | VisA BA | Held-out Real-IAD BA | Source eval JSON |
|---|---|---:|---:|---:|---|
| Base | Qwen2.5-VL-7B (zero-shot) | 69.01 | 53.79 | — | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/) |
| Control | Labels-only, same 6K images, ckpt-188 | 77.86 | 68.64 | — | [`results/sft_qwen25vl_7b_6k_noreason/checkpoint-188/`](results/sft_qwen25vl_7b_6k_noreason/checkpoint-188/) |
| SFT | 7B-frozen-6K, ckpt-564 | 80.16 | 64.78 | — | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/) |
| **KCR SFT (headline)** | 7B KCR / Arm-C, ckpt-376 | **82.80** | **72.07** | — | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/) |
| GRPO (headline RL) | 7B GRPO Run-2, ckpt-530 | 82.73 | 70.39 | **80.87** | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/) |
| Reference | IAD-R1 (recanonicalized) | 81.92 | 71.34 | — | [`results/iad_r1_qwen_recanon/`](results/iad_r1_qwen_recanon/) |

#### Cross-architecture, LLaVA-OneVision-7B-SI

Same harness, same subsets, same recipe. This is IAD-R1's own headline backbone, so it
is the hardest place to argue that our result is a Qwen artefact.

| Stage | Model / checkpoint | DS-MVTec BA | VisA BA | Source eval JSON |
|---|---|---:|---:|---|
| Base | LLaVA-OV-7B-SI zero-shot, fair yes/no prompt | 75.66 | 53.80 | [`results/llava_ov_7b_zeroshot_eval/`](results/llava_ov_7b_zeroshot_eval/) |
| SFT | 6K Gemini traces, epoch 1 | 85.91 | 68.26 | [`results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/`](results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/) |
| SFT + GRPO | from SFT ep1, ckpt-530 | 87.66 | 72.58 | [`results/grpo_llava_ov_from_ep1/checkpoint-530/`](results/grpo_llava_ov_from_ep1/checkpoint-530/) |
| **KCR SFT** | native KCR corpus, epoch 4 | **88.45** | **74.25** | [`results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/`](results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/) |

#### Corpus transfer, Qwen3-VL-8B-Instruct

| Stage | Model / checkpoint | DS-MVTec BA | VisA BA | Source eval JSON |
|---|---|---:|---:|---|
| Base | Qwen3-VL-8B-Instruct zero-shot | 78.68 | 64.45 | [`results/qwen3vl_8b_baseline_eval/`](results/qwen3vl_8b_baseline_eval/) |
| **KCR SFT** | Qwen KCR corpus, ckpt-376 | **85.82** | **76.52** | [`results/sft_qwen3vl_8b_armC/checkpoint-376/`](results/sft_qwen3vl_8b_armC/checkpoint-376/) |

#### Explanation quality (the point of the whole project)

Detection accuracy is only half of it. A Gemini-3-Flash judge scores the reasoning on
five axes (visual grounding, defect faithfulness, evidence before conclusion, coherence,
conciseness), 0 to 2 each, summed to 0 to 10. Median of three judge samples, n = 100
product-diverse traces per model per benchmark, each model scored on **its own** correctly
detected anomalies.

| Model | DS-MVTec /10 | VisA /10 |
|---|---:|---:|
| **KCR / Arm-C SFT** | **9.05** | **8.52** |
| IAD-R1, asked the same way (prompt-matched) | 6.14 | 7.04 |
| IAD-R1, under its own prompt | 4.50 | 6.32 |

The two IAD-R1 rows answer different questions and both belong here. Its own GRPO prompt
never asks for reasoning, so about 35 to 40 percent of its answers are a bare "Yes" and
score 0. That is explanation *reliability*. The prompt-matched row asks it the same way we
ask our models, all of its answers then carry a trace, and that is explanation *quality*.
Quoting only 4.50 understates IAD-R1. Full method and per-axis breakdown in
[`results/explainability_multi/README.md`](results/explainability_multi/README.md).

Benchmark sizes: DS-MVTec full *n* = 1670, VisA full *n* = 2141, held-out Real-IAD
*n* = 4236. The GRPO Run-2 checkpoint matches KCR on DS-MVTec while generalizing to a
Real-IAD split it never saw during SFT or RL.

### The cross-architecture result, and its caveat

**This is the strongest thing in the repository.** On LLaVA-OneVision-7B-SI the KCR corpus
reaches **88.45 / 74.25** from supervised fine-tuning alone. SFT plus GRPO on that same
backbone reaches **87.66 / 72.58**. Corpus curation beats reinforcement learning on IAD-R1's
own architecture, which is the same ordering we already measured on Qwen2.5-VL-7B. So it is
a property of the method, not of one backbone.

GRPO is not useless there. It lifts its own init (LLaVA 6K SFT epoch 1, 85.91 / 68.26) by
+1.75 DS-MVTec and +4.32 VisA. It just does not reach what a better corpus reaches for free.

The LLaVA KCR corpus is a **native** loop, not the Qwen corpus reused. The rollouts came
from a LLaVA policy and the teacher corrected and revised LLaVA's own failures. Only the
rollout sampler changed between the two backbones. Every judging, bucketing and correction
script is shared.

> **Contamination caveat, DS-MVTec only.** The LLaVA-OneVision training mixture
> (`lmms-lab/LLaVA-OneVision-Data`, config `vision_flan(filtered)`) contains **426 rows**
> whose id matches `%MVTecAD%`. VisA matches **0**. Every DS-MVTec number for a
> LLaVA-derived model in this repo therefore carries a pretraining-exposure asterisk,
> including IAD-R1's released checkpoint. **VisA does not.** Read the VisA column as the
> clean one. Verified live through the Hugging Face datasets-server filter.

> **One comparison we do not make.** Our IAD-R1 reference row (81.92 / 71.34) is their
> released **Qwen2.5-VL-7B** checkpoint re-evaluated on our harness. We never re-evaluated
> their LLaVA-OneVision checkpoint on the full subsets. So "we beat IAD-R1 on its own
> backbone" is not a claim this repo supports. What it supports is that our pipeline on
> their backbone scores 88.45 / 74.25 under the same harness that scores their released
> model 81.92 / 71.34.

### Corpus transfer to a newer backbone

The KCR corpus was curated with a Qwen2.5-VL-7B policy. On **Qwen3-VL-8B-Instruct**, a
backbone it was never rolled out from, it still lifts the model from **78.68 / 64.45** to
**85.82 / 76.52**, so **+7.14 DS-MVTec and +12.07 VisA**. Same corpus file, same recipe as
the Qwen2.5 Arm-C run, only the model and the chat template change.

**There is no GRPO comparison on that backbone.** Nothing was RL-trained on Qwen3-VL-8B.
This line supports a corpus-transfer claim and nothing more. The corpus-versus-RL comparison
exists only on Qwen2.5-VL-7B and on LLaVA-OneVision-7B-SI.

### How much of the gain is the reasoning?

The obvious control. Take the exact 6,000 images, strip every trace, keep only the yes/no
label, and train the same model with the same hyperparameters on the same checkpoint grid.
Labels-only peaks at **77.86 / 68.64**. KCR on the same images reaches **82.80 / 72.07**.
Stripping the reasoning supervision costs **-4.94 DS-MVTec and -3.43 VisA**. Fine-tuning on
our images alone does a lot of the work, and the reasoning supervision adds the rest.
Details in [`results/sft_qwen25vl_7b_6k_noreason/NOTE.md`](results/sft_qwen25vl_7b_6k_noreason/NOTE.md).

> The thesis text contained several transcription and table-construction errors that
> were **corrected this session** against the raw JSONs. If a number in the thesis PDF
> disagrees with a number here, **the data in `results/` is authoritative**. Every
> known mismatch is logged in [`UNVERIFIED.md`](UNVERIFIED.md) and
> [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md).

### Released on Hugging Face

The trained checkpoints and the trace corpus are openly released:

| Backbone | Artifact | DS-MVTec / VisA | Link |
|---|---|---:|---|
| — | **AnomalyThink** dataset (reasoning traces) | — | <https://huggingface.co/datasets/aacudad/AnomalyThink> |
| Qwen2.5-VL-7B | **KCR SFT**, thesis headline | 82.80 / 72.07 | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-KCR> |
| Qwen2.5-VL-7B | SFT-6K | 80.16 / 64.78 | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT> |
| Qwen2.5-VL-7B | SFT + GRPO | 82.73 / 70.39 | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO> |
| Qwen2.5-VL-7B | GRPO-on-KCR, research preview / future work | 82.95 / 72.62 | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-KCR-GRPO> |
| LLaVA-OneVision-7B-SI | **KCR SFT**, best model in this repo | **88.45 / 74.25** | <https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-KCR> |
| LLaVA-OneVision-7B-SI | SFT-6K | 85.91 / 68.26 | <https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT> |
| LLaVA-OneVision-7B-SI | SFT + GRPO | 87.66 / 72.58 | <https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT-GRPO> |

> **Two model ids were renamed.** `AnomalyThink-Qwen2.5-VL-7B` is now
> `AnomalyThink-Qwen2.5-VL-7B-KCR`, and `AnomalyThink-Qwen2.5-VL-7B-ArmC-GRPO` is now
> `AnomalyThink-Qwen2.5-VL-7B-KCR-GRPO`. Hugging Face redirects the old ids, but use the new
> ones in anything you write.

The **AnomalyThink** dataset now has a `llava_kcr/` folder holding the LLaVA arms of the KCR
loop: **A 8,998** (keep only), **B 9,124** (keep + correct) and **C 6,000** (keep + correct +
revise = KCR, 50/50 anomaly/normal). Arm C is the corpus behind the 88.45 / 74.25 model. The
Qwen arms stay in `teacher_ablation_abc/` and `iter2/`.

The dataset ships our reasoning traces only. The underlying Real-IAD images are not redistributed (obtain Real-IAD separately).

### Explore the reasoning (trace viewers)

Self-contained HTML viewers show, per model, EVERY DS-MVTec and VisA sample with the true defect region overlaid in red, alongside that model's own generated reasoning trace and verdict (correct/incorrect). Open any file directly in a browser (images embedded, nothing to fetch): [`docs/trace_viewers/index.html`](docs/trace_viewers/index.html) links one page per model per benchmark (e.g. `armC_dsmvtec.html`, `armC_visa.html`; base is DS-MVTec only). Regenerate with [`scripts/05_figures/build_trace_viewer.py`](scripts/05_figures/build_trace_viewer.py) (`--all` for every sample, `--per-bench N` for a diverse subset).

---

## 2. Repository layout

```
repository_tu_delft_vlms/
├── scripts/        # The pipeline, numbered in execution order
│   ├── 00_generate/      # Download source data + generate Gemini reasoning traces
│   │   └── trace_audit/  #   Gemini grounding audit of our own corpus (code only, sweep incomplete)
│   ├── 01_sft/           # LlamaFactory SFT launchers: Qwen2.5-VL 3B/7B (6K, 15K, Arm A/B/C),
│   │                     #   LLaVA-OneVision-7B, Qwen3-VL-8B, labels-only control
│   ├── 02_grpo/          # GRPO training (Qwen + LLaVA launchers) + type-embedding / Gemini judge servers
│   ├── 03_rollout_star/  # The KCR loop: rollout -> bucket -> judge -> correct/revise -> rebuild SFT.
│   │                     #   Qwen and LLaVA share every stage after the rollout sampler.
│   │                     #   Also Arm D, the STaR self-rationalisation arm (code, results pending)
│   ├── 04_eval/          # Evaluation harness + watchers + summarizers (DS-MVTec / VisA / Real-IAD),
│   │                     #   incl. the vLLM LLaVA path and the explainability judge
│   └── 05_figures/       # Thesis figure + HTML audit generators
├── configs/        # All training configs
│   ├── sft/              # LlamaFactory SFT YAMLs (one per run, all three backbones)
│   ├── grpo/             # GRPO config (currently empty placeholder, see 02_grpo/*.sh)
│   ├── deepspeed/        # ZeRO-3 (+ CPU offload) DeepSpeed configs
│   ├── dataset_info.json # LlamaFactory dataset registry mapping names -> trace JSONs
│   └── dataset_info_additions.json  # Newer registry keys, merge into the above before training
├── prompts/        # Inspector / "think" system prompts used for generation and eval
├── traces/         # TEXT-ONLY datasets (LlamaFactory {messages, images} format)
│   ├── anomalythink_6k/      # 6K SFT split (3000 anomaly + 3000 normal)
│   ├── anomalythink_15k/     # Full ~14,472-trace corpus + SFT/GRPO partition files
│   ├── teacher_ablation_abc/ # Arm A + Arm B teacher-distillation datasets (Qwen), and the 10K polish set
│   ├── iter2/                # The Qwen KCR corpus. sft_iter2_train.json (6,000) is Arm C,
│   │                         #   the file behind the 82.80 / 72.07 headline model
│   └── variety_star_6k/      # In-progress Real-IAD Variety STaR corpus
├── results/        # Eval JSONs + trainer_state.json per run, plus the BA tooling
│   ├── eval_ba_inventory.txt # Every (BA, n, path) we computed, the source of truth
│   ├── compute_ba.py         # Recompute BA from any eval JSON
│   ├── explainability_multi/ # 5-axis explanation-quality judge output + figures
│   └── <run>/checkpoint-X/eval_<bench>_full_<mode>.json
├── docs/           # pipeline_overview (PDF/TeX); data_card.md, model_card.md, prompt_modes.md
└── env/            # Conda / environment capture (see Quickstart)
```

Run `ls -R` from the repo root to see the exact tree. The directories above are stable.

**Where the new results live.** Every run family added for the cross-architecture work follows
the same `results/<run>/checkpoint-X/eval_<bench>_full_<mode>.json` layout, and each folder has a
`NOTE.md` with the per-epoch confusion matrices and the caveats:

| Folder | What |
|---|---|
| [`results/llava_ov_7b_zeroshot_eval/`](results/llava_ov_7b_zeroshot_eval/) | LLaVA-OV-7B-SI base row, fair yes/no prompt |
| [`results/sft_llava_ov_7b_frozen_iad_sft_6k_train/`](results/sft_llava_ov_7b_frozen_iad_sft_6k_train/) | LLaVA 6K Gemini SFT, 4 epochs |
| [`results/grpo_llava_ov_from_ep1/`](results/grpo_llava_ov_from_ep1/) | LLaVA SFT + GRPO, ckpt-530 |
| [`results/sft_llava_ov_7b_frozen_llava_iter1_C/`](results/sft_llava_ov_7b_frozen_llava_iter1_C/) | **LLaVA native KCR, 4 epochs, the 88.45 / 74.25 run** |
| [`results/qwen3vl_8b_baseline_eval/`](results/qwen3vl_8b_baseline_eval/) | Qwen3-VL-8B base row |
| [`results/sft_qwen3vl_8b_armC/`](results/sft_qwen3vl_8b_armC/) | Qwen3-VL-8B on the Qwen KCR corpus, 4 epochs |
| [`results/sft_qwen25vl_7b_6k_noreason/`](results/sft_qwen25vl_7b_6k_noreason/) | Labels-only control, 4 epochs |

---

## 3. What is and is NOT included

**Included (everything needed to re-derive the work):**
- All **training and eval code** (`scripts/`), all **configs** (`configs/`), all **prompts** (`prompts/`).
- The complete **reasoning-trace datasets** (`traces/`) as plain-text LlamaFactory
  JSONs. Each record is `{"messages": [...], "images": [...]}`. The `images` entries
  are **absolute cluster paths** (`/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/...`)
  — they work as-is on the TU Delft cluster; on any other machine, re-download the
  images (see below) and remap the prefix with
  [`traces/rewrite_image_paths.py`](traces/rewrite_image_paths.py). The actual
  pixels are not shipped.
- All **evaluation results** (`results/`): per-checkpoint eval JSONs with confusion-matrix
  counts, `trainer_state.json` loss/metric curves, the BA inventory, and `compute_ba.py`.

**NOT included (re-downloadable or re-trainable):**
- **Model weights.** The large base/SFT/GRPO weights are not committed here, but the
  headline checkpoints for all three backbones are released on Hugging Face (see
  *Released on Hugging Face* in Section 1 above). Alternatively, download Qwen2.5-VL,
  LLaVA-OneVision-7B-SI or Qwen3-VL-8B-Instruct and re-run SFT/GRPO with the shipped configs.
- **Source / overlay images.** Real-IAD, MVTec/DS-MVTec, and MMAD are all
  re-downloadable — use the `download_*.py` scripts in [`scripts/00_generate/`](scripts/00_generate/).
  After downloading, point the trace JSONs at your copy with
  `python traces/rewrite_image_paths.py --old-prefix /bulk/aacudad/reasoning_traces/reasoning_traces_gen/data --new-prefix <your-data-dir> traces/**/*.json`.
- **The laptop backup** and any large scratch artifacts.

---

## 4. The data corpus (AnomalyThink-15K)

The reasoning supervision is **~14,472 Gemini-2.5-Flash traces** (single-pass "C1-only"),
generated on Real-IAD parts (temp 1.0, target 120–200 words inside `<think>`, 3 images per
anomaly = original + overlay + reference, 1 image per normal). Quality control is **in-prompt
auto-reject rules plus JSON/Pydantic schema validation** — there is no second-reviewer model
and no rubric filter.

The corpus is partitioned into **three disjoint, stratified (product × normal/anomaly) splits**:

| Split | Size | Used for |
|---|---:|---|
| SFT | 6,000 | Supervised fine-tuning (the headline SFT model) |
| GRPO | 4,236 (2,118 anomaly + 2,118 normal) | RL training |
| Held-out Real-IAD | 4,236 | Generalization test set |

The **6K SFT split is a strict subset** of the 14,472, and both have the same mean trace
length (~141 words). So the empirical "**6K SFT beats 15K SFT**" result is a
**data-count / composition effect at fixed compute** — not a length, verbosity, or
quality-filtering effect. See [`docs/data_card.md`](docs/data_card.md) for the full
breakdown (products per split, generation settings, QC rules) and the *in-progress*
**Variety STaR** corpus (rollout Arm-C ckpt-376 → Gemini judge → correct/rewrite → SFT on a
6K stratified variety set), which is currently a thesis placeholder only.

### The KCR corpora

The 14,472 Gemini traces are the *starting* corpus. The headline models train on a KCR-curated
corpus built on top of them, one per backbone:

| Corpus | Size | Where |
|---|---:|---|
| Qwen KCR (Arm C), the 82.80 / 72.07 and 85.82 / 76.52 models | 6,000 (50/50) | [`traces/iter2/sft_iter2_train.json`](traces/iter2/sft_iter2_train.json), key `iad_sft_iter2` |
| Qwen Arm A (keep only) | 2,978 | [`traces/teacher_ablation_abc/sft_A_kept_balanced.json`](traces/teacher_ablation_abc/sft_A_kept_balanced.json) |
| Qwen Arm B (keep + correct) | 4,369 | [`traces/teacher_ablation_abc/sft_B_kept_corrected.json`](traces/teacher_ablation_abc/sft_B_kept_corrected.json) |
| LLaVA KCR (Arm C), the 88.45 / 74.25 model | 6,000 (50/50) | Hugging Face dataset, `llava_kcr/sft_llava_C_train.json` |
| LLaVA Arm A / Arm B | 8,998 / 9,124 | Hugging Face dataset, `llava_kcr/` |

As of 2026-09-02 the LLaVA arms live on Hugging Face and not under `traces/`. Arms A and B were
never trained on for LLaVA, so they are pool that supports no number in this repo, and shipping
only Arm C would make the folder read as if the other two did not exist. If a `traces/llava_kcr/`
folder later appears, it holds the same files.

The **bucket counts** of the loop itself, before any balancing. Qwen: 8,872 kept, 1,364
corrected, 2,406 rewritten. LLaVA: out of 10,236 rollouts, 9,179 kept, 1,057 needing correction
and 2,499 needing a rewrite locally, after which the Gemini judge demoted a further 133 from keep
to rewrite. The arm files above are **balanced, stratified subsets** of those buckets, which is
why Arm A is 2,978 and not 8,872. Both loops end at a 6,000-record 50/50 Arm C.

The ablation is what justifies the third step. On Qwen2.5-VL-7B, best-epoch DS-MVTec goes
**A 79.01, B 80.75, C 82.80**. Keeping alone is worth something, adding teacher correction is
worth about 1.7 pp more, and adding the revise step is worth about 2 pp on top of that. Full
reasoning in [`CLAIMS_EVIDENCE.md`](CLAIMS_EVIDENCE.md) §1.6.

---

## 5. The prompt-mode contract

The evaluation prompt that produced a result is encoded in the **eval JSON filename suffix**,
never inferred from context. The modes are:

- `_trainprompt` is the exact instruction template the model was trained with. Headline numbers use this.
- `_grpoprompt` is the GRPO-time prompt.
- `_bareprompt` is a minimal, unconditioned prompt.
- `_iadr1native` is IAD-R1's own prompt, used only on IAD-R1's released checkpoint.
- `_yesnouser` appends "Answer with yes or no." to the user turn. It exists for base models that do not emit the `<think>` / `<answer>` schema. Without it the strict parser reads every sample as "no" and returns BA exactly 50.00, which is a parser artefact and not a measurement. The LLaVA-OV base row uses this mode.
- `_noreasonprompt` is the labels-only control's prompt, which asks for a bare verdict and nothing else. It is character-identical to that run's training prompt.

So `eval_dsmvtec_full_trainprompt.json` is "DS-MVTec, full set, train-prompt mode".
**Comparing across modes is only fair within the same suffix.** Full details and the
exact prompt strings are in [`docs/prompt_modes.md`](docs/prompt_modes.md).

One extra tail exists and it is **not** a prompt mode. A trailing `_vllm`, as in
`eval_dsmvtec_full_trainprompt_vllm.json`, records that the model was **served through vLLM**
instead of the HuggingFace generate path. The prompt is unchanged. LLaVA-OneVision needs it
because the HF path costs about 9 s per sample against 0.71 s for vLLM. On a probe the two
paths agreed on 99 percent of samples.

---

## 6. Quickstart and where to go next

**Environment** (captured under [`env/`](env/)):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
# WORK_DIR is the workspace holding Training/, outputs/, hf_cache/ and LlamaFactory/.
# Every script added for the LLaVA / Qwen3 / audit lines defaults it to the parent of
# this repository, so set it only if your layout differs.
export WORK_DIR=$(cd .. && pwd)
export HF_HOME=$WORK_DIR/hf_cache
```

Training uses **LlamaFactory** (`llamafactory-cli train <config.yaml>`); the GRPO trainer is
derived from IAD-R1 / AnomalyR1. Gemini generation runs via Vertex
(`GOOGLE_APPLICATION_CREDENTIALS`) or the API (`GEMINI_API_KEYS` in `.env`).

Three backbones are trained with the same LlamaFactory path. Qwen2.5-VL-7B and Qwen3-VL-8B use
the HuggingFace generate path for evaluation. **LLaVA-OneVision-7B-SI is served through vLLM for
evaluation and for KCR rollouts**, because its anyres image budget is about 3,700 image tokens
and the HF path costs about 9 s per sample against 0.71 s for vLLM. Those results carry a `_vllm`
tail on the filename, see section 5.

GRPO Run-2 (the headline RL config, read from the saved `training_args.bin`):
base/ref = 7B-frozen-6K SFT **ckpt-564**; `num_generations` G = **4**; KL coef `beta` = **0**
(KL is *monitored* via Schulman k3, not added to the loss); LR `eta` = **1e-6**; clip `epsilon`
= **0.2**; **4,236** training samples; max **1,060** steps (save/eval every 250); reward weights
`(w_f, w_a, w_t, w_l) = (0.3, 0.3, 0.2, 0.2)`; the reasoning reward (judge server :5100) is
**not used** → flat 0.5; type-embedding model `nomic-embed-text-v2-moe` (server :5200);
per-device BS 2, grad-accum 4 (EBS 8); max prompt 2048 / completion 512; bf16; seed 42; 2× RTX A6000
with ZeRO-3 CPU offload.

> Older thesis values of G=2, beta=0.04, eta=5e-6, or "13K samples" are **stale** — use the
> actuals above. See [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md).

**Companion docs — read these next:**
- [`RUNBOOK.md`](RUNBOOK.md) — step-by-step: download → generate → SFT → GRPO → eval.
- [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md) — every headline number → exact JSON + config.
- [`CLAIMS_EVIDENCE.md`](CLAIMS_EVIDENCE.md) — each thesis claim mapped to its supporting artifact.
- [`UNVERIFIED.md`](UNVERIFIED.md) — known thesis/data mismatches and unverifiable cells (read before trusting any thesis table).
- [`docs/data_card.md`](docs/data_card.md) and [`docs/model_card.md`](docs/model_card.md) — dataset and model documentation.
- [`docs/pipeline_overview.pdf`](docs/pipeline_overview.pdf) — the end-to-end pipeline diagram.
- `results/<run>/NOTE.md` — per-run reading notes. Every run family added for the
  cross-architecture work has one, with the per-epoch confusion matrices, which checkpoint was
  released, and the caveats that apply to that run only.
- [`results/explainability_multi/README.md`](results/explainability_multi/README.md) — the
  explanation-quality method, the five axes, and why IAD-R1 appears twice.

---

## 7. SECURITY — read before pushing anything public

> **This repository is NOT yet safe to publish.** Live credentials are still embedded in
> some of the helper scripts and must be removed first.

At time of writing, **Hugging Face tokens** and the **Vertex service-account key** are still
present in some `download_*.py` scripts and in the judge servers under
[`scripts/02_grpo/`](scripts/02_grpo/) and [`scripts/00_generate/`](scripts/00_generate/).
**Before the first public commit you MUST:**

1. **Rotate** every credential — revoke the HF tokens and the Vertex service-account key,
   issue fresh ones, and load them only from environment variables / a git-ignored `.env`.
2. **Add a `.gitignore`** that excludes `.env`, `*.json` service-account keys, `hf_cache/`,
   model weights, and any local image data.
3. **Scan with [gitleaks](https://github.com/gitleaks/gitleaks)** (`gitleaks detect --source .`)
   and confirm zero findings — including the full git **history**, not just the working tree.

Do not push to a public remote until all three are done and verified.

**Status note, 2026-09-02.** Step 2 is done. `.gitignore` excludes `.env`, `*key*.json`,
`vertexai-*.json`, `*.pem`, `hf_cache/`, weights and local image data, and `.env` is present on
disk but untracked. A pattern scan over all **tracked** files today found no live credential.
The only hits are the two documented placeholders (`.env.example` and `RUNBOOK.md`) and, in
`docs/trace_viewers/*.html`, a handful of `AIza...` strings that are random substrings inside
base64-encoded images, not API keys. They are 34 and 45 characters long, and a Google API key is
exactly 39. Expect a naive secret scanner to flag them.

**Steps 1 and 3 are still open.** Nobody has confirmed the credentials were rotated, and nobody
has run gitleaks over the full **history**, which is what actually matters once a repo is public.
Do that before treating this warning as cleared.
