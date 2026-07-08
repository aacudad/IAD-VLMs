# Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection

> TU Delft MSc thesis (2026) — Adnane Acudad. This repository is the reproducible code-and-data companion to the thesis. It is meant to be the **front door** for the next student who inherits this work.

This project teaches a vision-language model (VLM) not just to say *whether* an
industrial part is defective, but to **explain why** in natural language. We start
from **Qwen2.5-VL** and improve it in three stages: (1) supervised fine-tuning (SFT)
on Gemini-2.5-Flash reasoning traces, (2) reinforcement learning with GRPO on a
disjoint split of those traces, and (3) an in-progress "Variety STaR" self-training
loop. Across the board this lifts balanced accuracy on held-out anomaly benchmarks
(DS-MVTec, VisA, Real-IAD) while producing human-readable inspection reasoning. All
headline numbers in this README were **recomputed from the raw evaluation JSONs**
shipped in `results/` — see [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt)
and the provenance docs linked below.

---

## 1. Headline results

**Balanced accuracy (BA)** on the two primary benchmarks, defined as
`BA = 0.5 * (TP/(TP+FN) + TN/(TN+FP))` and computed straight from the `metrics`
block (`tp, tn, fp, fn`) of each eval JSON. Recompute any cell with
[`results/compute_ba.py`](results/compute_ba.py); the full inventory of every
checkpoint we ever evaluated is in [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).

| Stage | Model / checkpoint | DS-MVTec BA | VisA BA | Held-out Real-IAD BA | Source eval JSON |
|---|---|---:|---:|---:|---|
| Base | Qwen2.5-VL-7B (zero-shot) | 69.01 | 53.79 | — | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/) |
| **SFT (headline)** | 7B-frozen-6K, ckpt-564 | **80.16** | **64.78** | — | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/) |
| SFT (Arm-C, teacher ablation) | 7B Arm-C, ckpt-376 | 82.80 | 72.07 | — | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/) |
| **GRPO (headline RL)** | 7B GRPO Run-2, ckpt-530 | **82.73** | **70.39** | **80.87** | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/) |
| Reference | IAD-R1 (recanonicalized) | 81.92 | 71.34 | — | [`results/iad_r1_qwen_recanon/`](results/iad_r1_qwen_recanon/) |

Benchmark sizes: DS-MVTec full *n* = 1670, VisA full *n* = 2141, held-out Real-IAD
*n* = 4236. The GRPO Run-2 checkpoint matches Arm-C on DS-MVTec while generalizing
to a Real-IAD split it never saw during SFT or RL.

> The thesis text contained several transcription and table-construction errors that
> were **corrected this session** against the raw JSONs. If a number in the thesis PDF
> disagrees with a number here, **the data in `results/` is authoritative**. Every
> known mismatch is logged in [`UNVERIFIED.md`](UNVERIFIED.md) and
> [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md).

### Released on Hugging Face

The trained checkpoints and the trace corpus are openly released:

| Artifact | Link |
|---|---|
| **AnomalyThink** dataset (reasoning traces) | <https://huggingface.co/datasets/aacudad/AnomalyThink> |
| **Arm-C SFT**, best model (82.80 / 72.07) | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B> |
| **SFT-6K** (80.16 / 64.78) | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT> |
| **SFT + GRPO** (82.73 / 70.39) | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO> |
| GRPO-on-Arm-C, research preview / future work (82.95 / 72.62) | <https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-ArmC-GRPO> |

The dataset ships our reasoning traces only. The underlying Real-IAD images are not redistributed (obtain Real-IAD separately).

### Explore the reasoning (trace viewers)

Self-contained HTML viewers show, per model, EVERY DS-MVTec and VisA sample with the true defect region overlaid in red, alongside that model's own generated reasoning trace and verdict (correct/incorrect). Open any file directly in a browser (images embedded, nothing to fetch): [`docs/trace_viewers/index.html`](docs/trace_viewers/index.html) links one page per model per benchmark (e.g. `armC_dsmvtec.html`, `armC_visa.html`; base is DS-MVTec only). Regenerate with [`scripts/05_figures/build_trace_viewer.py`](scripts/05_figures/build_trace_viewer.py) (`--all` for every sample, `--per-bench N` for a diverse subset).

---

## 2. Repository layout

```
repository_tu_delft_vlms/
├── scripts/        # The pipeline, numbered in execution order
│   ├── 00_generate/      # Download source data + generate Gemini reasoning traces
│   ├── 01_sft/           # LlamaFactory SFT launch scripts (3B/6K, 7B/6K, 15K, Arm A/B/C, iter2)
│   ├── 02_grpo/          # GRPO training + the type-embedding / Gemini judge servers
│   ├── 03_rollout_star/  # Iteration-2 / Variety STaR: rollout -> judge -> correct -> rebuild SFT
│   ├── 04_eval/          # Evaluation harness + summarizers (DS-MVTec / VisA / Real-IAD)
│   └── 05_figures/       # Thesis figure + HTML audit generators
├── configs/        # All training configs
│   ├── sft/              # LlamaFactory SFT YAMLs (one per run)
│   ├── grpo/             # GRPO config (currently empty placeholder; see 02_grpo/*.sh)
│   ├── deepspeed/        # ZeRO-3 (+ CPU offload) DeepSpeed configs
│   └── dataset_info.json # LlamaFactory dataset registry mapping names -> trace JSONs
├── prompts/        # Inspector / "think" system prompts used for generation and eval
├── traces/         # TEXT-ONLY datasets (LlamaFactory {messages, images} format)
│   ├── anomalythink_6k/      # 6K SFT split (3000 anomaly + 3000 normal)
│   ├── anomalythink_15k/     # Full ~14,472-trace corpus + SFT/GRPO partition files
│   ├── teacher_ablation_abc/ # Arm A/B/C teacher-distillation datasets
│   ├── iter2/                # Iteration-2 self-training datasets
│   └── variety_star_6k/      # In-progress Real-IAD Variety STaR corpus
├── results/        # Eval JSONs + trainer_state.json per run, plus the BA tooling
│   ├── eval_ba_inventory.txt # Every (BA, n, path) we computed — the source of truth
│   ├── compute_ba.py         # Recompute BA from any eval JSON
│   └── <run>/checkpoint-X/eval_<bench>_full_<mode>.json
├── docs/           # pipeline_overview (PDF/TeX); data_card.md, model_card.md, prompt_modes.md
└── env/            # Conda / environment capture (see Quickstart)
```

Run `ls -R` from the repo root to see the exact tree; the directories above are stable.

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
  headline checkpoints are released on Hugging Face (see *Released on Hugging Face* in
  Section 1 above). Alternatively, download Qwen2.5-VL and re-run SFT/GRPO with the
  shipped configs.
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

---

## 5. The prompt-mode contract

The evaluation prompt that produced a result is encoded in the **eval JSON filename suffix**,
never inferred from context. The three modes are:

- `_trainprompt` — the exact instruction template the model was trained with (headline numbers use this).
- `_grpoprompt` — the GRPO-time prompt.
- `_bareprompt` — a minimal, unconditioned prompt.

So `eval_dsmvtec_full_trainprompt.json` is "DS-MVTec, full set, train-prompt mode".
**Comparing across modes is only fair within the same suffix.** Full details and the
exact prompt strings are in [`docs/prompt_modes.md`](docs/prompt_modes.md).

---

## 6. Quickstart and where to go next

**Environment** (captured under [`env/`](env/)):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
```

Training uses **LlamaFactory** (`llamafactory-cli train <config.yaml>`); the GRPO trainer is
derived from IAD-R1 / AnomalyR1. Gemini generation runs via Vertex
(`GOOGLE_APPLICATION_CREDENTIALS`) or the API (`GEMINI_API_KEYS` in `.env`).

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
