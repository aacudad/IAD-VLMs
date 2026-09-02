# Real-IAD v4 reasoning-trace pipeline

This is the **canonical Real-IAD reasoning-trace generation pipeline** (v4) that produced the
headline **AnomalyThink** corpus, built around `generate_realiad_traces_v4.py`. The generator
loads `prompts/inspector_prompt_test_v2.txt` (the structured six-phase / XML / auto-reject
prompt — see repo `prompts/`) via Vertex AI.

> **Model note.** The headline **Real-IAD** AnomalyThink corpus was generated with **Gemini 2.5-Flash**
> (as reported in the thesis). This bundled v4 script is configured for `google/gemini-3-flash-preview`
> (a later iteration); the **Real-IAD-Variety** extension used Gemini 3 Flash preview. For consistency
> the thesis reports Gemini 2.5-Flash throughout.

> ⚠️ **Credentials are NOT included in this repository.** The original bundle shipped a `.env`
> (~20 Gemini API keys + an OpenAI key) and `vertex-service-account.json` (a Vertex AI service-account
> key); both are **excluded here**, and the HF token in `download_realiad.py` has been replaced
> with an `HF_TOKEN` env lookup. To run: create your own `.env` with `GEMINI_API_KEYS=...`, point
> `GOOGLE_APPLICATION_CREDENTIALS` at your own Vertex key, and `export HF_TOKEN=...` (see the
> repository root README §Security).

---

## The pipeline at a glance

```
download_realiad.py        --> data/Real-IAD/images/   (HuggingFace: Real-IAD/Real-IAD,
        |                                                30 products, C-angles + masks)
        |
sample_realiad_v4.py       --> 3k_realiad_v4/grpo_train.json  (3000 entries)
   (C1-only, mask-only,            3k_realiad_v4/sft_train.json   (3000 entries)
    50/50 anomaly/normal,
    stratified, seed 42)
        |
generate_realiad_traces_v4.py  <-- prompts/inspector_prompt_test_v2.txt   (system prompt)
   |-- config_mmad.py             <-- .env  (GEMINI_API_KEYS)
   |-- utils_realiad.py           (create_overlay, ANOMALY_MAP)
   |-- vertex-service-account.json     (Vertex AI auth)
   |-- model: google/gemini-3-flash-preview  (via Vertex AI)
        |
        v
   data/output/reasoning_traces_realiad_v4/{grpo,sft}/trace_*.json
        |
compile_traces_v4.py       --> 3k_realiad_v4_compiled/{grpo,sft}_train.json   (final output)
```

---

## File-by-file

### Core generator
- **`generate_realiad_traces_v4.py`** — the main script. Reads the 3K SFT + 3K GRPO
  index files, builds prompts, and calls Gemini 3 Flash (via Vertex AI) to generate a
  structured reasoning trace per image. Output is one `trace_<image_id>.json` per image.
  - For **anomalous** images it sends 3 images to the model: the original, a red
    mask-overlay (region cue), and a normal reference image of the same product/camera.
  - For **normal** images it sends just the original image.
  - Supports sharding (`--shard_id`, `--total_shards`) and is resumable (skips traces
    that already exist). `--debug` saves prompts/images without calling the API.

### Imported modules
- **`config_mmad.py`** — central paths (`DATA_DIR`, `PROMPTS_DIR`, `LOGS_DIR`),
  `BATCH_SIZE = 10`, `MAX_RETRIES = 3`. Loads `.env` and requires `GEMINI_API_KEYS`.
- **`utils_realiad.py`** — `create_overlay()` draws the red mask + bounding box on the
  image; `ANOMALY_MAP` maps Real-IAD defect codes (AK, BX, CH, HS, ...) to readable
  names; plus mask-location helpers.

### Credentials (sensitive)
- **`.env`** — `GEMINI_API_KEYS` (comma-separated key pool), `GEMINI_MODELS`
  (preference list), `OPENAI_API_KEY` (used by the OpenAI-based verification scripts,
  not by the v4 generator itself).
- **Vertex AI service-account key** — supply your own JSON and point
  `GOOGLE_APPLICATION_CREDENTIALS` at it. Set `GOOGLE_CLOUD_PROJECT` too, the generator
  reads the project id from the environment and has no default.

### System prompt
- **`prompts/inspector_prompt_test_v2.txt`** — the system instruction given to Gemini.
  Defines the XML trace format `<think>` (120-200 words) + `<location>` (3x3 grid) +
  `<type>` + `<answer>Yes/No</answer>`, the defect-type vocabulary, and a full
  scoring/acceptance rubric the model self-checks against.

### Dataset sampling (makes the input index files)
- **`sample_realiad_v4.py`** — builds `3k_realiad_v4/grpo_train.json` and
  `sft_train.json`. Rules: C1 camera angle only, NG images only if they have a
  ground-truth mask, 50/50 anomaly/normal, stratified across all 30 products,
  zero overlap between splits, `SEED = 42`.

### Input index files
- **`3k_realiad_v4/grpo_train.json`** (3000) — GRPO entries (image_path, product,
  is_anomaly, question, gt_label).
- **`3k_realiad_v4/sft_train.json`** (3000) — SFT entries in chat-message format with
  an empty assistant turn to be filled by the generated trace.
  - Note: these store **absolute Linux paths** (`/bulk/aacudad/...`). The generator
    rewrites them to local paths on the fly by anchoring on `reasoning_traces_gen/`.

### Raw data source
- **`download_realiad.py`** — downloads the Real-IAD dataset from HuggingFace
  (`Real-IAD/Real-IAD`, resolution `realiad_1024`, all 30 categories) into
  `data/Real-IAD/`. Contains a hardcoded HF token — replace/rotate as needed.

### Post-processing
- **`compile_traces_v4.py`** — merges the individual `trace_*.json` files back together
  with the original index entries into the final
  `3k_realiad_v4_compiled/{grpo,sft}_train.json` (training data with the reasoning
  trace filled into `answer` / the assistant message).

---

## How to run

```bash
# NOTE: .env and vertex-service-account.json are NOT included in this repo; supply your own
# (GEMINI_API_KEYS in .env, GOOGLE_APPLICATION_CREDENTIALS -> your Vertex key).
# 1. download the dataset
python download_realiad.py
# 2. build the 3K/3K index files
python sample_realiad_v4.py
# 3. generate traces (optionally sharded across processes)
python generate_realiad_traces_v4.py --shard_id 0 --total_shards 4
# 4. compile into final training data
python compile_traces_v4.py
```
