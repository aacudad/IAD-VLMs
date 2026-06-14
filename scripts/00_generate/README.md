# Stage 0 — Trace generation

How the **AnomalyThink** reasoning-trace corpora were produced. Two pipelines:

## `realiad_v4/` — canonical Real-IAD pipeline (headline corpus)
The authoritative generator for the headline **AnomalyThink** corpus (the 6K SFT / 4.2K GRPO /
4.2K held-out splits). Flow:
```
download_realiad.py        -> Real-IAD images + masks (HF: Real-IAD/Real-IAD, 30 products)
sample_realiad_v4.py       -> C1-only, mask-filtered, 50/50, stratified, seed 42
generate_realiad_traces_v4.py  <- prompts/inspector_prompt_test_v2.txt   (structured six-phase/XML prompt)
                               <- google/gemini-3-flash-preview via Vertex AI
compile_traces_v4.py       -> final ShareGPT-format SFT/GRPO JSONs
```
Helpers: `config_mmad.py` (env/keys), `utils_realiad.py` (overlay + ANOMALY_MAP). The prompt it
loads is in the repo at [`prompts/inspector_prompt_test_v2.txt`](../../prompts/inspector_prompt_test_v2.txt).
**Model: `google/gemini-3-flash-preview`** (note: the thesis text currently says "Gemini 2.5-Flash"
— see NUMBER_PROVENANCE.md, model-provenance entry).

## Real-IAD-**Variety** extension (this folder, `*variety*`)
The 160-product Real-IAD-Variety extension (Variety-STaR). Scripts:
`download_variety.py`, `sample_variety.py`, `generate_variety_traces.py`,
`validate_variety_traces.py`, `assemble_variety_sft_pool.py`, `run_variety_download.sh`,
`run_variety_generate.sh`. Same generator design as v4 (Vertex AI, **`google/gemini-3-flash-preview`**,
`inspector_prompt_test_v2.txt`). Output corpus shipped at [`traces/variety_star_6k/`](../../traces/variety_star_6k/).

## Older / auxiliary (top level)
`generate_realiad_traces.py`, `generate_mmad_traces.py`, `download_mmad.py`, `parallel_realiad_gen.py`,
`trace_loader.py`, `utils_mmad.py` — earlier/auxiliary versions. For the headline Real-IAD corpus,
**`realiad_v4/` supersedes `generate_realiad_traces.py`.**

## Credentials (not shipped)
No API keys or service-account files are included. Supply your own: `.env` with `GEMINI_API_KEYS=...`,
`GOOGLE_APPLICATION_CREDENTIALS` -> your Vertex key, `export HF_TOKEN=...`. See the repo root README §Security.
