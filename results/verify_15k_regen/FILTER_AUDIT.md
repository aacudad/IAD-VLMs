# GPT-5-mini Verifier Audit of the 15K SFT Corpus

**Date:** 2026-06-16 · **Script:** `repository_tu_delft_vlms/scripts/verify_15k_traces.py`
**Verifier:** GPT-5-mini (OpenAI Responses API, reasoning_effort=low), reference-guided (original image + RED GT-mask overlay + trace), literature-updated decomposed rubric (5 format + 8 content criteria, status ∈ {completely_correct, correct, wrong, completely_wrong}).
**Corpus:** `Training/15k_dataset_regenerated/combined_sft_train.json` = **14,472 traces** — the exact set the headline 15K SFT cell trained on (`iad_sft_15k_regen_combined`). Cost ≈ **$10** (within the $20 budget).

## Result — the 15K corpus is ~98% clean
| Status | Count | % |
|---|---|---|
| completely_correct | 14,015 | 96.8 |
| correct | 143 | 1.0 |
| **KEEP (cc + correct)** | **14,158** | **97.8** |
| wrong | 313 | 2.2 |
| completely_wrong | 1 | 0.0 |
| **DROP** | **314** | **2.2** |

**Top issue codes among the 314 dropped traces:**
`illogical_reasoning` (150) · `incoherent_content` (88) · `extra_location_tag` (45) · `wrong_product_reference` (18) · `missing_location_tag` (12) · `has_batch_reference` (11).
→ The filter catches genuine *content* problems (illogical/incoherent reasoning, wrong product, batch-leakage), not just formatting.

## What this means for the thesis (the key takeaway)
An **independent reference-guided verifier judges the 15K corpus 97.8% correct** — i.e. the corpus is **not noisy**. Therefore the headline "6K curated split beats the full 15K union" result is a **composition / training-budget effect, NOT a data-noise / quality-filter effect**. This directly resolves the thesis's contested "quality" framing flagged in the review: there is no large pool of bad traces to filter out, so the 6K<15K gap cannot be attributed to the 15K being dirty.

**Implication for the planned filter→re-SFT:** a naive "keep completely_correct+correct → re-train" removes only ~2.2% → a near-tie with the unfiltered run (confirmed-clean, not a gain). The meaningful experiment is the **matched-count** design (filtered vs equal-size random-unfiltered, matched epochs/compute) and/or a **harsher issue-code hard-drop** (treat `illogical_reasoning`/`incoherent_content`/`wrong_product_reference` as drops) to move off saturation — reported as future work unless run.

## Per-product keep-rate (32 products) — uniformly clean
No product concentrates the noise; the lowest keep-rates are still ~96%:
`toy_brick` 95.9% (493/514) · `wooden_beads` 96.2% · `mounts` 96.7% · `plastic_nut` 96.8% · `end_cap` 96.8% · `sim_card_set` 96.9% · `audiojack` 97.0% · `pcb` 97.0% — all others ≥97%.
This even distribution is itself evidence that the drop is broad low-rate noise, not a corrupt subset.

## Robustness / reproducibility
- Per-trace verdicts saved immediately (`verify_<id>.json`); 3-retry per batch then skip-as-error; errors retried on resume (resumable). Logs every batch in `run.log`.
- Anomaly traces: original + red GT-mask overlay (99–100% mask coverage); normals: original only (no mask).

## Lineage / methodology (cite in thesis)
Reference-guided verifier rejection-sampling filter in the **STaR → RFT → RAFT → V-STaR** line:
STaR (2203.14465), RFT (2308.01825), RAFT (2304.06767), V-STaR (2402.06457), Let's-Verify (2305.20050), GenRM (2408.15240), MM-Verify (2502.13383); data-quality selection LIMA (2305.11206)/AlpaGasus (2307.08701)/LIMO (2502.03387); domain precedent **IAD-R1** (2508.09178) folds verification into its GRPO reward — we instead run it as an explicit offline trace-curation stage.
