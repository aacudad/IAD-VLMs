# GPT-5-mini Verifier Audit — Non-Regenerated C1 Union (the 6K's actual traces)

**Date:** 2026-06-16 · **Script:** `scripts/verify_c1_union.py` · **Verifier:** GPT-5-mini, reference-guided (image + RED GT-mask overlay + trace), **identical criteria to the regen-15K run** (byte-identical system prompt, same model/decoding/issue-codes). 8 parallel shards.

**Corpus:** the **non-regenerated C1 union** = `combined_sft_c1_train.json` (10,236) + `grpo_train.json` (4,236) = **14,472 unique (image_id, trace) pairs**. This is the corpus whose traces the **6K SFT split is a 100% subset of** (unlike the regenerated `iad_sft_15k_regen_combined`, whose traces differ ~59%).

## Breakdown
| Set | completely_correct | correct | wrong | completely_wrong | **KEEP** |
|---|---|---|---|---|---|
| **6K SFT split** (the headline 6K) | 5,797 | 57 | 146 | 0 | **97.6%** |
| Full union (14,472) | 13,626 | 129 | 716 | 1 | **95.0%** |
| — SFT pool (10,236) | 9,669 | 102 | 464 | 1 | 95.5% |
| — GRPO split (4,236) | 3,957 | 27 | 252 | 0 | 94.1% |

## Findings
- **The 6K curated split is the cleanest part of the corpus (97.6% keep)** — higher than the full SFT pool (95.5%), the GRPO split (94.1%), and the union (95.0%). So the 6K is not merely a subset; it is a modestly **higher-quality** subset (the non-6K traces are noisier).
- The **non-regenerated** union (95.0%) is ~2.8 pp noisier than the **regenerated** 14,472 (97.8%, see `../verify_15k_regen/`) — consistent with the regeneration pass having cleaned up traces.
- Nuance for the thesis: this is a *modest* quality gradient (6K 97.6% vs union 95.0%); the dominant 6K-vs-15K effect remains count/composition (both are ≥95% clean), but the 6K being the cleanest slice is a genuine, reportable secondary signal.

## Files
- `verdicts.jsonl` — all 14,472 verdicts (image_id, status, issues, issue_elaboration, confidence, src)
- `summary.json` — counts for union / 6K-subset / by-source
- Per-trace originals (not committed): `outputs/verify_c1_union_results/`
