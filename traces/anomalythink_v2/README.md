# AnomalyThink-v2

The 6,000-image SFT corpus with grounding-audit corrections applied.
Not uploaded to HuggingFace. Local only.

## What changed from v1

Same 6,000 Real-IAD images. Same user prompts. Only the assistant trace differs,
and only where the grounding audit flagged an ungrounded claim.

| action | items | share |
|---|---|---|
| kept unchanged (audit found nothing) | 5,042 | 84.03% |
| corrected | 950 | 15.83% |
| flagged but unfixable, v1 trace retained | 8 | 0.13% |

Corpus size is held at exactly 6,000 so that a v1-vs-v2 SFT run is a controlled
A/B on trace quality alone.

## How the corrections were produced

Model: `gemini-3.7-flash`. For each anomalous item it received three images: the
clean query image, the overlay with the ground-truth mask in red, and a reference
normal of the same product and camera view. Normal items received the query image
only. It was shown the original Gemini-2.5-Flash trace and instructed to correct
it in place, not to rewrite it. Prompt: `Training/trace_audit_gemini/prompt_grounding_v2.txt`.

The edit really is minimal. Median rewrite/original word ratio is 0.977, and 78.6%
of rewrites land within 10% of the original length. Total word count across the 950
corrected traces falls by 5.48%, which is the audit stripping unsupported claims.

## Issue types fixed

| issue type | corrected traces |
|---|---|
| defect_type_error | 525 |
| location_error | 276 |
| overlay_leak | 176 |
| count_error | 127 |
| colour_finish_error | 50 |
| invented_defect | 46 |
| feature_absent | 24 |
| marking_text_error | 11 |
| missed_defect | 6 |
| viewpoint_error | 3 |
| other | 1 |

A trace can carry more than one issue type, so the column sums above 950.
Note that `viewpoint_error` is almost absent here. It is concentrated in the GRPO
split, which this corpus does not cover.

## Validation performed

- 6,000 items, images and user prompts byte-identical to v1.
- 950 assistant traces differ. Nothing else does.
- All 953 candidate rewrites carry an XML tag skeleton identical to the original,
  so every trace is a drop-in replacement for the training format.
- Every trace in v2 has a parseable `<answer>` tag and it matches the gold label.
  Zero mismatches.
- No trace in v1 or v2 mentions the overlay, the mask, or the annotation.

## The 8 unfixable items

These were flagged `completely_wrong` but the audit deliberately withheld a rewrite,
because the defect is not visible in the evaluated camera view. Writing a correction
would mean asserting a defect that cannot be seen. The v1 trace is retained so the
corpus stays at 6,000. They are listed in the manifest with
`action = unfixable_kept_original`.

```
toy_NG_QS_S0113          end_cap_NG_ZW_S0075      pcb_NG_QS_S0083
bottle_cap_NG_ZW_S0103   toy_NG_QS_S0099          eraser_NG_QS_S0084
mint_NG_ZW_S0078         end_cap_NG_ZW_S0113
```

If you would rather drop them than retain a known-bad trace, filter the manifest on
that action. It costs the exact-6,000 property of the A/B.

## Caveat to state in any writeup

These are teacher-judge flags, not verified ground truth. The judge over-flags on
`defect_type_error` (it marks a physically correct description wrong when the wording
does not match the dataset's gold category string) and on `location_error` (it splits
bottom-left against bottom-center on masks that straddle the boundary). Treat both
counts as upper bounds. See `defense_prep_20260901/AUDIT_BIGGEST_PROBLEMS.md`.

## Files

- `anomalythink_v2_train.json` — the corpus, LLaMA-Factory format, 6,000 items
- `anomalythink_v2_manifest.jsonl` — per-item provenance: action, audit category,
  issue types, word counts
- `../build_anomalythink_v2.py` — the builder, deterministic and re-runnable
