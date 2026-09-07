# Gemini 3.x Flash zero-shot references (thesis Table 6.14), evaluated 7 Sep 2026

Same harness, structured prompt and images as every other row (`scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py --gemini`),
one process per model and benchmark, launcher `scripts/04_eval/run_gemini_flash_family.sh`. Thinking level `low`, the lowest
the four models accept (`minimal` returns empty text on 3.5 and is rejected by 3.7 and 3.8). Strict scoring, unparsed counted wrong.

| model (release) | DS-MVTec (sens / spec) | VisA (sens / spec) | unparsed DS / VisA |
|---|---|---|---|
| gemini-3.5-flash (19 May 2026) | 89.28 (95.7 / 82.9) | 76.44 (90.5 / 62.4) | 0 / 2 |
| gemini-3.6-flash (21 Jul 2026) | 90.60 (94.0 / 87.2) | 82.01 (87.2 / 76.8) | 0 / 0 |
| gemini-3.7-flash (13 Aug 2026) | 89.32 (96.7 / 82.0) | 81.04 (88.6 / 73.5) | 0 / 0 |
| gemini-3.8-flash (2 Sep 2026)  | 89.52 (96.2 / 82.9) | 79.66 (89.0 / 70.3) | 0 / 36 |

The June run of gemini-3-flash-preview at thinking level `minimal` (93.09 / 80.25, `results/gemini3flash_eval/`) is not in
the thesis; its 92.8 DS-MVTec specificity is not reproduced by any of the four later models (82 to 87).
