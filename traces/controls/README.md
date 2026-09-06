# Control corpora

- `filtered_6k_cc_train.json` (5,797): the 6K SFT traces that the GPT-5-mini verifier marked answer-correct.
  Thesis §7.2, model `results/sft_filtered6kcc_from_base/`.
- `sft_iter2_balanced_train.json` (192): the Balanced-192 refinement set of thesis Table 6.8
  (96 normal + 96 anomalous patches), model `results/sft_qwen25vl_7b_iter2_balanced/`.
- the labels-only (no reasoning) 6K file, if present: thesis §6.3 verdict-only control,
  model `results/sft_qwen25vl_7b_6k_noreason/`.
