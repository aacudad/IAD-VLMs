# Qwen KCR corpus (Arm C)

## Two versions of `sft_iter2_train.json` (added 2026-09-06)

- `sft_iter2_train.json`: the current file, with the `<location>` spelling fix applied (316 traces changed
  `center-right` / `center-left` style cells to the nine canonical grid names, one further trace edited).
  This is the version the thesis Chapter 3 vocabulary describes and the version registered as `iad_sft_iter2`.
- `sft_iter2_train_v1_prepatch.json`: the file previously shipped here, before that fix.
Both hold the same 6,000 images and the same 3,000 / 3,000 verdicts. The Arm-C model
(`results/sft_qwen25vl_7b_abc_C_full_patched/`) was trained from the `iad_sft_iter2` registry key; no
training log records which of the two spellings was on disk at launch, so treat the two as equivalent for
the verdict and differing only in 316 location strings.
