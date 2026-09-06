# LLaVA-OneVision KCR corpora

- `sft_llava_C_original_train.json` (6,000): the corrected build, the corpus behind the thesis LLaVA KCR row
  (87.32 / 72.65 at epoch 2, 74.29 VisA at epoch 4). All 6,000 images inside the 6,000-image SFT split,
  balanced on the `<answer>` verdict (3,000 / 3,000). Built by
  `scripts/03_rollout_star/build_llava_arms_original.py` from `traces/rollout_pools/llava_phase0_10k/`
  (seed 42, pool restricted to the SFT split, `min(yes, no, 3000)` per class).
- `sft_llava_C_first_build_train.json` (6,000): the first build (thesis Appendix M). Drawn from the whole
  10,236-image pool (3,516 images in the SFT split, 2,484 from the GRPO split) and balanced on the folder
  name, which leaves it at 45.0 % anomalous by verdict. Trained model: `results/sft_llava_ov_7b_frozen_llava_iter1_C/`.
Same LlamaFactory `{messages, images}` format and image-path convention as `traces/anomalythink_6k/`.
