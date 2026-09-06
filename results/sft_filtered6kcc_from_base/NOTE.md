# Retrain from base on only the verifier-passing 6K traces (thesis §7.2 control)

Corpus: `traces/controls/filtered_6k_cc_train.json` (5,797 of the 6,000 SFT traces, the answer-correct
subset). Best-balanced epoch ckpt-364: DS-MVTec 79.60, VisA 65.99, against 80.16 / 64.78 for the unfiltered 6K.
The verifier itself is `results/verify_15k_regen/` (GPT-5-mini keeps 14,158 of 14,472, 97.8 %).
