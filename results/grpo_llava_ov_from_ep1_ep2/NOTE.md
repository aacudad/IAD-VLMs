# LLaVA SFT + GRPO, restart run (second GRPO checkpoint)

Same recipe as `results/grpo_llava_ov_from_ep1/`, restarted. ckpt-530: DS-MVTec 87.86 (vLLM), VisA 71.89
(`eval_visa_parallel.json`, vLLM). Thesis §6.9 uses it to show the SFT-vs-GRPO tie does not depend on
which GRPO checkpoint is chosen.
