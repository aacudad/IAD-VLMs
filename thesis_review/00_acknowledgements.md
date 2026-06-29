# Acknowledgements — readable render
*(Source of truth = Overleaf `chapters/00_acknowledgements.tex`, commit `5beb406`. Macros resolved, ¶ markers added. ✅ Applied: full rewrite, dash-free.)*

---

**¶1 — Opening**

This thesis is the product of nine months of work at the Department of Microelectronics of the Faculty of Electrical Engineering, Mathematics and Computer Science at TU Delft.

**¶2 — Supervisors**

First and foremost, I thank my supervisors, **Prof. dr. ir. Justin Dauwels** and **Prof. dr. ir. Willem D. van Driel**, and my daily supervisor, **Amir Ghorbani Ghezeljehmeidan**, for the trust they placed in me to pursue this topic that bridges machine learning, computer vision, and industrial inspection. I also thank them for guidance that was always sharp on the science while leaving me considerable room to explore.

**¶3 — Compute**

The computational experiments described in Chapters 4 and 5, particularly the fine-tuning and post-training of Qwen2.5-VL-7B, consume substantially more GPU-hours than a single workstation can deliver. I am grateful to the Microelectronics department for granting me reliable access to the RTX A6000 cluster on which the experimental runs were performed.

**¶4 — Open-source community**

I owe a special debt to the open-source community whose code and models this work builds upon: the Alibaba Qwen team for Qwen2.5-VL [li2025… → bai2025qwen25vl], the DeepSeek group for the GRPO algorithm [shao2024deepseekmath], the HuggingFace `transformers`/`trl` maintainers, and the authors of LLaMA-Factory and Unsloth. Reasoning-trace generation relied mainly on Google's Gemini 2.5-Flash. I also thank the authors of IAD-R1 [li2025iadr1] for releasing their model weights, which enabled a fair head-to-head comparison on the benchmarks.

**¶5 — Family**

Finally, and most importantly, I thank my mother and father, my siblings, and the rest of my family for their unwavering support throughout my studies and through every late night, failed run, and moment of doubt this thesis demanded.

*Adnane Acudad — Delft, \today*

---

## Status
- ✅ **Applied + pushed** (commit `5beb406`): full rewrite per your draft — nine months; supervisors = Dauwels + van Driel; daily supervisor = Amir; RTX A6000 cluster; IAD-R1 thanks added (cited `li2025iadr1`); ¶5 Option-2 ending ("every late night, failed run, and moment of doubt this thesis demanded"); all en-dashes removed.
- Decisions locked: van Driel = **supervisor**; ¶5 ending = **Option 2**.
