# Abstract — readable render
*(Source of truth = Overleaf `chapters/00_abstract.tex`. This reflects the **de-em-dashed** version pushed 2026-06-20 — em-dashes replaced by commas / sentence-splits. Macros resolved: `\qwenvl`→Qwen2.5-VL, etc. Paragraph markers ¶ added for reference. ~700 words.)*

---

**¶1 — Motivation**

Industrial quality control demands inspection systems that are *accurate*, *explainable* and *adaptable* to new product categories with minimal labelled data. Traditional deep-learning detectors (such as memory-bank, normalising-flow and reconstruction-based methods) saturate at near-perfect AUROC on the established MVTec AD benchmark. Yet they emit only a binary defect/no-defect verdict and a heatmap. This gives operators no interpretable account of *why* a sample was flagged, *where* the defect is, or *what type* of defect it is. That lack of interpretability is a primary obstacle to deploying automated inspection in safety- and audit-critical manufacturing environments.

**¶2 — Framework & corpus**

This thesis investigates whether modern vision-language models (VLMs) can close that explanation gap while remaining competitive with, and in places exceeding, the state of the art. We develop an end-to-end framework that fine-tunes the open-weights Qwen2.5-VL model family on *structured chain-of-thought reasoning traces*. Each trace follows a six-phase assessment template: *Framing*, *Scan*, *Focus*, *Evaluate*, *Alternatives* and *Decide*. It emits an XML output containing a free-text rationale (`<think>`), a categorical defect-type tag (`<type>`), a spatial location (`<location>`), and a binary verdict (`<answer>`). We generate **14,472** such traces from the Real-IAD multi-view industrial dataset using Gemini 2.5-Flash, under a structured prompt that enforces the output format and a set of auto-reject rules. We partition them into product- and class-stratified splits: a 6,000-trace SFT pool, a 4,236-trace GRPO pool, and a 4,236-trace held-out split.

**¶3 — Controlled comparison & evaluation setup**

We then run a controlled comparison of fine-tuning recipes that vary three factors, Qwen2.5-VL-3B versus 7B, vision encoder frozen versus unfrozen, and 6K-versus-15K supervised data. This is followed by Group Relative Policy Optimisation (GRPO) with an unweighted two-function reward (a format/consistency reward and an accuracy reward, together scoring four bounded sub-signals: format, verdict, semantic type, location). Evaluation is performed on the DS-MVTec (1,670 samples) and VisA (2,141 samples) subsets of the MMAD benchmark, plus a held-out, in-domain 4,236-sample Real-IAD split.

**¶4 — Central finding + six results**

> **The central finding is that the quality of the supervised reasoning corpus, not reinforcement learning, is the decisive lever: a carefully curated SFT corpus is the single best detector in this thesis and matches or exceeds the SFT+GRPO pipeline, while GRPO improves only a weak SFT baseline and adds no headroom on a strong one.**

Six main results are reported.

- **(i)** A four-factor supervised study identifies a strong SFT operating point (7B, frozen vision encoder, 6K traces, epoch 3) reaching **80.16 % balanced accuracy** on DS-MVTec. This is already within ~1.8 pp of the released IAD-R1 SFT+GRPO checkpoint measured under a common evaluation harness (§res-sota).

- **(ii)** A more carefully constructed SFT corpus, built by rolling out the SFT+GRPO policy and then correcting and rewriting its traces with a teacher model (Gemini 2.5-Flash) under the STaR (Zelikman et al., 2022) and Saunders critique-and-revise (Saunders et al., 2022) recipes (6,000 items), reaches **82.80 % on DS-MVTec and 72.07 % on VisA using supervised fine-tuning alone**; this is the single best detector in the thesis, and it exceeds the released IAD-R1 checkpoint on *both* subsets under identical evaluation conditions (+0.88 pp on DS-MVTec and +0.73 pp on VisA, vs. 81.92 % / 71.34 %).

- **(iii)** GRPO is double-edged. Applied to the original SFT-Iter1 baseline it lifts performance to 82.73 % / 70.39 %; but applied to the strong SFT corpus of (ii) it provides *no benefit*. Across checkpoints it sits ~2 pp *below* the SFT initialisation on DS-MVTec while training reward rises, and a controlled three-estimator ablation (vanilla z-score, Dr.GRPO, G²RPO) confirms the degradation is independent of the advantage estimator. The published SFT+GRPO lift therefore reflects the weakness of the baseline it was measured against; on a strong baseline a well-curated SFT corpus is the dominant lever and single-pass GRPO adds no binary-accuracy headroom.

- **(iv)** Dataset composition matters more than sheer quantity: the 6,000-trace SFT split of (i) outperforms the full ≈14,500-trace stratified superset by ~8 balanced-accuracy points after SFT (a data-count effect, not trace length; Chapter 6). *(✅ deduped: 137-word "same-length" clause collapsed → cite Ch.6.)*

- **(v)** Freezing the vision encoder consistently outperforms unfreezing it, especially on the harder VisA benchmark.

- **(vi)** Per-product analysis shows the largest gains on previously difficult categories, `metal_nut` +25.8, `cable` +19.2, `pill` +16.0 percentage points, while already-saturated categories regress only marginally (`tile` −2.1, `wood` −0.1).

**¶5 — Reproducibility & deliverable**

The framework is reproducible on commodity hardware (2×RTX A6000 with ~300 GB of host RAM for ZeRO-3 CPU offloading) and yields a single 7B-parameter model that simultaneously *detects*, *localises* and *explains* industrial defects in natural language, while requiring no per-product retraining and no pixel-level masks at inference time.

**Keywords:** industrial anomaly detection, vision-language models, chain-of-thought, supervised fine-tuning, GRPO, Real-IAD, MMAD, Qwen2.5-VL, explainable AI, dataset, benchmark.

---

## Status / open questions for this pass
- ✅ Done + pushed: em-dashes removed; ¶1–¶3; ¶4 (ii) GRPO-rollout transparency; (iv) "6K of (i)" disambiguation; (vi) wood/tile regressions; ¶5 ~300 GB RAM; keywords +dataset/benchmark.
- **PENDING — central sentence:** *"not reinforcement learning"* → *"not a final reinforcement-learning step"* (so it doesn't read as "RL did nothing"). Awaiting your OK.
- **PENDING — length:** ~700 words; (i)–(vi) reads like a contributions list. Keep all six, or trim to top-3?
- **All Qwen2.5-VL-7B** — the Qwen3-VL-8B Arm-C SFT is running now (lands ~19:00); if strong, a one-line addition belongs here.
- **Figure** — none *in* the abstract (TUD style); the system-overview diagram goes at the top of Ch.1.
