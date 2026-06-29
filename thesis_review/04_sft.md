# Ch.4 Supervised Fine-Tuning — readable render (FULL prose, 1:1 with Overleaf)
*(Source = Overleaf `chapters/04_sft.tex`, commit `6ef1943`. Verbatim prose, macros resolved (`\qwenvl`→Qwen2.5-VL, `\dsmvtec`→DS-MVTec, `\visa`→VisA, `\bal`→balanced accuracy; `\citep{k}`→[k]); equations/listings/tables noted. ~1,847 words. **Already dash-free (0 en-dashes).**)*

> **📐 Figure (present): `fig:pipeline-overview`** (`figures/pipeline_overview.pdf`) — the end-to-end pipeline: (1) Real-IAD → Gemini 2.5-Flash traces → AnomalyThink-6K; (2) four-factor SFT grid → 7B-frozen-6K 80.16% + Arm-C 82.80/72.07 (best); (3) GRPO lifts a weak baseline, no gain on Arm-C; (4) eval on DS-MVTec/VisA + held-out Real-IAD. *(This is your system-overview figure — it exists. Caption now says "Gemini 2.5-Flash", consistent.)*

---

**Intro ¶.** This chapter describes the SFT stage, in which the Qwen2.5-VL backbone is trained to emit the structured six-phase trace of Ch.3. Roadmap: §4.1 objective; §4.2 model + frozen/unfrozen ViT; §4.3 data formatting; §4.4 infrastructure (DeepSpeed ZeRO-3 + CPU offload on 2×A6000); §4.5 full hyperparameters; §4.6 the four-factor ablation design; §4.7 the evaluation protocol.

## §4.1 Training Objective — ✅ your voice (`93cb4bd`)
SFT minimises the standard auto-regressive negative-log-likelihood on the trace tokens. For an example (I, y) — test image I, gold trace y tokenised with the Qwen2.5-VL tokenizer — the loss is **[Eq 4.1: L_SFT(θ) = −Σ_t log π_θ(y_t | I, x_usr, y_{<t})]**, where x_usr is the fixed user question and only assistant tokens contribute (user tokens masked). The model emits *the entire trace, including all four tags*, so supervision covers both the natural-language reasoning and the structured tags.

## §4.2 Model and Vision Encoder Treatment — ✅ your voice (`93cb4bd`)
**Backbone.** Qwen2.5-VL [bai2025qwen25vl] at two sizes: Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct (open weights, Alibaba). Shared 3-component architecture: window-attention ViT encoder → multimodal projector → Qwen LM trunk. **3B:** 36 layers / 16 heads / 2048 hidden. **7B:** 28 layers / 28 heads / 3584 hidden. Native dynamic resolution: image → variable number of 14×14 patches → variable-length visual-token sequence; we cap `image_max_pixels = 262,144` (≈512×512), and the eval harness applies the same cap for train/test consistency.

**Why full fine-tuning, not LoRA.** Following [li2025iadr1], who report LoRA [hu2022lora] fails to convey enough information for long reasoning traces (the `<think>` block collapses to short generic templates because the low-rank adapter can't represent the six-phase diversity); **+ your addition: our own early LoRA experiments likewise showed no improvement over the base model.** So **full** SFT of the LM trunk; the ViT is an ablation factor.

**Frozen vs unfrozen ViT.** The pretrained ViTs are strong on natural images but industrial images are OOD (uniform backgrounds, macro shots, unfamiliar materials). *Frozen:* only LM trunk + projector updated, ViT fixed — preserves the pretrained representation, avoids catastrophic forgetting. *Unfrozen:* ViT updated end-to-end — adapts to industrial statistics but risks overfitting on the small (6K–15K) set. Treated as an ablation factor; **frozen wins decisively at 6K and 15K, especially on VisA** (§7 discussion).

## §4.3 Data Format and Prompt — ✅ your voice (`962657b`)
> ✅ **Verified your "is there a mismatch?" question:** train data turns = `['user','assistant']` (no system turn); the headline eval (`else` branch) re-uses the **byte-identical** `make_train_prompt` user question + adds only `system: "Please answer by yes or no"`. So the only train↔eval difference is that eval-only system line. Your wording is accurate. *(The "Are there any defects…" questions in the code are other eval modes — iadr1_native / bare_question / grpo_eval — not headline.)*
Each example → a multimodal ShareGPT-style conversation (LlamaFactory / HF `trl` format). Deliberately **zero-shot**: a single test image, no reference image, two turns, **no system turn at training time**.
- **User turn:** `<image>` + the fixed product-conditioned question: *"Analyze the provided image of the {product_name}. Determine if there are any anomalies present. If an anomaly is detected, specify its type and location, and provide a detailed reasoning for your conclusion."*
- **Assistant turn:** the gold Gemini trace (`<think>` + for anomalies `<location>`/`<type>` + `<answer>`).

User tokens masked; images loaded lazily.

**Prompt consistency (train↔eval).** The eval harness re-uses this exact user question, adding only a one-line system message ("Please answer by yes or no") to anchor the verdict — the **train-prompt** eval mode used for every headline number in Ch.6. Keeping the prompt identical between train and inference measures *capability*, not instruction-following robustness, and avoids prompt-drift regressions.

## §4.4 Training Infrastructure
**Hardware.** Single node, **2× RTX A6000 (48 GB each)** + ample RAM + NVMe. 3B-unfrozen fits one GPU at batch 2 with gradient checkpointing; 7B-unfrozen does **not** fit even on both GPUs without offloading → **DeepSpeed ZeRO-3** [rasley2020deepspeed, rajbhandari2020zero] with CPU offload.
**ZeRO-3 + CPU offload.** Partition gradients/optimizer-states/16-bit params across the 2 GPUs; offload optimizer states + params to pinned CPU memory. **bf16** (more stable than fp16 for 7B; avoids loss-scaler tuning). [Listing 4.1: key ZeRO-3 settings — stage 3, offload_optimizer + offload_param to CPU pinned, overlap_comm, contiguous_gradients; full JSON in App. A.]
**Wall-clock:** 7B-frozen-6K ×4 epochs ≈ **6.9 h** on the 2 A6000s; 7B-unfrozen-15K ≈ **12.2 h**; 3B-frozen-6K ≈ **1.6 h**.

## §4.5 Hyperparameters
[Table 4.1 — full table in App. A; headline values:]
- Optimizer AdamW [loshchilov2019adamw], β=(0.9, 0.999); **LR 1e-5 (frozen) / 1e-6 (unfrozen)**; cosine schedule, 20-step (6K) / 50-step (15K) warmup; weight decay 0; **effective batch 32**; seq-len cap 12,144; bf16; grad-clip 1.0; **4 epochs**; seed 42.
- Per-device batch / grad-accum vary per cell (4/4 for 6K + all 3B; 16/1 for 7B-15K) but always = effective batch 32 on 2 GPUs (App. A).

**Why a low, encoder-dependent LR:** trunk (+ projector in frozen cells) at 1e-5; when the ViT is **unfrozen**, drop to 1e-6 — an early 7B-unfrozen pilot at 1e-5 on 6K peaked at 72.8% DS then *degraded* with more training (§6), so the smaller step keeps the pretrained ViT stable. Frozen cells train robustly at 1e-5.

> ⚠️ **TODO comment in source (lines 120–124):** the controlled **trace-length ablation** (same count, short-V2 vs long-V1 traces) was *removed pending execution* — note: the 6K-vs-15K result is a data-count effect (6K is a same-length subset of 15K), **not** a length effect, so a single-factor length study is still required before any trace-length claim. (Pre-registered design in git history.)

## §4.6 Ablation Design
Four-factor grid: **Model size** {3B, 7B} × **Vision encoder** {frozen, unfrozen} × **Dataset** {6K, 15K} × **Epoch** {1,2,3,4}. A full factorial = 2×2×2×4 = 32 cells/benchmark; we run the **unfrozen encoder at 15K only** (the 15K-unfrozen cells already show unfreezing gives no DS gain and hurts VisA, per [li2025iadr1]'s frozen-ViT recommendation), so the two unfrozen-6K configs are skipped → **6 configurations** (24 epoch-cells/benchmark). Single seed/cell, averaged over the full eval set (1,670 DS-MVTec, 2,141 VisA). One factor moves at a time → clean "effect-of-X" comparisons, now a **table** (`tab:sft-effects`, ✅ commit `3505557`): **size** 3B-frozen-6K vs 7B-frozen-6K; **encoder** 7B-frozen-**15K** vs 7B-unfrozen-**15K** *(fixed — 7B-unfrozen-6K was never run; verified vs eval dirs + §6 table)*; **dataset** 7B-frozen-6K vs 7B-frozen-15K. Results in §6 (no numbers duplicated here).

## §4.7 Evaluation Protocol
- **Inference inputs:** single test image + the same user question (zero-shot, no reference), preceded by "Please answer by yes or no"; image at the 262,144-pixel cap.
- **Decoding:** greedy (`do_sample=False`), max 1024 new tokens (reproducible; sampling didn't materially change balanced acc).
- **Answer extraction:** deterministic regex pulling `<think>`/`<location>`/`<type>`/`<answer>`; failure to emit a parseable `<answer>` = incorrect. <0.5% of post-SFT outputs fail this parse.
- **Metrics:** **Balanced accuracy** = ½(TPR + TNR) — primary, because IAD class distributions vary and an "always yes" model scores deceptively high plain accuracy. Also plain acc/precision/recall/F1 (literature comparison) + per-product balanced acc on the 15 DS-MVTec products (App. perprod). All Ch.6 percentages to 2 decimals.

> **[takeaway box]** — ✅ your voice (`962657b`). Three distinguishing features of this SFT pipeline: *full* fine-tuning (not LoRA), the visual encoder as an ablation factor (frozen vs unfrozen), and commodity 2×A6000 hardware via ZeRO-3 + CPU offload. With the controlled four-factor design, this enables clean "effect-of-X" statements that are hard to extract from the literature, where SFT is usually a footnote ahead of an RL stage.
> ⚠️ **Your Q "confirm the literature?"** — I kept **"hard to extract from the literature"** (the *contribution* framing: we provide clean ablations the field lacks). "Confirm" would be only half-true: frozen>unfrozen *does* confirm IAD-R1, but 6K-beats-15K is a *new* finding, not a confirmation. Say the word if you'd rather flip to "confirm". **Q "footnote ahead of an RL stage?"** — kept; it's defensible (IAD-R1 / AnomalyR1 treat SFT as the cold-start before GRPO, rarely studied on its own).

---

## Notes / flags for the walkthrough
- ✅ **Already dash-free** (0 en-dashes) — no de-dash needed.
- **📐 Figure:** `pipeline_overview.pdf` (your system-overview diagram) is present + referenced; caption says "Gemini 2.5-Flash" (consistent). *(The "arch/grid placeholder" from the plan — a visual of the 4-factor grid — is NOT present; optional to add.)*
- ⚠️ **Source TODO** (§4.5): the trace-length ablation is removed-pending-execution — decide whether to (a) run it, (b) keep the honest "data-count not length" framing without it, or (c) cut the TODO note.
- **Repetition-audit overlaps** (per `docs/repetition_audit.md`) that live here and recur elsewhere: frozen-encoder-wins, 2×A6000/ZeRO-3 hardware, full-vs-LoRA, 6K-vs-15K, balanced-accuracy rationale. §4 is the natural *canonical home* for the hardware + LoRA + ablation-design details; the abstract/intro/conclusion should cross-ref here.
- **✅ Numbers VERIFIED (2026-06-22) against run artifacts:**
  - Wall-clock 7B-frozen-6K = 24,937 s = **6.93 h** ✓; 7B-unfrozen-15K = 43,876 s = **12.19 h** ✓ (`train_results.json`).
  - 3B-frozen-6K: full run is 6 epochs / 2.35 h, but thesis reports **4 epochs → ≈1.6 h** (= 8452×4/6÷3600 ≈ 1.57 h) per your "report the 4 epochs" call. ✓ consistent.
  - LR: frozen = 9.996e-6 ≈ **1e-5** ✓; unfrozen = 9.999e-7 ≈ **1e-6** ✓; 7B runs **4 epochs**, effective batch **32** ✓ (`trainer_state.json`).
  - **72.8% unfrozen-pilot peak — VERIFIED.** Artifact = `outputs/sft_qwen25vl_7b_zeroshot_6k/` (unfrozen: 100% trainable params, training log 2026-03-19; LR 1e-5). DS-MVTec BA: ckpt-100 60.22 → **ckpt-200 72.82** → ckpt-376 69.27 (peaked then degraded — matches the §4.5 sentence exactly). Eval file: `checkpoint-200/eval_dsmvtec_full_trainprompt_all.json`.
  - Still unchecked (low priority): "<0.5% parse failure", architecture dims (3B 36L/16H/2048; 7B 28L/28H/3584).
- Content is solid + already the most "effect-of-X"-disciplined chapter. Mostly a voice/wording pass when you're ready.
