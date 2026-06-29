# Ch.6 Results and Analysis — render (mirrors Overleaf `7681742`) + my review comments

*(Source = `chapters/06_results.tex`, 612 lines, 13 sections, 11 tables. Macros resolved: `\qwenvl`→Qwen2.5-VL, `\dsmvtec`→DS-MVTec, `\visa`→VisA, `\realiad`→Real-IAD, `\mmad`→MMAD, `\visiontower`→ViT, tags shown literally. Tables represented by their key numbers. The §5.8-twin orphan (`sec:res-compound`) has been removed.)*

---

# 🔎 MY COMMENTS — what raises my attention (from everything we just discussed)

**The good news: §6 is much stronger than I feared, and it already does most of the right things.** It establishes Arm-C (82.80/72.07) as the best model (§6.8 SOTA, §6.12 teacher-ablation), shows GRPO *degrades* the strong Arm-C init (§6.13, with a clean 3-estimator ablation), and — at line 466 — **explicitly states the reset-vs-continue distinction** I was worried was missing.

But there are **5 things to reconcile**, all flowing from what we discussed:

**1. 🔴 The reset-vs-continue insight is *buried* — and it's a clean A/B.** §6.5 ("Post-GRPO SFT refinement") trains 3 compositions that **continue from the GRPO ckpt** → all regress → its takeaway says *"only GRPO can extract value from the cleaned data."* §6.12 then trains **Arm-C from base** on what line 466 calls *"identical image set and recipe to the iter-2-clean corpus … but trained from base here rather than from a GRPO checkpoint"* → **82.80, the best model.** Same ~6K data, the only difference is **from-base (82.80) vs continue-from-ckpt (81.07)** = **+1.73 pp for reset-to-base.** We *verified* this is exactly the OpenVLThinker standard (every SFT round resets to base; only data carries forward). So §6.5's "only GRPO can extract value" is *contradicted* by §6.12 unless the reader connects the parenthetical at line 466. **→ Make reset-vs-continue prominent; soften §6.5's takeaway.**

**2. 🟠 The headline framing is mixed across the chapter.** Early sections (§6.3 GRPO, §6.10 held-out, §6.11 dynamics-takeaway line 436) still present **SFT+GRPO 82.73/70.39 as the headline**; later sections (§6.8 SOTA, §6.12, §6.13) correctly crown **Arm-C 82.80/72.07** and show GRPO doesn't help. The chapter *builds* to the right answer, but the first half reads as the old (pre-reversal) story. **→ Propagate the Arm-C headline into the early sections** (esp. the §6.11 "final aggregate result" takeaway, which still says "adding GRPO lifts it to 82.73/70.39").

**3. 🔴 Location-cleaning discrepancy (from B.4).** §6.5 (line 196) claims *"2,700 type-tag substitutions and 1,167 location normalisations,"* and §6.12 (line 466) says Arm-C = the *cleaned* iter-2 corpus. But we **verified** the location fix was applied to the **GRPO file only** — Arm-C's actual training file (`sft_iter2_train.json`) still has **317 off-grid `<location>` tags** (center-right ×220, center-left ×89). So the headline model trained on **un-normalised** locations, and the "1,167 normalisations" number is unverified (only ~13 tags were fixed, on the GRPO data). **→ This contradicts both §6.12's "cleaned corpus" claim and §3.2's "clean 3×3 grid." Verify the type-fix vs location-fix scope before trusting line 196.**

**4. 🟠 GRPO IS upstream of the headline — and §6.12 *does* say so** (line 443: *"all three re-train on rollouts of the SFT+GRPO policy"*). So the abstract's "corpus quality, not RL, is decisive" is defensible *only with* this nuance: GRPO's rollouts are the *raw material* for Arm-C; the *curation* (Gemini critique-revise) is the lever; GRPO *as a final stage* adds nothing. Right now that nuance lives only in §6.12 — worth surfacing earlier.

**5. ⚠️ The judge-model question (C.8) is live in §6.6.** §6.6 uses a **Gemini-3-Flash** judge for explanation quality, while the trace teacher is presented everywhere as **Gemini-2.5-Flash**. If the teacher and judge are the *same* underlying model, that's a self-evaluation concern; if different, the v2.5/v3 naming is inconsistent. **→ Settle at this section** (the call you flagged).

**Also solid (no action):** §6.13's GRPO-on-Arm-C degradation + the 3-way advantage-estimator ablation (z-score / Dr.GRPO / G²RPO all fail to beat the SFT init) is a genuinely strong, well-controlled result. C.7 verified: every number is attributed to the right model (82.80/72.07 = Arm-C SFT, 82.73/70.39 = SFT+GRPO) — no mis-attribution.

---

# Full render

**Intro / roadmap.** Reports: §6.1 baseline → §6.2 SFT four-factor ablation → §6.3 GRPO production → §6.4 iterative/variant RL → §6.5 post-GRPO SFT-refine → §6.6 explanation quality → §6.7 Variety continuation → §6.8 SOTA → §6.9 per-product → §6.10 held-out Real-IAD → §6.11 dynamics → §6.12 teacher ablation → §6.13 GRPO-on-Arm-C. All % to 2 decimals; bold rows = headline; bal-acc = balanced accuracy.

## §6.1 Baseline Performance
> **📋 Table 6.1 (`tab:baseline`):** un-fine-tuned. DS-MVTec: 3B base 56.14 bal-acc (always-yes, recall 99.67); **7B base 69.01** (precision 98.97, recall 39.15). VisA: 3B 50.21; **7B 53.79** (recall 7.69 — severely conservative).
Two pathologies: 3B always says "yes" (bal-acc ≈ chance), 7B is over-conservative (catches 7.7% of VisA defects). Justifies bal-acc as the primary metric.

## §6.2 SFT Ablation — Four-Factor Grid
> **📋 Table 6.2 (`tab:sft-summary`):** 6 cells (best epoch). 3B-frozen-6K 69.08/57.22; 3B-frozen-15K 69.56/59.65; 3B-unfrozen-15K 68.58/59.60; 7B-frozen-15K 72.60/66.94; **7B-frozen-6K (ep3) = 80.16/64.78 (headline SFT)**; 7B-unfrozen-15K 72.08/58.60. (Figs: dsmvtec_chart, visa_chart.)

Three findings: **(1) Quality ≫ quantity** — 7B-frozen-6K 80.16 vs 15K 71.66 (same ep3) = 8.5 pp gap; data-count/composition effect (6K is a same-length 100% subset of the 15K, both ~137 words), *not* a length or quality-filter effect. **(2) Frozen > unfrozen** — esp. 7B VisA (66.94 vs 58.60, 8.3 pp); the early 7B-unfrozen-6K pilot at the frozen LR peaked 72.8 (ep1) then degraded to 69.3 (ep2) → lowered the unfrozen LR; still below frozen. **(3) 7B > 3B everywhere** (3–11 pp on DS).
> **[takeaway]** Headline SFT = 7B-frozen-6K-ep3 = **80.16% DS** (85.76 F1, 91.65 precision). Within ~1.8 pp of IAD-R1's SFT+GRPO on our harness, with SFT alone.

> *(TODO comment in source: trace-length ablation pending — removed because 6K-vs-15K is a data-count effect, 6K being a same-length subset.)*

## §6.3 GRPO Production Run
> **📋 Table 6.3 (`tab:grpo-results`):** Run-1 ckpt-315 80.63/70.25, ckpt-530 81.65/69.39. **Run-2 ckpt-530 = 82.73/70.39 (headline)**, ckpt-1060 82.03/69.86. SFT-only best 80.16/64.78. **Δ GRPO vs SFT = +2.57 DS / +5.61 VisA.**
Run-2 dominates Run-1 (seed sensitivity). ckpt-530 (ep1) > ckpt-1060 (ep2) by ~0.7 pp → mild reward over-optimisation → early-stop at 530. Larger VisA gain (+5.61) because VisA SFT baseline was weaker (more headroom).
> 🟠 *My note: this section presents 82.73/70.39 as "the headline" — but §6.8/§6.12 later crown Arm-C 82.80/72.07. The reversal isn't propagated here.*

## §6.4 Iterative SFT↔RL and Alternative RL Variants
> **📋 Table 6.4 (`tab:iter2-progression`):** iter-2 v1 (GRPO prompt): SFT₁ 77.95, GRPO₁ 76.89–77.16. iter-2 v2: SFT₁ ckpt-378 77.61/72.00, ckpt-504 79.54/72.02; **GRPO₁ ckpt-530 = 80.64/68.65** (peak), then declines.
> **📋 Table 6.5 (`tab:rl-variants`):** Single-stage GRPO 82.73/70.39 (headline) > G²RPO 81.94 > iter-2 v2 80.64 — both elaborations **below** single-stage on DS.
**"More machinery did not help here."** Iterative cycle (80.64) + G²RPO (81.94) both < single-stage (82.73). Read as a scale/task effect (narrow binary task; single GRPO pass ≈ ceiling for 7B); rollout-and-filter concentrates the SFT corpus on easy already-correct examples (known rejection-sampling failure). **Whether GRPO is needed at all on a strong SFT corpus → answered "no" in §6.13.**
> *Prompt-format caveat:* iter-2 GRPO ckpts evaluated under the GRPO prompt (G), the headline under train prompt (T); they agree within ~2 pp where both available.

## §6.5 Post-GRPO SFT Refinement on Cleaned, Gemini-Patched Data
Motivated by two data flaws (Ch.3): ~30% of NG `<type>` tags carried raw Real-IAD codes (AK/ZW/HS) not readable names; `<location>` mixed center-left vs middle-left spellings. **Claimed corrected: 2,700 type substitutions + 1,167 location normalisations.** 🔴 *(B.4: the location fix hit the GRPO file only; Arm-C's file still has 317 off-grid tags — verify this 1,167 number.)*
Rolled out cleaned 6K + 4K GRPO pool at k=8 (81,888 rollouts) from run-2 ckpt-530 → Gemini-3 judged → 3,770 Gemini-patched traces (94.8% NG).
> **📋 Table 6.6 (`tab:rollout-failure-modes`):** all-8-wrong: 807 NG (15.8%) + 176 OK (3.4%). Failures NG-skewed.
> **📋 Table 6.7 (`tab:sft-iter2-cleaned`):** baseline (run-2 ckpt-530) 82.73/70.39. **6K-pool (3,557 kept + 2,443 Gemini-patched, continue-from-ckpt) = 81.07/70.05 (−1.00)**; Balanced-192 71.30 (−10.65, catastrophic forgetting); Held-out+197/197 75.77 (−5.47). **All regress.**
**Root cause: gradient asymmetry** — kept-as-is items (own rollouts) ≈ zero loss; patched items 94.8% NG, high loss → effective signal ~95% NG → drives NG-recall up, OK-recall down.
> **[takeaway]** All 3 regressed; gradient asymmetry; *"only a bilateral reward signal (GRPO) can extract value from the cleaned data."* 🔴 **My flag: this 6K-pool is the SAME data as Arm-C (§6.12) — the only difference is it CONTINUES from the GRPO ckpt (81.07) whereas Arm-C resets to base (82.80). So "only GRPO can extract value" is wrong; reset-to-base SFT extracts +1.73. This takeaway needs rewriting.**

## §6.6 Explanation Quality: a Reasoning-Faithfulness Comparison
Arm-C vs IAD-R1 on *matched* correctly-detected anomalies (both say "yes"), 100 product-diverse images/benchmark. Each trace scored by a **Gemini-3-Flash judge** (blind to author, told the verdict is correct), 5 axes ×0–2 = /10. Location + Type scored mechanically (Nomic similarity).
> **📋 Table 6.8 (`tab:explainability`):** DS: IAD-R1 4.29 (Loc 0.43, Type 0.34) vs **Arm-C 9.14 (0.82, 0.56)**. VisA: IAD-R1 6.40 (Loc **0.64**, 0.54) vs **Arm-C 8.67 (0.56, 0.53)**.
Arm-C reasoning judged much more faithful: **+4.85 DS, +2.27 VisA**. Honest limits stated: grounding-consistency not mechanistic faithfulness; single judge sample.
> ⚠️ *My flag (C.8): the judge is **Gemini-3-Flash** while the teacher is presented as **Gemini-2.5-Flash**. Decide: rename for consistency vs keep accurate (and whether teacher=judge=same model → self-evaluation concern).*

## §6.7 Real-IAD-Variety Continuation SFT
Built a 6K Variety STaR corpus (Arm-C recipe, 160-cat C1) + continuation-SFT from the two strongest ckpts. **Every probe regressed** (best, subset+rehearsal, ~1.2 pp *below* its Arm-C init; no probe recovered 82.80/72.07). Reported as a **negative** result; reinforces "curated in-distribution corpus is the lever." Proper broad-category study = future work.

## §6.8 Comparison to State of the Art
> **📋 Table 6.9 (`tab:sota`)** (all under one harness, same 1,670/2,141 samples): 7B base 69.01/53.80; **Gemini-2.5-Flash (teacher) 81.52/75.18**; GPT-5-mini 77.10/68.23; Ours-15K-SFT 71.66/64.28; Ours-6K-SFT 80.16/64.78; **IAD-R1 (released) SFT+GRPO 81.92/71.34**; Ours-6K+GRPO 82.73/70.39; **★ Ours Arm-C curated-6K SFT-only = 82.80/72.07 (best in both columns).**
Headlines: **Arm-C beats IAD-R1 on both** (+0.88 DS, +0.73 VisA) with **SFT alone**; SFT+GRPO beats IAD-R1 on DS (+0.81) but trails VisA (−0.95). **Student surpasses teacher on DS** (Arm-C 82.80 > Gemini-2.5 81.52) but teacher leads VisA (75.18, recall-conservative). GPT-5-mini below both (recall-aggressive). Contamination caveat noted.

## §6.9 Per-Product Breakdown on DS-MVTec
> **📋 Fig (`per_product_chart`):** base vs SFT (ep3 frozen 6K). Biggest gains where base was worst: metal_nut +25.8 (57.9→83.6), cable +19.2, pill +16.0, toothbrush +15.8, screw +14.9, transistor +13.3. Texture products (wood −0.1, tile −2.1) flat/slightly regress. VisA: 11/12 improve.
> **📋 Fig (`qual-examples`):** verbatim Arm-C traces (bottle NG → top-right/Missing Parts/Yes ✓; transistor OK → No ✓), showing evidence-before-conclusion.

## §6.10 Held-Out Real-IAD Evaluation
Sample-level held-out (4,236 Real-IAD images never seen in trace-gen/SFT/GRPO; 23 products, *seen* categories).
> **📋 Table 6.10 (`tab:realiad-ood`):** **bal-acc 80.87** (acc 80.81, precision 91.28, recall 68.34, TPR/TNR 68.34/93.40).
Within 2 pp of the DS number; very high TNR (93.4) → rarely false-flags. 🟠 *Evaluated on the **GRPO** checkpoint, not Arm-C — another spot still centering the GRPO model.* Product-disjoint split = future work.

## §6.11 Training and Reward Dynamics
SFT loss 1.14→0.58→0.34 (ep4); val bal-acc peaks ep3 (ckpt-564). GRPO reward: combined 1.66→2.22→2.13 (steps 10/100/530), accuracy 0.875→1.25 (+42.9%), format 0.781→0.875 (+12%), KL bounded [0.003,0.095], completion 163–174 tok. ckpt-1060 < ckpt-530 by ~0.7 pp → early-stop 530.
> **[takeaway]** *"Final aggregate result. … SFT lifts to 80.16/64.78; adding GRPO lifts it further to 82.73/70.39."* 🔴 **My flag: this 'final aggregate' takeaway still ends on SFT+GRPO 82.73 — it should end on Arm-C 82.80/72.07 (the actual best, per §6.12). Stale headline.** *(Also: KL range here [0.003,0.095] differs from §5.5's [0.0,0.117]/13-of-530 — reconcile which window.)*

## §6.12 Teacher Ablation: External Distillation vs. Self-Distillation ⭐
**The key section.** Three arms = three *curation strategies* for the **same OpenVLThinker self-distillation loop** — all re-train on rollouts of the SFT+GRPO policy (ckpt-530) on the fixed 6K image pool; only teacher-intervention level varies. KD framing: Hinton (external), self-distillation (Zhang), rejection-sampling (RAFT/ReST), STaR (rationalise-on-failure), Saunders (critique-and-revise).
- **Arm A** (pure rejection sampling, kept-only, 2,978): best ep4 **79.01/68.83**.
- **Arm B** (STaR, kept + 812 Gemini-corrected, 4,369): best ep3 **80.75/67.48**.
- **★ Arm C** (STaR + Saunders critique-revise, +1,631 rewritten, 6,000, 50/50): best ep2 **82.80/72.07.** ← **headline.**
> **Line 466 (critical):** Arm-C *"is identical in image set and recipe to the iter-2-clean corpus used in §6.5, **but trained from base here rather than from a GRPO checkpoint**."* 🔴 **← This is the reset-vs-continue distinction. It explains why Arm-C (82.80) beats §6.5's 6K-pool (81.07) on the same data — and it matches the verified OpenVLThinker standard. It deserves to be a headline point, not a parenthetical.**
> **📋 Tables 6.11/6.12 (`tab:abc-teacher`, `tab:abc-headline-comparison`):** A −1.15 DS/+4.05 VisA; B +0.59/+2.70; **C +2.64 DS/+7.29 VisA** vs the SFT baseline.
4 findings: (1) C>B>A on DS (each Gemini layer +~2 pp); (2) **C beats SFT+GRPO on VisA (+1.68) with no GRPO**; (3) pure self-distillation (A) beats Gemini-only headline on VisA (+4.05) but not DS — Gemini traces are DS-favourable, model rollouts VisA-favourable; (4) on VisA C>A>B (B's Gemini-corrected NG traces carry a VisA-unfavourable signal).

## §6.13 GRPO on the Best SFT Baseline (Arm-C ckpt-376) ⭐
*Does GRPO help on top of a strong SFT init?*
> **📋 Table 6.13 (`tab:grpo-on-c`):** init Arm-C 82.80/72.07. Every GRPO-on-C ckpt **below** init on DS (−1.6 to −4.0); best VisA ckpt-265 +0.61 then falls. **GRPO degrades the strong baseline** while training reward rises (→3.25) = reward/task decoupling.
3-way **advantage-estimator ablation** (same Arm-C init, 120-step bursts, 400-sample probe): z-score / Dr.GRPO / G²RPO — **none exceeds the SFT init** on DS at any step (Table 6.14 `tab:grpo-estimator-ablation`; G²RPO crashed at step 100). → "the cause is not the normalisation but the absence of binary-accuracy headroom above a well-constructed SFT checkpoint" (cites RLVR-sharpens-not-extends, SFT-is-the-binding-constraint).
> **[takeaway]** *"The single best model in this thesis is the **Arm-C SFT checkpoint itself (82.80/72.07)**, not any GRPO variant. The published SFT+GRPO lift was real against the weak SFT-Iter1 baseline but does not reproduce against a strong one."* ✅ *This is the correct, fully-reversed conclusion — §6 lands here; the early sections just need to be brought in line with it.*

---

## Status
- Mirrors Overleaf `7681742` (orphan `sec:res-compound` removed; 0 dangling refs).
- **5 reconciliation items** in the comments above — none are number errors (C.7 confirmed attribution is clean); they're **framing** (headline propagation, reset-vs-continue prominence) + **two facts to verify** (the 1,167 location-normalisations claim vs B.4; the Gemini-2.5-vs-3 judge).
- This is a *review* render — no edits made to §6 beyond the orphan removal. Tell me which of the 5 to action and we walk it alinea-by-alinea like Ch.5.
