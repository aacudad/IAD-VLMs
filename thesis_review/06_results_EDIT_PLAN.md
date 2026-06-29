# Ch.6 edit plan — your commentary, section by section

Working doc for the §6 walk. Your voice is applied to every proposed passage (typos/grammar fixed, numbers/cites preserved, **no em or en dashes**). Nothing is written to Overleaf until you greenlight each item.

Legend: ✅ ready to apply · ❓ needs your decision · 🔬 verified against disk this session · 🖼️ figure/diagram to make

---

## TO-DO LIST (consolidated)

**Ready to apply (your voice, mechanical):**
- [ ] Roadmap: "single best detector" -> "single best model"; adopt your rephrasings
- [ ] §6.1: your rewrite + resolve the appendix pointer (prompts = App B)
- [ ] §6.2: de-dash the title; name the 4 factors (done already in roadmap); reframe "quality > quantity" -> "more data does not mean better performance"
- [ ] §6.2 frozen>unfrozen: fix 1-decimal numbers to 2 (72.82 / 69.27 / 80.16); drop "headline" from "the grid"
- [ ] §6.3: retitle "GRPO Production Run" -> "GRPO Run"; **fix seed + cadence claims** (see 🔬)
- [ ] §6.4: your rewrites; fix "89.54"->"79.54"; gloss machinery/corpora
- [ ] §6.5: gloss "post-hoc"; add the qwen-style rationale sentence

**Needs your decision (❓):**
- [ ] Q1 §6.7 Variety: report it or cut to an appendix/footnote?
- [ ] Q2 §6.9: add VisA per-product (currently DS-MVTec only; VisA per-product lives in App D)?
- [ ] Q3 §6.10 held-out: keep in main text or move to appendix?
- [ ] Q4 §6.2: **drop "it even resulted in overfitting"** (contradicts §7: 15K is *under*-trained, not overfit) — confirm
- [ ] Q5 §6.3: keep both runs (fix the seed claim) OR report only run-2?
- [ ] Q6 §6.4: drop the iter-2 v1 rows from Table 6.4 (weights gone, G-prompt only, superseded by v2)?
- [ ] Q7 §6.5 location: how to phrase "1,167 normalisations" / "cleaned corpus" given the headline likely trained pre-fix (needs git/backup check of the pre-fix file)
- [ ] Q8 §6.12 discoverability: rename to "Teacher Ablation: the Arm-C Best Model" and/or add a forward-pointer from §6.3/§6.5 so the Arm-C result is findable before page ~last

**Figures/diagrams to make (🖼️):**
- [ ] §6.2: decide figure design (stacked bars? split 3B/7B and 6K/15K?) — recommendation below
- [ ] §6.5: a diagram of the 5-level faithfulness rubric + a figure
- [ ] §6.5 Fig 6.5: de-dash caption + reselect better sample images (walk through together)
- [ ] earlier chapters (Ch.3 dataset-generation): better figures (noted for later)

---

## Roadmap (chapter intro)

**Your asks:** "single best detector" -> model/checkpoint; reword; the inline "(report this?)" notes are questions, answered below.

✅ **Proposed (your voice):**
> This chapter reports the experimental results of this thesis. Section 6.1 sets the performance of the base Qwen2.5-VL models. Section 6.2 reports the SFT four-factor ablation. Section 6.3 reports the GRPO run and its effect on the SFT baseline. Sections 6.4 and 6.5 report two further post-training variants. Section 6.6 compares explanation quality against IAD-R1 with a reasoning-faithfulness judge, and Section 6.7 tests continuation training on Real-IAD-Variety. Section 6.8 then compares performance against IAD-R1 [cite], the current SOTA. Section 6.9 reports the per-product breakdown on DS-MVTec **and VisA**. Section 6.10 reports the held-out Real-IAD test result. Section 6.11 reports the training and reward dynamics. Finally, Section 6.12 presents the teacher ablation that yields the **single best model** in this thesis, the curated Arm-C SFT checkpoint, and Section 6.13 asks whether GRPO adds anything on top of the strong baseline.

❓ "and VisA" in the 6.9 line is contingent on Q2. The 6.7/6.10 wording is contingent on Q1/Q3.

**Answer "why detector?":** "detector" is technically fine (the model *is* an anomaly detector), but "model" reads cleaner and matches how §6.12/§6.13 refer to it. Recommend **"model"** (or "checkpoint" if you prefer the artifact noun).

---

## §6.1 Baseline Performance

✅ **Proposed (your voice):**
> We first test the zero-shot performance of Qwen2.5-VL-3B and Qwen2.5-VL-7B without any fine-tuning. This sets the headroom available for the rest of the chapter. The base models are prompted with the same prompt and two-image user turn as the fine-tuned models; the exact prompt is in Appendix B.
>
> Two observations are worth flagging. First, the 3B baseline almost always says "yes" (recall to 100%, precision around 56 to 76%). This inflates raw accuracy, but in balanced accuracy it sits near 50%. Second, the 7B baseline does the opposite: it is so conservative that on VisA it catches only 7.7% of true defects. The bal-acc metric exposes both pathologies, which justifies its use as our primary metric in the rest of the chapter.

- "(come with an appendix)" resolved: the eval prompt is in **Appendix B (prompts)**. I will confirm the exact label (`app:prompts`) before writing.
- ❓ You dropped the original sentence "the same regex-based answer extraction is applied." Keep it (it is a real detail) or leave it out? (Recommend keeping a short version.)

---

## §6.2 SFT Ablation (de-dash the title)

❓ **Title:** "SFT Ablation -- Four-Factor Grid" has an en dash. Options: **"SFT Ablation: Four-Factor Grid"** (recommended) or "Four-Factor SFT Ablation Grid". Label `sec:res-sft` is unchanged, so cross-refs stay intact.

✅ **First paragraph (your voice):** the four factors are already named (matches the roadmap edit). Final sentence: "Each is then checkpointed and evaluated on both DS-MVTec and VisA at every epoch. We report bal-acc by epoch in Figures X and Y, and summarise the best checkpoint per cell in Table 6.2."

🖼️ **Figure-design question (Q):** the two line charts (`dsmvtec_chart`, `visa_chart`) plot bal-acc by epoch for all 6 cells, which is busy. Options:
- **A. Keep two line charts** (one per benchmark) — simplest, shows the epoch trajectories.
- **B. Grouped/stacked bar chart** of best-epoch bal-acc per cell — cleaner for "which config wins" but loses the epoch story.
- **C. Split by factor:** a 3B panel vs a 7B panel (and/or 6K vs 15K) — clearest for reading each factor's effect.
- **My recommendation:** keep the **two line charts** (they carry the epoch/over-fit story) but **split each into a 3B panel and a 7B panel** so the 6-cell tangle becomes readable, and add a small **best-epoch bar chart** as the summary. We can prototype it.

🔴 **"quality > quantity" -> reframed (you are right).** Both corpora are the same unfiltered Gemini traces and the 6K is a literal subset of the 15K, so "quality" is not the lever. Proposed header + body:

✅ **Proposed (your voice):**
> **More data does not mean better performance at the SFT stage.** The 7B-frozen-6K model reaches 80.16% bal-acc on DS-MVTec. The same configuration at 15K training samples reaches 71.66% at the same (epoch-3) checkpoint, an 8.5-point gap. The gap is smaller when each corpus is taken at its own best epoch (6K ep3 80.16% vs 15K ep4 72.60%), but still 7.6 pp. Either way it is the largest single effect in the ablation, and it is opposite to the conventional intuition that more data is always helpful. Importantly, the two datasets are not a concise-versus-verbose contrast: the 6K SFT split is a 100% subset of the roughly 14.5K set (the full stratified C1-angle partition), and both have the same mean trace length (about 137 words). So the gap is a pure data-count and data-composition effect at fixed compute (composition = which traces are in the set: the 6K is the clean stratified SFT split, whereas the 15K additionally folds the GRPO and held-out splits in as SFT data). Training on the curated 6K SFT split alone matches or beats training on the full union. We do not attribute this to a single mechanism, and it is not a quality-ranking filter (both corpora are unfiltered Gemini output, Section 3.4). The additional roughly 8.5K traces provide no benefit and slightly hurt at a fixed 4-epoch budget.

- **Answer "what does composition mean?":** the *mix/makeup* of the dataset (which products, which class balance, which splits are folded in), as opposed to the raw count. I glossed it inline above.
- ❓ **Q4:** I **removed your "In this case it even resulted in overfitting."** §7 attributes the 15K underperformance to *under-training* (its loss is still descending at epoch 4 because 15K needs more passes), which is the opposite of overfitting. Confirm you are OK dropping it (or we phrase the real mechanism).

✅ **Frozen > unfrozen (your voice, numbers fixed to 2 decimals):**
> **Frozen > unfrozen.** Freezing the visual encoder outperforms unfreezing it on DS-MVTec at both model sizes (3B 69.56 vs 68.58; 7B 72.60 vs 72.08, on 15K). The effect is most visible on VisA for the 7B model: frozen 66.94% vs unfrozen 58.60%, an 8.3 pp gap; at 3B the VisA difference is negligible (59.65 vs 59.60). We interpret the 7B VisA effect as classical overfitting: the visual encoder's parameters are highly expressive and small industrial training sets cannot supervise them robustly. An early 7B-unfrozen pilot on the 6K pool, trained at the frozen cells' learning rate, made the instability concrete: it peaked at **72.82%** DS-MVTec at the end of epoch 1 and then degraded to **69.27%** by epoch 2 (vs **80.16%** for the frozen cell). We therefore lowered the unfrozen learning rate by an order of magnitude to stabilise training; unfreezing nonetheless stayed well below frozen at every learning rate we tried, and prior work [IAD-R1] reaches the same conclusion. The reported unfrozen cells therefore use the 15K dataset, and we do not add unfrozen-6K cells to the grid.

- **Answer "2-decimal inconsistency":** you were right. 72.8/69.3/80.2 are now 72.82/69.27/80.16 (the real pilot values).
- **Answer "check the ref":** yes, it is **IAD-R1 (`\citep{li2025iadr1}`)** — IAD-R1 recommends a frozen ViT.
- "headline grid" -> "the grid" (per your note).

---

## §6.3 GRPO Run (not "Production Run")

🔬 **Verified, both your hunches correct:**
- Both run-1 and run-2 used **seed 42** -> the "different random seeds" claim is **wrong**.
- Save cadence: **run-1 every 315 steps, run-2 every 530** -> "every 250 steps" is **wrong**.
- (This matches the §7 limitation, which already says the two runs share seed 42 and differ only by rollout non-determinism.)

❓ **Q5 — two options:**
- **(a) Keep both runs, fix the claims:** "we ran two GRPO runs at the **same seed (42)** and identical hyperparameters; the run-to-run difference therefore reflects sampling/rollout non-determinism rather than seed variance. Run 1 saved every 315 steps and run 2 every 530." This is honest and supports the §7 single-seed limitation.
- **(b) Report only run-2** as the GRPO result and mention run-1 once as a replication.
- **Recommend (a)** (the run-to-run gap is a useful honesty point), but your call.

✅ **Retitle:** "GRPO Production Run" -> **"GRPO Run"**.
✅ **Over-optimisation sentence:** "the further-trained ckpt-1060 (end of epoch 2) underperforms ckpt-530 (end of epoch 1) by ~0.7 **pp**" (you wrote "epoch"; it is percentage points). "We revisit this in Section 7.x" via the existing `\ref`.

---

## §6.4 Iterative SFT<->RL and Alternative RL Variants  *(your notes mislabeled this "7.4"; it is §6.4)*

🔬 **Verified:**
- **iter-2 v1 weights are gone** (only `iter2_v1_eval_archive_.../` with eval JSONs survives). So Table 6.4's v1 rows (ckpt-196 etc., evaluated under the GRPO prompt) **cannot be re-evaluated** under the train prompt without retraining.
- **iter-2 v2 weights exist** (`sft_..._iter2_v2_frozen` ckpt-126..504; `grpo_..._iter2_v2_full` ckpt-265..1060).
- **Init / comparability:** G2RPO (`g2rpo_v2_full`) initialises from the same 6K-frozen SFT as single-stage GRPO, so it *is* comparable. iter-2, by design, **resets to base and re-SFTs**, so its GRPO stage initialises from iter-2's *own* SFT1 (ckpt-504), a different init. So Table 6.5's "all initialised from the same SFT cold-start" is **loose** for iter-2.

❓ **Q6 — recommend dropping the iter-2 v1 rows** from Table 6.4 (weights rotated, GRPO-prompt only, superseded by v2). That simultaneously fixes your "why was train prompt not used for ckpt-196 / should we include it" concern. Keep v2 only.

✅ **Two-patterns paragraph (your voice, "89.54" -> "79.54"):**
> Two patterns recur from the single-stage run. First, the iter-2 GRPO stage again peaks at the end of epoch 1 and then declines with further training, the same mild over-optimisation diagnosed in Section 6.11. Second, the RL stage lifts DS-MVTec over its own iter-2 SFT base (80.64% vs 79.54%, +1.10 pp) but costs a few points on VisA (68.65% vs 72.02%), the same precision/recall rebalancing that GRPO performs.

- **Answer "(how?)" precision/recall rebalancing:** GRPO's reward gives credit for a correct "yes" verdict, so the policy learns to call "defect" more readily; this raises recall/TPR (catches more defects) at the cost of precision/specificity (more false alarms). On DS-MVTec that net-helps; on VisA, where the SFT model was already recall-shy, pushing toward "yes" over-corrects and the OK class suffers, so VisA bal-acc dips. I can add a one-line gloss.
- **Answer "machinery":** the extra algorithmic complexity (additional training stages and the alternative advantage estimator). I will reword to "more elaborate methods did not help here" to avoid the colloquialism.
- **Answer "corpora":** plural of "corpus" = the training datasets/text collections. I will say "datasets".

✅ **Finding paragraph:** your voice, with "machinery"->"elaborate methods", "corpora"->"datasets", and the cross-ref to §6.13 kept.

❓ **Q (caption):** reframe Table 6.5 caption from "all initialised from the same SFT cold-start" to "each strategy's best checkpoint (single-stage and G2RPO share the SFT init; iter-2 resets to base and re-SFTs, so it is the best of that pipeline rather than a same-init comparison)."

---

## §6.5 Post-GRPO SFT Refinement on Cleaned, Gemini-Patched Data

- **Answer "post-hoc":** "after the fact" — the two data flaws were found *after* the GRPO run, not before it. I will write "two systematic data flaws found after the fact (Chapter 3)".

🔴 **Q7 — the location-cleaning claim (your item 3 / B.4). Verified this session:**
- Headline Arm-C (`abc_C_full_patched`, ckpt-376) and §6.5's 6K-pool both train on `iad_sft_iter2` = `Training/datasets_sft_iter2/sft_iter2_train.json`.
- That file *now* uses the 3x3 spelling (middle-left/middle-right/center, no center-left/center-right), so the "center-left vs middle-left" spelling fix **is** reflected in the current file. BUT it still contains **hundreds of multi-region comma tags** (e.g. "center,middle-right") and a few `\r\n` artifacts, so it is not a perfectly clean single-cell 3x3 set.
- There is a **separate `abc_C_locfixed` run** on the same dataset name, which strongly implies the headline `_full_patched` trained on a **pre-fix** version of the file (the fix was applied later and re-trained as `_locfixed`).
- **Therefore:** before we state "1,167 location normalisations" and call Arm-C the "cleaned corpus", we should (i) confirm the pre-fix file content (git/backup of `sft_iter2_train.json`), and (ii) decide whether the headline number (82.80/72.07) belongs to the pre-fix or post-fix run. **I recommend we soften §6.5 to "type-tag substitutions (and a location-spelling normalisation)" without the unverified 1,167 count, and reconcile §6.12's "cleaned corpus" wording**, pending that check. Your call on how far to chase it.

✅ **Add your qwen-style rationale** to the rewrite step: "...were rewritten with minimal-change instructions, to keep the answers in the Qwen2.5-VL-7B style rather than the Gemini-3-Flash style."

🖼️ **§6.5 figures:**
- Rubric: make a **diagram of the 5-level faithfulness rubric** (Hallucinated / Poor / Acceptable / Good / Excellent) plus a distribution figure.
- **Fig 6.5:** de-dash the caption, and reselect better sample images (we walk through together).
- Note: better figures wanted in the earlier dataset-generation chapter(s) too.

✅ **Root-cause paragraph (your voice):** apply as written (typos fixed: "3000 ng + 3000 ok").

❓ **Q8 — "WE MISS ARM-C RESULTS SECTION?":** No, it is **§6.12 (Teacher Ablation)** — Arm C is defined and crowned there (82.80/72.07). But the fact that you, the author, expected an Arm-C section and did not find one by §6.5 is strong evidence it is **buried**. Options: (i) rename §6.12 to "Teacher Ablation: the Arm-C Best Model"; (ii) add a forward-pointer in §6.5's takeaway ("the same corpus, trained from base, is the thesis's best model; see §6.12"); (iii) both. This ties into the earlier items 1 and 2 (reset-vs-continue prominence + headline propagation).

---

---

# POST-VERIFICATION FINAL PROPOSALS (your 8 decisions + 3 verified facts)

## 🔬 Q4 — "did the 15K overfit?" Verified: NO. (you are not right here, and I can show why)
Per-epoch DS-MVTec bal-acc (recomputed from eval JSONs):
- **6K:** ep1 79.24, ep2 76.66, **ep3 80.16 (peak)**, ep4 77.07 -> peaks then DROPS = textbook overfit shape.
- **15K:** ep1 71.00, ep2 65.40, ep3 71.66, **ep4 72.60 (best = last epoch)** -> still climbing, never near 6K's 80.16.
- Training loss converges to the SAME floor for both (6K 0.344, 15K 0.365), so 15K is not under-optimised either.

So the overfit signature (peak early, decline) is the **6K** run, not the 15K. The 15K is **uniformly worse at every epoch** and its best is the final one. That is exactly the data-count/composition story, not overfitting. **Proposed: drop "it even resulted in overfitting"; optionally ADD a sentence that turns this into a strength:** "This is not a checkpoint-selection artifact: the 15K run is behind at every epoch and its best checkpoint is the final one, so it is not overfitting; if anything the 6K run shows the mild peak-then-drop (ep3 80.16, ep4 77.07)." (Do NOT call the extra traces "lower quality" — that contradicts our own "unfiltered, same-quality subset" point.) Add your "especially for a 7B model" clause.

## 🔬 Q2 — VisA per-product (verified vs App D, base -> SFT 6K ckpt-564, n=2141)
macro avg base 53.81 -> SFT 66.89 (**+13.08**). Only **pcb4 regresses** (61.00 -> 54.83, -6.18, the one product base was already above chance). Top gains: pipe_fryum +39.24, chewinggum +27.28, fryum +26.41, macaroni1 +18.50, cashew +15.50, pcb2 +13.93, candle +11.50; small: pcb1 +4.96, pcb3 +3.10, macaroni2 +1.50, capsules +1.17. -> add a VisA per-product figure (mirror the DS one) + 2 sentences in §6.9, pointing to App D for the full table. (Need to generate `per_product_visa_chart.png`.)

## 🔬 Q7 — Arm-C location provenance. Verified, DECISIVE:
- Headline Arm-C = `abc_C_full_patched/ckpt-376` = **82.80/72.07**, and it **trained on UN-normalised locations** (the `.old` / committed-repo file, which still has 317 off-grid center-left/center-right tags).
- The normalisation was applied **in place afterwards** and re-trained as `abc_C_locfixed` = **82.00/71.48** -> the location fix made results **slightly worse**, not better.
- The actual location-tag change is **317**, not 1,167. And **0 type tags changed** between pre/post files (both already code-free) -> the "2,700 type substitutions" did not happen in this edit (it was upstream at generation). **Both the 2,700 and 1,167 counts are unsupported for this file.**
- **Proposed (your decision #7, drop counts):** "Both were subsequently corrected: raw Real-IAD defect codes were mapped to the human-readable type vocabulary, and the `<location>` spelling was normalised to the single 3x3 grid." 
- **Plus a consistency fix you should decide on:** §6.12 currently calls Arm-C "identical ... to the iter-2-clean corpus." Since the headline trained PRE-fix and the fix didn't help, soften to "the same corpus (a later location-normalised re-run gave 82.00/71.48, within noise)" OR just drop the "cleaned" descriptor. (Optional honest caveat available if you want it.)

## Your decisions -> concrete edits
- **#1 Variety (§6.7): CUT.** Remove the section; remove it from the roadmap; keep Variety only as future work in §8 (F5). Must grep `sec:res-variety-cont` for dangling refs first.
- **#2 §6.9: ADD VisA per-product** (numbers above) + figure; point to App D.
- **#3 §6.10 held-out: MOVE to an appendix** (it's evaluated on the GRPO model; a sample-level test). Remove from roadmap; add a one-line forward-pointer if desired. Must repoint `sec:res-realiad-ood` refs.
- **#5 §6.3: report ONLY run-2.** Drop the "two runs / different seeds / every 250 steps" sentence entirely; Table 6.3 keeps only run-2 (ckpt-530 bold, ckpt-1060) + SFT-only + Δ rows. Retitle "GRPO Production Run" -> "GRPO Run". Keep the run-to-run variance as a §7 limitation (it's the only place run-1 then appears). NOTE: this means the seed claim never appears, so nothing to correct.
- **#6 §6.4: drop the iter-2 v1 rows** from Table 6.4 (weights rotated, G-prompt only). Keep v2; simplify "iter-2 v2" -> "iter-2". (The live text already says 79.54, not 89.54 — your "89.54" was a commentary typo; no fix needed.)
- **#8 ORDERING (the real issue): REORDER so Arm-C is shown before §6.5 references it.** Proposed new order:
  1. 6.1 Baseline
  2. 6.2 SFT four-factor ablation
  3. 6.3 GRPO run
  4. 6.4 Iterative / variant RL
  5. **Teacher ablation (Arm-C, the best corpus)** [was 6.12]
  6. **GRPO on Arm-C** [was 6.13]
  7. **Post-GRPO SFT refinement** [was 6.5] -> now AFTER Arm-C, so its reset-vs-continue point lands (and its takeaway is rewritten: continuing from the RL'd checkpoint regresses; the same data trained from base is Arm-C, the best model -> the lesson is reset-to-base, not that GRPO is required)
  8. Explanation quality [was 6.6]
  9. SOTA [was 6.8]
  10. Per-product DS + VisA [was 6.9]
  11. Training & reward dynamics [was 6.11]
  (Variety cut; held-out -> appendix.)
  Execution: done via the `/tmp/ol_thesis` git clone (move blocks, fix `\ref`s, commit, push), NOT write_section.

---

## Sections not yet reviewed by you
§6.6 (explanation quality, where you stopped at "The results so far"), §6.7, §6.8, §6.9, §6.10, §6.11, §6.12, §6.13. The 5 items from your earlier review render (reset-vs-continue, headline propagation, location fix, GRPO-upstream nuance, Gemini-2.5-vs-3 judge) still stand for those.
