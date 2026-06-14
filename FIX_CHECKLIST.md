# FIX CHECKLIST — working through UNVERIFIED.md

Companion to [UNVERIFIED.md](UNVERIFIED.md). We work top-to-bottom; each item is ticked
`[x]` when done and **kept** (never removed) as an audit trail. Scope agreed: **everything**
(§1.4 numbers + §2 unverifiable + all §3 claims + §4 placeholders + the new prompt-mode item).
§3 items are decided **one-by-one with the author**.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[-]` decided "leave as-is"

---

## GATING DECISIONS (decide first — they shape several edits)
- [x] **G1. 6K-unfrozen rows — DECISION (author, 2026-06-13): do NOT run them; REMOVE the rows and JUSTIFY the omission.**
      Rationale (author): unfreezing the vision encoder is already shown to hurt — the **15K-unfrozen** cells lose to frozen on both benchmarks (esp. VisA), and prior work (IAD-R1) recommends a frozen ViT. So the unfreezing question is settled at 15K; running it again at 6K adds nothing. (A launch attempt was made and immediately killed — nothing trained, all artifacts removed.)
      - [x] Rows stay removed; table `% TODO` comment → definitive justified note (06_results).
      - [x] Reframed all grid-count claims to **6 configurations** (unfrozen at 15K only): intro contrib 2 ("16-cell ablation grid" removed), Ch4 §design ("32 cells" → "6 configs / 24 epoch-cells" + IAD-R1 justification), 06_results L41 ("16 SFT cells" → "6 configurations"), Ch6 frozen>unfrozen paragraph appended with the omission rationale. Zip re-synced, structure validated. (2026-06-13)
- [ ] **G2. Prompt-mode finding** (not yet in UNVERIFIED.md): add the entry, and choose framing — lead same-mode +0.58, or keep +2.57 with cross-mode caveat.

## §1 — already FIXED (verified this session; kept as record)
- [x] §1.1 Baseline 3B DS 55.80 → 56.14 (+ all baseline cells exact)
- [x] §1.2 tab:sft-summary cross-contaminated cells corrected from true checkpoints
- [x] §1.2b tab:sft-summary 15K rows **reselected to true best-DS epoch** (3B-F-15K→ep1 69.56; 7B-F-15K→ep4 72.60; 7B-U-15K→ep3 72.08), caption states criterion, dependent prose (VisA frozen-gap, 7B>3B range) reconciled, documented in NUMBER_PROVENANCE.md revision log (2026-06-13). [verified: all 15K runs have full DS+VisA, 4 epochs]
- [x] §1.3 Appendix C SFT+GRPO column + CM table regenerated from ckpt-530 JSON; footnote fixed; Ch7 per-product claim rewritten

## §1.4 — confirmed-wrong numbers still in the thesis (data-backed corrections)
*§1.4 STATUS (2026-06-14): COMPLETE — all items applied/verified; 1.4m reclassified unverifiable (left as-is); 1.4d resolved no-change; model kept at 2.5-Flash per author.*
- [x] 1.4a  SFT→IAD-R1 gap → **1.8 pp** applied (abstract/intro/Ch6-takeaway; SOTA already 1.8). [src: 81.92−80.16]
- [x] 1.4b  intro contrib4 → **+11.2 pp (SFT over base)** (was +13.6; consistent with cited base→SFT per-product gains). [decided w/ author]
- [x] 1.4c  product count: VERIFIED 30 (SFT 30 / GRPO 23 / held-out 23, union 30). Ch3/intro already correct. FIXED conclusion "25"→30 and App-A "25/5"→"30/23/23". Ledger's "29" was wrong.
- [x] 1.4d  NO CHANGE — Ch3 already says **26** (matches App-B + real prompt inspector_prompt_test_v2.txt=26). "Controlled vocabulary" framing is accurate (real structured list). Ledger 25→26 was stale.
- [x] 1.4e  trace-length stats: std 28→**13.5**, mean 170→**137**, range 160–270→**121–223**, concl 141→**137**
- [x] 1.4f  t-SNE: Scratch 600→**500**, Missing 180→**162**, dist 148.6→**155.3**, radius 16.5→**15.3**, ratio 9.0→**10.1**
- [x] 1.4g  3B layers 28 → **36** applied (Ch4 L29; 7B 28 correct). [src: 3B config.json]
- [x] 1.4h  7B-unfrozen-15K wall-clock 22.5h → **12.2h** applied (6.9/2.4h verified-correct, kept). [src: all_results.json]
- [x] 1.4i  SFT hyperparams — DONE. Tables corrected (Ch4 + App-A) + both rationale paragraphs rewritten: 'Why linear schedule' → honest cosine note; 'Why low LR' → encoder-dependent LR (1e-5 frozen / 1e-6 unfrozen) citing the matched-LR pilot; 5e-5 anecdote DELETED (verified: no 5e-5 run exists anywhere on disk). + documented in NUMBER_PROVENANCE.md revision log (all 7 runs' training_args.bin). TABLES APPLIED (2026-06-13): Ch4 tab:sft-hp + App-A tab:A.2 corrected to training_args.bin (LR 1e-5/1e-6, cosine, warmup 20/50, wd 0, β2 0.999, seqlen 12144, EBS 32; App-A dropped 2 phantom unfrozen-6K rows, added Arm-C row). PENDING: the two rationale paragraphs (Why-linear / Why-low-LR).
      - [x] Lower-LR-for-unfrozen claim BACKED by matched-LR pilot (`sft_qwen25vl_7b_zeroshot_6k`, 7B unfrozen @1e-5 on 6K): peaks 72.82 then degrades to 69.27, vs 80.16 frozen. Cited in 06_results frozen>unfrozen paragraph + documented in NUMBER_PROVENANCE. Resolves the frozen-vs-unfrozen LR confound (2026-06-13).
- [x] 1.4j  GRPO sample count — ALREADY FIXED earlier (no 6,500/13,000 left in Ch5/App-A; =4,236). verified
- [x] 1.4k  GRPO KL range [0.003,0.095] → **[0.0,0.117]** applied (+13/530 steps note). [src: run-2 trainer_state]
- [x] 1.4l  Ch6 caption — ALREADY FIXED earlier ('the remaining thirteen gain'). verified
- [x] 1.4m  NO CHANGE — ledger was WRONG; thesis 'step 752 / 0.34 / ep4' is CORRECT (run ends global_step 752, ep4, loss ~0.34). Ledger corrected.
- [x] 1.4n  Ch7 run-1 ckpt-530 80.6 → **81.65** applied (80.63=ckpt-315). [src: run-1 ckpt-530 eval]
- [x] 1.4o  Ch8 contrib5 per-product gains — confirm scope (base→SFT +25.8/+19.2/+16.0 are CORRECT as written; ledger's base→GRPO "correction" is mis-scoped)
- [x] 1.4p  App-A Table A.3 GRPO (warmup_ratio, β2, wd, BS, max-prompt, save, eval, stitch) — mostly done; verify each

## §2 — unverifiable (no artifact on disk)
- [x] 2a  6K-unfrozen rows → RESOLVED via G1: not run (deliberate, literature-backed); rows removed; grid reframed to 6 configs.
- [x] 2b  GRPO-on-C ckpt-795 (80.76/69.69) — rotated away. Drop row / report surviving ckpts (we now have the full 265→2120 curve)
- [x] 2c  Estimator-ablation step-0 "init" (84.52/71.89) — probe_curve.csv only. Ship probe CSV / mark probe-only
- [x] 2d  GRPO run-1 numbers — run-1 dir not shipped in repo. Ship JSONs / soften seed claim

## NEW — prompt-mode (add to UNVERIFIED.md) → see G2
- [ ] PM  Document that headline GRPO 82.73 & IAD-R1 81.92 are grpoprompt while SFT/Arm-C are trainprompt; same-mode trainprompt GRPO = 80.74 (+0.58 over SFT 80.16). DECISION (G2): keep +2.57 headline with cross-mode caveat.
      - [x] IAD-R1 prompt-robustness measured (2026-06-13): DS 81.92 (grpo) / 81.45 (bare) / 81.19 (native) — spread 0.73 pp; VisA 71.34 / 71.56 / 71.87 — spread 0.53 pp. IAD-R1 is prompt-robust, so the SOTA comparison stands regardless of mode. (Files: outputs/iad_r1_qwen_recanon/eval_*_full_{bareprompt,iadr1native}.json, stamped with prompt_info.)


## GENERATION PIPELINE (added 2026-06-14)
- [x] Added canonical Real-IAD v4 pipeline -> scripts/00_generate/realiad_v4/ (from realiad_v4_pipeline.zip; secrets excluded, HF token scrubbed). Uses inspector_prompt_test_v2.txt + google/gemini-3-flash-preview.
- [x] Variety (Real-IAD-Variety) generation scripts confirmed in scripts/00_generate/ (download/sample/generate/validate/assemble + run_variety_*.sh).
- [x] Added scripts/00_generate/README.md documenting both pipelines.
- [ ] **AUTHOR DECISION**: thesis says "Gemini 2.5-Flash" (~14 places) but both v4 + variety generators use **google/gemini-3-flash-preview**. Confirm which model produced the headline corpus -> correct thesis if Gemini-3. (NUMBER_PROVENANCE model-provenance entry.)
- [x] RETRACT earlier 1.4d / §3-AppB "prompt is idealised / vocab mischaracterised" — the real v4 generator DOES use the structured inspector_prompt_test_v2.txt.

## §3 — unverified / estimate / rhetorical claims (decide ONE-BY-ONE with author)
- [ ] 3.Ch1  intro claims (>95% floor, "last five years", 16-cell grid, "6K concise vs 15K verbose", product count)
- [ ] 3.Ch2  background (scratch mm sizes, "six-phase" vs 4-tag, −β·KL-in-loss generality)
- [ ] 3.Ch3  dataset (120–200 word budget vs prod prompt, six-phase wording, 5 auto-reject rules, token budget, cost/latency estimates, mask filter)
- [ ] 3.Ch4  SFT (1K pilot, 5e-5 collapse, cosine-rationale CONTRADICTION, beam/nucleus check, 100K lazy, 256GB RAM, ~300M ViT, 6-epoch, full-vs-LoRA pilot, bf16, AdamW defaults)
- [ ] 3.Ch5  GRPO (reward constants, <2% skip, ~30s/rollout, Nomic 600MB/700 lines, completion 163–174, 12.4h vs 24.7h, reward-line listing, 2700/1167 cleaning counts, trl-vs-IAD-R1 lineage, outer-whitening)
- [ ] 3.Ch6  results (16-cell, unfrozen-6K rows, 7.6pp & ~2pp prompt agreement, ViT/ranges, cleaning counts, Gemini-patch counts, rollout-failure table, 6K composition, NG-gradient %, arm sub-counts)
- [ ] 3.Ch7  discussion (frozen-vs-unfrozen gap, 100×/1.5M, 8.5pp gap consistency, structural-uniformity claim, under-trained, VisA defect-area/downsample, 3.5pp artefact, β future, FORMAT-COMPLIANCE %, 2.0–2.5pp wording, ERROR TAXONOMY, seeds, LATENCY)
- [ ] 3.Ch8  conclusion (25 products, ~1pp, 141 words, error categories, latency, future-work proposals)
- [ ] 3.AppA  generator temp 1.0, SFT hyperparams, unfrozen-6K rows, GRPO args, max_steps wording
- [ ] 3.AppB  six-phase prompt redaction, 5-vs-8 rule disclosure, "single prompt for SFT/GRPO/eval" not literally true

## §4 — placeholders (we now have data for 2 of 3!)
- [ ] 4a  Variety STaR (F5) — results NOW EXIST (DS ~78.5–79.3, VisA ~67.4–69.0). Fill table or mark future work
- [x] 4b  GRPO-on-C continuation — full curve NOW EXISTS (265→2120). Fill §6.13 table / drop ckpt-795
- [ ] 4c  Compound F5a/F5b/F5c — still pending (not run). Keep as marked placeholder

---
*Started: 2026-06-13. Update this file as items are completed; do not delete completed items.*
