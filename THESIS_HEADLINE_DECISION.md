# Thesis headline decision & narrative (decided 2026-06-20)

Authoritative record of the thesis's central claim, the headline-model decision, what
stays in Future Work, and where every supporting number lives. If the thesis text and
this doc ever disagree, **this doc is the intended story** — fix the text to match.

## DECISION: the single headline model is **Arm-C SFT**, NOT SFT+GRPO

- **Headline best model = Arm-C SFT `checkpoint-376` = 82.80 DS-MVTec / 72.07 VisA (balanced acc).**
  Path: `outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376` ·
  evals `checkpoint-376/eval_{dsmvtec,visa}_full_trainprompt.json`.
- **SFT+GRPO (run-2, 82.73 / 70.39) is DEMOTED** from "the headline" to "a weak-init GRPO
  result" — it is *not* the best model (it trails Arm-C on both metrics; the late
  teacher-ablation established this). Path: `outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530`.
- **Reason:** the teacher ablation made Arm-C SFT the best checkpoint; GRPO on the strong
  init does not beat it under the recipe the thesis describes. The pre-reversal framing
  (SFT+GRPO = headline) survived in the abstract/intro/conclusion and must be propagated out.

## The claim set (the story to tell)
1. Base VLM is a weak, opaque IAD reasoner (Qwen2.5-VL-7B base ≈ **69.0 / 53.8**).
2. **Curated reasoning-trace SFT is the decisive lever** → best = **Arm-C 82.80 / 72.07**, which
   **beats the IAD-R1 reference on VisA (+0.73 pp)** under our common harness.
3. **Quality > quantity:** 6K curated SFT beats 15K by ~8.5 pp on DS — a *count/composition*
   effect, **not** noise: GPT-5-mini verifier rates the corpus **97.8% correct**, and
   filtering the 6K to 5,797 "completely-correct" traces is a **wash** (72.79 vs 72.47 avg).
4. Frozen ViT > unfrozen.
5. **GRPO (RL) is a secondary, conditional refinement:** it lifts a **weak** SFT init
   (80.16 → 82.73), but under the written recipe does **not** beat the **strong** Arm-C init
   (over-optimization / data ceiling) — across β, estimators, and the Gemini-judge reward.
6. Explainability/faithfulness is the differentiator (faithfulness-evaluated traces).

## FUTURE WORK (do NOT write as a current result): prompt-aligned GRPO on the strong init
- **Do NOT claim in the thesis body that we ran GRPO on Arm-C and beat it.** Frame as future work:
  *"Preliminary: GRPO may give a small further gain on the strong SFT init when the RL
  training prompt is aligned with the evaluation prompt — to be reported once finalized."*
- The in-progress run (`grpo_sftprompt_kl0.1_sys_3ep`, eval-aligned `sft_sys` prompt, β=0.1,
  3 epochs) edges the ceiling: **DS 82.88 (tie), VisA 73.70 (+1.6), best-avg ckpt-636 77.56 > 77.44 init** —
  but it is **in progress (epoch ~1.6 of 3), a different prompt config than the thesis, and unreplicated.**
- Report it only when: the run completes, holds (no epoch-3 over-optimization drift), and ideally replicates.
- Data/where it lives: `outputs/grpo_sftprompt_kl0.1_sys_3ep/` (train.log, `checkpoint-*/eval_*.json`);
  figures `results/grpo_sftprompt_kl0.1_sys_3ep/reward_kl_ba_curve.png`,
  `results/grpo_gemini_vs_evalaligned/`; method docs `docs/grpo_sftprompt_runs.md`,
  `docs/grpo_beta_kl_explainer.md`, `docs/grpo_kl_drift_test.md`.

## Thesis text to fix to propagate the Arm-C headline (from the multi-agent review, 2026-06-19)
Line refs are vs the local working copy `/tmp/thesis_review/thesis_tud`; **the live Overleaf is
the source of truth — read each section before editing** (it can differ from the local copy/zips).
- **P0** `06_results.tex:91,513` + `tab:sft-summary`: retract "SFT+GRPO = production/thesis headline";
  make Arm-C SFT the single headline; forward-point to the teacher-ablation section.
- **P0** `06_results.tex:513` vs `407` & `02_background.tex:121`: fix the IAD-R1 "beat published
  numbers" self-contradiction → "the IAD-R1 checkpoint under our common harness".
- **P0** `07_discussion.tex:125` Limitations box: **rewrite** — it names SFT+GRPO as strongest and
  says "VisA trails IAD-R1", both false (best is Arm-C SFT, which *exceeds* IAD-R1 on VisA +0.73).
- **P0** `08_conclusion.tex:51` (F5): resolve the "[Placeholder]" — the Variety probes are done & negative.
- **P1** Results/Discussion roadmaps; Intro contribution-3 + RQ2 (still pre-assume the GRPO-win story).
- Abstract: align verb (GRPO "adds no headroom" vs body "degrades"); trim to ~250-350 words.

## Provenance table (where every headline number comes from)
| claim | value | source path / doc |
|---|---|---|
| base VLM | 69.0 / 53.8 | `outputs/qwen25vl_baseline_eval/` (see `NUMBER_PROVENANCE.md`) |
| **Arm-C SFT (HEADLINE)** | **82.80 / 72.07** | `outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_{dsmvtec,visa}_full_trainprompt.json` |
| SFT+GRPO run-2 (demoted) | 82.73 / 70.39 | `outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_*` |
| weak SFT init | 80.16 / 64.78 | `outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_*` |
| IAD-R1 reference (our harness) | see doc | `outputs/iad_r1_qwen_recanon/` (Arm-C exceeds it on VisA +0.73) |
| verifier corpus-clean (regen 15k) | 97.8% keep | `results/verify_15k_regen/` |
| verifier non-regen union | 95.0% (6K subset 97.6%) | `results/verify_c1_union/FILTER_AUDIT_c1.md` |
| filtered-6K (wash) | 79.60/65.99 best (avg 72.79) vs 80.16/64.78 | `results/sft_filtered6kcc/ba_summary.txt`; written into §`sec:disc-quality` of `07_discussion.tex` |
| held-out 4k curated pool | 3673 good / 563 corr / 704 rewr | `docs/heldout_4k_rollout.md` |
| reward design | acc + format (unweighted, max 3.0) | `docs/grpo_beta_kl_explainer.md`, thesis App E |
| prompt-aligned GRPO (FUTURE WORK) | DS 82.88 / VisA 73.70 (in progress) | `outputs/grpo_sftprompt_kl0.1_sys_3ep/`; `docs/grpo_sftprompt_runs.md` |

See also `NUMBER_PROVENANCE.md`, `CLAIMS_EVIDENCE.md`, `FIX_CHECKLIST.md`, `UNVERIFIED.md`.
