# STATUS / HANDOFF — 2026-06-20 (evening)

Snapshot for context-switching away and coming back later. What is running, the intermediate
GRPO numbers, the Qwen3-VL plan, and what still needs to land in the thesis.

---

## 1. What is RUNNING / SCHEDULED right now

| Job | GPU | State | ETA / trigger |
|---|---|---|---|
| **GRPO 3-epoch, eval-aligned `sft_sys`, β=0.1** (FUTURE WORK) | cuda 1,2 | running, step **1092/1590 (~69%)** | ~**12h25m left → ~08:45 Sun 21** |
| GRPO eval watcher (DS→VisA per ckpt) | cuda 3 | running | finishes ~1h after training |
| **Qwen3-VL-8B Arm-C SFT scheduler** (`schedule_qwen3vl_after_grpo.sh`, PID 2968744) | — | armed, polling 5-min | fires when GRPO frees **cuda 1,2** → 2-step smoke → if pass → 4-epoch SFT on cuda 1,2 + eval watcher on cuda 3 |
| Qwen3-VL-8B **baseline** eval | cuda 0 | ✅ DONE | — |

cuda 0 is free. All schedulers/watchers are detached (`nohup`), survive logout.

---

## 2. GRPO 3-epoch intermediate results  (`outputs/grpo_sftprompt_kl0.1_sys_3ep/`)

**This is the prompt-aligned GRPO on the strong Arm-C init — FUTURE WORK, NOT a thesis result**
(see `THESIS_HEADLINE_DECISION.md`). Tests whether aligning the RL training prompt to the eval
prompt (`sft_sys` = system "Please answer by yes or no" + Analyze) lets GRPO beat the Arm-C SFT
ceiling **82.80 / 72.07 (avg 77.44)**. Reward = accuracy + consistency (unweighted, max 3.0).

Per-checkpoint balanced accuracy (default eval prompt, same harness as the 82.80 baseline):

| ckpt | DS-MVTec | VisA | avg | note |
|---|---|---|---|---|
| init (Arm-C 376) | 82.80 | 72.07 | 77.44 | the ceiling to beat |
| 106 | 80.18 | 69.51 | 74.84 | early dip |
| 212 | 81.52 | 70.22 | 75.87 | |
| 318 | 81.83 | 72.25 | 77.04 | |
| 424 | 81.49 | 73.01 | 77.25 | |
| 530 (≈ep1) | **82.88** | 71.34 | 77.11 | DS edges ceiling |
| 636 | 81.41 | **73.70** | 77.56 | best VisA |
| 742 | 82.35 | 72.17 | 77.26 | |
| 848 | 80.75 | 72.73 | 76.74 | |
| 954 | **82.95** | 72.62 | **77.78** | **best avg so far** |
| 1060 | 81.59 | (pending) | — | |

**Read:** the eval-aligned run *edges* the ceiling on avg (ckpt-954 **77.78 > 77.44**), with the best
DS (82.95) and best VisA (73.70) each slightly above the Arm-C init — but it bounces (no monotone
climb), so the gain is small and not yet clearly stable across epoch 3. Verdict still: **report only
as future work once the run completes + ideally replicates.** Do NOT write "GRPO beat Arm-C" as a result.

**KL (Schulman k3 vs frozen Arm-C init, β=0.1)** — 1093 steps logged:
- **median 0.026**, steady-state recent ~0.03–0.06 → β=0.1 holds the policy very close to the init (as designed).
- 133 steps > 0.2; **9 steps > 1.0**, all in one unstable patch around **epoch 0.30–0.35** (grad_norm up to 380),
  with a single **max spike kl = 23.55** there. The run recovered (back to ~0.03). So: one early
  transient instability, otherwise a tightly-leashed policy.

**Reward:** rises from init toward **~2.16–2.58** (max observed **2.578** / 3.0); accuracy_reward ~1.3–1.4,
consistency_reward steady ~0.875. Reward IS optimised — but, as expected, only modest eval gain (the
classic small-KL / near-data-ceiling regime; cf. `docs/grpo_beta_kl_explainer.md`).

Figures/docs: `results/grpo_sftprompt_kl0.1_sys_3ep/`, `docs/grpo_sftprompt_runs.md`.

---

## 3. Qwen3-VL-8B-Instruct (backbone-independence / Limitation L4)

- **Baseline (no fine-tuning), our harness, default prompt:** **DS 78.68 / VisA 64.45**
  (`outputs/qwen3vl_8b_baseline_eval/`). vs Qwen2.5-VL-7B base 69.01/53.80 → a *much* stronger start.
- **Arm-C SFT scheduled overnight** (`Training/sft_qwen3vl_8b_armC.yaml`): identical to the 7B Arm-C
  recipe (frozen ViT, full FT of LLM+projector, ZeRO-3 CPU offload, `save_only_model`, 4 epochs,
  Arm-C 6K `iad_sft_iter2`) except model=`Qwen/Qwen3-VL-8B-Instruct`, `template: qwen3_vl`, batch 4×4
  (eff. 32), HF caches → `/bulk/.../hf_cache`. Runs on **cuda 1,2** after GRPO; eval watcher on cuda 3.
- **Expectation:** likely new thesis-best, point estimate **~84 / ~74**; VisA is the honest signal
  (more headroom, less contamination-prone), DS gain carries the MVTec-familiarity caveat. Worst case
  ~7B-Arm-C numbers (data ceiling) — still retires L4.
- Files: `Training/{sft_qwen3vl_8b_armC.yaml, run_qwen3vl_8b_armC.sh, schedule_qwen3vl_after_grpo.sh,
  watch_and_eval_qwen3vl_cuda3.sh, run_qwen3vl_baseline_eval.sh}`.

---

## 4. Thesis — what landed this session, what STILL needs adding

**Done this session (pushed to Overleaf earlier; the MCP later disconnected — see note):**
- Added **GPT-5-mini** zero-shot row + paragraph to `tab:sota` (DS 77.10 / VisA 68.23). Gemini-2.5 (81.52/75.18) already there.
- Fixed the **Limitations takeaway box** (was wrongly "SFT+GRPO best / VisA trails IAD-R1" → now Arm-C SFT beats IAD-R1 on both).
- **Condensed** §res-variety-cont to one honest negative paragraph; **reframed F5** to future work; fixed "four→five".
- **Open release** wired: code → `github.com/aacudad/IAD-VLMs`, dataset → `huggingface.co/datasets/aacudad/anomalythink` (HF upload pending token).

**STILL TO ADD / decide (the "forgot/pending" list):**
1. **Qwen3-VL-8B results** (baseline 78.68/64.45 + Arm-C SFT, pending overnight) — NOT in thesis yet.
   When the SFT finishes: add as the **backbone-independence** result that retires **L4**; decide headline vs L4-discussion placement.
2. **Prompt-aligned GRPO 3-epoch** (§2 above) — stays **future work**; update `docs/grpo_sftprompt_runs.md`
   with final per-ckpt BA + the KL story when it completes. Do not add as a body result.
3. **Gemini-3-Flash-preview (93.20/80.29)** — deliberately **repo-only** (contamination); see
   `docs/contamination_gemini3.md`. Optional one-line cautionary mention in L5; currently held out.
4. **HuggingFace dataset upload** — needs HF MCP OAuth (still "needs authentication") + the write token;
   then push the configs (15K distilled / rollout+corrected+rewritten / Variety STaR-6K / GPT-5-mini-verified / Arm-C-6K) and confirm the URL in the conclusion (`% TODO[release]`).

**⚠️ Overleaf MCP note:** the Overleaf MCP **disconnected this session** (needs a Claude Code reload to
restore). I could NOT re-read the live thesis at handoff time. The "done this session" list above reflects
my earlier live reads + the `write_section` pushes that returned "pushed successfully" this session.
On return: reload Claude Code → the `overleaf` MCP reconnects → re-read to confirm everything rendered.

---

## Pointers
- `THESIS_HEADLINE_DECISION.md` — Arm-C SFT (82.80/72.07) is the single headline; GRPO secondary/future-work.
- `docs/proprietary-zeroshot-baselines` (memory) + `docs/contamination_gemini3.md` — the zero-shot reference rows.
- `docs/grpo_sftprompt_runs.md`, `docs/grpo_beta_kl_explainer.md` — the GRPO-prompt experiments + KL theory.
