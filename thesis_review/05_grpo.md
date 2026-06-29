# Ch.5 GRPO Reinforcement Learning — render (mirrors Overleaf `2969d8e`)

> **Legend:** **✅ YOUR WORDS** = applied as you dictated. **✍️ MY STYLING** = restyled in your short-sentence voice (you didn't dictate it — double-check).
> Macros resolved (`\qwenvl`→Qwen2.5-VL, `\dsmvtec`→DS-MVTec, `\visa`→VisA, `\realiad`→Real-IAD, tags shown literally). Equations/tables/listings/algorithm-box noted, not re-typeset. **0 spaced en-dashes; ε=0.2.** This now mirrors the pushed `.tex`.

---

## ✅ Intro — YOUR WORDS
Post-SFT RL stage. We take the best SFT checkpoint of Chapter 4 and refine it with Group Relative Policy Optimisation (GRPO, §2-bg-grpo). Roadmap: §5.1 the algorithm as implemented; §5.2 the two reward functions and their unweighted aggregation; §5.3 data preparation; §5.4 the training loop and hyperparameters; §5.5 implementation details and production-run statistics.

## ✅ §5.1 GRPO as Implemented — YOUR WORDS
GRPO replaces PPO's value network with a Monte-Carlo group baseline. For a given prompt q, a group of G completions {o_1,…,o_G} is sampled from the rollout policy π_θold. Then we score each completion with R(o_i; q) and form the group-relative advantage **[Eq 5.1: Â_{i,t} = (R(o_i;q) − μ_G)/σ_G]**, broadcast as a constant over the tokens of o_i. The policy is then updated by the clipped surrogate of [Eq:grpo-obj] **[Eq 5.2: J_GRPO, with the min/clip surrogate and the −β·KL^{k₃} term]**, with token-level importance ratio ρ_{i,t}(θ) = π_θ(o_{i,t}|q,o_{i,<t}) / π_θold(o_{i,t}|q,o_{i,<t}) and the Schulman **k₃** KL term KL_t^{k₃} of [Eq:k3]. It is placed in the loss (weighted by β) rather than a reward bonus.

**Hyperparameter choices.** We use G = 4 rollouts per prompt. This trades a less-noisy baseline against the cost of full Qwen2.5-VL-7B generations at <2048 tokens on one A6000. Clipping ε = **0.2** (the PPO default). KL coefficient β = 0. The k₃ KL of [Eq:k3] is monitored (stays small) but not added to the loss. Reference policy π_ref fixed to the SFT checkpoint. It is never updated.

> **📐 Algorithm box (`alg:grpo`):** GRPO training loop (single iteration). Require: SFT ckpt θ_0, dataset, reward R, G, clip ε, KL coef β, steps T. (1) θ_ref←θ_0; θ←θ_0. (2) for t=1…T: (3) sample prompt q; (4) sample group {o_i} from π_θold; (5) score r_i←R(o_i;q) (four-component, §5.2); (6) compute μ_G, σ_G + advantages via Eq 5.1; (7) compute surrogate J via Eq 5.2; (8) one AdamW step.

## §5.2 Reward Function
### ✅ §5.2 intro — YOUR WORDS
Following the IAD-R1 reward design [li2025iadr1] our trainer is built on, the per-rollout reward is the **unweighted** sum of two reward functions: a *consistency* (format) reward and an *accuracy* reward. This makes it consist of four sub-signals (format, verdict, type, location): **[Eq 5.3: R = R_cons + R_acc, R_cons∈[0,1], R_acc∈[0,2]]**. So the maximum reward is 3.0. No per-component weight tuning: the two functions are summed with equal unit weight, exactly as in the IAD-R1 code we adapt. Each is detailed below.

### ✅ §5.2.1 Consistency (format) reward R_cons — YOUR WORDS
R_cons ∈ {0, 1}. A binary reward on the full match of the XML schema, and there is a difference between the anomalous and normal output schemas. For an anomalous trace the output must contain `<think>`, `<location>`, `<type>`, `<answer>` in order. For a normal trace it must contain `<think>` then `<answer>`. `<location>` and `<type>` are absent and not enforced, as normal samples do not have a location or type. For this binary reward, a full match scores 1 and any deviation from it scores 0. No partial credit is given. This catches the two failure modes seen in mid-training rollouts: free-form outputs with no XML, and normal samples that emit loc/type tags.

### ✅ §5.2.2 Accuracy reward R_acc — YOUR WORDS (+ verified detail)
R_acc ∈ [0, 2]. This reward rewards the final verdict, the type of anomaly, and the location. We parse the `<answer>` tag permissively ({yes, y, true, A}→yes; {no, n, false, B}→no), then compute:
- *Normal samples* (gt = no): R_acc = 1 if the predicted verdict is "no", else 0.
- *Anomalous samples* (gt = yes): R_acc = ½(s_typ + s_loc), plus 1.0 if the predicted verdict is "yes". So a unit reward for the correct verdict, plus the average of a type and a location sub-score. The type/location credit is granted regardless of the verdict.

The *type* sub-score s_typ ∈ [0,1] is the semantic similarity between the predicted and ground-truth `<type>` strings. It uses the Nomic embedding model **[nomic2025moe — cite FIXED: was v1 (2402.01613), now the v2-moe paper 2502.07972]** (`nomic-embed-text-v2-moe`, served on CPU), and maps the cosine similarity to fixed bins: ≥0.90→1.0, ≥0.80→0.9, ≥0.70→0.7, ≥0.55→0.5, ≥0.40→0.2, else 0. These boundaries were calibrated on a hand-built taxonomy of defect-type string pairs grouped by expected similarity tier (exact, synonym, same-category, related, weak, none) — script `Training/test_nomic_embeddings.py`. It is 0 unless both prediction and gold emit a `<type>` tag. The *location* sub-score s_loc ∈ {0,1} is 1 if and only if the predicted `<location>` maps to the same cell of a 3×3 region grid as gold. Else 0. It is 0 unless both emit a `<location>` tag.

### ✍️ §5.2.3 Total reward and run variants — MY STYLING (double-check)
Summed without weighting (Eq 5.3), this gives four effective sub-signals. Maximum contributions: format = 1, verdict = 1, type = 0.5, location = 0.5 (total 3.0). The verdict and the XML each carry unit weight. Type and location are auxiliary, at half weight through the average inside R_acc. All sub-scores are bounded, so the group-normalised advantage of Eq 5.1 stays well-scaled. The headline production run (run 2) uses exactly these two reward functions. The GRPO-on-C and estimator-ablation probes (§6) add a third *reasoning* reward. The Gemini-judge variant adds a Gemini-scored reasoning reward. All are combined in the same unweighted way.

## ✅ §5.3 Data Preparation — YOUR WORDS (+ verified counts)
**Source.** The *complementary* 4K GRPO split (cf. §3.5) is drawn from the same trace dataset as SFT. There is no overlap with the SFT training set. The split covers **23 of the 30 Real-IAD products**. We use these 4,236 prompts, stratified to 50/50 anomalous/normal both overall and within each product (per-product totals are *not* equalised; they range from 8 to 302). As in §3.5, we focus on the top-down C1 camera angle.

**Image input.** The same *zero-shot, single-image* setup as SFT (Ch.4). One test image, no reference image (trainer flag `single_img=1`). The dynamic-resolution budget is capped at `max_pixels=480,000` (about 0.48 MP): an input image is downscaled if needed so its pixel count stays under this cap, which bounds the number of visual tokens the encoder produces. No system message (`use_system_prompt=false`). The user turn is the fixed question reproduced verbatim in Appendix (prompts) (*"…Are there any defects in the query image?"*). The single-image format keeps the per-rollout cost low, one image-token sequence per prompt. That matters when every training step generates G full completions.

*(the "Filtering during loading" alinea was removed per your instruction.)*

## ✅ §5.4 Training Loop and Hyperparameters — your words (why-G4 / why-η) + verified
> **📋 Table 5.1 (`tab:grpo-hp`):** init θ_0 = best 7B-frozen-6K SFT ckpt-564 (= epoch 3); π_ref = same, fixed; **G=4; ε=0.2; β=0** (monitored, not in loss); **η=1e-6**, linear; AdamW (0.9, 0.999); weight decay 0; per-device batch 1 × grad-accum 4 = **effective batch 8** (2 GPUs); max prompt 4096 tok; max completion 512 tok; max pixels 480,000; **T = 1,060 steps (2 epochs over 4,236)**; save every 530; eval post-hoc on DS-MVTec/VisA; bf16; seed 42.

**Why G = 4.** Larger group sizes give a less noisy advantage estimate and are standard in DeepSeekMath (G = 64). But on a 7B VLM with dynamic-resolution image inputs (capped at the ≈0.48 MP `max_pixels` budget of §5.3) and CPU-offloaded ZeRO-3, each rollout costs on the order of 20–30 seconds. So we use G = 4: enough rollouts for a stable group baseline while staying within the per-step memory and wall-clock budget on a single A6000.

**Why η = 1e-6.** GRPO is far more sensitive to learning rate than SFT. At higher rates the policy drifts from the reference rapidly and the reward signal becomes **unstable**. At η = 1e-6 the monitored KL stays small for the entire run while reward keeps climbing.

## ✅ §5.5 Implementation Details and Production-Run Statistics — your words + verified vs run-2 log
**Framework.** Hugging Face's `trl` library's GRPO trainer. We modified it to support the custom four-component reward function and the Nomic embedding-based type similarity, building on the IAD-R1 implementation [li2025iadr1] that our code is based on. *(lines-of-code / file-list dropped per your instruction.)*

**Nomic embedding on CPU.** The Nomic model (`nomic-embed-text-v2-moe`, ≈475M parameters) is loaded in float32, adding approximately **1.9 GB to host RAM** *(corrected from "600 MB" — verified: 475M × 4 bytes ≈ 1.9 GB fp32, all 8 experts resident on CPU; 600 MB matched no plausible state)*. We keep it on CPU to leave VRAM for the rollouts. The controlled vocabulary's defect-type embeddings are pre-computed once at trainer start. Only the predicted type string is embedded on each rollout. The per-step latency overhead is < 50 ms.

**Image lazy-loading.** Images are loaded on the fly via PIL. The first rollout of a given image caches it for the duration of the group, then **frees it from memory**.

**Production-run statistics.** The production run (run 2, headline) processed 2,118 anomaly + 2,118 normal samples (4,236 total) over 2 epochs (1,060 update steps). A checkpoint is saved at the end of each epoch (every 530 steps). *All numbers below ✅ verified against `train.log` / `trainer_state.json`.*
- *Mean combined reward* (the unweighted sum of §5.2, max 3.0): 1.66 (step 10) → 2.22 (step 100) → 2.13 (step 530). ✅ exact logged values.
- *Accuracy reward* (the bundled verdict + type + location signal of §5.2): from 0.875 to 1.25, a +42.9% relative improvement. *(removed the "~73% verdict-correct → ~100%" gloss — over-interpretation: the reward bundles type+location, not a clean verdict %.)*
- *Format consistency reward*: 0.781 → 0.875 (+12.0%). ✅
- *Mean KL divergence*: through epoch 1 (up to `ckpt-530`) bounded in [0.0, 0.117], exceeding 0.095 on only 13 of those 530 steps; it climbs further in epoch 2. ✅ *(qualified — over the full 1,060 steps KL drifts higher, max ~0.197.)*
- *Completion length*: ~153–186-token band, mean ~167. ✅
- *Wall-clock*: ≈11.3 h to `ckpt-530` (end ep1); ≈24.7 h for the full two-epoch run on 2×A6000 + ZeRO-3. ✅

> **📄 Listing 5.1 (`lst:reward-line`):** `R = 2.81 = consistency(1.00) + accuracy(1.81)`; `accuracy 1.81 = verdict(1.0) + (type 0.62 + location 1.0)/2`; `batch mean R: 2.13 std: 0.27 KL: 0.0009`.

**Checkpoint selection.** We evaluate both saved checkpoints (end of epoch 1 and epoch 2) on the DS-MVTec and VisA test sets and pick the highest balanced accuracy. The headline is `checkpoint-530` (epoch 1), best on both: **DS-MVTec 82.73, VisA 70.39**. `checkpoint-1060` (epoch 2) underperforms by 0.70 pp DS-MVTec (0.53 pp VisA). Mild over-optimisation, discussed in §7. *(Only two checkpoints exist, so "report epoch 1 and 2" = report all. NB: 82.73/70.39 is the GRPO run-2 number, NOT the Arm-C SFT 82.80/72.07.)*

## ✅ §5.6 Iterative SFT↔RL: A Self-Distillation Cycle — your words
The single SFT→GRPO pass of §5.4 treats the GRPO policy as the end product. [deng2025openvlthinker] instead *iterate* the two stages. The RL-improved policy generates fresh reasoning traces. These are filtered for correctness and fed into a new SFT round. Then RL is run again. The intuition is self-distillation: once the policy reasons better than the original teacher on a subset of inputs, its own outputs become a higher-quality SFT **dataset** than the teacher's. We implemented one such cycle, referred to as **iter-2**. It tests directly whether the loop that helps general multimodal reasoning also helps the narrow IAD task. We report it as a **controlled experiment**, not the headline recipe, for reasons the results of §6 make clear.

**Procedure** (from the headline GRPO `ckpt-530`): (1) roll out completions on the GRPO prompt set with the iter-2 policy; (2) a rollout-and-filter (rejection-sampling) step keeps only trajectories whose verdict matches gold and whose XML passes the format regex, yielding 4,026 self-traces; (3) reset to the Qwen2.5-VL-7B base and fresh-SFT on these (frozen ViT, η=1e-5, batch 4, 4 epochs; ckpts at ep3 = ckpt-378, ep4 = ckpt-504); (4) run GRPO again from the iter-2 SFT checkpoint (2 epochs, same reward, num_generations=4, save every 265). The full sequence base→SFT₀→GRPO₀→rollout-and-filter→base→SFT₁→GRPO₁ is the IAD instantiation of the OpenVLThinker self-improvement loop [deng2025openvlthinker]. The rejection-sampling filter plays the role of the verifier.

**Two variants.** We ran the cycle twice. *iter-2 v1* used the first rollout-and-filter pass. *iter-2 v2* tightened the filter (stricter format matching and de-duplication of near-identical traces) before the second SFT. Both are reported in §6. The takeaway, previewed: the iterative cycle improves over its own intermediate iter-2 SFT checkpoint, but does **not** surpass the simpler single-stage GRPO of §5.4. We therefore retain single-stage run 2 `ckpt-530` as the strongest *GRPO* checkpoint. Whether any GRPO checkpoint should be the deployed model, as opposed to a well-curated SFT checkpoint, is revisited in §6.

## ✅ §5.7 Gaussian-Advantage GRPO (G²RPO) — your words + verified
We also evaluate the advantage-estimator modification introduced by OpenVLThinker-v2 [hu2026openvlthinkerv2], which we call **G²RPO**. The vanilla GRPO advantage in Eq 5.1 normalises group-relative rewards by their empirical mean and standard deviation. This z-score estimator becomes unstable when the in-group reward distribution is **heavy-tailed** (a few rollouts score far from the rest) or **quantised** (the reward takes only a handful of discrete values). Both are typical of IAD GRPO. With G=4 rollouts and a reward taking only a few discrete values in [0,3] (format, verdict and location are binary, the type sub-signal is binned), it is common for three of the four rollouts to share a value while the fourth is an outlier. The z-score then divides by a tiny σ and the advantages can explode.

**The modification.** G²RPO replaces the z-score with a non-parametric rank-to-quantile transform onto the standard normal **[Eq 5.4: Â_i^{G²RPO} = Φ⁻¹((rank(r_i) − 0.5)/G)]**. The −0.5 midpoint places the smallest reward at quantile 1/(2G) rather than 0, avoiding the ±∞ tails. So the worst rollout always receives Φ⁻¹(1/(2G)) ≈ −1.15 (for G=4), regardless of the absolute reward gap. The best receives Φ⁻¹((2G−1)/(2G)) ≈ +1.15. The middle two are evenly spaced between them. The estimator is therefore scale-invariant (it depends only on relative ranks) and tail-robust (an outlier gets the same advantage as a moderately-better reward, since both are simply "the best in their group").

**Tie handling.** For our discrete-valued reward vector r ∈ [0,3]^G, ties are common and matter. If two of the four rollouts both receive r=2.0, naive rank assignment would give them advantages Φ⁻¹(2/8) and Φ⁻¹(3/8), which are not equal. We instead average the target quantiles of tied rewards. So algebraically-equal rewards receive algebraically-equal advantages. A small but important fix: without it, the gradient would prefer one trivially-indistinguishable rollout over another.

**Implementation.** The flag `--use_g2rpo=true` swaps the advantage in Eq 5.1 for Eq 5.4; the surrogate loss Eq 5.2 and the KL term are unchanged. The estimator is a direct port of OpenVLThinker-v2's per-group rank-to-quantile transform. Everything else is identical to the vanilla GRPO of §5.4 (SFT init, reward functions, KL coefficient β, group size G=4, prompt format, dataset, hyperparameters), so the experiment is a clean A/B on the advantage estimator alone. *(Python file-paths cut per your instruction.)*

**Optional outer whitening.** A secondary flag `--g2rpo_outer_whitening=true` adds a rank-to-quantile pass over the entire batch (across groups) after the per-group transform. This is OpenVLThinker-v2's headline configuration. It is a second whitening that makes the cross-group distribution of advantages also match N(0,1). We tested both variants in pilot runs and saw minimal difference on IAD. So the reported G²RPO numbers in §6 use per-group OT without outer whitening.

**Why we tested it.** The reward of §5.2 is discrete-valued in [0,3], exactly the quantised heavy-tailed regime that motivated G²RPO. The hypothesis: the rank-based estimator would give a cleaner training signal on IAD groups where most rollouts cluster at a few reward values, and that would translate into a better final checkpoint. As we report in §6, the variant improved over the vanilla advantage in some pilots but did not beat the single-stage GRPO headline. The effect size of the advantage estimator is small relative to the reward design and data quality of §5.2.

## ✍️ Takeaway box — MY STYLING (double-check)
> The GRPO stage of this thesis is a deliberately minimal implementation of group-relative policy optimisation: G=4 rollouts, two unweighted reward functions scoring four bounded sub-signals (format, verdict, type, location; max 3.0), the k₃ KL estimator monitored but not penalised (β=0), and two epochs over ~4,236 prompts. The next chapter shows that, despite this minimalism, it adds **+2.57 pp on DS-MVTec and +5.61 pp on VisA** over the already-strong SFT baseline. A more elaborate iterative SFT↔RL cycle (§5.6) or a Gaussian-advantage substitution (§5.7) does not beat it.

---

## Status + still-open decisions
- **Mirrors Overleaf `2969d8e`.** Your words / your direction now: Intro, §5.1 (+ box), §5.2 (intro, .1, .2), §5.3, §5.4, §5.5, §5.6, §5.7. Only **§5.2.3 + the takeaway** remain pure ✍️-restyling to double-check. **ε=0.2; 0 spaced en-dashes; §5.8 removed.**
- ✅ **§5.5 stats all verified vs the run-2 log** (reward 1.66→2.22→2.13, format +12%, completion 153–186/167, wall-clock 11.3/24.7 h, 2 epochs, ckpt-530 best DS 82.73/VisA 70.39 by 0.70/0.53 pp — all exact). **Corrected:** Nomic 600 MB→**1.9 GB**; dropped the over-claimed "73% verdict-correct" gloss; qualified the KL bound to epoch 1.
- ✅ **Verified this pass (code-checked):** Nomic cite fixed v1→v2-moe (`nomic2025moe`, both Ch.5 + Ch.6); type-similarity bins + calibration script added; R_acc asymmetry confirmed (normal max 1, anomalous max 2 — fine, group-normalised); §5.3 = 23/30 products, 50/50 per-product (totals 8–302), C1-only; `max_pixels` glossed.
- 🟠 **Location-vocab mismatch (NEW finding — your call):** §3.2 describes a clean 9-cell 3×3 grid, but the trace-generator actually loaded a *15-value loose* vocab (adds bare left/right/top/bottom + center-left/center-right alongside middle-left/right); ~1,165 traces are off the clean grid. The reward's `s_loc` substring-maps these to cells gracefully (so "center-left"/"middle-left"/"left" all → middle-left cell — no scoring break), but §3.2's "clean 3×3 grid" claim overstates what was enforced. Decide whether to soften §3.2.
- 🔴 **Headline framing** not yet added (takeaway sells +2.57/+5.61 over the *weak* SFT, not your Arm-C headline).
- ⚠️ **§6 counterpart still present:** `06_results.tex` has the matching `sec:res-compound` "[Placeholder — results pending]" + a Compound-experiment **results table** (lines ~267–275). With §5.8 gone, that's an orphaned results section — should be removed too (your call; flagged for the Ch.6 walk).
