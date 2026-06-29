# Ch.3 Reasoning Trace Generation — readable render (FULL prose, 1:1 with Overleaf)
*(Source = Overleaf `chapters/03_dataset_traces.tex`, commit `6c042c6`. Verbatim prose, macros resolved. **✅ = walked + applied in your voice. ⏳ = still original prose, not yet walked.** 24 spaced en-dashes `--` remain (all in ⏳ sections); numeric ranges 120--200, C1--C5, 4--5 mm stay.)*

---

## ✅ Intro ¶ — DONE
This chapter describes how the supervised training data of this thesis is produced. The dataset, which we name **AnomalyThink**, consists of **14,472** structured chain-of-thought (CoT) reasoning traces generated over the Real-IAD dataset using Gemini 2.5-Flash. It is divided into three disjoint, product- and class-stratified splits: a **6K SFT split** (**AnomalyThink-6K**), a **4K GRPO split**, and a **4K held-out split**. We also experiment with the full union (≈14.5K), which we call **AnomalyThink-15K**, as a data-quantity ablation in Chapter 6. [§-by-§ roadmap follows.]

## ✅ §3.1 Source Dataset — DONE (your voice + de-dashed + code table)
All **reasoning** traces are generated from Real-IAD [wang2024realiad] (§2.6). For trace generation we use two of its properties directly: the *five camera viewpoints* per physical part (cameras `C1`–`C5`) and the pixel-precise binary defect masks on the anomalous samples. Each sample is grouped under a directory `Product/{OK,NG}/{defect_code}/S####/{image}.jpg`, where `OK` marks defect-free items, `NG` (*No-Good*) marks defective items, the `defect_code` is one of eight two-letter Real-IAD anomaly codes mapped to a readable defect type (**Table 3.1**), and `S####` is the sample identifier. Camera angle is encoded in the filename (`_C1_`, …, `_C5_`).

> **📋 Table 3.1 — Real-IAD anomaly codes** (NEW, authoritative from `utils_realiad.py`): AK=Pit, BX=Deformation, CH=Abrasion, HS=Scratch, PS=Damage, QS=Missing Parts, YW=Foreign Objects, ZW=Contamination (OK=Normal).

For trace generation we use all 30 Real-IAD products and restrict to anomalous samples that have a ground-truth defect mask (NG images without a mask are skipped). A pre-computed mask-analysis pipeline additionally labels each retained mask as `Single`, `Multi-Close` or `Spread`: **`Single` is** one localised region, **`Multi-Close`** several nearby regions, and **`Spread`** scattered non-contiguous regions (Fig. mask-spread). We use this taxonomy only *descriptively* and do *not* filter, weight or stratify the **dataset** on it (inclusion depends solely on mask presence). For normal samples no mask filter is applied. We sample uniformly across the available `OK` directories.

> **[Figures]** Real-IAD audiojack triplet (normal / anomalous **Deformation/BX** [caption fixed] / red overlay); mask-spread categories (Single/Multi-Close/Spread). Both image files present.

## ✅ §3.2 Output Format: Structured Six-Phase Trace — DONE (your voice + de-dashed + location table)
Every **generated and predicted trace sticks to** a fixed XML output format, which is machine-checkable and parseable with simple regular expressions. **It** lends itself to per-component reward design (§5). For anomalous samples the trace has four tags:
```
<think>[120-200 word inspection rationale following the six-phase template]</think>
<location>top-right and center</location>
<type>Scratch</type>
<answer>Yes</answer>
```
For normal samples the trace ends at `<answer>` with no `<location>` or `<type>`:
```
<think>[justification why the inspected image is defect-free]</think>
<answer>No</answer>
```

**Acknowledgement of shared schema with IAD-R1.** This four-tag schema is identical to the one introduced by IAD-R1 [li2025iadr1] for SC-GRPO. Specifically, the four-component reward decomposition R_acc, R_loc, R_type, R_con (§5) maps one-to-one onto the four tags, which we adopt unchanged so that our results on DS-MVTec and VisA are directly comparable to the IAD-R1 published numbers (Table sota). Our contributions *over* IAD-R1 are (i) the *trace content* (the six-phase template); (ii) the *source data*, the openly-released **AnomalyThink dataset** over Real-IAD (§3.7) rather than the closed Expert-AD corpus; and (iii) the GRPO configuration of Ch.5 (group size, KL coefficient, reward composition, learning rate). The surface XML format is deliberately shared so that downstream comparisons isolate these three factors as the locus of the contribution.

The contents of `<think>` follow the **six-phase template**:
1. *Framing.* Name the product and viewing angle, recall the dominant nominal features (e.g. "threaded cylinder, four notches, central hole").
2. *Scan.* Sweep the image surface-by-surface, noting any region of interest without yet asserting a defect.
3. *Focus.* Zoom in on the most suspicious region and describe the visual cue ("sharp specular highlight cutting across the texture").
4. *Evaluate.* Compare the cue to a plausible defect category: deeper than the surrounding micro-texture? Aligned with a known failure mode?
5. *Alternatives.* Briefly rule out competing explanations, **such as** specular glare, dust, or a feature that is supposed to be there.
6. *Decide.* Conclude: defect / no defect. If defect, name the type and location using the controlled vocabulary.

The total budget is **120–200 words**. This explicit target replaces an earlier, looser "3–5 sentences" instruction used in pilot generation. It is what produces the tightly-clustered trace lengths reported in §3.6 (mean ≈137 words).

**Location vocabulary.** A short list of canonical positions is enforced to make `<location>` easy to score with a 3×3 grid-style reward (§5). The nine allowed values map directly onto the grid cells (**Table 3.2**); comma-separated compounds such as `top-right, center` denote defects spanning multiple cells.

> **📋 Table 3.2 — Location vocabulary** (NEW, as a 3×3 grid): top-left / top-center / top-right · middle-left / center / middle-right · bottom-left / bottom-center / bottom-right.

Free-form descriptions ("next to the lower screw") and bare single-axis terms (`left`, `top`) are disallowed by the prompt; restricting to these nine cells is what makes `<location>` easy to score with the 3×3 grid reward.

**Defect-type vocabulary.** `<type>` draws from a controlled list of **26** defect categories in four groups: Surface (Scratch, Crack, Dent, Pit, Abrasion, Discoloration, Contamination, Corrosion, Stain, Burr, Flash); Structural (Broken, Fracture, Deformation, Warping, Chip, Damage, Bend); Completeness (Missing Parts, Missing Component, Extra Component, Incomplete Assembly); Other (Misalignment, Foreign Object, Poor Finish, Overmolding). Broader than Real-IAD's two-letter codes; the translation to a generic vocabulary enables zero-shot transfer to MVTec/VisA categories.

## ✅ §3.3 Generation Pipeline — DONE (Gemini 2.5-Flash; verdict+type disclosed; de-dashed) — `c3a30be`
[Figure: trace-gen pipeline TikZ — Real-IAD image+mask → build prompt → **Gemini 2.5-Flash** generate+self-check → structured trace; batch 5–10.] **Inputs:** batch ≤10; for each anomalous sample three images (original, red-mask overlay, reference normal from same product and, *when available*, same camera); normal samples = original only. **Privileged inputs = the overlay (marks *where*) PLUS two internal text hints: the ground-truth verdict and the gold defect type** — so the teacher explains an already-known defect rather than detecting it; all discarded at inference (student sees only the test image). **System prompt** (App. B): role, six-phase template, two vocabularies, JSON schema, six auto-reject rules, and the self-scoring rubric. **Auto-reject rules:** (1) no batch leakage, (2) no metadata leakage, (3) evidence before conclusion, (4) output-structure correctness, (5) causal restraint, (6) no unsupported defect claim. **Generation engine:** Gemini 2.5-Flash [geminiteam2025gemini25] — native multi-image + low batched cost. *(Removed the thinking-mode reason + the $0.0001/11s cost figures as unverifiable.)*
> 💡 Figure-TODO: enhance the pipeline diagram with real input/output sample thumbnails.
> ✅ Model kept as Gemini 2.5-Flash throughout for consistency (your call); the 2.5 `\citep` is restored.

## ✅ §3.4 Generation-Time Quality Control — DONE (de-dashed; "no second model" verified) — commit `6b137c0`
QC is applied *during* generation, inside the single Gemini call: **no second model, no post-hoc score-ranked selection** (verified — the rubric is in-prompt Gemini self-scoring; gpt-5-mini appears only in unused 20-sample test harnesses, never the shipped corpus). **In-prompt acceptance filter:** eight-category rubric (0/1/2, ≥13/16, rewrite-on-fail), embedded in `inspector_prompt_test_v2.txt` (App. B §348–451). **Parse-level validation** (Pydantic JSON). Applied uniformly, *not* rank-and-select → AnomalyThink-6K is a stratified partition, not a top-scored subset (6K-beats-15K = data-count effect, not differential filtering).

## ✅ §3.5 Dataset Assembly — DONE (your voice; "held-out split", de-dashed) — `1dbc072`
Three disjoint splits, stratified by product, balanced by class: a 6K SFT split (AnomalyThink-6K), a 4K GRPO split (Ch.5), and a **4K held-out split**. Three invariants: class balance (50/50), per-product stratification, zero image overlap. AnomalyThink-15K = union (≈14.5K), used as the data-quantity ablation; *kept* the integrity note that the 15K ablation is evaluated only on DS-MVTec/VisA, so pooling the held-out split in contaminates no held-out number. **Camera-angle filtering:** all splits restricted to top-down `_C1_` (future work: other 4 views). **File format:** ShareGPT JSON (SFT) + custom GRPO format (App. A).

## ✅ §3.6 Corpus Statistics — DONE (reference datasheet for counts; drop unverifiable cost; 165 tokens) — `4023776`
Per-split composition (30 products SFT / 23 GRPO / 23 held-out, all ≈50/50; total 14,472) now **referenced to the datasheet, Appendix~I (`app:datasheet`)** instead of restated (per your "isn't this repetitive?"). Kept: **mean ≈165 tokens, range 121–223**, within the 120–200-word target (using tokens only, per your call — no "137 words"). **Dropped** the unverifiable ≈1.6 s/trace + <\$20 cost figures (no constant in code). t-SNE figure + the two trace examples retained (see t-SNE discussion above — candidate to cut/reframe).

## ✅ §3.7 Positioning vs Existing Reasoning-Trace **Datasets** — DONE (retitled; condensed; de-dashed) — `4023776`
[Table: AnomalyThink vs Anomaly-Instruct-125k vs Expert-AD.] **Vs Anomaly-Instruct-125k:** open + larger (125K) but MVTec/VisA-sourced (our eval sets) + multi-turn QA (no parseable tags); complementary, future work. **Vs Expert-AD:** the direct comparable (SFT+GRPO structured CoT) but **not publicly released**. AnomalyThink = first openly-released reasoning-trace **dataset** built on Real-IAD + purpose-designed for structured GRPO; prompt/QC/splits all disclosed, released with the thesis. [takeaway box, de-dashed.]

---

## Where we are
- ✅ **Done (in your voice, de-dashed, pushed `6c042c6`):** Intro, §3.1, §3.2 — incl. two new tables (anomaly codes, location 3×3 grid) and the BX caption fix.
- ⏳ **Still to walk:** §3.3 (pipeline), §3.4 (QC), §3.5 (assembly), §3.6 (stats), §3.7 (positioning) — original prose, **24 en-dashes** remain here.
- **Consistency flags for those sections:** "held-out **test** split" (§3.5/§3.6) vs your "held-out split"; "**165 tokens**" (§3.6) vs "137 words" (§3.2); "≈14,500" (§3.6) vs exact "14,472" (intro); §3.7 title "**Corpora**" vs "datasets".
- **Figure-TODO:** enhance the §3.3 pipeline diagram with real input/output sample thumbnails (your idea).
