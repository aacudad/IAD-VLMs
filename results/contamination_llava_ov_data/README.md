# MVTec contamination in the LLaVA-OneVision training data: settled count and evidence

Counted from scratch on 2 Sep 2026, over the actual parquet data, not over a sampling or filter endpoint.

**Verdict: 1,999 is right. 426 is wrong. 426 is what you get when you read the number off a
truncated index that covers only the first 20 % of the config.**

---

## 1. The count

Dataset `lmms-lab/LLaVA-OneVision-Data`, config `vision_flan(filtered)`, split `train`.

| quantity | value |
|---|---|
| parquet shards in the config | **50** (`train-00000-of-00050.parquet` ... `train-00049-of-00050.parquet`) |
| total rows | **186,060** |
| rows whose `id` matches MVTec-AD | **1,999** |
| distinct matching `id` strings | 361 |
| share of the config | 1,999 / 186,060 = **1.07 %** |
| total parquet bytes | 24,750,561,877 (24.75 GB) |

**Exact matching rule.** Python `re.compile("mvtec", re.IGNORECASE).search(row["id"])`, applied to the
`id` column of every one of the 186,060 rows. Substring, case-insensitive, no anchoring. All 1,999
hits start with the literal prefix `vision-flan_MVTecAD+`, and their `data_source` column is
`vision_flan(filtered)` for all 1,999. The patterns `mvtec`, `mvtec-?ad` and `mvtecad` return the
identical 1,999. The pattern `mv_tec` returns 0, so no alternate spelling is being missed.

**How it was counted.** The config is 24.75 GB because the `image` column holds the raw image bytes,
so downloading all 50 shards was not sensible. Instead each shard was opened over HTTP range
requests through `huggingface_hub.HfFileSystem` and only the `id` and `data_source` columns were
materialised (`pyarrow.parquet.ParquetFile.read(columns=["id","data_source"])`). The parquet footer
gives `num_rows` per shard independently of the rows actually read, and the sum of the 50 footer
`num_rows` values equals the 186,060 rows actually materialised. Shard 0 holds 3,722 rows, shard 1
holds 3,722, and so on. Runtime 3 min 47 s.

**Two independent cross-checks on the total.**

1. `https://datasets-server.huggingface.co/size?dataset=lmms-lab/LLaVA-OneVision-Data&config=vision_flan(filtered)`
   returns `num_rows: 186060` and `num_bytes_parquet_files: 24750561877`.
2. Summing the 50 `x-linked-size` HTTP headers gives 24,750,561,877 bytes, the same number to the byte.

So the denominator 186,060 is confirmed by two sources that do not share a code path.

---

## 2. How many are anomaly detection, not incidental mentions

Every one of the 1,999 ids has the shape `vision-flan_MVTecAD+<task>+<index>`. Splitting on `+`:

| task field | rows | distinct ids |
|---|---|---|
| `image_classification` | 1,000 | 308 |
| `anomaly_detection` | **999** | 53 |
| total | 1,999 | 361 |

**999 of the 1,999 rows are anomaly detection instructions.** None of the 1,999 is an incidental
mention. The other 1,000 are not incidental either. They are object-category classification over the
same MVTec-AD industrial images ("bottle, cable, carpet, and more"), so they still expose the
backbone to the exact image distribution DS-MVTec tests on. They just do not ask for a defect verdict.

### Three matching ids, verbatim

```
vision-flan_MVTecAD+anomaly_detection+002
vision-flan_MVTecAD+anomaly_detection+007
vision-flan_MVTecAD+image_classification+186
```

### What those rows actually contain

Pulled the `conversations` column for the 43 matching rows in shard 0.

`vision-flan_MVTecAD+anomaly_detection+002`, human turn:

> `<image>`
> The primary objective of this task is to accurately identify the type and cause of anomalies in the
> object present in the provided image. The image depicts a specific category of object and texture,
> and within this category, there are defect-free images as well as images exhibiting different types
> of defects. Your task is to carefully examine the image and meticulously identify the specific type
> and cause of any deviations from the normal appearance of the object or texture. [...]

Assistant turns observed on `anomaly_detection` rows in shard 0, verbatim:

> `The anomaly is faulty imprint.`
> `The anomaly is squeeze.`
> `The anomaly is poke insulation.`
> `No anomaly.`

`vision-flan_MVTecAD+image_classification+186`, human turn asks to "classify an image based on its
corresponding object category [...] a bottle, cable, carpet, and more", assistant answers
`The object is a capsule.`

This is not a loose keyword hit. `faulty imprint`, `squeeze` and `poke insulation` are literally
MVTec-AD's own per-category defect class names (`faulty_imprint`, `squeeze` for capsule,
`poke_insulation` for cable), and `No anomaly.` is the normal-class answer. The task format is the
same binary-plus-defect-type question DS-MVTec asks.

### The images are genuinely MVTec-AD

Decoded four of the image blobs from shard 0 and looked at them.

- `scratch_contam/mvtec_sample_1.png`, 1000x1000, id `...anomaly_detection+002`, answer
  "The anomaly is faulty imprint." It is the MVTec-AD **capsule** category: a black-and-orange
  capsule on a grey ground, imprint reading "50" plus a logo where it should read "500". MVTec-AD's
  capsule images are 1000x1000, which matches exactly.
- `scratch_contam/mvtec_sample_3.png`, 1024x1024, id `...anomaly_detection+007`, answer
  "No anomaly." It is the MVTec-AD **cable** category: the standard cross-section of a three-wire
  cable, green/yellow, blue and brown conductors in a white sheath. MVTec-AD cable images are
  1024x1024, which matches exactly.

### The duplicate ids are distinct training examples

This is the point that matters for the count. Within shard 0, the 43 matching rows have **43 distinct
image MD5 hashes**, so no image repeats. And the id `vision-flan_MVTecAD+anomaly_detection+002`
appears twice in shard 0 on two different images with two different answers:

```
row 100  img_md5=b678f98d389b  1,163,309 bytes  answer='The anomaly is faulty imprint.'
row 547  img_md5=9dfff80caf03  1,337,156 bytes  answer='The anomaly is poke insulation.'
```

So the `id` field is a **prompt template index, not a sample identifier**. Across the whole config
the most frequent ids repeat 55, 54 and 54 times. Counting distinct ids (361) counts prompt
templates and undercounts training examples by more than 5x. The number of contaminated
training examples is 1,999. That is the number to quote.

---

## 3. VisA

Same rule, `re.compile("visa", re.IGNORECASE)`, over the same 186,060 rows.

**Result: 1 matching row, and it is a false positive. Zero genuine VisA rows.**

The single hit is, verbatim:

```
multiinstruct_wikihow_wikihow_next_step_10353636_apply-for-a-pakistan-visa
```

That is a WikiHow "how to apply for a Pakistan visa" instruction from the multiinstruct source. It
matches on the travel-document sense of the word "visa", not the VisA anomaly detection benchmark.
It is the only false positive the substring rule produces, and it is the reason a naive
case-insensitive count would say 1 rather than 0. Ruling it out by hand leaves **0**.

The pattern `visa[-_]?anomaly` returns 0. The case-sensitive SQL form `"id" LIKE '%VisA%'` also
returns 0, because the false positive is lowercase.

**The VisA column is clean.**

---

## 4. Where 426 came from

Not distinct-vs-rows. **Distinct ids are 361, not 426**, so that hypothesis is ruled out by the data.

The real cause is that the HuggingFace datasets-server `/filter` endpoint answers from a **partial
index**, and the earlier agent read `num_rows_total` off it without reading the `partial` flag next to it.

Evidence, captured live:

```
GET https://datasets-server.huggingface.co/filter
    ?dataset=lmms-lab/LLaVA-OneVision-Data
    &config=vision_flan(filtered)&split=train
    &where="id" LIKE '%'          <-- matches everything
->  num_rows_total : 37220
    partial        : true
```

The endpoint can only see **37,220 rows**, not 186,060. It indexes roughly the first 5 GB of a config
and sets `"partial": true` when it truncates.

Now line that up with my full scan. Cumulative MVTec rows and cumulative bytes, by shard:

| through shard | cumulative rows | cumulative bytes | cumulative MVTec rows |
|---|---|---|---|
| 7 | 29,776 | 4.043 GB | 352 |
| 8 | 33,498 | 4.530 GB | 387 |
| **9** | **37,220** | **5.076 GB** | **426** |
| 10 | 40,941 | 5.540 GB | 470 |
| 49 | 186,060 | 24.751 GB | 1,999 |

Three numbers land on top of each other:

- The endpoint sees 37,220 rows. Shards 0 through 9 hold exactly 37,220 rows.
- Shards 0 through 9 are 5.076 GB, the first shard boundary at or above the 5 GB truncation point.
- Shards 0 through 9 contain **exactly 426** rows matching MVTec.

So 426 is the true MVTec count of the first 10 of 50 shards. The MVTec rows are spread evenly across
the corpus, about 40 per shard, so the endpoint saw 20.0 % of the data and reported 21.3 % of the
MVTec rows. The earlier agent's query was correct. The endpoint's answer was correct for what it
indexed. The mistake was treating a `partial: true` answer as a whole-dataset count.

The same truncation is why the VisA answer from that endpoint (`0`) happened to be right by luck: it
was 0 in the first 10 shards and it is also 0 in all 50.

Practical rule for the defense: if a datasets-server `/filter` or `/search` response carries
`"partial": true`, the count is a floor over an unknown prefix of the data, not a count.

---

## 5. Does "my model beats the base model anyway, so who cares" hold?

**Partly. It holds for the claim you actually need, and it fails for exactly one sentence in Chapter 6.
Fixing that sentence costs you nothing in results.**

Be precise about the mechanism. The contamination sits in the LLaVA-OneVision instruction mixture.
Every model you built on LLaVA-OV-7B-SI inherits it, including the untuned base. It gives a model
prior exposure to MVTec-AD images and to MVTec anomaly-detection questions with MVTec's own defect
labels as answers. So a DS-MVTec score on that backbone is part inspection skill and part recall of
material the backbone already saw. It is not a VisA problem, because there is no VisA material.

**Where your reasoning is right.**

1. **Any LLaVA-vs-LLaVA comparison is safe.** The contamination is a constant that both sides carry.
   LLaVA SFT 87.70 against LLaVA-OV base 75.66 is a within-backbone delta and the contaminated prior
   is on both sides of it. Same for Arm-C 88.45 against SFT+GRPO 87.66 and against the restart 87.86.
2. **That covers the central claim of the replication.** The claim in §6 is that the supervised stage
   is the last and best stage on this backbone too. That is an ordering among four LLaVA-derived
   checkpoints. Contamination does not reorder them. The replication survives intact.
3. **VisA is untouched.** 0 rows. Every VisA number and every VisA comparison, cross-architecture
   included, is clean.
4. **The thesis headline is untouched.** The headline is the Qwen2.5-VL-7B Arm-C checkpoint at
   82.80 / 72.07. Qwen2.5-VL was not trained on `vision_flan(filtered)`, so this finding says nothing
   about it. Chapter 6 already says so at line 301.

**Where your reasoning fails.**

The one comparison this bites is **a LLaVA-derived model against a model whose backbone never saw
this mixture, read on DS-MVTec**. Chapter 6 makes that comparison. Line 301 states that IAD-R1's
released checkpoint is a Qwen2.5-VL-7B model. So "our LLaVA model reaches 88.45 on DS-MVTec, above
IAD-R1's 81.92" puts a contaminated backbone against a clean one on the contaminated benchmark. That
gap is not safe to read as a capability gap. The same applies to the DS-MVTec column against the
Gemini-2.5-Flash and GPT-5-mini zero-shot rows, which are also not LLaVA-derived.

Note this cuts against you and for you at once. The DS-MVTec half of that IAD-R1 comparison is
weakened. The VisA half of it, 74.25 against 71.34, is clean and still stands. So you do not lose the
"we beat IAD-R1 on its own kind of task" claim. You lose the right to lean on the DS-MVTec column
while making it.

**What to say if a committee member pushes.** Do not defend the DS-MVTec cross-architecture gap.
Concede it in one sentence and redirect: the contamination is a property of the backbone, it is
present in the baseline as well as in every model trained from it, so it cannot manufacture the
ordering the chapter claims, and the cross-architecture claim rests on VisA, which is clean. That is
a stronger position than arguing the contamination is small. It is not small in kind. 999 anomaly
detection instructions on MVTec images with MVTec's own defect vocabulary is precisely the task.

**One honesty limit, worth volunteering before you are asked.** I proved MVTec-AD images and MVTec
anomaly-detection instructions are in the mixture. I did not prove that the specific DS-MVTec test
images appear in it, which would need decoding all 24.75 GB and hashing against the MMAD subset.
"Possible train-on-test contamination", the phrasing already in the thesis, is the correct strength.
Do not upgrade it to "confirmed test-set leakage".

**Two things the current text overstates or leaves loose.** The count is a lower bound for a reason
worth stating: I matched on identifiers only, and I searched one config of 89. I checked a second
vision-flan-derived config, `allava_instruct_vflan4v` (19,990 rows, 6 shards), and it returns 0 for
both patterns, but its ids are opaque (`allava_vflan_inst_100010`), so an identifier match there
cannot detect MVTec even if it is present. Identifier matching only works where the id carries the
source name. The true whole-mixture figure is unknown and can only be larger.

---

## 6. Exact sentence for the thesis

Replacing the `\paragraph{Contamination caveat.}` block at `chapters/06_results.tex:303`. Three
changes: "roughly half" becomes the exact 999, "the canonical parquet export" becomes the explicit
all-50-shards scope, and "returns zero rows for VisA" becomes "returns no VisA rows" because a
case-insensitive match does return one row and it is a WikiHow item about travel visas.

**Old** (`chapters/06_results.tex:303`):

> \paragraph{Contamination caveat.} The public LLaVA-OneVision instruction mixture contains MVTec-AD material, so every \dsmvtec\ number on this backbone carries a possible train-on-test contamination. Counted over the canonical parquet export of \texttt{lmms-lab/LLaVA-OneVision-Data}, the \texttt{vision\_flan(filtered)} configuration alone holds 1{,}999 of 186{,}060 rows whose sample identifier matches \texttt{MVTecAD}, and roughly half of those are anomaly-detection instructions. The same query returns zero rows for \visa. We searched one configuration, so the count is a lower bound. The \visa\ column is the one this comparison leans on, and it is clean.

**New**:

> \paragraph{Contamination caveat.} The public LLaVA-OneVision instruction mixture contains MVTec-AD material, so every \dsmvtec\ number on this backbone carries a possible train-on-test contamination. We counted it over all 50 parquet shards of \texttt{lmms-lab/LLaVA-OneVision-Data} rather than over a sampling endpoint. The \texttt{vision\_flan(filtered)} configuration alone holds 1{,}999 of its 186{,}060 rows whose sample identifier contains \texttt{MVTecAD} under a case-insensitive match. Of those, 999 are anomaly-detection instructions whose answers are MVTec-AD's own defect class names or ``No anomaly.'', and the remaining 1{,}000 are object-category questions over the same industrial images. The same match returns no \visa\ rows. We searched one configuration and matched on identifiers only, so the count is a lower bound. The contamination is carried by the backbone, so it is present in the LLaVA-OneVision baseline as well as in every model we train from it, and it therefore cannot account for the ordering among those models. It does weaken any \dsmvtec\ comparison between a LLaVA-derived model and a model built on a different backbone. The \visa\ column is the one this comparison leans on, and it is clean.

If a shorter version is wanted, the minimum correct edit is to change
`roughly half of those are anomaly-detection instructions` to
`999 of those are anomaly-detection instructions` and
`returns zero rows for \visa` to `returns no \visa\ rows`.

**Also check.** `chapters/08_conclusion.tex:27` and `chapters/01_introduction.tex:131` both state the
caveat without a number. Both are fine as written and need no change.

---

## Files and URLs used

### Scripts written (all under `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/`)
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/probe_schema.py` — schema and id probe of shard 0
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/count_shards.py` — reads the `id` and `data_source` columns of all 50 shards over HTTP range requests
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/analyze.py` — MVTec and VisA matching
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/breakdown.py` — task-type split
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/cumul.py` — cumulative MVTec rows per shard, the 426 reconstruction
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/convs.py` — conversation text for shard-0 matches
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/imgcheck.py` — image MD5s and blob extraction
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/vflan4v.py` — second-config check

### Data produced
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/all_ids.jsonl` — all 186,060 ids with `data_source`
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/totals.json` — per-shard row counts and totals
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/mvtec_ids.json` — the 1,999 matching ids
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/shard0_mvtec_convs.json` — full conversations for the 43 shard-0 matches
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/shard_sizes.json` — the 50 shard byte sizes
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/mvtec_sample_{0,1,2,3}.png` — decoded MVTec images
- `/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/small_{0,1,2,3}.png` — downscaled for viewing

### Thesis files read (not modified)
- `/bulk/aacudad/ol_thesis/chapters/06_results.tex` lines 301, 303
- `/bulk/aacudad/ol_thesis/chapters/08_conclusion.tex` line 27
- `/bulk/aacudad/ol_thesis/chapters/01_introduction.tex` line 131
- `/bulk/aacudad/ol_thesis/chapters/07_discussion.tex` line 137 (limitation L5, the separate Gemini-teacher leakage caveat)

### URLs
- `https://huggingface.co/api/datasets/lmms-lab/LLaVA-OneVision-Data` — file listing, 650 files, 89 configs, 50 `vision_flan(filtered)` shards
- `https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/resolve/main/vision_flan(filtered)/train-000NN-of-00050.parquet` — the 50 shards, read by range request, and `x-linked-size` headers
- `https://datasets-server.huggingface.co/size?dataset=lmms-lab/LLaVA-OneVision-Data&config=vision_flan(filtered)` — independent confirmation of 186,060 rows and 24,750,561,877 bytes
- `https://datasets-server.huggingface.co/filter?dataset=lmms-lab/LLaVA-OneVision-Data&config=vision_flan(filtered)&split=train&where="id" LIKE '%'` — returned `num_rows_total: 37220`, `partial: true`, the source of the 426 error
- `https://datasets-server.huggingface.co/filter?...&where="id" LIKE '%VisA%'` — returned `num_rows_total: 0`, `partial: true`

### Environment
- `/bulk/aacudad/reasoning_traces/envs/grpo_fast312/bin/python3` (pyarrow 25.0.1, huggingface_hub 0.36.2). CPU only, no GPU touched.

### Not used
No local copy of this dataset exists. `/bulk/aacudad/reasoning_traces/hf_cache/` holds only model
repos (`llava-onevision-qwen2-7b-si-hf`, `Qwen2.5-VL-7B-Instruct`, `Qwen3-VL-8B-Instruct`, the three
`aacudad/AnomalyThink-*` repos, two nomic embedding models) and a `datasets/json` cache. Nothing
matching `LLaVA-OneVision-Data` or `vision_flan` is on disk anywhere under `/bulk/aacudad`.
