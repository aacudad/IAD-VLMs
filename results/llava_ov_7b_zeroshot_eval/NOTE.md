# LLaVA-OneVision-7B-SI zero-shot (the base row of the LLaVA line)

`lmms-lab/llava-onevision-qwen2-7b-si` with no fine-tuning, on the same DS-MVTec and VisA subsets
as every other row in this repo.

| Benchmark | BA | tp | tn | fp | fn | n | Eval JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| DS-MVTec | 75.66 | 632 | 443 | 1 | 594 | 1670 | [`eval_dsmvtec_full_yesnouser.json`](eval_dsmvtec_full_yesnouser.json) |
| VisA | 53.80 | 91 | 944 | 0 | 1106 | 2141 | [`eval_visa_full_yesnouser.json`](eval_visa_full_yesnouser.json) |

## Why the `_yesnouser` prompt mode and not `_trainprompt`

The base model does not emit the `<think>` / `<answer>` schema that the train prompt asks for. Under
`_trainprompt` the strict parser finds no verdict at all, scores every sample as "no", and returns
tp=0, fp=0, BA exactly 50.00 on both benchmarks. That is a parser artefact, not a measurement of the
model.

`--yesno-user` appends a plain "Answer with yes or no." to the user turn. It gives the base model a
question it can actually answer, and it is the fair base row. The harness flag lives in
[`scripts/04_eval/`](../../scripts/04_eval/) and the mode is documented in
[`docs/prompt_modes.md`](../../docs/prompt_modes.md).

The two degenerate `_trainprompt` JSONs (3.1 MB of all-"no" predictions) are left in the source tree
and are not shipped here. They carry no information this paragraph does not.

## What the numbers say

The base model is extremely conservative. It fires almost no false positives (1 on DS-MVTec, 0 on
VisA) and misses nearly half of the DS-MVTec anomalies and 92 percent of the VisA anomalies. VisA at
53.80 is barely above chance. Everything the fine-tuned LLaVA models gain is recall.

## Contamination caveat, DS-MVTec only

The LLaVA-OneVision training mixture contains 426 rows whose id matches `%MVTecAD%` (VisA: 0). The
75.66 already includes whatever that exposure is worth. The 53.80 VisA number does not.
