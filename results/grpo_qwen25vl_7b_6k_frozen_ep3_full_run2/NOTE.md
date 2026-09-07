
## 2026-09-07: prompt of the checkpoint-530 files

`checkpoint-530/eval_{dsmvtec,visa}_full_trainprompt.json` (82.73 / 70.39, the thesis row) were produced with `--grpo-eval`
(GRPO training prompt, no system message) despite the file name; `eval_run.log` lines 16 and 2082 record `GRPO eval prompt:True`.
The supervised train-prompt evaluations are `eval_dsmvtec_full_trainprompt_default.json` (80.74) and
`eval_visa_full_trainprompt_default.json` (70.50, run 7 Sep 2026). The thesis states both (6.1, Table 6.14 note, Appendix J).
