# Qwen3-VL-8B-Instruct zero-shot (the base row of the corpus-transfer test)

`Qwen/Qwen3-VL-8B-Instruct` with no fine-tuning, on the same subsets and the same `_trainprompt`
mode as the fine-tuned run in [`../sft_qwen3vl_8b_armC/`](../sft_qwen3vl_8b_armC/).

| Benchmark | BA | tp | tn | fp | fn | n | Eval JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| DS-MVTec | 78.68 | 880 | 380 | 64 | 346 | 1670 | [`eval_dsmvtec_full_trainprompt.json`](eval_dsmvtec_full_trainprompt.json) |
| VisA | 64.45 | 446 | 865 | 79 | 751 | 2141 | [`eval_visa_full_trainprompt.json`](eval_visa_full_trainprompt.json) |

Unlike LLaVA-OneVision, this base model already follows the train prompt well enough to be parsed,
so no fair-prompt variant was needed and the base row is directly comparable to the fine-tuned rows.

For scale, the Qwen2.5-VL-7B base row on the same harness is 69.01 / 53.79
([`../qwen25vl_baseline_eval/`](../qwen25vl_baseline_eval/)). Qwen3-VL-8B starts a long way ahead of
it, which is exactly why the corpus-transfer question is worth asking: a stronger base leaves less
headroom, and the KCR corpus still adds +7.14 DS-MVTec and +12.07 VisA on top of it.
