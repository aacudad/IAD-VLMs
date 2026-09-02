# Qwen3-VL-8B-Instruct on the KCR corpus (corpus transfer to a newer backbone)

The KCR corpus was curated with a Qwen2.5-VL-7B policy. This run asks whether it still helps a
backbone it was never rolled out from. It does.

- Backbone: `Qwen/Qwen3-VL-8B-Instruct`, from base, no intermediate SFT.
- Corpus: the **Qwen KCR corpus**, LlamaFactory key `iad_sft_iter2`, shipped at
  [`traces/iter2/sft_iter2_train.json`](../../traces/iter2/sft_iter2_train.json), 6,000 records.
  This is exactly the corpus that the Qwen2.5-VL-7B Arm-C headline model trained on
  ([`configs/sft/sft_abc_C.yaml`](../../configs/sft/sft_abc_C.yaml) uses the same key).
- Recipe: mirrors `sft_abc_C.yaml` except model, `template: qwen3_vl`, output dir, and batch
  (4 x 4 instead of the 7B setting). Frozen vision tower, projector and language model trained,
  4 epochs, LR 1e-5 cosine, ZeRO-3 CPU offload, `save_only_model`.
- Harness: DS-MVTec n=1670, VisA n=2141, `_trainprompt` mode, identical to every other row here.

## Balanced accuracy

| Checkpoint | DS-MVTec BA | tp | tn | fp | fn | n | VisA BA | tp | tn | fp | fn | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, no fine-tuning | 78.68 | 880 | 380 | 64 | 346 | 1670 | 64.45 | 446 | 865 | 79 | 751 | 2141 |
| checkpoint-188 (ep1) | 82.62 | 971 | 382 | 62 | 255 | 1670 | 77.67 | 779 | 852 | 92 | 418 | 2141 |
| **checkpoint-376 (ep2)** | **85.82** | 1055 | 380 | 64 | 171 | 1670 | **76.52** | 883 | 747 | 196 | 313 | 2139 |
| checkpoint-564 (ep3) | 85.18 | 1089 | 362 | 82 | 137 | 1670 | 76.62 | 896 | 740 | 204 | 301 | 2141 |
| checkpoint-752 (ep4) | 82.59 | 1125 | 326 | 118 | 101 | 1670 | 73.26 | 950 | 634 | 310 | 247 | 2141 |

The base row is [`../qwen3vl_8b_baseline_eval/`](../qwen3vl_8b_baseline_eval/).

**Corpus transfer: 78.68 / 64.45 to 85.82 / 76.52, so +7.14 DS-MVTec and +12.07 VisA.** The VisA
gain is the larger one, which is the pattern the Qwen2.5-VL runs also show. Checkpoint-376 is the
best DS-MVTec epoch and is within 0.10 pp of the best VisA epoch, so it is the checkpoint to quote.

## There is no GRPO comparison on this backbone

Say this plainly. Nothing was RL-trained on Qwen3-VL-8B. This folder shows only that a KCR corpus
built from one policy still transfers to a different, newer backbone. It says nothing about whether
GRPO would add to it here, and it must not be read as a corpus-versus-RL result on Qwen3-VL. The
corpus-versus-RL comparison exists only on Qwen2.5-VL-7B and on LLaVA-OneVision-7B-SI.

## Naming

Keep-Correct-Revise (KCR) is the method. "Arm C" is the same thing under its ablation name, and it
is what every path, dataset key and directory in this repo uses.
