#!/usr/bin/env python3
"""Recompute corpus descriptors used in the thesis, from on-disk artefacts.

Backs the Ch.3 trace-length and t-SNE type-cluster numbers (see NUMBER_PROVENANCE.md).

Sources (all under /bulk/aacudad/reasoning_traces):
  - Trace JSONs: Training/datasets_small_new_v4/combined_6k_train.json (6K),
                 Training/15k_dataset_regenerated/combined_sft_train.json (15K union)
                 -> <think>-block length in Qwen2.5-VL tokens and in words.
  - Type-cluster geometry: Training/anomaly_type_analysis/types.json (8908 raw type strings)
                           + .../embeddings_2d.npy (8908x2 t-SNE coords, aligned).

Run: conda activate llama_sft && python recompute_corpus_stats.py
"""
import json, re, os, statistics
import numpy as np

ROOT = "/bulk/aacudad/reasoning_traces"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf_cache")


def think_texts(path):
    d = json.load(open(path)); out = []
    for r in d:
        for m in r.get("messages", []):
            if m.get("role") == "assistant":
                t = re.search(r"<think>(.*?)</think>", m["content"], re.S)
                if t:
                    out.append(t.group(1).strip())
    return out


print("==== 1. <think> length: tokens (Qwen2.5-VL) and words ====")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    for label, p in [("6K ", "Training/datasets_small_new_v4/combined_6k_train.json"),
                     ("15K", "Training/15k_dataset_regenerated/combined_sft_train.json")]:
        texts = think_texts(f"{ROOT}/{p}")
        toks = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
        words = [len(t.split()) for t in texts]
        print(f"  {label}: n={len(toks)} | TOKENS mean={statistics.mean(toks):.1f} "
              f"std={statistics.pstdev(toks):.1f} range[{min(toks)},{max(toks)}] "
              f"p5={int(np.percentile(toks,5))} p95={int(np.percentile(toks,95))} | "
              f"WORDS mean={statistics.mean(words):.1f} std={statistics.pstdev(words):.1f}")
except Exception as ex:
    print("  TOKEN STATS FAILED:", repr(ex))

print("\n==== 2. t-SNE type-cluster geometry ====")
types = [str(t) for t in json.load(open(f"{ROOT}/Training/anomaly_type_analysis/types.json"))]
emb = np.load(f"{ROOT}/Training/anomaly_type_analysis/embeddings_2d.npy")
assert len(types) == len(emb), (len(types), len(emb))
print(f"  N labelled traces = {len(types)}")


def stats(mask):
    pts = emb[mask]
    if len(pts) == 0:
        return 0, np.array([0., 0.]), 0.
    c = pts.mean(0)
    r = float(np.linalg.norm(pts - c, axis=1).mean())
    return len(pts), c, r


def exact(label):
    return np.array([t.strip().lower() == label.lower() for t in types])


def substr(s):
    return np.array([s.lower() in t.lower() for t in types])


for name, m in [("Scratch (exact)", exact("Scratch")),
                ("Scratch (substr)", substr("scratch")),
                ("Missing component (exact)", exact("Missing component")),
                ("Missing* (substr)", substr("missing"))]:
    n, c, r = stats(m)
    print(f"  {name:26s}: n={n:4d}  centroid=({c[0]:7.1f},{c[1]:7.1f})  within-radius={r:.2f}")

ns, cs, rs = stats(exact("Scratch"))
nm, cm, rm = stats(exact("Missing component"))
dist = float(np.linalg.norm(cs - cm))
radius = (rs + rm) / 2
print(f"\n  Scratch<->Missing-component centroid distance = {dist:.1f}")
print(f"  mean within-cluster radius (Scratch, Missing)  = {radius:.1f}")
print(f"  separation ratio (distance / radius)           = {dist / radius:.1f}" if radius else "  ratio: n/a")
print("DONE")
