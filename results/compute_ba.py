#!/usr/bin/env python3
"""Recompute balanced accuracy (and acc/F1) from any eval_*.json with a `metrics`
block {tp,tn,fp,fn}.  Usage: python compute_ba.py path/to/eval_*.json [...]"""
import json, sys
def ba(tp, tn, fp, fn):
    pos, neg = tp + fn, tn + fp
    return 50 * ((tp / pos if pos else 0) + (tn / neg if neg else 0))
for p in sys.argv[1:]:
    m = json.load(open(p)).get("metrics", {})
    tp, tn, fp, fn = (m.get(k, 0) for k in ("tp", "tn", "fp", "fn"))
    n = tp + tn + fp + fn
    if not n:
        print(f"{p}: no tp/tn/fp/fn"); continue
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * P * R / (P + R) if (P + R) else 0
    print(f"{p}\n  n={n}  TP{tp} TN{tn} FP{fp} FN{fn}"
          f"  BA={ba(tp,tn,fp,fn):.2f}  acc={100*(tp+tn)/n:.2f}  F1={100*f1:.2f}")
