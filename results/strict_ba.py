#!/usr/bin/env python3
"""Strict balanced accuracy from an eval_*.json: every sample counts, an unparsed prediction is wrong."""
import json,sys
for p in sys.argv[1:]:
    r=json.load(open(p))['results']; P=sum(x['gt_answer']=='yes' for x in r); N=len(r)-P
    tp=sum(x['gt_answer']=='yes' and x['pred_answer']=='yes' for x in r); tn=sum(x['gt_answer']=='no' and x['pred_answer']=='no' for x in r)
    un=sum(x['pred_answer'] not in ('yes','no') for x in r)
    print(f"{p}: n={len(r)} unparsed={un} BA={50*(tp/P+tn/N):.2f}")
