import json, re
from collections import Counter

rows = [json.loads(l) for l in open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/all_ids.jsonl")]
print("total rows:", len(rows))
ids = [r["id"] for r in rows]

def show(name, pat):
    rx = re.compile(pat, re.I)
    m = [r for r in rows if rx.search(r["id"])]
    d = set(x["id"] for x in m)
    print(f"\n--- {name}  pattern={pat!r} (case-insensitive) ---")
    print(f"  matching ROWS: {len(m)}")
    print(f"  DISTINCT matching ids: {len(d)}")
    print(f"  data_source values: {Counter(x['data_source'] for x in m)}")
    return m, d

# MVTec: try several spellings
for pat in [r"mvtec", r"mvtec-?ad", r"mvtecad", r"mv_tec"]:
    rx = re.compile(pat, re.I)
    m = [r for r in rows if rx.search(r["id"])]
    print(f"pattern {pat!r}: rows={len(m)} distinct={len(set(x['id'] for x in m))}")

m_mv, d_mv = show("MVTec", r"mvtec")
print("\n  first 12 matching ids:")
for x in m_mv[:12]:
    print("   ", x["id"])

# duplicate structure among mvtec matches
c = Counter(x["id"] for x in m_mv)
print("\n  multiplicity histogram of mvtec ids:", Counter(c.values()))
print("  most common:", c.most_common(3))

# VisA
for pat in [r"visa", r"\bvisa\b", r"visa[-_]?anomaly", r"visa"]:
    rx = re.compile(pat, re.I)
    m = [r for r in rows if rx.search(r["id"])]
    print(f"\nVisA pattern {pat!r}: rows={len(m)} distinct={len(set(x['id'] for x in m))}")
    for x in m[:15]:
        print("    ", x["id"])

json.dump({"mvtec_rows":[x["id"] for x in m_mv]}, open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/mvtec_ids.json","w"), indent=1)
