import json, re
from collections import Counter
rows=[json.loads(l) for l in open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/all_ids.jsonl")]
rx=re.compile("mvtec", re.I)
m=[r["id"] for r in rows if rx.search(r["id"])]
print("mvtec rows:", len(m), " distinct:", len(set(m)))

def task(i):
    # ids look like vision-flan_MVTecAD+<task>+<n>
    parts=i.split("+")
    return parts[1] if len(parts)>1 else "(no +)"

c=Counter(task(i) for i in m)
print("\nROWS by task field:")
for k,v in c.most_common(): print(f"  {k:35s} {v:6d}")
cd=Counter(task(i) for i in set(m))
print("\nDISTINCT ids by task field:")
for k,v in cd.most_common(): print(f"  {k:35s} {v:6d}")

print("\nprefix check, all mvtec ids start with 'vision-flan_MVTecAD+':",
      all(i.startswith("vision-flan_MVTecAD+") for i in m))
print("\nsorted distinct ids, first 8:")
for i in sorted(set(m))[:8]: print("   ", i)
