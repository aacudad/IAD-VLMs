import json, re, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from collections import Counter

CFG="allava_instruct_vflan4v"; N=6
def do(i):
    fs=HfFileSystem()
    p=f"datasets/lmms-lab/LLaVA-OneVision-Data/{CFG}/train-{i:05d}-of-{N:05d}.parquet"
    for a in range(4):
        try:
            with fs.open(p,"rb") as f:
                pf=pq.ParquetFile(f); n=pf.metadata.num_rows
                t=pf.read(columns=["id"])
            return i,n,t.column("id").to_pylist()
        except Exception:
            if a==3: raise
            time.sleep(5*(a+1))
res={}
with ThreadPoolExecutor(max_workers=6) as ex:
    for i,n,ids in ex.map(do,range(N)):
        res[i]=(n,ids); print("shard",i,n,flush=True)
allids=[]
for i in range(N): allids.extend(res[i][1])
print("\nCONFIG:",CFG)
print("shards:",N,"total rows:",len(allids),"distinct ids:",len(set(allids)))
for pat in ["mvtec","visa"]:
    rx=re.compile(pat,re.I)
    m=[x for x in allids if rx.search(x)]
    print(f"  {pat!r}: rows={len(m)} distinct={len(set(m))}")
    for x in list(dict.fromkeys(m))[:6]: print("     ",x)
print("\nsample ids:", allids[:3])
json.dump({"total":len(allids)}, open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/vflan4v.json","w"))
