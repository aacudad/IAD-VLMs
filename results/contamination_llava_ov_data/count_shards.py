import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/lmms-lab/LLaVA-OneVision-Data"
CFG = "vision_flan(filtered)"
N = 50

def do_shard(i):
    fs = HfFileSystem()
    path = f"{REPO}/{CFG}/train-{i:05d}-of-{N:05d}.parquet"
    for attempt in range(4):
        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                nrows = pf.metadata.num_rows
                tbl = pf.read(columns=["id", "data_source"])
            ids = tbl.column("id").to_pylist()
            ds = tbl.column("data_source").to_pylist()
            assert len(ids) == nrows, (len(ids), nrows)
            return i, nrows, ids, ds
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))

t0 = time.time()
results = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, nrows, ids, ds in ex.map(do_shard, range(N)):
        results[i] = (nrows, ids, ds)
        print(f"shard {i:02d}: {nrows} rows  ({time.time()-t0:.0f}s)", flush=True)

all_ids = []
all_ds = []
per_shard = {}
for i in range(N):
    nrows, ids, ds = results[i]
    per_shard[i] = nrows
    all_ids.extend(ids)
    all_ds.extend(ds)

total = len(all_ids)
print("\n=== TOTALS ===")
print("shards:", N)
print("total rows:", total)
print("sum of per-shard metadata num_rows:", sum(per_shard.values()))
print("distinct ids overall:", len(set(all_ids)))

out = {
  "shards": N,
  "per_shard_rows": per_shard,
  "total_rows": total,
  "distinct_ids_total": len(set(all_ids)),
}

with open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/all_ids.jsonl", "w") as f:
    for a, b in zip(all_ids, all_ds):
        f.write(json.dumps({"id": a, "data_source": b}) + "\n")

with open("/bulk/aacudad/reasoning_traces/defense_prep_20260901/scratch_contam/totals.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote all_ids.jsonl and totals.json")
