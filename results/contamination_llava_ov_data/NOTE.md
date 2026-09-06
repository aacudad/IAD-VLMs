# MVTec-AD rows in the LLaVA-OneVision training mixture (thesis §6.1 caveat)

Counted over the parquet footers and `id` column of `lmms-lab/LLaVA-OneVision-Data`, config
`vision_flan(filtered)`: 1,999 of 186,060 rows match `MVTecAD`, 0 match VisA. `README.md` is the full
evidence note, `totals.json` the per-shard counts, `count_shards.py` / `vflan4v.py` the counting code.
The earlier figure of 426 came from a truncated index and is wrong.
