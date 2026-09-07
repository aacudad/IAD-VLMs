# Type-string analysis on the final corpus (thesis Table 5.1, Figure 5.2, Appendix E heat map), 7 Sep 2026

`types_final_anom.json`: (split, <type> string) for the 7,237 anomalous traces of the final corpus that carry a tag
(SFT 6K 3,000, GRPO split 2,108 of 2,118, held-out 2,129); 36 distinct strings.

Two embedding paths, and they differ:
- `embed_tsne_search_document.py`: `search_document:` prefix, batches of 64, unmasked mean pooling (the path of the
  original `analyze_anomaly_types.py`). Identical strings get slightly different vectors per batch, which is why the
  t-SNE (`tsne_2d_search_document_unmasked.npy`, thesis Figure 5.2) shows clouds. Stated in the caption.
- `reward_path_tables_and_tsne.py`: the reward server's path (`search_query:` prefix, one string at a time, masked mean
  pooling, L2 norm), checked to 4 decimals against the live server. Gives `embeddings_reward_path.npy`,
  `synonym_table_reward_path.json` (Table 5.1), `top10_cosine_reward_path.json` (Appendix E heat map). Under this path
  the 36 strings collapse to 36 points, so no t-SNE is shown for it.

The old `../types.json` (8,908 strings) is an unidentified early pool with every string duplicated; superseded.
