# Data behind the Chapter 6 figures and the two derived tables

Produced by `scripts/05_figures/thesis_figures_v2/`.

- `figdata.json`: `export_data.py` output. Ladder (base / SFT / SFT+GRPO / KCR on both backbones and
  benchmarks, with TPR / TNR), per-product balanced accuracy for the three trained stages, and the
  teacher-ablation arms per epoch. Every Chapter 6 figure reads this file.
- `loc_hit.json`, `tab_loc_hit.tex`: `loc_hit_table.py`. Share of anomalies whose predicted `<location>`
  overlaps the ground-truth mask cells (thesis Table 6.12), full benchmarks, per stage and backbone.
- `type_sim.json`, `tab_type_sim.tex`: `type_sim_table.py`. Nomic similarity and exact-match rate of the
  predicted `<type>` against the MMAD defect label (thesis Appendix J.5).
- `grpo_curve.json`: run-2 reward / KL / length series for the dynamics figure.

The registry of which eval JSON feeds which cell is `scripts/05_figures/thesis_figures_v2/registry.py`.
