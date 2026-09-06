# Chapter 6 figure pipeline

Every figure in thesis Chapter 6 (and the ladder numbers in Figure 1.1) is drawn from one data file by
small SVG generators, then cropped, stripped of in-canvas prose and converted to PDF for the thesis.

```
WORK_DIR=/path/to/workspace   # parent of this repo, holds outputs/ (or point registry.py at results/)
python export_data.py         # registry.py -> figdata.json  (ladder, per-product, arms)
python loc_hit_table.py       # -> loc_hit.json, tab_loc_hit.tex   (thesis Table 6.12)
python type_sim_table.py      # -> type_sim.json, tab_type_sim.tex (thesis Appendix J.5)
python f1_ladder.py f2_arms.py f3_operating.py f4_perproduct.py f5_explain.py f6_gallery.py
python g1_sft.py g2_decoupling.py g3_iter2.py g4_dynamics.py g5_flow.py
THESIS_FIGURES=/path/to/thesis/figures python thesisify.py   # crops, strips captions, writes thesis/*.pdf and installs
```

- `registry.py` maps every (backbone, stage, benchmark) cell to its eval JSON. Change a checkpoint here and
  nothing else. The LLaVA SFT+GRPO and KCR rows point at the vLLM files by decision.
- `svgkit.py` holds the house style (Verdana, colours, markers). `thesisify.py` needs `cairosvg`
  (`CAIROSVG` env if not on PATH).
- `stages/f7_stages.py` draws Figure 4.1 (SFT stage next to GRPO stage). `svg_final/` holds the final SVGs
  as installed, including the Figure 1.1 overview (`method_overview_v4.svg`, exported to PDF by hand).
- `prep_gallery.py` and `prep_images.py` prepare the case images for `f6_gallery.py`.
- The generated data files are also shipped under `results/thesis_figure_data/`.

The renderer for the SFT-vs-KCR pair images (Figure 6.9, Appendix F.5) is not part of this folder, see
`results/sft_vs_kcr_pairs/README.md`.
