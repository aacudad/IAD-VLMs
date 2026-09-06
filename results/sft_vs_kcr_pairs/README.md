# SFT-vs-KCR trace pairs (thesis Figure 6.9 and Appendix F.5)

Pool rule: DS-MVTec or VisA anomalies on which the 6K SFT model (ckpt-564) answers no, the Arm-C KCR model
(ckpt-376) answers yes, and the KCR `<location>` overlaps the ground-truth mask cells. 65 images satisfy it
(`index_all65.json`). `index_selected21.json` is the hand-picked set of 21 shown in the appendix,
`index_first30.json` the first 30 rendered. The PNGs live in the thesis repository under `figures/pairs/`.
The renderer that drew the arrow overlays was written in a session scratch area and is not on disk; the
index files hold every path and tag needed to re-render.


## 2026-09-07: figures redrawn

Figure 6.9 and the 21 Appendix F.5 pairs are now drawn by `scripts/05_figures/thesis_figures_v2/f8_pairs.py`
(house-style SVG, image beside both traces, the decisive sentence shaded). The Appendix F.1 to F.4 and G.1 to G.5
example figures come from `f9_appendix_examples.py`; `appendix_examples_selection.json` lists every image id used
and the rule that chose it (see the script header).
