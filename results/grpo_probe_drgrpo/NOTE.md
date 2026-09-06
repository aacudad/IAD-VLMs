# Advantage-estimator probe from Arm-C ckpt-376 (thesis Table 6.10, Figure 6.6)

120 steps, checkpoint every 20, all from the same initialisation, identical except for the advantage:
`grpo_probe_ctrl` = vanilla group z-score, `grpo_probe_drgrpo` = mean-centring only, `grpo_probe_g2rpo` =
rank-to-quantile (G2RPO). Each `checkpoint-N/probe_dsmvtec.json` and `probe_visa.json` is the fixed
seed-deterministic 400-sample probe. The probe reads about 1.7 points above the full evaluation, so these
numbers are not comparable to full-subset numbers. Balanced accuracy recomputed with `results/compute_ba.py`
reproduces every cell of the thesis table.
