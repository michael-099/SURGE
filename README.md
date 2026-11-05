# SURGE

Video saliency prediction research.

- **`SURGE_final_proposal.md`**: the final, fact-checked proposal (SURGE-D). It contains the fact-check ledger for the original design and reality-check documents, an independent replication of the V-JEPA 2 surprise diagnostics, the distilled architecture, the experiments and decision gates, and the compute budget.
- **`bench/`**: replication scripts and results.
  - `count_flops.py`: FLOPs and parameter counts of the V-JEPA 2 ViT-L encoder and predictor.
  - `replicate_surprise.py`: surprise diagnostics on real weights (11 clips × 2 frame rates).
  - `replication.json`: raw per-clip results.

The original documents (`SURGE_video_saliency_research_design.md`, `SURGE_reality_check_and_compute.md`) were removed after they were consolidated. They remain in git history at commit `199de7a`.

To reproduce:

```
pip install torch transformers av numpy
python bench/count_flops.py
python bench/replicate_surprise.py <folder of .mp4 clips> bench/replication.json
```
