# CrackSAFE-MRC Demo with Explainability

This package was generated from the Phase-14A Kaggle notebook.

## Included
- app.py
- checkpoint.pt
- phase14a_demo_config.json
- examples/
- requirements.txt

## Model
- Selected ablation: Boundary + clDice + Lovasz
- Encoder: efficientnet-b3
- Temperature: 1.556495
- Recommended mask threshold: 0.55

## Outputs
- overlay
- binary mask
- explanation heatmap (HiResCAM / GradCAM++ / GradCAM)
- detections JSON
- short summary
