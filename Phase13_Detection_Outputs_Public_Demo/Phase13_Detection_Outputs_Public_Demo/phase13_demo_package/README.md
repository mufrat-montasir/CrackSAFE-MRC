# CrackSAFE-MRC Prototype Demo

This folder was exported from the Phase-13 Kaggle notebook.

## Included
- `app.py`
- `checkpoint.pt`
- `phase13_demo_config.json`
- `examples/`
- `requirements.txt`

## Model
- Selected ablation: **Boundary + clDice + Lovasz**
- Encoder: **efficientnet-b3**
- Temperature: **1.556495**
- Recommended mask threshold: **0.55**

## What the demo returns
- crack mask
- overlay on original image
- crack/no-crack decision
- crack area proxy
- calibrated confidence
- AABB / OBB detections
- short plain-language remark

## Deploy to Hugging Face Spaces
1. Create a new **Gradio Space**
2. Upload all files in this folder
3. Let Spaces build from `requirements.txt`
4. The app will run from `app.py`
