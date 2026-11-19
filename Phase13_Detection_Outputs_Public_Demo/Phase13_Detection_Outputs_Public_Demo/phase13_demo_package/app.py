
import json
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent

with open(ROOT / "phase13_demo_config.json", "r") as f:
    CFG = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=CFG["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    ckpt = torch.load(ROOT / CFG["checkpoint_path"], map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(DEVICE).eval()
    preprocess_fn = smp.encoders.get_preprocessing_fn(CFG["encoder_name"], "imagenet")
    return model, preprocess_fn

MODEL, PREPROCESS_FN = build_model()

def preprocess_rgb(rgb_img, image_size):
    resized = cv2.resize(rgb_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = PREPROCESS_FN(resized.astype(np.float32))
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return x

@torch.no_grad()
def predict_prob_map(rgb_img):
    h, w = rgb_img.shape[:2]
    x = preprocess_rgb(rgb_img, CFG["image_size"])
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    with amp_ctx:
        logits = MODEL(x).float()[0, 0]

    prob = torch.sigmoid(logits / float(CFG["temperature"])).cpu().numpy().astype(np.float32)
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob

def clean_binary_mask(bin_mask):
    out = bin_mask.astype(np.uint8).copy()

    if CFG["apply_packaging_close"] and CFG["packaging_close_kernel"] > 1:
        kernel = np.ones((CFG["packaging_close_kernel"], CFG["packaging_close_kernel"]), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    min_area = int(CFG["packaging_min_component_area"])
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == lab] = 1
        out = cleaned

    return out

def mask_to_components(prob_map, bin_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    components = []

    for lab in range(1, num_labels):
        comp_mask = (labels == lab).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        area_px = int(comp_mask.sum())
        scores = prob_map[comp_mask > 0]
        score_mean = float(scores.mean()) if scores.size > 0 else 0.0
        score_p95 = float(np.percentile(scores, 95)) if scores.size > 0 else 0.0

        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect
        box = cv2.boxPoints(rect).astype(np.int32).tolist()

        components.append({
            "area_px": area_px,
            "score_mean": score_mean,
            "score_p95": score_p95,
            "aabb": {
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "width": int(x2 - x1 + 1),
                "height": int(y2 - y1 + 1)
            },
            "obb": {
                "center_x": float(cx),
                "center_y": float(cy),
                "width": float(w),
                "height": float(h),
                "angle_deg": float(angle),
                "points": box
            }
        })

    components = sorted(components, key=lambda d: d["area_px"], reverse=True)
    return components

def draw_overlay(rgb_img, bin_mask, components, alpha=0.35):
    overlay = rgb_img.copy()
    green = np.zeros_like(rgb_img)
    green[..., 1] = 255
    overlay = np.where(bin_mask[..., None] > 0, (1 - alpha) * overlay + alpha * green, overlay).astype(np.uint8)

    for i, comp in enumerate(components, start=1):
        a = comp["aabb"]
        cv2.rectangle(overlay, (a["x1"], a["y1"]), (a["x2"], a["y2"]), (255, 0, 0), 2)

        obb_pts = np.array(comp["obb"]["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [obb_pts], isClosed=True, color=(255, 255, 0), thickness=2)

        label = f"{i}:{comp['score_p95']:.2f}"
        cv2.putText(overlay, label, (a["x1"], max(15, a["y1"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay

def predict(image, confidence_gate):
    if image is None:
        return None, None, {}, "Please upload an image."

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    rgb = image.astype(np.uint8)
    prob_map = predict_prob_map(rgb)
    bin_mask = (prob_map >= float(CFG["mask_threshold"])).astype(np.uint8)
    bin_mask = clean_binary_mask(bin_mask)
    components = mask_to_components(prob_map, bin_mask)

    pred_any = int(bin_mask.sum() > 0)
    max_prob = float(prob_map.max()) if prob_map.size > 0 else 0.0
    gate_pass = int(max_prob >= confidence_gate)

    overlay = draw_overlay(rgb, bin_mask, components)
    mask_rgb = np.stack([bin_mask * 255, bin_mask * 255, bin_mask * 255], axis=-1).astype(np.uint8)

    crack_area_px = int(bin_mask.sum())
    crack_area_ratio = float(crack_area_px / max(bin_mask.size, 1))

    remark = []
    if pred_any:
        remark.append("Crack-like pixels detected.")
    else:
        remark.append("No crack-like pixels detected at the current threshold.")

    if gate_pass:
        remark.append(f"Confidence gate passed at {confidence_gate:.2f}.")
    else:
        remark.append(f"Confidence gate not passed at {confidence_gate:.2f}.")

    summary_md = f"""
### CrackSAFE-MRC Prototype Output
- **Predicted crack/no-crack:** {"Crack" if pred_any else "No-crack"}
- **Mask threshold:** {CFG["mask_threshold"]:.2f}
- **Confidence gate:** {confidence_gate:.2f}
- **Max calibrated confidence:** {max_prob:.4f}
- **Crack area (pixels):** {crack_area_px}
- **Crack area ratio:** {crack_area_ratio:.6f}
- **Detected components:** {len(components)}
- **Remark:** {' '.join(remark)}
"""

    result_json = {
        "pred_any": pred_any,
        "confidence_gate_pass": gate_pass,
        "max_calibrated_confidence": max_prob,
        "crack_area_px": crack_area_px,
        "crack_area_ratio": crack_area_ratio,
        "num_components": len(components),
        "components": components
    }

    return overlay, mask_rgb, result_json, summary_md

examples = []
examples_dir = ROOT / "examples"
if examples_dir.exists():
    for p in sorted(examples_dir.glob("*.jpg")):
        examples.append([str(p), 0.90])

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload wall/concrete image"),
        gr.Slider(minimum=0.50, maximum=0.99, value=0.90, step=0.01, label="Confidence gate")
    ],
    outputs=[
        gr.Image(type="numpy", label="Overlay"),
        gr.Image(type="numpy", label="Binary mask"),
        gr.JSON(label="Detections"),
        gr.Markdown(label="Summary")
    ],
    title="CrackSAFE-MRC Prototype Demo",
    description="Upload a wall/concrete image to get calibrated crack segmentation, AABB/OBB detections, crack area proxy, and confidence-aware output.",
    examples=examples if len(examples) > 0 else None,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
