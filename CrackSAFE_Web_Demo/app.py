import json
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent

with open(ROOT / "phase14a_demo_config.json", "r") as f:
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


def preprocess_rgb(rgb_img):
    resized = cv2.resize(
        rgb_img,
        (CFG["image_size"], CFG["image_size"]),
        interpolation=cv2.INTER_LINEAR
    )
    x = PREPROCESS_FN(resized.astype(np.float32))
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return resized, x


def clean_binary_mask(bin_mask):
    out = bin_mask.astype(np.uint8).copy()

    if CFG["apply_packaging_close"] and CFG["packaging_close_kernel"] > 1:
        kernel = np.ones(
            (CFG["packaging_close_kernel"], CFG["packaging_close_kernel"]),
            np.uint8
        )
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


def find_target_layer(model):
    if hasattr(model.encoder, "_blocks") and len(model.encoder._blocks) > 0:
        return model.encoder._blocks[-1]

    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


TARGET_LAYER = find_target_layer(MODEL)


class BinarySegmentationTarget:
    def __init__(self, target_mask):
        self.target_mask = torch.from_numpy(target_mask.astype(np.float32))

    def __call__(self, model_output):
        if model_output.ndim == 4:
            out = model_output[0, 0]
        else:
            out = model_output[0]
        return (out * self.target_mask.to(out.device)).sum()


def get_cam_method(name):
    if name == "HiResCAM":
        return HiResCAM(model=MODEL, target_layers=[TARGET_LAYER])
    elif name == "GradCAMPlusPlus":
        return GradCAMPlusPlus(model=MODEL, target_layers=[TARGET_LAYER])
    else:
        return GradCAM(model=MODEL, target_layers=[TARGET_LAYER])


@torch.no_grad()
def predict_prob_map(input_tensor):
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if USE_AMP else nullcontext()
    )
    with amp_ctx:
        logits = MODEL(input_tensor).float()[0, 0]

    prob = torch.sigmoid(logits / float(CFG["temperature"])).cpu().numpy().astype(np.float32)
    return prob


def build_components(prob_map, bin_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8),
        connectivity=8
    )
    components = []

    for lab in range(1, num_labels):
        comp_mask = (labels == lab).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        scores = prob_map[comp_mask > 0]
        score_p95 = float(np.percentile(scores, 95)) if scores.size > 0 else 0.0

        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.int32).tolist()

        components.append({
            "area_px": int(comp_mask.sum()),
            "score_p95": score_p95,
            "aabb": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "obb": {"points": box}
        })

    return components


def draw_overlay(rgb_img, bin_mask, components, alpha=0.35):
    overlay = rgb_img.copy()
    green = np.zeros_like(rgb_img)
    green[..., 1] = 255

    overlay = np.where(
        bin_mask[..., None] > 0,
        (1 - alpha) * overlay + alpha * green,
        overlay
    ).astype(np.uint8)

    for i, comp in enumerate(components, start=1):
        a = comp["aabb"]
        cv2.rectangle(overlay, (a["x1"], a["y1"]), (a["x2"], a["y2"]), (255, 0, 0), 2)

        obb_pts = np.array(comp["obb"]["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [obb_pts], True, (255, 255, 0), 2)

        cv2.putText(
            overlay,
            f"{i}:{comp['score_p95']:.2f}",
            (a["x1"], max(15, a["y1"] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return overlay


def predict(image, cam_method, confidence_gate):
    if image is None:
        return None, None, None, "{}", "Upload an image to begin analysis."

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[..., :3]

    rgb = image.astype(np.uint8)
    rgb_448, x = preprocess_rgb(rgb)
    input_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    prob = predict_prob_map(input_tensor)
    pred = (prob >= float(CFG["mask_threshold"])).astype(np.uint8)
    pred = clean_binary_mask(pred)

    if pred.sum() > 0:
        target_mask = pred
    else:
        target_mask = (prob >= np.percentile(prob, 99.5)).astype(np.uint8)

    target = BinarySegmentationTarget(target_mask)
    with get_cam_method(cam_method) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]

    cam_vis = show_cam_on_image(rgb_448.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)

    components = build_components(prob, pred)
    overlay = draw_overlay(rgb_448, pred, components)
    mask_rgb = np.stack([pred * 255, pred * 255, pred * 255], axis=-1).astype(np.uint8)

    pred_any = int(pred.sum() > 0)
    max_prob = float(prob.max()) if prob.size > 0 else 0.0
    crack_area_px = int(pred.sum())
    crack_area_ratio = float(pred.mean())
    gate_pass = int(max_prob >= confidence_gate)

    summary = {
        "pred_any": pred_any,
        "confidence_gate_pass": gate_pass,
        "max_calibrated_confidence": max_prob,
        "crack_area_px": crack_area_px,
        "crack_area_ratio": crack_area_ratio,
        "num_components": len(components),
        "cam_method": cam_method,
        "components": components
    }

    summary_json = json.dumps(summary, indent=2)

    summary_md = f"""
### Operational Summary
- **Prediction:** {"Crack" if pred_any else "No-crack"}
- **Mask threshold:** {CFG["mask_threshold"]:.2f}
- **Confidence gate:** {confidence_gate:.2f}
- **Gate passed:** {"Yes" if gate_pass else "No"}
- **Max calibrated confidence:** {max_prob:.4f}
- **Crack area (pixels):** {crack_area_px}
- **Crack area ratio:** {crack_area_ratio:.6f}
- **Explainability method:** {cam_method}
- **Detected components:** {len(components)}
"""

    return overlay, mask_rgb, cam_vis, summary_json, summary_md


def reset_ui():
    return None, "HiResCAM", 0.90, None, None, None, "{}", "Upload an image to begin analysis."


examples = []
examples_dir = ROOT / "examples"
if examples_dir.exists():
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in sorted(examples_dir.glob(ext)):
            examples.append([str(p), "HiResCAM", 0.90])


theme = gr.themes.Soft()

css = """
:root {
  --brand-accent: #2563eb;
  --brand-accent-2: #0f172a;
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
}

.hero-wrap {
  padding: 18px 10px 6px 10px;
}

.hero-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 24px;
  padding: 28px 28px 22px 28px;
  background:
    radial-gradient(circle at top right, rgba(37,99,235,0.18), transparent 30%),
    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.96));
  box-shadow: 0 14px 40px rgba(0,0,0,0.25);
}

.hero-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  color: #dbeafe;
  font-size: 13px;
  margin-bottom: 14px;
}

.hero-title {
  font-size: 34px;
  font-weight: 800;
  line-height: 1.1;
  margin: 0 0 10px 0;
}

.hero-subtitle {
  font-size: 16px;
  opacity: 0.92;
  margin: 0;
}

.app-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px;
  padding: 14px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.section-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
}

.metric-note {
  font-size: 14px;
  opacity: 0.88;
}

.footer-note {
  text-align: center;
  font-size: 13px;
  opacity: 0.7;
  padding: 10px 0 18px 0;
}

#run-btn, #clear-btn {
  min-height: 46px;
  border-radius: 14px !important;
  font-weight: 700 !important;
}

#run-btn {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
}

#summary-box, #json-box {
  border-radius: 18px !important;
}
"""


with gr.Blocks(theme=theme, css=css, title="CrackSAFE-MRC Demo + Explainability") as demo:
    gr.HTML(
        """
        <div class="hero-wrap">
          <div class="hero-card">
            <div class="hero-badge">CrackSAFE-MRC · Web Demo</div>
            <div class="hero-title">Crack Segmentation + Detection + Explainability</div>
            <p class="hero-subtitle">
              Upload a wall or concrete image to generate a calibrated crack mask,
              operational detections, and CAM-based explanation views suitable for
              demonstration, review, and poster presentation.
            </p>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            with gr.Group(elem_classes=["app-card"]):
                gr.Markdown("### Input & Controls")
                input_image = gr.Image(
                    type="numpy",
                    label="Upload wall/concrete image",
                    height=360
                )
                cam_method = gr.Dropdown(
                    choices=["HiResCAM", "GradCAMPlusPlus", "GradCAM"],
                    value="HiResCAM",
                    label="CAM method"
                )
                confidence_gate = gr.Slider(
                    minimum=0.50,
                    maximum=0.99,
                    value=0.90,
                    step=0.01,
                    label="Confidence gate"
                )
                with gr.Row():
                    run_btn = gr.Button("Run Analysis", variant="primary", elem_id="run-btn")
                    clear_btn = gr.Button("Reset", elem_id="clear-btn")

            with gr.Accordion("Model Details", open=False):
                gr.Markdown(
                    f"""
**Selected ablation:** {CFG.get("selected_ablation", "Finalist model")}  
**Encoder:** {CFG["encoder_name"]}  
**Temperature:** {float(CFG["temperature"]):.6f}  
**Mask threshold:** {float(CFG["mask_threshold"]):.2f}  
**Input size:** {int(CFG["image_size"])} × {int(CFG["image_size"])}  
**Packaging close:** {bool(CFG["apply_packaging_close"])}  
**Minimum component area:** {int(CFG["packaging_min_component_area"])}
"""
                )

            if examples:
                with gr.Accordion("Quick Example Images", open=True):
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image, cam_method, confidence_gate],
                        cache_examples=False,
                        label="Load one of the packaged demo examples"
                    )

        with gr.Column(scale=6, min_width=560):
            with gr.Tabs():
                with gr.Tab("Overlay"):
                    overlay_img = gr.Image(label="Overlay", height=360)
                with gr.Tab("Binary Mask"):
                    mask_img = gr.Image(label="Binary mask", height=360)
                with gr.Tab("Explanation Heatmap"):
                    cam_img = gr.Image(label="Explanation heatmap", height=360)

            with gr.Row():
                with gr.Column(scale=5):
                    detections_json = gr.Code(
                        label="Detections JSON",
                        language="json",
                        lines=20,
                        elem_id="json-box"
                    )
                with gr.Column(scale=4):
                    summary_md = gr.Markdown(
                        value="Upload an image to begin analysis.",
                        elem_id="summary-box"
                    )

    gr.HTML(
        """
        <div class="footer-note">
          Designed for capstone demonstration, explainability review, and poster-ready visualization.
        </div>
        """
    )

    run_btn.click(
        fn=predict,
        inputs=[input_image, cam_method, confidence_gate],
        outputs=[overlay_img, mask_img, cam_img, detections_json, summary_md]
    )

    clear_btn.click(
        fn=reset_ui,
        inputs=[],
        outputs=[input_image, cam_method, confidence_gate, overlay_img, mask_img, cam_img, detections_json, summary_md]
    )


if __name__ == "__main__":
    demo.queue().launch()