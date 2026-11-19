
import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                state_dict = ckpt_obj[key]
                break
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned

def build_model(cfg, checkpoint_path, device):
    model = smp.UnetPlusPlus(
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        activation=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    preprocess_fn = smp.encoders.get_preprocessing_fn(cfg["encoder_name"], "imagenet")
    return model, preprocess_fn

def preprocess_rgb(rgb_img, preprocess_fn, image_size):
    resized = cv2.resize(rgb_img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = preprocess_fn(resized.astype(np.float32))
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return resized, x

def clean_binary_mask(bin_mask, cfg):
    out = bin_mask.astype(np.uint8).copy()
    pp = cfg["packaging_postprocess"]

    if pp["apply_packaging_close"] and pp["packaging_close_kernel"] > 1:
        kernel = np.ones((pp["packaging_close_kernel"], pp["packaging_close_kernel"]), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if pp["packaging_min_component_area"] > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= pp["packaging_min_component_area"]:
                cleaned[labels == lab] = 1
        out = cleaned

    return out

def build_components(prob_map, bin_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    components = []
    for lab in range(1, num_labels):
        comp_mask = (labels == lab).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        scores = prob_map[comp_mask > 0]

        components.append({
            "area_px": int(comp_mask.sum()),
            "score_mean": float(scores.mean()) if scores.size > 0 else 0.0,
            "score_p95": float(np.percentile(scores, 95)) if scores.size > 0 else 0.0,
            "aabb": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
    components = sorted(components, key=lambda d: d["area_px"], reverse=True)
    return components

def draw_overlay(rgb_img, bin_mask, components):
    overlay = rgb_img.copy()
    green = np.zeros_like(rgb_img)
    green[..., 1] = 255
    overlay = np.where(bin_mask[..., None] > 0, (0.65 * overlay + 0.35 * green), overlay).astype(np.uint8)

    for comp in components:
        a = comp["aabb"]
        cv2.rectangle(overlay, (a["x1"], a["y1"]), (a["x2"], a["y2"]), (255, 0, 0), 2)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(package_dir / "configs" / "phase14b_repro_config.json", "r") as f:
        cfg = json.load(f)

    checkpoint_path = package_dir / "models" / "final_checkpoint.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model, preprocess_fn = build_model(cfg, checkpoint_path, device)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(sorted(input_dir.glob(ext)))

    records = []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_448, x = preprocess_rgb(rgb, preprocess_fn, cfg["image_size"])
        input_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with amp_ctx:
            logits = model(input_tensor).float()[0, 0]

        prob = torch.sigmoid(logits / float(cfg["temperature"])).cpu().numpy().astype(np.float32)
        pred = (prob >= float(cfg["recommended_mask_threshold"])).astype(np.uint8)
        pred = clean_binary_mask(pred, cfg)

        components = build_components(prob, pred)
        overlay = draw_overlay(rgb_448, pred, components)
        mask_rgb = np.stack([pred * 255, pred * 255, pred * 255], axis=-1).astype(np.uint8)

        stem = p.stem
        cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

        records.append({
            "filename": p.name,
            "pred_any": int(pred.sum() > 0),
            "crack_area_px": int(pred.sum()),
            "crack_area_ratio": float(pred.mean()),
            "max_prob": float(prob.max()),
            "num_components": int(len(components)),
            "components": components,
        })

    with open(output_dir / "predictions.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\\n")

    print(f"Saved outputs to: {output_dir}")

if __name__ == "__main__":
    main()
