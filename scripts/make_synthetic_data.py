"""Generate a small synthetic product-photo dataset with known defect locations.

Why synthetic? It is consented, free, and lets us split train/dev/test by *product instance*
(every image gets a unique product colour/shape/seed, so nothing leaks between splits).

Usage:
    python scripts/make_synthetic_data.py --out data/eval --n 40
    python scripts/make_synthetic_data.py --out samples --demo
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _background(rng, h, w):
    base = rng.integers(225, 246)
    bg = np.full((h, w, 3), base, np.float32)
    grad = np.linspace(-8, 8, w, dtype=np.float32)[None, :, None]
    return bg + grad


def _product(rng, h, w):
    """Draw a rounded-rectangle product with a label and soft shading. Returns (img, box)."""
    img = _background(rng, h, w)
    pw, ph = int(w * rng.uniform(0.38, 0.5)), int(h * rng.uniform(0.55, 0.7))
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x0 + 25, y0), (x0 + pw - 25, y0 + ph), 255, -1)
    cv2.rectangle(mask, (x0, y0 + 25), (x0 + pw, y0 + ph - 25), 255, -1)
    for cx, cy in [(x0 + 25, y0 + 25), (x0 + pw - 25, y0 + 25), (x0 + 25, y0 + ph - 25), (x0 + pw - 25, y0 + ph - 25)]:
        cv2.circle(mask, (cx, cy), 25, 255, -1)
    color = np.array([rng.integers(40, 150), rng.integers(40, 150), rng.integers(40, 150)], np.float32)
    shade = np.linspace(1.15, 0.8, pw, dtype=np.float32)[None, :, None]
    body = np.broadcast_to(color, (ph, pw, 3)) * shade
    layer = img.copy()
    layer[y0:y0 + ph, x0:x0 + pw] = body
    img = np.where(mask[..., None] > 0, layer, img)
    ly0, ly1 = y0 + int(ph * 0.35), y0 + int(ph * 0.62)
    img[ly0:ly1, x0 + 10:x0 + pw - 10] = 238
    return img, (x0, y0, pw, ph)


def _add_crack(rng, img, box):
    x0, y0, pw, ph = box
    x, y = int(x0 + pw * rng.uniform(0.25, 0.7)), int(y0 + ph * rng.uniform(0.1, 0.25))
    pts = [(x, y)]
    for _ in range(int(rng.integers(9, 14))):
        x += int(rng.integers(-9, 10))
        y += int(rng.integers(7, 14))
        pts.append((x, y))
    cv2.polylines(img, [np.array(pts)], False, (20, 20, 20), 2, cv2.LINE_AA)
    return cv2.boundingRect(np.array(pts))


def _add_scratch(rng, img, box):
    x0, y0, pw, ph = box
    zone = (0.06, 0.28) if rng.random() < 0.5 else (0.7, 0.9)  # keep scratches off the light label panel
    x, y = int(x0 + pw * rng.uniform(0.1, 0.4)), int(y0 + ph * rng.uniform(*zone))
    x2, y2 = x + int(pw * rng.uniform(0.35, 0.5)), y + int(rng.integers(-25, 26))
    for off in (0, 3):
        cv2.line(img, (x, y + off), (x2, y2 + off), (235, 235, 235), 2, cv2.LINE_AA)
    return cv2.boundingRect(np.array([(x, y), (x2, y2 + 3)]))


def _add_stain(rng, img, box):
    x0, y0, pw, ph = box
    cx, cy = int(x0 + pw * rng.uniform(0.3, 0.7)), int(y0 + ph * rng.uniform(0.7, 0.85))
    layer = np.zeros(img.shape[:2], np.float32)
    for _ in range(5):
        cv2.ellipse(layer, (cx + int(rng.integers(-12, 13)), cy + int(rng.integers(-8, 9))),
                    (int(rng.integers(14, 26)), int(rng.integers(10, 20))), int(rng.integers(0, 180)), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 3)[..., None]
    stain = np.array([30, 70, 120], np.float32)  # brownish (BGR)
    img[:] = img * (1 - 0.75 * layer) + stain * 0.75 * layer
    ys, xs = np.where(layer[..., 0] > 0.3)
    return cv2.boundingRect(np.stack([xs, ys], 1).astype(np.int32))


DEFECTS = {"crack": _add_crack, "scratch": _add_scratch, "stain": _add_stain}


def make_image(seed, defect=None, size=(480, 480)):
    rng = np.random.default_rng(seed)
    h, w = size
    img, box = _product(rng, h, w)
    bbox = DEFECTS[defect](rng, img, box) if defect else None
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)  # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    label = None
    if bbox:
        x, y, bw, bh = bbox
        label = {"x": x / w, "y": y / h, "w": bw / w, "h": bh / h}
    return img, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--n", type=int, default=40, help="images per class (intact / defective)")
    ap.add_argument("--demo", action="store_true", help="write a small named demo set")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)

    if args.demo:
        out.mkdir(parents=True, exist_ok=True)
        plan = [("intact_1", None), ("intact_2", None), ("crack", "crack"), ("scratch", "scratch"), ("stain", "stain")]
        for i, (name, d) in enumerate(plan):
            img, _ = make_image(args.seed * 1000 + i, d)
            cv2.imwrite(str(out / f"{name}.png"), img)
        print(f"Wrote {len(plan)} demo images to {out}")
        return

    labels = {}
    (out / "intact").mkdir(parents=True, exist_ok=True)
    (out / "defective").mkdir(parents=True, exist_ok=True)
    kinds = list(DEFECTS)
    for i in range(args.n):
        img, _ = make_image(args.seed * 100000 + i, None)
        name = f"intact/intact_{i:03d}.png"
        cv2.imwrite(str(out / name), img)
        labels[name] = {"label": "intact", "bbox": None, "type": None}
    for i in range(args.n):
        kind = kinds[i % len(kinds)]
        img, bbox = make_image(args.seed * 100000 + 50000 + i, kind)
        name = f"defective/{kind}_{i:03d}.png"
        cv2.imwrite(str(out / name), img)
        labels[name] = {"label": "defective", "bbox": bbox, "type": kind}
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"Wrote {len(labels)} images + labels.json to {out}")


if __name__ == "__main__":
    main()
