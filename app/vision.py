"""Vision evidence: decide whether a product photo shows a defect, and where.

Three backends return the same structure. Clothing orders use the trained clothes model when its
weights exist; everything else uses heuristic or clip:

* clothes-yolo - two YOLO models trained on garment photos (hole, stain), from CV/scripts/predict.py.
                 Needs ultralytics and the weights in CV/runs/deploy.

* heuristic  - classical CV (Otsu thresholding, morphology, local colour/contrast anomaly,
               connected components). No model download, runs anywhere. Great on clean photos,
               weak on busy backgrounds or textured products.
* clip       - zero-shot CLIP (ViT) scored on the whole photo and on a tile grid. No training
               needed; handles real-world photos better. Needs torch + transformers.

Result schema (regions use coordinates normalised to 0-1 so any UI size works):
    {backend, label: defective|intact|inconclusive, confidence, defect_type,
     regions: [{x, y, w, h, score, type}], note}
"""
import logging
from functools import lru_cache

import cv2
import numpy as np

from . import config

log = logging.getLogger("returnguard.vision")
MAX_SIDE = 640


def _load(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image file")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- heuristic backend
def _product_mask(gray):
    """Segment the product from the background: Otsu threshold -> close -> biggest blob, holes filled."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    if border.mean() > 127:  # background came out white -> invert so product is foreground
        m = cv2.bitwise_not(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * m.size:
        return None
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 255, -1)
    return filled


def analyze_heuristic(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _product_mask(gray)
    if mask is None:
        return {"backend": "heuristic", "label": "inconclusive", "confidence": 0.0, "defect_type": None,
                "regions": [], "note": "Could not find a product in the photo."}

    # Ignore the product outline: erode so the silhouette is not mistaken for damage.
    r = max(5, int(0.03 * min(h, w)))
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    if interior.sum() < 0.02 * h * w:
        return {"backend": "heuristic", "label": "inconclusive", "confidence": 0.0, "defect_type": None,
                "regions": [], "note": "Product too small in the photo."}

    # Expected appearance = large median filter. A median keeps big flat panels and step edges
    # (labels, shading) but erases thin cracks/scratches and small stains, so
    # |pixel - median| lights up exactly the local damage.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    filled = lab.copy()
    filled[~(mask > 0)] = np.median(lab[interior], axis=0).astype(np.uint8)  # background -> neutral
    ksize = int(max(15, (min(h, w) // 16) | 1))
    expected = cv2.medianBlur(filled, ksize)
    local = cv2.GaussianBlur(lab, (0, 0), 1.0)
    dev = np.linalg.norm(local.astype(np.float32) - expected.astype(np.float32), axis=2)

    # Median filters round off the corners of flat panels, which looks like damage there.
    # Mask a thin band around strong edges of the *expected* image to avoid those false alarms.
    gx = cv2.Sobel(expected.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(expected.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.sqrt((gx ** 2 + gy ** 2).sum(axis=2)) > 60).astype(np.uint8)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize // 2 | 1, ksize // 2 | 1)))
    valid = interior & (edge == 0)
    dev[~valid] = 0

    vals = dev[valid] if valid.any() else dev[interior]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    thr = max(med + 6 * mad, 9.0)  # floor stops sensor noise firing on clean products

    anomaly = ((dev > thr) & valid).astype(np.uint8) * 255
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly, connectivity=8)

    min_area = max(30, 0.0015 * interior.sum())
    L, L_exp = lab[..., 0].astype(np.float32), expected[..., 0].astype(np.float32)
    regions = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        comp = labels == i
        strength = float(dev[comp].mean())
        score = float(np.clip(strength / (2.5 * thr), 0.0, 1.0))
        elong = max(bw, bh) / max(1, min(bw, bh))
        fill = area / max(1, bw * bh)
        darker = float((L[comp] - L_exp[comp]).mean()) < 0
        # Stains shift colour (chroma) while cracks/scratches mostly change brightness, so the
        # chroma-to-brightness ratio tells them apart. Thickness separates cracks from other damage.
        diff = local[comp].astype(np.float32) - expected[comp].astype(np.float32)
        chroma_ratio = float(np.linalg.norm(diff[:, 1:], axis=1).mean() / (np.abs(diff[:, 0]).mean() + 1e-6))
        thickness = float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3).max())
        if chroma_ratio > 0.9:
            dtype = "stain"
        elif thickness < 0.015 * min(h, w):
            dtype = "crack" if darker else "scratch"
        else:
            dtype = "surface damage"
        regions.append({"x": x / w, "y": y / h, "w": bw / w, "h": bh / h, "score": round(score, 3), "type": dtype})
    regions.sort(key=lambda r: r["score"], reverse=True)
    regions = regions[:5]

    if not regions:
        p99 = float(np.percentile(vals, 99.5))
        conf = float(np.clip(1.0 - p99 / (2 * thr), 0.5, 0.99))
        return {"backend": "heuristic", "label": "intact", "confidence": round(conf, 3), "defect_type": None,
                "regions": [], "note": "No local anomaly found on the product surface."}

    top = regions[0]
    if top["score"] >= 0.4:
        return {"backend": "heuristic", "label": "defective", "confidence": round(min(0.99, 0.5 + top["score"] / 2), 3),
                "defect_type": top["type"], "regions": regions, "note": f"{len(regions)} anomalous region(s) found."}
    return {"backend": "heuristic", "label": "inconclusive", "confidence": round(top["score"], 3),
            "defect_type": top["type"], "regions": regions, "note": "Only weak anomalies found; a human should look."}


# --------------------------------------------------------------------------- CLIP backend
INTACT_PROMPTS = ["a photo of a product in perfect condition", "a photo of a brand new undamaged item"]
DEFECT_PROMPTS = {
    "crack": "a photo of a product with a crack",
    "scratch": "a photo of a product with a scratch",
    "stain": "a photo of a product with a stain",
    "dent": "a photo of a dented product",
    "tear": "a photo of a torn or ripped product",
}


class ClipBackend:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
        self.proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        prompts = INTACT_PROMPTS + list(DEFECT_PROMPTS.values())
        with torch.no_grad():
            tok = self.proc(text=prompts, return_tensors="pt", padding=True)
            t = self.model.get_text_features(**tok)
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.n_intact = len(INTACT_PROMPTS)

    def _probs(self, crops):
        """Return per-crop (p_defect, best_defect_type)."""
        from PIL import Image

        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with self.torch.no_grad():
            inp = self.proc(images=pil, return_tensors="pt")
            f = self.model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100.0 * f @ self.text.T
            intact = logits[:, : self.n_intact].mean(dim=1, keepdim=True)
            merged = self.torch.cat([intact, logits[:, self.n_intact:]], dim=1)
            p = merged.softmax(dim=1).numpy()
        types = list(DEFECT_PROMPTS)
        return [(float(1 - row[0]), types[int(np.argmax(row[1:]))]) for row in p]

    def analyze(self, img):
        h, w = img.shape[:2]
        boxes = [(0, 0, w, h)]
        for g in (2, 3):  # tile grids give coarse localisation without any training
            for gy in range(g):
                for gx in range(g):
                    boxes.append((gx * w // g, gy * h // g, w // g, h // g))
        crops = [img[y:y + bh, x:x + bw] for x, y, bw, bh in boxes]
        res = self._probs(crops)
        whole_p, whole_type = res[0]
        tiles = sorted(
            [(p, t, b) for (p, t), b in zip(res[1:], boxes[1:])], key=lambda r: r[0], reverse=True
        )
        regions = [
            {"x": b[0] / w, "y": b[1] / h, "w": b[2] / w, "h": b[3] / h, "score": round(p, 3), "type": t}
            for p, t, b in tiles[:3] if p >= 0.6
        ]
        peak = max([whole_p] + [t[0] for t in tiles])
        if whole_p >= 0.6 or peak >= 0.85:
            label, conf = "defective", max(whole_p, peak)
        elif whole_p <= 0.35 and peak < 0.6:
            label, conf = "intact", 1 - max(whole_p, peak)
        else:
            label, conf = "inconclusive", 1 - abs(whole_p - 0.5) * 2
        return {"backend": "clip", "label": label, "confidence": round(float(conf), 3),
                "defect_type": whole_type if label != "intact" else None, "regions": regions,
                "note": "Zero-shot CLIP scores; thresholds should be calibrated on pilot data."}


@lru_cache(maxsize=1)
def _clip():
    return ClipBackend()


def clip_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def active_backend() -> str:
    if config.VISION_BACKEND in ("clip", "heuristic"):
        return config.VISION_BACKEND
    return "clip" if clip_available() else "heuristic"


# --------------------------------------------------------------------------- clothes backend
# Trained two-model YOLO predictor from CV/scripts/predict.py (hole and stain on garments).
CLOTHES_LABELS = {"damaged": "defective", "unsure": "inconclusive", "clean": "intact", "no_garment": "inconclusive"}
VERDICT_FROM_LABEL = {"defective": "damaged", "inconclusive": "unsure", "intact": "clean"}
VERDICT_RANK = {"damaged": 3, "unsure": 2, "clean": 1, "no_garment": 0}


def _contained(inner: dict, outer: dict) -> float:
    """Share of `inner`'s area that lies inside `outer` (boxes normalised 0-1)."""
    w = min(inner["x"] + inner["w"], outer["x"] + outer["w"]) - max(inner["x"], outer["x"])
    h = min(inner["y"] + inner["h"], outer["y"] + outer["h"]) - max(inner["y"], outer["y"])
    return max(0.0, w) * max(0.0, h) / max(inner["w"] * inner["h"], 1e-9)


def clean_regions(regions: list[dict]) -> list[dict]:
    """Tidy the boxes shown to people and sent to the vision check (the damaged/clean verdict is unchanged).

    On real photos the model often adds very large, low-scoring boxes over half a sleeve while the real
    hole gets a small box with a lower score, so a higher score cut-off would hide the wrong one. Instead:
    drop boxes covering more than REGION_MAX_AREA of the photo, drop boxes mostly inside a stronger one,
    and keep boxes from REGION_MIN_SCORE up. If only weaker boxes are left, keep the best one marked as weak.
    If every box is oversized, none is kept (the vision check then looks at the whole photo).
    """
    ordered = sorted(regions, key=lambda r: r["score"], reverse=True)
    sized = [r for r in ordered if r["w"] * r["h"] <= config.REGION_MAX_AREA]
    kept = []
    for r in sized:
        if all(_contained(r, k) < 0.6 for k in kept):
            kept.append(r)
    strong = [r for r in kept if r["score"] >= config.REGION_MIN_SCORE]
    out = strong or kept[:1]
    return [{**r, "weak": r["score"] < config.REGION_STRONG_SCORE} for r in out[:5]]


class ClothesBackend:
    def __init__(self):
        import importlib.util
        import os
        import sys

        if config.CLOTHES_SECOND_OPINION:  # transformers lives in CV/.pydeps, apart from the app venv
            os.environ.setdefault("HF_HOME", str(config.CV_DIR / ".hf_cache"))
            sys.path.append(str(config.CV_DIR / ".pydeps"))
        spec = importlib.util.spec_from_file_location("cv_predict", config.CV_DIR / "scripts" / "predict.py")
        predict = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict)

        device = config.CLOTHES_DEVICE
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.predictor = predict.DefectPredictor(config.CLOTHES_DETECTOR, config.CLOTHES_LOCATOR, device=device,
                                                 second_opinion=config.CLOTHES_SECOND_OPINION)

    def analyze(self, img, tiles: bool = False):
        r = self._tiled(img) if tiles else self.predictor.predict(img)
        score = r["damage_score"]
        # The predictor's thresholds are low (damaged from 0.2), so map scores onto the 0.5-1 range the
        # decision rules expect; unsure and no-garment photos always go to a human.
        conf = {"damaged": 0.5 + score / 2, "clean": 1 - score}.get(r["verdict"], score)
        regions = clean_regions([{k: reg[k] for k in ("x", "y", "w", "h", "score", "type")} for reg in r["regions"]])
        return {"backend": "clothes-yolo", "mode": "tiles" if tiles else "full", "verdict": r["verdict"],
                "label": CLOTHES_LABELS[r["verdict"]], "confidence": round(float(conf), 3),
                "damage_score": score, "detector_score": r["detector_score"], "locator_score": r["locator_score"],
                "garment_found": r["garment_score"] is not None or tiles and r.get("garment_found", False),
                "defect_type": r["defect_type"], "regions": regions, "note": r["note"]}

    def _tiled(self, img):
        """Retry at higher zoom: run the predictor on four overlapping quarters of the photo and map the
        regions back. Used when the first pass is unsure or the claim and the first pass disagree."""
        h, w = img.shape[:2]
        tw, th = int(w * 0.6), int(h * 0.6)
        best, regions, garment = None, [], False
        for x0 in (0, w - tw):
            for y0 in (0, h - th):
                r = self.predictor.predict(img[y0:y0 + th, x0:x0 + tw])
                garment = garment or r["garment_score"] is not None
                for reg in r["regions"]:
                    regions.append({**reg, "x": (x0 + reg["x"] * tw) / w, "y": (y0 + reg["y"] * th) / h,
                                    "w": reg["w"] * tw / w, "h": reg["h"] * th / h})
                if best is None or (VERDICT_RANK[r["verdict"]], r["damage_score"]) > (VERDICT_RANK[best["verdict"]], best["damage_score"]):
                    best = r
        regions.sort(key=lambda reg: reg["score"], reverse=True)
        verdict = best["verdict"] if best["verdict"] != "no_garment" or not garment else "clean"
        return {**best, "verdict": verdict, "regions": regions[:15], "garment_found": garment,
                "defect_type": regions[0]["type"] if regions and verdict in ("damaged", "unsure") else None}


@lru_cache(maxsize=1)
def _clothes():
    return ClothesBackend()


def clothes_available() -> bool:
    if config.CLOTHES_MODEL == "off" or not config.CLOTHES_DETECTOR.is_file():
        return False
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def uses_clothes_model(category: str | None) -> bool:
    return (category or "").lower() in config.CLOTHING_CATEGORIES and clothes_available()


def _with_verdict(out: dict) -> dict:
    """Give heuristic/CLIP results the same fields as the clothes model."""
    out.setdefault("verdict", VERDICT_FROM_LABEL.get(out["label"], "unsure"))
    out.setdefault("garment_found", True)
    out.setdefault("mode", "full")
    return out


def analyze(path, backend: str | None = None, category: str | None = None, tiles: bool = False) -> dict:
    if backend in (None, "clothes") and uses_clothes_model(category):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # full resolution: the predictor zooms into the garment
        if img is None:
            raise ValueError("Could not read the image file")
        try:
            return _clothes().analyze(img, tiles=tiles)
        except Exception as exc:
            log.warning("Clothes model failed (%s); falling back to %s backend", exc, active_backend())
            out = analyze(path, backend=active_backend())
            out["note"] = f"Clothes model unavailable ({type(exc).__name__}). " + out["note"]
            return out
    img = _load(path)
    backend = backend or active_backend()
    if backend == "clip":
        try:
            return _with_verdict(_clip().analyze(img))
        except Exception as exc:  # model download blocked, out of memory, ...
            log.warning("CLIP failed (%s); falling back to heuristic backend", exc)
            out = analyze_heuristic(img)
            out["note"] = f"CLIP unavailable ({type(exc).__name__}); used classical CV. " + out["note"]
            return _with_verdict(out)
    return _with_verdict(analyze_heuristic(img))
