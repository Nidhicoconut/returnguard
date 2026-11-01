"""Two-model clothing-defect predictor for single photos.

- The detector (YOLO11s box model trained on our data plus defect-clothes-ukrks v3) decides
  whether the photo shows damage. It flagged none of the 14 clean validation photos at 0.2.
- The locator (YOLO11m box model, same training data) marks where the damage is. It finds the
  most defects per wrong box (defect mAP@50 0.36) but flags some clean garments, so it only
  runs when the detector says the photo is (possibly) damaged.
- On photos larger than 1000 px, both models also run on a crop of the garment, so small defects
  survive the resize to 640 px. On 640 px photos the crop pass only added false alarms.

Thresholds come from the validation comparison in CV/WORKING_CONTEXT.md.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/predict.py path/to/photo.jpg [more.jpg | folder ...] [--out CV/runs/predict]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CV_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = CV_DIR / "runs" / "deploy" / "clothes_damage_detector.pt"
DEFAULT_LOCATOR = CV_DIR / "runs" / "deploy" / "clothes_damage_locator.pt"
GARMENT, DEFECTS = "clothes", ("hole", "stain")
COLORS = {"hole": (60, 60, 235), "stain": (200, 60, 220)}  # BGR
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
GDINO_PROMPT = "a hole in fabric. a stain on fabric."


class GroundingDino:
    """Open-vocabulary detector (needs `transformers`, kept in CV/.pydeps; weights cached in CV/.hf_cache)."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(self.device).eval()

    def detect(self, pil, threshold: float) -> list[dict]:
        """Return defect-like boxes (label mentions hole or stain) in pil pixel coordinates."""
        small = pil.copy()
        small.thumbnail((1024, 1024))
        sx, sy = pil.width / small.width, pil.height / small.height
        inputs = self.proc(images=small, text=GDINO_PROMPT, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[small.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = []
        for (x1, y1, x2, y2), s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            if "hole" in str(l) or "stain" in str(l):
                dets.append({"score": float(s), "xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy], "label": str(l)})
        return dets


class DefectPredictor:
    def __init__(
        self,
        detector: Path = DEFAULT_DETECTOR,
        locator: Path | None = DEFAULT_LOCATOR,
        device: str = "0",
        damage_conf: float = 0.2,
        unsure_conf: float = 0.15,
        region_conf: float = 0.15,
        locator_damage_conf: float = 0.4,
        locator_unsure_conf: float = 0.25,
        max_regions: int = 15,
        garment_crop: bool = True,
        second_opinion: bool = False,
        second_opinion_conf: float = 0.5,
    ):
        self.max_regions, self.garment_crop = max_regions, garment_crop
        # Grounding DINO cannot separate clean from damaged on its own, so it can only turn a
        # "clean" verdict into "unsure" (validated: no damaged-photo change, 2 more clean -> unsure).
        self.gdino = GroundingDino(device) if second_opinion else None
        self.second_opinion_conf = second_opinion_conf
        self.locator_damage_conf, self.locator_unsure_conf = locator_damage_conf, locator_unsure_conf
        self.detector = YOLO(str(detector))
        self.locator = YOLO(str(locator)) if locator and Path(locator).is_file() else None
        self.device, self.damage_conf, self.unsure_conf, self.region_conf = device, damage_conf, unsure_conf, region_conf

    def _run(self, model: YOLO, img: np.ndarray, conf: float, offset=(0, 0)) -> list[dict]:
        r = model.predict(img, conf=conf, imgsz=640, device=self.device, verbose=False)[0]
        ox, oy = offset
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        dets = []
        for box, c, s, poly in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), polys):
            dets.append({
                "type": model.names[int(c)],
                "score": float(s),
                "xyxy": [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy],
                "polygon": [[float(x) + ox, float(y) + oy] for x, y in poly] if poly is not None and len(poly) else None,
            })
        return dets

    def _with_garment_crop(self, model: YOLO, img: np.ndarray, conf: float) -> tuple[list[dict], dict | None]:
        """Run on the whole photo, then on the garment crop when the garment fills under 70% of the photo."""
        h, w = img.shape[:2]
        dets = self._run(model, img, conf)
        garments = sorted((d for d in dets if d["type"] == GARMENT), key=lambda d: d["score"], reverse=True)
        garment = garments[0] if garments else None
        if garment:
            x1, y1, x2, y2 = garment["xyxy"]
            mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
            cx1, cy1, cx2, cy2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))
            # Only worth it when the photo is large enough that the 640 px resize loses detail.
            if self.garment_crop and max(h, w) > 1000 and (cx2 - cx1) * (cy2 - cy1) < 0.7 * w * h:
                dets += [d for d in self._run(model, img[cy1:cy2, cx1:cx2], conf, (cx1, cy1)) if d["type"] != GARMENT]
            # Damage must be on the garment: drop detections centred on skin, faces or background.
            dets = [d for d in dets if cx1 <= (d["xyxy"][0] + d["xyxy"][2]) / 2 <= cx2
                    and cy1 <= (d["xyxy"][1] + d["xyxy"][3]) / 2 <= cy2]
        return [d for d in dets if d["type"] in DEFECTS], garment

    @staticmethod
    def _merge(dets: list[dict]) -> list[dict]:
        kept = []
        for d in sorted(dets, key=lambda d: d["score"], reverse=True):
            if all(k["type"] != d["type"] or _iou(k["xyxy"], d["xyxy"]) < 0.5 for k in kept):
                kept.append(d)
        return kept

    def predict(self, img: np.ndarray) -> dict:
        """img is a BGR array (cv2.imread). Returns the verdict and defect regions (normalised 0-1)."""
        h, w = img.shape[:2]
        found, garment = self._with_garment_crop(self.detector, img, self.unsure_conf)
        located = self._with_garment_crop(self.locator, img, self.region_conf)[0] if self.locator else []
        top = max((d["score"] for d in found), default=0.0)
        loc_top = max((d["score"] for d in located), default=0.0)
        best = lambda dets, c: max((d["score"] for d in dets if d["type"] == c), default=0.0)

        # Both models vote: the detector rarely flags clean garments, the locator sees more damage
        # on real photos. Either one being confident is enough (validated: 26/29 damaged, 0/14 clean).
        if top >= self.damage_conf or loc_top >= self.locator_damage_conf:
            verdict = "damaged"
        elif top >= self.unsure_conf or loc_top >= self.locator_unsure_conf:
            verdict = "unsure"
        else:
            verdict = "clean" if garment else "no_garment"

        regions = []
        if verdict in ("damaged", "unsure"):
            regions = self._merge([d for d in found if d["score"] >= self.unsure_conf] + located)[: self.max_regions]

        second = []
        if verdict == "clean" and self.gdino:
            from PIL import Image

            for d in self.gdino.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), self.second_opinion_conf):
                x1, y1, x2, y2 = d["xyxy"]
                on_garment = garment is None or (garment["xyxy"][0] <= (x1 + x2) / 2 <= garment["xyxy"][2]
                                                 and garment["xyxy"][1] <= (y1 + y2) / 2 <= garment["xyxy"][3])
                if on_garment and (x2 - x1) * (y2 - y1) <= 0.25 * w * h:  # skip whole-garment boxes
                    kind = "stain" if "stain" in d["label"] else "hole"
                    second.append({"type": kind, "score": d["score"], "xyxy": d["xyxy"], "polygon": None})
            if second:
                verdict, regions = "unsure", self._merge(second)[: self.max_regions]

        def norm(d: dict) -> dict:
            x1, y1, x2, y2 = d["xyxy"]
            return {"type": d["type"], "score": round(d["score"], 3),
                    "x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h,
                    "polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in d["polygon"]] if d["polygon"] else None}

        notes = {
            "damaged": f"Damage found ({len(regions)} region(s)).",
            "unsure": ("The second-opinion model sees possible damage that the main models missed. "
                       "A closer, well-lit photo of the marked area would help." if second else
                       "Weak signs of damage. A closer, well-lit photo of the damaged area would help."),
            "clean": "No hole or stain found on the garment.",
            "no_garment": "No garment found. Photograph the whole item, then the damage close up.",
        }
        return {
            "verdict": verdict,
            "damage_score": round(max(top, loc_top), 3),
            "detector_score": round(top, 3),
            "locator_score": round(loc_top, 3),
            "second_opinion_score": round(max((d["score"] for d in second), default=0.0), 3) if self.gdino else None,
            # Highest score per defect class from either model.
            "class_scores": {c: round(max(best(found, c), best(located, c)), 3) for c in DEFECTS},
            "defect_types": [c for c in DEFECTS
                             if best(found, c) >= self.damage_conf or best(located, c) >= self.locator_damage_conf],
            "defect_type": regions[0]["type"] if regions else None,
            "regions": [norm(d) for d in regions],
            "garment_score": round(garment["score"], 3) if garment else None,
            "note": notes[verdict],
        }


def draw(img: np.ndarray, result: dict) -> np.ndarray:
    """Return a copy of the BGR image with masks, boxes, labels and a verdict banner."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    overlay = out.copy()
    for r in result["regions"]:
        if r["polygon"]:
            pts = np.array([[x * w, y * h] for x, y in r["polygon"]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], COLORS[r["type"]])
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    font_scale = max(0.5, min(h, w) / 900)
    for i, r in enumerate(result["regions"]):  # regions are sorted by score
        x1, y1 = int(r["x"] * w), int(r["y"] * h)
        x2, y2 = int((r["x"] + r["w"]) * w), int((r["y"] + r["h"]) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[r["type"]], thick)
        if i >= 8:  # label only the strongest regions to keep the photo readable
            continue
        label = f"{r['type']} {r['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 - 4 if y1 - th - 8 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    banner_color = {"damaged": (40, 40, 200), "unsure": (0, 150, 220), "clean": (60, 150, 40), "no_garment": (90, 90, 90)}
    text = f"{result['verdict'].upper()}  (damage score {result['damage_score']:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), banner_color[result["verdict"]], -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Photos or folders of photos")
    parser.add_argument("--out", type=Path, default=CV_DIR / "runs" / "predict")
    parser.add_argument("--device", default="0")
    parser.add_argument("--second-opinion", action="store_true",
                        help="Use Grounding DINO to turn missed 'clean' photos into 'unsure' "
                             "(run with PYTHONPATH=CV/.pydeps and HF_HOME=CV/.hf_cache)")
    args = parser.parse_args()

    files = []
    for p in args.inputs:
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES) if p.is_dir() else [p]
    predictor = DefectPredictor(device=args.device, second_opinion=args.second_opinion)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"Skipped {f}: not a readable image")
            continue
        result = predictor.predict(img)
        cv2.imwrite(str(args.out / f"{f.stem}_pred.jpg"), draw(img, result))
        (args.out / f"{f.stem}_pred.json").write_text(json.dumps(result, indent=2))
        print(f"{f.name}: {result['verdict']} (score {result['damage_score']}, {len(result['regions'])} regions)")
    print(f"Outputs in {args.out}")


if __name__ == "__main__":
    main()
