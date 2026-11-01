"""Score a checkpoint and report defect-only metrics (hole + stain, excluding clothes).

Reports mAP (box and mask) plus photo-level rates, which match how ReturnGuard
uses the detector: per class, the share of photos containing that defect that the
model flags (recall), and the share of photos without it that it falsely flags.

Use --split valid while developing. Run --split test once, after the model and
thresholds are locked.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/evaluate.py --weights CV/runs/yolo11s_seg/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

CV_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = CV_DIR / "datasets" / "defect_clothes_v2" / "prepared" / "data.yaml"
DEFECT_CLASSES = ("hole", "stain")


def photo_level(model: YOLO, data: Path, split: str, conf: float, imgsz: int, device: str) -> dict[str, dict[str, float]]:
    cfg = yaml.safe_load(data.read_text())
    root = Path(cfg.get("path") or data.parent)
    image_dir = root / cfg["val" if split == "valid" else "test"]
    label_dir = image_dir.parent / "labels"
    ids = {name: i for i, name in model.names.items()}
    counts = {n: {"pos": 0, "hit": 0, "neg": 0, "false_flag": 0} for n in DEFECT_CLASSES}
    for img in sorted(image_dir.iterdir()):
        label = label_dir / (img.stem + ".txt")
        truth = {int(l.split()[0]) for l in label.read_text().splitlines() if l.strip()} if label.exists() else set()
        result = model.predict(str(img), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        flagged = {int(c) for c in result.boxes.cls.tolist()}
        for n in DEFECT_CLASSES:
            c = counts[n]
            if ids[n] in truth:
                c["pos"] += 1
                c["hit"] += ids[n] in flagged
            else:
                c["neg"] += 1
                c["false_flag"] += ids[n] in flagged
    return {
        n: {
            "photos_with_defect": c["pos"],
            "photo_recall": c["hit"] / c["pos"] if c["pos"] else float("nan"),
            "photos_without_defect": c["neg"],
            "photo_false_flag_rate": c["false_flag"] / c["neg"] if c["neg"] else float("nan"),
        }
        for n, c in counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--photo-conf", type=float, default=0.25, help="Confidence threshold for photo-level rates")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split="val" if args.split == "valid" else "test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(CV_DIR / "runs" / "eval"),
        name=f"{args.weights.parent.parent.name}_{args.split}",
        exist_ok=True,
    )

    names = model.names
    results: dict[str, dict[str, float]] = {}
    for kind in ("box", "seg"):
        m = getattr(metrics, kind, None)
        if m is None:
            continue
        per_class = {names[c]: {"mAP50": float(m.ap50[i]), "mAP50-95": float(m.ap[i])} for i, c in enumerate(m.ap_class_index)}
        defects = [per_class[n] for n in DEFECT_CLASSES if n in per_class]
        results[kind] = {
            **{f"{n}_mAP50": v["mAP50"] for n, v in per_class.items()},
            "defect_mAP50": sum(d["mAP50"] for d in defects) / len(defects),
            "defect_mAP50-95": sum(d["mAP50-95"] for d in defects) / len(defects),
        }
    results[f"photo@conf{args.photo_conf}"] = photo_level(model, args.data, args.split, args.photo_conf, args.imgsz, args.device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
