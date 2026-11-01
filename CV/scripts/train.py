"""Fine-tune a COCO-pretrained YOLO segmentation model on the prepared dataset.

Validates on the valid split only; the test split is never touched here.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/train.py
    .venv/Scripts/python.exe CV/scripts/train.py --model yolo11s-seg.pt --epochs 150 --name yolo11s_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

CV_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = CV_DIR / "datasets" / "defect_clothes_v2" / "prepared" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11s_seg")
    parser.add_argument("--project", type=Path, default=CV_DIR / "runs")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Random zoom range (+/- gain); higher values show defects larger, like close-ups")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"{args.data} not found; run CV/scripts/prepare.py first")

    YOLO(args.model).train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        cos_lr=True,
        # Roboflow already baked in flips/rot90/crop/shear; add vertical flip and
        # copy-paste, which helps small instance-segmentation objects.
        flipud=0.5,
        copy_paste=0.3,
        scale=args.scale,
        plots=True,
    )
    print(f"Best checkpoint: {args.project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
