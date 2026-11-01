"""Photo-level check of the full predictor (predict.py) on a labelled split.

Counts how damaged and clean photos are classified (damaged / unsure / clean / no garment).

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/evaluate_predictor.py
    PYTHONPATH=CV/.pydeps HF_HOME=CV/.hf_cache .venv/Scripts/python.exe CV/scripts/evaluate_predictor.py --second-opinion
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from predict import CV_DIR, DEFAULT_DETECTOR, DEFAULT_LOCATOR, DefectPredictor

VERDICTS = ("damaged", "unsure", "clean", "no_garment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=CV_DIR / "datasets" / "defect_clothes_v2" / "prepared" / "valid")
    parser.add_argument("--device", default="0")
    parser.add_argument("--second-opinion", action="store_true")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--locator", type=Path, default=DEFAULT_LOCATOR)
    args = parser.parse_args()

    predictor = DefectPredictor(args.detector, args.locator, device=args.device, second_opinion=args.second_opinion)
    counts = Counter()
    for img in sorted((args.split_dir / "images").iterdir()):
        label = args.split_dir / "labels" / (img.stem + ".txt")
        damaged = any(l.split()[0] in ("1", "2") for l in label.read_text().splitlines() if l.strip())
        counts[("damaged" if damaged else "clean", predictor.predict(cv2.imread(str(img)))["verdict"])] += 1

    for truth in ("damaged", "clean"):
        row = {v: counts[(truth, v)] for v in VERDICTS}
        print(f"{truth} photos ({sum(row.values())}): {row}")


if __name__ == "__main__":
    main()
