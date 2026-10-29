"""Audit a Roboflow YOLO export before using it for training.

Reports classes, per-split image and instance counts, polygon vs box rows, how many
original photos remain after grouping Roboflow's augmented copies, and overlap with a
reference dataset (exact bytes and near-duplicates by perceptual hash), split by the
reference split so leaks into our valid/test are visible.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/audit_dataset.py CV/datasets/extra_1/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

CV_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = CV_DIR / "datasets" / "defect_clothes_v2" / "prepared"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: robust to resizing and recompression, not to rotation or crops."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def images_by_split(root: Path) -> dict[str, list[Path]]:
    out = {}
    for split in ("train", "valid", "test"):
        d = root / split / "images"
        if d.is_dir():
            out[split] = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Folder containing data.yaml and train/valid/test")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Dataset to check overlap against")
    parser.add_argument("--near", type=int, default=6, help="Max dHash bit difference counted as a near-duplicate")
    args = parser.parse_args()

    cfg = yaml.safe_load((args.dataset / "data.yaml").read_text())
    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]
    report: dict = {"dataset": args.dataset.as_posix(), "classes": names, "roboflow": cfg.get("roboflow"), "splits": {}}

    new_images = images_by_split(args.dataset)
    for split, imgs in new_images.items():
        inst, rows, empty = Counter(), Counter(), 0
        stems = set()
        for img in imgs:
            stems.add(img.name.split(".rf.")[0])
            label = args.dataset / split / "labels" / (img.stem + ".txt")
            lines = [l.split() for l in label.read_text().splitlines() if l.strip()] if label.exists() else []
            empty += not lines
            for parts in lines:
                inst[names[int(parts[0])]] += 1
                rows["box" if len(parts) == 5 else "polygon"] += 1
        report["splits"][split] = {"images": len(imgs), "original_photos": len(stems), "empty_labels": empty,
                                   "instances": dict(inst), "label_rows": dict(rows)}

    # Overlap with the reference dataset.
    ref_bytes, ref_hash = {}, []
    for split, imgs in images_by_split(args.reference).items():
        for img in imgs:
            ref_bytes[hashlib.md5(img.read_bytes()).hexdigest()] = split
            ref_hash.append((dhash(img), split))
    ref_h = np.array([h for h, _ in ref_hash], dtype=np.uint64)
    ref_s = [s for _, s in ref_hash]

    overlap = defaultdict(Counter)
    for split, imgs in new_images.items():
        for img in imgs:
            md5 = hashlib.md5(img.read_bytes()).hexdigest()
            if md5 in ref_bytes:
                overlap[split][f"exact_in_ref_{ref_bytes[md5]}"] += 1
                continue
            dist = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(ref_h, np.uint64(dhash(img)))])
            j = int(dist.argmin())
            if dist[j] <= args.near:
                overlap[split][f"near_dup_in_ref_{ref_s[j]}"] += 1
    report["overlap_with_reference"] = {k: dict(v) for k, v in overlap.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
