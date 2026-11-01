"""Build a box-detection dataset: our prepared split plus extra Roboflow exports (train only).

- Our prepared train/valid/test are kept as they are, with polygons converted to boxes.
  Valid and test stay exactly our photos, so scores remain comparable with earlier runs.
- Extra datasets only add training images. An extra image is dropped when:
    * its original filename (stem before ".rf.") matches one of our photos, or
    * it is an exact or near duplicate (dHash) of any copy of our photos, including
      Roboflow's augmented copies in the raw export, or
    * it contains a class that is not mapped (e.g. "cut", which marks weave flaws).
- Extra class names are mapped onto clothes / hole / stain.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/prepare_merged.py --extra CV/datasets/extra_1/raw
    .venv/Scripts/python.exe CV/scripts/prepare_merged.py --out CV/datasets/ours_det        # no extras: baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from audit_dataset import IMAGE_SUFFIXES, dhash

CV_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = CV_DIR / "datasets"
CLASS_NAMES = ["clothes", "hole", "stain"]
CLASS_MAP = {"clothes": "clothes", "hole": "hole", "stain": "stain"}  # extra name -> our name; others are dropped
SPLITS = ("train", "valid", "test")


def stem(p: Path) -> str:
    return p.name.split(".rf.")[0]


def to_box_row(parts: list[str], cls: int) -> str:
    coords = [float(v) for v in parts[1:]]
    if len(coords) == 4:
        cx, cy, w, h = coords
    else:
        xs, ys = coords[0::2], coords[1::2]
        x1, x2, y1, y2 = max(0.0, min(xs)), min(1.0, max(xs)), max(0.0, min(ys)), min(1.0, max(ys))
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=DATASET_DIR / "defect_clothes_v2" / "prepared")
    parser.add_argument("--ours-raw", type=Path, default=DATASET_DIR / "defect_clothes_v2" / "raw",
                        help="Raw export of our dataset; all its copies of our photos are used for duplicate checks")
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="Extra Roboflow YOLO exports (folders with data.yaml)")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "merged_det")
    parser.add_argument("--near", type=int, default=6, help="Max dHash bit difference counted as a duplicate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()

    if args.out.exists():
        if not args.overwrite:
            parser.error(f"{args.out} exists; pass --overwrite to rebuild it")
        shutil.rmtree(args.out)
    for split in SPLITS:
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)

    report: dict = {"ours": args.ours.as_posix(), "extra": {}, "splits": {s: Counter() for s in SPLITS}}

    def write(img: Path, rows: list[str], split: str, prefix: str = "") -> None:
        name = prefix + img.name
        shutil.copy2(img, args.out / split / "images" / name)
        (args.out / split / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        report["splits"][split]["images"] += 1
        report["splits"][split].update(CLASS_NAMES[int(r.split()[0])] for r in rows)

    # Our data, polygons -> boxes.
    our_stems = set()
    for split in SPLITS:
        for img in images(args.ours / split / "images"):
            our_stems.add(stem(img))
            label = args.ours / split / "labels" / (img.stem + ".txt")
            rows = [to_box_row(l.split(), int(l.split()[0])) for l in label.read_text().splitlines() if l.strip()]
            write(img, rows, split)

    # Every copy of our photos (prepared + all raw augmented copies) for duplicate checks.
    known = [p for s in SPLITS for p in images(args.ours / s / "images")]
    known += [p for s in SPLITS for p in images(args.ours_raw / s / "images") if stem(p) in our_stems]
    known_md5 = {hashlib.md5(p.read_bytes()).hexdigest() for p in known}
    known_hash = np.array([dhash(p) for p in known], dtype=np.uint64)
    seen_md5 = set(known_md5)

    for k, extra in enumerate(args.extra):
        cfg = yaml.safe_load((extra / "data.yaml").read_text())
        names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][i] for i in sorted(cfg["names"])]
        mapping = {i: CLASS_NAMES.index(CLASS_MAP[n]) for i, n in enumerate(names) if n in CLASS_MAP}
        stats = Counter()
        for split in SPLITS:
            for img in images(extra / split / "images"):
                stats["seen"] += 1
                label = extra / split / "labels" / (img.stem + ".txt")
                lines = [l.split() for l in label.read_text().splitlines() if l.strip()] if label.exists() else []
                if any(int(p[0]) not in mapping for p in lines):
                    stats["dropped_unmapped_class"] += 1
                    continue
                if stem(img) in our_stems:
                    stats["dropped_same_name_as_ours"] += 1
                    continue
                md5 = hashlib.md5(img.read_bytes()).hexdigest()
                if md5 in seen_md5:
                    stats["dropped_exact_duplicate"] += 1
                    continue
                h = np.uint64(dhash(img))
                if len(known_hash) and min(bin(int(x)).count("1") for x in np.bitwise_xor(known_hash, h)) <= args.near:
                    stats["dropped_near_duplicate"] += 1
                    continue
                seen_md5.add(md5)
                known_hash = np.append(known_hash, h)
                write(img, [to_box_row(p, mapping[int(p[0])]) for p in lines], "train", prefix=f"x{k}_")
                stats["added_to_train"] += 1
        report["extra"][extra.as_posix()] = dict(stats)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )
    report["splits"] = {s: dict(c) for s, c in report["splits"].items()}
    (args.out / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
