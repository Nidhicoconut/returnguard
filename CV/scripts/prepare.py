"""Build the training dataset from the untouched Roboflow export.

- Pools train/valid/test and groups images by source photo (the filename stem
  before ".rf."), so Roboflow's augmented copies of one photo never cross splits.
- Re-splits groups 70/15/15, stratified by which defect classes they contain.
- Train keeps every augmented copy; valid/test keep one copy per source photo.
- Converts box-only label rows to rectangle polygons so every row is a
  segmentation polygon.

Usage (from repository root):
    .venv/Scripts/python.exe CV/scripts/prepare.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CV_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = CV_DIR / "datasets" / "defect_clothes_v2"
CLASS_NAMES = ["clothes", "hole", "stain"]
SPLITS = ("train", "valid", "test")


def to_polygon_row(line: str) -> str | None:
    parts = line.split()
    if not parts:
        return None
    cls, coords = parts[0], [float(v) for v in parts[1:]]
    if len(coords) == 4:  # cx cy w h -> rectangle polygon
        cx, cy, w, h = coords
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    elif len(coords) < 6 or len(coords) % 2:
        raise ValueError(f"Malformed label row: {line!r}")
    coords = [min(max(v, 0.0), 1.0) for v in coords]
    return cls + " " + " ".join(f"{v:.6f}" for v in coords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATASET_DIR / "raw")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exclude-box-defects",
        action="store_true",
        help="Drop source photos whose hole/stain labels include box rows (loose cluster boxes). "
        "Split membership of the remaining photos is unchanged.",
    )
    parser.add_argument(
        "--relabel-dir",
        type=Path,
        help="Folder (searched recursively) of re-labelled images + YOLO labels. A source photo found "
        "there uses that single image and label instead of its raw copies, and is never excluded.",
    )
    parser.add_argument(
        "--relabel-splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Only apply --relabel-dir to photos assigned to these splits",
    )
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            parser.error(f"{args.out} exists; pass --overwrite to rebuild it")
        shutil.rmtree(args.out)

    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in SPLITS:
        for img in sorted((args.raw / split / "images").iterdir()):
            label = args.raw / split / "labels" / (img.stem + ".txt")
            groups[img.name.split(".rf.")[0]].append((img, label))

    def signature(items: list[tuple[Path, Path]]) -> tuple[bool, bool]:
        classes = {l.split()[0] for _, lbl in items if lbl.exists() for l in lbl.read_text().splitlines() if l.strip()}
        return ("1" in classes, "2" in classes)

    strata: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for key, items in groups.items():
        strata[signature(items)].append(key)

    rng = random.Random(args.seed)
    assignment: dict[str, str] = {}
    for keys in strata.values():
        keys.sort()
        rng.shuffle(keys)
        n_val, n_test = round(len(keys) * args.val_frac), round(len(keys) * args.test_frac)
        for i, key in enumerate(keys):
            assignment[key] = "valid" if i < n_val else "test" if i < n_val + n_test else "train"

    def has_box_defect(items: list[tuple[Path, Path]]) -> bool:
        return any(
            len(parts) == 5 and parts[0] != "0"
            for _, lbl in items if lbl.exists()
            for parts in (l.split() for l in lbl.read_text().splitlines())
        )

    relabelled: set[str] = set()
    if args.relabel_dir:
        # Re-exported files keep the original stem as a filename prefix; match the longest key.
        keys = sorted(groups, key=len, reverse=True)
        for img in sorted(p for p in args.relabel_dir.rglob("*") if p.parent.name == "images"):
            key = next((k for k in keys if img.name.startswith(k)), None)
            if key and assignment[key] in args.relabel_splits and key not in relabelled:
                groups[key] = [(img, img.parent.parent / "labels" / (img.stem + ".txt"))]
                relabelled.add(key)
        print(f"Using re-labelled images for {len(relabelled)} source photos")

    if args.exclude_box_defects:
        groups = {k: v for k, v in groups.items() if k in relabelled or not has_box_defect(v)}

    report = {
        "source": "Roboflow hawadz/defect-clothes-v2 v2 (CC BY 4.0)",
        "seed": args.seed,
        "exclude_box_defects": args.exclude_box_defects,
        "relabel_dir": args.relabel_dir.as_posix() if args.relabel_dir else None,
        "relabel_splits": args.relabel_splits if args.relabel_dir else None,
        "relabelled_photos": len(relabelled),
        "splits": {},
    }
    for split in SPLITS:
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)
        report["splits"][split] = {"source_photos": 0, "images": 0, "instances": Counter()}

    for key, items in sorted(groups.items()):
        split = assignment[key]
        if split != "train":
            items = items[:1]
        stats = report["splits"][split]
        stats["source_photos"] += 1
        for img, label in items:
            shutil.copy2(img, args.out / split / "images" / img.name)
            rows = [r for r in map(to_polygon_row, label.read_text().splitlines()) if r] if label.exists() else []
            (args.out / split / "labels" / (img.stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
            stats["images"] += 1
            stats["instances"].update(CLASS_NAMES[int(r.split()[0])] for r in rows)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )
    (args.out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
