"""Evaluate ReturnGuard: vision metrics per backend, then end-to-end product metrics.

    python scripts/make_synthetic_data.py --out data/eval --n 40
    python scripts/evaluate.py --data data/eval --backends heuristic,clip

Writes reports/eval_report.md. Splits are by product instance: every image uses its own seed,
so no product appears twice. Replace data/eval with your own consented photos + labels.json
(same format) for the real numbers you put in the report.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/eval")
ap.add_argument("--backends", default="heuristic")
ap.add_argument("--out", default="reports")
args = ap.parse_args()

os.environ["RG_DATA_DIR"] = tempfile.mkdtemp(prefix="rg_eval_")  # keep evaluation cases out of the real DB
import cv2  # noqa: E402

from app import agent, config, db, vision  # noqa: E402

data = ROOT / args.data
labels = json.loads((data / "labels.json").read_text())


def iou(a, b):
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw, ih = max(0, min(ax2, bx2) - max(a["x"], b["x"])), max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter + 1e-9)


def vision_eval(backend):
    tp = fp = fn = tn = inc = 0
    ious, type_ok, lat = [], 0, []
    for name, gt in labels.items():
        t0 = time.time()
        r = vision.analyze(data / name, backend)
        lat.append(time.time() - t0)
        pred = r["label"]
        if pred == "inconclusive":
            inc += 1
        if gt["label"] == "defective":
            if pred == "defective":
                tp += 1
                ious.append(max((iou(g, gt["bbox"]) for g in r["regions"]), default=0.0))
                type_ok += r["defect_type"] == gt["type"]
            else:
                fn += 1
        else:
            fp += pred == "defective"
            tn += pred != "defective"
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"backend": backend, "precision": prec, "recall": rec, "f1": f1, "inconclusive_rate": inc / len(labels),
            "mean_iou": statistics.mean(ious) if ious else 0.0, "iou>=0.3": sum(i >= 0.3 for i in ious) / max(1, tp),
            "type_acc": type_ok / max(1, tp), "ms_per_image": 1000 * statistics.mean(lat), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def product_eval():
    """Run every image through the full agent in four scenarios with known correct outcomes."""
    db.init_db(reset=True)
    scenarios = {  # (order, reason template for defective photos, reason for intact photos)
        "honest defect claim -> approve": ("ORD-1001", "defective", "The item arrived damaged and cracked", "approve"),
        "honest change of mind -> approve": ("ORD-1002", "intact", "I changed my mind about the colour", "approve"),
        "defect claim on intact photo -> escalate": ("ORD-1007", "intact", "The item is broken and scratched", "escalate"),
        "unmentioned damage -> escalate": ("ORD-1003", "defective", "I changed my mind about the colour", "escalate"),
        "outside window -> reject": ("ORD-1004", "any", "It is cracked", "reject"),
    }
    rows, lat = {}, []
    for scen, (order, wants, reason, expected) in scenarios.items():
        ok = n = esc = 0
        for name, gt in labels.items():
            if wants != "any" and gt["label"] != wants:
                continue
            img = cv2.imread(str(data / name))
            fname = f"{abs(hash((scen, name)))}.jpg"
            cv2.imwrite(str(config.UPLOAD_DIR / fname), img)
            cid = db.create_case(order, reason, fname)
            t0 = time.time()
            case = agent.run_case(cid)
            lat.append(time.time() - t0)
            n += 1
            ok += case["decision"] == expected
            esc += case["decision"] == "escalate"
        rows[scen] = {"n": n, "accuracy": ok / max(1, n), "escalated": esc / max(1, n)}
    cites_valid = all(set(c["citations"] or []) <= {"R1", "R2", "R3", "R4", "R5", "R6", "R7"} for c in db.list_cases(10000))
    return rows, statistics.mean(lat) * 1000, cites_valid


backends = [b.strip() for b in args.backends.split(",") if b.strip()]
vis = []
for b in backends:
    if b == "clip" and not vision.clip_available():
        print("Skipping clip: torch/transformers not installed")
        continue
    vis.append(vision_eval(b))
rows, ms, cites = product_eval()

lines = ["# ReturnGuard evaluation", "", f"Images: {len(labels)} ({sum(g['label']=='defective' for g in labels.values())} defective). "
         f"Active backend for end-to-end run: {vision.active_backend()}. LLM provider: {os.getenv('LLM_PROVIDER', 'auto')}.", "",
         "## Vision (defective vs intact)", "",
         "| Backend | Precision | Recall | F1 | Inconclusive | Mean IoU | IoU>=0.3 | Type acc | ms/img |", "|---|---|---|---|---|---|---|---|---|"]
for v in vis:
    lines.append(f"| {v['backend']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['inconclusive_rate']:.2f} | "
                 f"{v['mean_iou']:.2f} | {v['iou>=0.3']:.2f} | {v['type_acc']:.2f} | {v['ms_per_image']:.0f} |")
lines += ["", "## End-to-end decisions", "", "| Scenario | Cases | Correct decision | Escalated |", "|---|---|---|---|"]
for scen, r in rows.items():
    lines.append(f"| {scen} | {r['n']} | {r['accuracy']:.2f} | {r['escalated']:.2f} |")
lines += ["", f"Mean agent latency: {ms:.0f} ms per case. All cited clauses exist in the policy: {cites}.", ""]
out = ROOT / args.out
out.mkdir(exist_ok=True)
(out / "eval_report.md").write_text("\n".join(lines))
print("\n".join(lines))
