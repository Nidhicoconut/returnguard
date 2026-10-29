# ReturnGuard CV: clothing defect detection

Detects **holes** and **stains** on clothing photos, and outlines the garment (`clothes`). This file collects every training and evaluation result to date (2026-10-29). Working notes and the full decision log are in [WORKING_CONTEXT.md](WORKING_CONTEXT.md).

## Summary

| | |
|---|---|
| Model in use | Two models, both trained on our data + defect-clothes-ukrks v3: a **YOLO11s box detector** decides *whether* a photo shows damage (threshold 0.20), a **YOLO11m box detector** marks *where* (regions ≥ 0.15) |
| Damaged photos caught | **25 of 29 (86%)** on the validation set, plus 1 marked unsure |
| Clean photos wrongly called damaged | **0 of 14**, with 2 marked unsure (the previous model: 3 of 14) |
| Defect mAP@50 (boxes, hole + stain) | 0.32 for the detector on all 43 validation photos (YOLO11s-seg: 0.29; 0.44 on the 26 consistently labelled ones) |
| Project target | Defect mAP@50 ≥ 0.55: **not met yet** |
| Main limitation | Training data: about 290 real photos, labelled in two inconsistent styles |

## Data

Source: Roboflow [defect-clothes-v2](https://universe.roboflow.com/hawadz/defect-clothes-v2/dataset/2), version 2, CC BY 4.0. It has 780 images, but those come from **289 original photos**: Roboflow made 3 augmented copies of each training photo. All images are stretched to 640 × 640.

The data was re-split by original photo (70/15/15, seed 42, stratified by hole/stain), so no augmented copy of a photo appears in two splits.

| Split | Original photos | Images | clothes | hole | stain |
|---|---:|---:|---:|---:|---:|
| Train | 203 | 548 | 556 | 1,976 | 784 |
| Validation | 43 | 43 | 41 | 118 | 56 |
| Test (not used yet) | 43 | 43 | 44 | 215 | 30 |

Validation has 29 photos with damage (15 with holes, 15 with stains) and 14 clean photos.

**Labelling problem:** 111 photos have loose rectangles drawn around *clusters* of holes, while 156 have tight outlines around each hole. A model that outlines each hole correctly gets scored as wrong on the rectangle photos, which pulls mAP down. The "consistent" validation subset (26 photos) keeps only tightly outlined photos. It has only **3 photos with holes**, so its hole numbers are rough.

## Models trained

All models start from COCO-pretrained Ultralytics YOLO11 segmentation weights, at 640 px, batch 16, up to 150 epochs with early stopping after 30 epochs without improvement, on an RTX 5080 Laptop GPU.

| Run | Model | Training data | Epochs run | Best epoch | Training time |
|---|---|---|---:|---:|---:|
| `yolo11s_seg` | YOLO11s-seg (10.1 M params, 32.9 GFLOPs) | All 203 train photos, original labels | 113 | 83 | 10 min |
| `yolo11m_seg` | YOLO11m-seg | Same as above | 88 | 58 | 15 min |
| `yolo11s_seg_polyonly` | YOLO11s-seg | Only the 124 consistently labelled train photos | 82 | 52 | 5 min |
| `yolo11s_seg_proposals` | YOLO11s-seg | All train photos; rectangle labels replaced by auto-generated outlines (model + SAM 2), not reviewed by a person | 99 | 69 | 7 min |
| `yolo11s_det_ours` | YOLO11s (boxes) | All 203 train photos, outlines converted to boxes | 150 | 145 | 9 min |
| **`yolo11s_det_merged`** | YOLO11s (boxes) | Ours + 949 photos from defect-clothes-ukrks v3 (≈1,150 original photos; see below) | 108 | 78 | 16 min |
| **`yolo11m_det_merged`** | YOLO11m (boxes) | Same as above | 150 | 122 | 47 min |
| `yolo11m_det_merged_zoom` | YOLO11m (boxes) | Same, stronger zoom augmentation (scale 0.9) | 121 | 91 | 35 min |
| Experiment 2 (old) | YOLO11n detect | Roboflow's original split | 50 | — | — |

Only the two bold runs (the deployed models) are kept on disk. The others were deleted on 2026-10-29, and their results stay here for reference, as do the Amazon retrain results below.

**Merged data:** [defect-clothes-ukrks v3](https://universe.roboflow.com/hawadz/defect-clothes-ukrks/dataset/3) (CC BY 4.0, same author as ours) has 1,482 un-augmented images with box labels. `scripts/prepare_merged.py` added 949 of them to **training only**. It dropped 205 containing `cut` (weave flaws in factory fabric, not tears), 133 with the same original filename as one of our photos (including 25 of our validation and 30 of our test photos), and 195 exact or near duplicates. Validation and test stay our same 43 + 43 photos.

### Validation scores at the best epoch (all classes, from training logs)

These include the easy `clothes` class, so they're higher than the defect-only scores below. The first two runs used the full 43-photo validation set; the last two used the 26-photo consistent subset, so the two pairs can't be compared with each other.

| Run | Validation set | Box P | Box R | Box mAP@50 | Box mAP@50-95 | Mask P | Mask R | Mask mAP@50 | Mask mAP@50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolo11s_seg` | 43 photos | 0.679 | 0.492 | 0.519 | 0.421 | 0.643 | 0.466 | 0.482 | 0.322 |
| `yolo11m_seg` | 43 photos | 0.552 | 0.627 | 0.557 | 0.416 | 0.586 | 0.550 | 0.515 | 0.326 |
| `yolo11s_seg_polyonly` | 26 consistent | 0.670 | 0.567 | 0.592 | 0.461 | 0.631 | 0.615 | 0.594 | 0.419 |
| `yolo11s_seg_proposals` | 26 consistent | 0.788 | 0.580 | 0.646 | 0.502 | 0.657 | 0.630 | 0.618 | 0.442 |

## Defect detection metrics (hole + stain only)

### YOLO11s on all 43 validation photos

| Class | Instances | Box P | Box R | Box mAP@50 | Box mAP@50-95 | Mask P | Mask R | Mask mAP@50 | Mask mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hole | 118 | 0.383 | 0.263 | 0.263 | 0.179 | 0.335 | 0.231 | 0.227 | 0.121 |
| stain | 56 | 0.708 | 0.214 | 0.314 | 0.214 | 0.702 | 0.214 | 0.297 | 0.168 |
| **defects (mean)** | 174 | | | **0.289** | **0.196** | | | **0.262** | **0.144** |
| clothes (not a defect) | 41 | 0.940 | 1.000 | 0.979 | 0.866 | 0.894 | 0.951 | 0.923 | 0.673 |

### All models on the 26 consistently labelled photos

| Model | Hole box mAP@50 | Stain box mAP@50 | **Defect box mAP@50** | Defect box mAP@50-95 | Defect mask mAP@50 | Defect mask mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s | 0.299 | 0.578 | **0.439** | 0.330 | 0.436 | 0.273 |
| YOLO11m | 0.337 | 0.441 | **0.389** | 0.267 | — | — |
| YOLO11s, consistent photos only | 0.302 | 0.482 | **0.392** | 0.266 | 0.409 | 0.234 |
| YOLO11s + auto outlines | 0.397 | 0.551 | **0.474** | 0.312 | 0.447 | 0.268 |

"—" means not measured.

## Photo-level metrics (what the app needs)

"Is there a hole or stain in this photo?" Confidence 0.3, all 43 validation photos.

| Model | Damaged photos caught | Clean photos wrongly flagged | Hole photos caught | Photos wrongly flagged for holes | Stain photos caught | Photos wrongly flagged for stains |
|---|---:|---:|---:|---:|---:|---:|
| **YOLO11s** | **25 / 29** | **3 / 14** | 14 / 15 | **4 / 28** | **11 / 15** | 2 / 28 |
| YOLO11m | 25 / 29 | 5 / 14 | **15 / 15** | 10 / 28 | 9 / 15 | **1 / 28** |
| YOLO11s + auto outlines | 24 / 29 | 7 / 14 | 13 / 15 | 11 / 28 | 7 / 15 | 2 / 28 |
| YOLO11s boxes, ours | **27 / 29** | 1 / 14 | 13 / 15 | 2 / 28 | 13 / 15 | 3 / 28 |
| YOLO11s boxes, merged | 22 / 29 | **0 / 14** | 10 / 15 | 2 / 28 | 11 / 15 | 2 / 28 |

### Damaged photos caught / clean photos wrongly flagged, by threshold

| Model | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---:|---:|---:|---:|
| YOLO11s-seg | 27 / 4 | 26 / 3 | 25 / 3 | 25 / 3 |
| YOLO11m-seg | 27 / 11 | 26 / 10 | 26 / 7 | 25 / 5 |
| YOLO11s + auto outlines | 28 / 11 | 27 / 10 | 25 / 9 | 24 / 7 |
| YOLO11s boxes, ours | 28 / 7 | 28 / 5 | 27 / 2 | 27 / 1 |
| **YOLO11s boxes, merged** | 28 / 7 | 26 / 4 | **25 / 0** | 22 / 0 |

### Defect mAP of the box models (all 43 photos, boxes)

| Model | Hole mAP@50 | Stain mAP@50 | Defect mAP@50 | Defect mAP@50-95 |
|---|---:|---:|---:|---:|
| YOLO11s-seg (for reference) | 0.263 | 0.314 | 0.289 | 0.196 |
| YOLO11s boxes, ours | 0.367 | 0.255 | 0.311 | 0.187 |
| YOLO11s boxes, merged | 0.318 | 0.329 | 0.323 | 0.197 |
| **YOLO11m boxes, merged** | **0.424** | 0.301 | **0.362** | 0.196 |
| YOLO11m boxes, merged + zoom | 0.349 | 0.335 | 0.342 | 0.204 |

At equal wrong boxes per photo (2 / 3 / 5), YOLO11m boxes-merged finds 67 / 75 / 93 of the 174 defects: the most of any model (YOLO11m-seg: 57 / 75 / 85). Damaged photos caught / clean photos flagged: 23 / 2 at 0.2 (YOLO11m-seg: 26 / 7). So it became the locator, and YOLO11s boxes-merged stays the detector.

## Finding each defect at equal false alarms

Lowering the threshold finds more defects but adds wrong boxes. This compares how many of the **174 real defects** each model finds when allowed the same number of wrong boxes per photo (box IoU ≥ 0.5, all 43 photos).

| Wrong boxes per photo | YOLO11s | **YOLO11m** | YOLO11s + auto outlines |
|---:|---:|---:|---:|
| 1 | 41 (24%) | **47 (27%)** | 45 (26%) |
| 2 | 49 (28%) | **57 (33%)** | 56 (32%) |
| 3 | 54 (31%) | **75 (43%)** | 59 (34%) |
| 5 | 59 (34%) | **85 (49%)** | 63 (36%) |

Box models at the same budgets: boxes-ours 41 / 46 / 56 / 65 / 73 and boxes-merged 31 / 37 / 49 / 60 / 70 defects for 0.5 / 1 / 2 / 3 / 5 wrong boxes per photo. YOLO11m stays the best at marking individual defects, so it remains the locator.

### Recall by defect size (YOLO11s, confidence 0.25)

| Size (share of photo) | Holes found | Stains found |
|---|---:|---:|
| Large, ≥ 1% | 9 / 40 | 9 / 25 |
| Small, 0.1–1% | 15 / 32 | 8 / 21 |
| Tiny, < 0.1% | 14 / 46 | 0 / 10 |

Large holes score worst mainly because of the loose rectangle labels: the model outlines individual holes inside a rectangle that covers a whole cluster.

## The deployed two-model predictor

`scripts/predict.py`, also used by `defect_check.ipynb`:

1. The merged YOLO11s box detector (`runs/deploy/clothes_damage_detector.pt`) classifies the photo. A defect score ≥ 0.20 means **damaged**; 0.15–0.20 means **unsure**. Each defect type gets its own score.
2. If damaged or unsure, the YOLO11m box locator (`runs/deploy/clothes_damage_locator.pt`) marks regions (score ≥ 0.15), merged with the detector's boxes. It replaced YOLO11m-seg, so regions are boxes, not outlines.
3. Detections centred outside the garment (skin, faces, background) are dropped; at most 15 regions are kept.
4. On photos larger than 1000 px, both models also run on a crop of the garment. On the 640 px validation photos this pass only added a false alarm, so it's skipped there.

5. **Both models vote:** damaged if the detector scores ≥ 0.20 **or** the locator ≥ 0.40; unsure if detector ≥ 0.15 or locator ≥ 0.25. Real customer-style photos showed the cautious detector missing damage the locator saw.
6. **Optional second opinion (Grounding DINO, open-vocabulary, `second_opinion=True`):** if both models say clean but it sees a likely hole or stain on the garment (≥ 0.50), the verdict becomes *unsure* (ask for a closer photo). It never makes a photo *damaged*: on its own it can't separate clean from damaged garments.

Result on the 43 validation photos:

| Setup | Damaged photos: damaged / unsure / clean | Clean photos: damaged / unsure / clean |
|---|---|---|
| **Both models voting** | **26** / 0 / 3 | **0** / 3 / 11 |
| + second opinion (notebook default) | 26 / 0 / 3 | 0 / 5 / 9 |
| Detector only (before) | 25 / 1 / 3 | 0 / 2 / 12 |
| YOLO11s-seg detector at 0.30 (first version) | 25 / 0 / 4 | 3 / 1 / 10 |

On two of the user's own photos (`real_test/`), the detector alone called both clean. Now the striped sweater with holes is *damaged* (locator 0.62), and the black trousers with a tear are *unsure* with the tear boxed (second opinion 0.53).

Speed: YOLO11s takes about 6.5 ms per image on the GPU (inference only).

## Amazon pseudo-label retrain (tried 2026-10-29, not adopted)

Retrained both models on `merged_amazon_det`: the existing training data plus 322 auto-labelled Amazon review photos (18% of training images, labels from our detector and Grounding DINO, class from the review text). Validation and test are unchanged and human-labelled.

| Run | Best box mAP@50 (all classes) | v1 equivalent |
|---|---:|---:|
| yolo11s_det_amazon (epoch 91, stopped at 121) | 0.560 | 0.540 |
| yolo11m_det_amazon (epoch 91, stopped at 123) | 0.549 | 0.582 |

Full predictor, 43 validation photos, no second opinion (damaged / unsure / clean):

| Detector + locator | Damaged photos (29) | Clean photos (14) |
|---|---|---|
| **v1 + v1 (kept)** | **26 / 0 / 3** | **0 / 3 / 11** |
| new + v1 | 25 / 1 / 3 | 1 / 2 / 11 |
| v1 + new | 26 / 1 / 2 | 1 / 1 / 12 |
| new + new | 25 / 1 / 3 | 1 / 1 / 12 |

Real photos (`CV/real_test/`, no second opinion): v1 gives the trousers clean and the sweater damaged. Every new combination gives the trousers unsure, and the new detector scores the sweater 0.58 (v1: 0.00). With the second opinion on, v1 already gives the trousers unsure.

**Decision: keep v1.** Every combination with a new model calls a clean validation photo damaged (`10_jpg.rf.77269e…`, a plain green T-shirt). The boxes sit on a seam, a printed logo and the hanger tag, which are the kinds of junk boxes the auto-labels contain. Wrongly flagging a clean return costs more than the small gain on real photos. Weights are kept in `CV/runs/yolo11{s,m}_det_amazon/` for reference.

## Earlier baseline (Experiment 2, deleted)

| | Hole mAP@50 | Stain mAP@50 | Defect mAP@50 | Note |
|---|---:|---:|---:|---|
| YOLO11n, Roboflow's original 30-photo validation | 0.62 | 0.42 | ≈ 0.52 | Smaller, easier validation set; not comparable |
| Same model on the new validation set | 0.91 | 0.94 | 0.92 | **Invalid**: 36 of 43 photos were in its training data |

## Conclusions

- **More data helped where it matters most.** The box detector trained with defect-clothes-ukrks v3 flags none of the 14 clean validation photos at 0.20 while catching the same 25 damaged photos, so it replaced YOLO11s-seg as the "is it damaged?" model. Switching from outlines to boxes alone also helped.
- **YOLO11m finds more of the individual defects** for the same number of wrong boxes, but flags many clean garments. That's why it only marks regions after YOLO11s says "damaged".
- **A bigger model or auto-generated outlines didn't fix accuracy.** The limit is data quantity and label consistency.
- **Known weak spots:** close-ups of a single large hole are missed, fold creases get flagged as holes, and tiny stains are rarely found.
- **Next steps:** merge more Roboflow clothing datasets (needs a Roboflow API key in `CV/.env`), add about 100 labelled team phone photos, retrain, and evaluate the held-out test set once at the end.

## Reproduce

From the repository root, with the project venv:

```
.venv/Scripts/python.exe CV/scripts/prepare.py                                   # build our split (train/valid/test)
.venv/Scripts/python.exe CV/scripts/audit_dataset.py CV/datasets/extra_1/raw     # check an extra dataset
.venv/Scripts/python.exe CV/scripts/prepare_merged.py --extra CV/datasets/extra_1/raw   # -> merged_det
.venv/Scripts/python.exe CV/scripts/train.py --model yolo11s.pt --data CV/datasets/merged_det/data.yaml --name yolo11s_det_merged   # detector
.venv/Scripts/python.exe CV/scripts/train.py --model yolo11m.pt --data CV/datasets/merged_det/data.yaml --name yolo11m_det_merged   # locator
.venv/Scripts/python.exe CV/scripts/evaluate.py --weights CV/runs/yolo11s_det_merged/weights/best.pt --data CV/datasets/merged_det/data.yaml
.venv/Scripts/python.exe CV/scripts/evaluate_predictor.py [--second-opinion]     # two-model photo-level check
.venv/Scripts/python.exe CV/scripts/predict.py path/to/photo.jpg                 # single photos
```

The equal-false-alarm tables above came from `export_predictions.py` and `compare_tradeoff.py`, which were removed on 2026-10-29 along with the runs they compared.

The interactive review page with every validation photo: https://claude.ai/artifact/DE54phgbd99Nsuu5BWU87K
