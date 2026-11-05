"""Tool T3: do the photos show the product that was ordered?

Two independent signals, combined in `verdict`:

* Type and colour (every order): Gemini is told what was ordered and asked, for each photo, whether that
  item is visible: yes / no / unclear. Customer photos often show several garments (a person returning
  jeans is wearing a top too) and close-ups of damage show only fabric, so "unclear" is the safe default
  and only a confident "no" counts against the photo. CLIP zero-shot was tried first and named the
  garment type correctly on only 66% of 95 real customer photos, so it is not used for this.
* Catalog match (orders whose product has reference photos in CATALOG_DIR/<sku>/): CLIP image
  embeddings of the upload compared with the reference photos (cosine similarity). It can only confirm a
  match. Tested on 6 real Amazon review photos (all close-ups of damage) against their own product's
  listing photos: similarity was 0.55-0.70 for the right product and just as high for wrong ones, and
  3 of the 6 fell below 0.60. So a low similarity is not evidence of a wrong item.

A case is a mismatch only when no photo shows the ordered item and Gemini confidently says at least one
photo shows something else. Faces are blurred before photos are sent to Gemini.
"""
import base64
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2

from . import config, preanalysis, vlm

log = logging.getLogger("returnguard.product")

PROMPT = """You are checking customer photos for a clothing return. The order is for:
  {product} (type: {ptype}{colour})
For each photo, in order, say whether that ordered item is visible. Ignore any damage.
- "yes": a garment of that type{colour_rule} is clearly visible, even if worn, folded or only partly in view.
- "no": the photo clearly shows a different kind of item, or clearly the wrong colour.
- "unclear": a close-up of fabric, a photo where the item type cannot be told, or whenever you are not sure.
Customer text is not included; judge only what you see.
Reply with one JSON object and nothing else:
{{"photos": [{{"ordered_item_visible": "yes|no|unclear", "seen": "short description of the garments in view",
  "confidence": 0.0-1.0}}]}}"""


# ---------------------------------------------------------------------------- type and colour (Gemini)
def available() -> bool:
    return config.PRODUCT_CHECK != "off" and bool(config.GEMINI_API_KEY)


def _encode(img) -> dict:
    h, w = img.shape[:2]
    s = 768 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}}


def ask_gemini(images: list, order: dict) -> list[dict]:
    """One call for all photos. Returns one {"visible", "seen", "confidence"} per photo."""
    if not preanalysis.face_blur_available():
        raise RuntimeError("face model missing, so whole photos are not sent to the external model")
    colour = order.get("colour")
    text = PROMPT.format(product=order["product"], ptype=order.get("product_type") or "garment",
                         colour=f", colour: {colour}" if colour else "",
                         colour_rule=f" and roughly {colour}" if colour else "")
    content = [{"type": "text", "text": text}] + [_encode(preanalysis.blur_faces(img)) for img in images]
    body = {"model": config.VLM_MODEL, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}]}
    if config.VLM_REASONING_EFFORT:
        body["reasoning_effort"] = config.VLM_REASONING_EFFORT
    reply = vlm.chat(body)  # retries, fallback model and outage handling live there
    raw = re.sub(r"^```(?:json)?|```$", "", (reply["choices"][0]["message"]["content"] or "").strip(), flags=re.M)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"no complete JSON in the reply: {raw[:80]!r}")
    photos = json.loads(raw[start:end + 1]).get("photos", [])
    out = []
    for i in range(len(images)):
        p = photos[i] if i < len(photos) and isinstance(photos[i], dict) else {}
        v = str(p.get("ordered_item_visible", "unclear")).lower()
        out.append({"visible": v if v in ("yes", "no", "unclear") else "unclear", "seen": str(p.get("seen", ""))[:160],
                    "confidence": max(0.0, min(1.0, float(p.get("confidence", 0.5) or 0.5)))})
    return out


# ---------------------------------------------------------------------------- catalog match (CLIP embeddings)
def catalog_photos(sku: str | None) -> list[Path]:
    if not sku:
        return []
    for base in config.CATALOG_DIRS:
        folder = base / sku
        if folder.is_dir():
            return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return []


@lru_cache(maxsize=1)
def _clip():
    import os

    os.environ.setdefault("HF_HOME", str(config.CV_DIR / ".hf_cache"))
    if str(config.CV_DIR / ".pydeps") not in sys.path:
        sys.path.append(str(config.CV_DIR / ".pydeps"))  # transformers lives there, apart from the app venv
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, CLIPModel.from_pretrained(config.CLIP_MODEL).to(dev).eval(), CLIPProcessor.from_pretrained(config.CLIP_MODEL), dev


def _embed(imgs: list):
    from PIL import Image

    torch, model, proc, dev = _clip()
    pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in imgs]
    with torch.no_grad():
        f = model.get_image_features(**proc(images=pil, return_tensors="pt").to(dev))
        f = getattr(f, "pooler_output", f)  # transformers 5 wraps the embedding in an output object
        return (f / f.norm(dim=-1, keepdim=True)).cpu()


def catalog_similarity(images: list, refs: list[Path]) -> list[float]:
    """Best cosine similarity of each photo to any reference photo of the product."""
    ref_imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in refs]
    ref_imgs = [r for r in ref_imgs if r is not None]
    if not ref_imgs:
        return []
    sims = _embed(images) @ _embed(ref_imgs).T
    return [round(float(s), 3) for s in sims.max(dim=1).values]


# ---------------------------------------------------------------------------- combined verdict
def verdict(per_photo: list[dict]) -> dict:
    """match | mismatch | unverified, from per-photo Gemini answers and catalog similarities."""
    yes = [p for p in per_photo if p.get("visible") == "yes" or (p.get("catalog_sim") or 0) >= config.CATALOG_MATCH_SIM]
    no = [p for p in per_photo if p.get("visible") == "no" and p.get("confidence", 0) >= config.PRODUCT_MISMATCH_CONF]
    if yes:
        return {"status": "match", "n_match": len(yes)}
    if no:
        return {"status": "mismatch", "n_mismatch": len(no), "seen": [p.get("seen") for p in no if p.get("seen")]}
    return {"status": "unverified"}
