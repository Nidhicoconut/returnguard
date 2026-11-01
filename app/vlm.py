"""Tool T9: check a damage region with a vision-language model (Gemini, OpenAI-compatible endpoint).

Blind first: the model is shown only the cropped region and asked what it sees. It never sees the
customer's claim, so it cannot simply agree with it. The comparison with the claim is done in code
(see `compare`). Faces are blurred before anything leaves the server.
"""
import base64
import json
import logging
import re
import time

import cv2
import httpx

from . import config, preanalysis

log = logging.getLogger("returnguard.vlm")

PROMPT = """You are inspecting a photo of a clothing item for a returns desk. Look only at what is visible.
Is there damage to the fabric: a hole, tear, rip, split seam, stain or discolouration?
Ignore designed features (prints, distressing, buttonholes, seams that are intact, labels) and folds or shadows.
Reply with one JSON object and nothing else:
{"damage_visible": true|false, "damage_type": "hole|tear|stain|discoloration|none|other", "confidence": 0.0-1.0,
 "description": "one short factual sentence"}"""

TYPE_TO_CLASS = {"hole": "tear_hole", "tear": "tear_hole", "stain": "stain_discoloration", "discoloration": "stain_discoloration"}


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def crop_region(img, region: dict | None, min_px: int = 256, context: float = 1.0):
    """Crop a normalised box with some surrounding context (the whole photo when region is None)."""
    h, w = img.shape[:2]
    if region is None:
        return img
    cx, cy = (region["x"] + region["w"] / 2) * w, (region["y"] + region["h"] / 2) * h
    half = max(region["w"] * w, region["h"] * h, min_px) * (1 + context) / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return img[y0:y1, x0:x1]


def _encode(img) -> str:
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"no complete JSON in the vision model's reply: {text[:80]!r}")
    data = json.loads(text[start:end + 1])
    return {"damage_visible": bool(data.get("damage_visible")), "damage_type": str(data.get("damage_type", "none")).lower(),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))), "description": str(data.get("description", ""))[:300]}


def describe(img, region: dict | None) -> dict:
    """Blind description of one region (or the whole photo). Raises on API or parsing errors."""
    crop = crop_region(img, region)
    if preanalysis.face_blur_available():
        crop = preanalysis.blur_faces(crop)
    elif region is None:
        raise RuntimeError("face model missing, so the whole photo is not sent to the external model")
    # Gemini 2.5 "thinks" before answering and the thinking counts against max_tokens; with a small
    # limit the JSON reply was cut off. Thinking is not needed for this short visual question.
    body = {"model": config.VLM_MODEL, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                      {"type": "image_url", "image_url": {"url": _encode(crop)}}]}]}
    if config.VLM_REASONING_EFFORT:
        body["reasoning_effort"] = config.VLM_REASONING_EFFORT
    for attempt in range(2):
        r = httpx.post(f"{config.VLM_BASE_URL}/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {config.GEMINI_API_KEY}"}, timeout=60)
        if r.status_code == 429 and attempt == 0:  # free-tier rate limit: wait once, then retry
            time.sleep(min(float(r.headers.get("retry-after", 8) or 8), 15))
            continue
        r.raise_for_status()
        return _parse(r.json()["choices"][0]["message"]["content"])


def compare(observations: list[dict], claimed_class: str | None) -> dict:
    """Targeted comparison with the claim, done in code from the blind descriptions."""
    seen = [o for o in observations if o.get("damage_visible")]
    if not observations:
        return {"status": "not_run", "confirms": None}
    if not seen:
        return {"status": "no_damage_seen", "confirms": False}
    classes = {TYPE_TO_CLASS.get(o["damage_type"]) for o in seen}
    matches = claimed_class is None or claimed_class in classes
    return {"status": "damage_seen", "confirms": True, "matches_claim": matches,
            "types_seen": sorted(o["damage_type"] for o in seen), "confidence": max(o["confidence"] for o in seen)}
