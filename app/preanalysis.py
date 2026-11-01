"""Image pre-analysis, run on every uploaded photo before any defect detection.

* Quality gate (tool T1): resolution, sharpness (variance of the Laplacian), exposure. A failed
  photo gets a specific instruction the customer can act on.
* Integrity checks (tool T7): exact and near-duplicate photos already used in other cases
  (SHA-256 and a 256-bit difference hash), and camera metadata that does not fit the order
  (taken before delivery, saved by editing software). Flags are neutral observations for a
  reviewer, never an accusation.
* Privacy: EXIF is read here and then dropped (uploads are re-encoded); faces are blurred
  before any crop leaves the server for the vision-language check.
"""
import datetime as dt
import hashlib
import io
import re

import cv2
import numpy as np

from . import config, db

INSTRUCTIONS = {
    "too_small": "The photo is too small. Send the original photo from your phone camera, not a screenshot or thumbnail.",
    "blurry": "The photo is blurry. Hold the phone steady, tap the screen to focus on the garment, and take it again.",
    "too_dark": "The photo is too dark. Take it in daylight or under a bright light.",
    "too_bright": "The photo is overexposed. Move out of direct light or turn off the flash.",
}
EDITING_SOFTWARE = re.compile(r"photoshop|gimp|lightroom|snapseed|picsart|canva|facetune|editor|pixlr", re.I)


def decode(data: bytes):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return arr


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(img) -> str:
    """256-bit difference hash. Checked on real photos: a re-saved, resized copy differs by at most 8 bits,
    different garment photos by 89 or more (a 64-bit hash confused similar flat-lay photos)."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (17, 16), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def hash_distance(a: str, b: str) -> int:
    if len(a) != len(b):  # hashes from an older format cannot be compared
        return 256
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def read_exif(data: bytes) -> dict:
    """Return the few EXIF fields the integrity check uses (empty when the photo has none)."""
    try:
        from PIL import Image

        exif = Image.open(io.BytesIO(data)).getexif()
        sub = exif.get_ifd(0x8769)  # Exif sub-IFD holds DateTimeOriginal
        taken = sub.get(36867) or exif.get(306)
        out = {}
        if taken:
            out["taken"] = str(taken)
        if exif.get(305):
            out["software"] = str(exif.get(305))
        if exif.get(272):
            out["camera"] = str(exif.get(272))
        return out
    except Exception:
        return {}


def assess_quality(img) -> dict:
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())
    problems = []
    if min(h, w) < config.MIN_SIDE_PX:
        problems.append("too_small")
    if sharpness < config.MIN_SHARPNESS:
        problems.append("blurry")
    if brightness < config.DARK_MEAN:
        problems.append("too_dark")
    elif brightness > config.BRIGHT_MEAN:
        problems.append("too_bright")
    return {"width": w, "height": h, "sharpness": round(sharpness, 1), "brightness": round(brightness, 1),
            "ok": not problems, "problems": problems, "instructions": [INSTRUCTIONS[p] for p in problems]}


def check_integrity(case_id: int, sha: str, dh: str, exif: dict, delivered_date: str | None) -> dict:
    flags, notes = [], []
    duplicate_of = None
    for other_case, other_sha, other_dh in db.other_media_hashes(case_id):
        if other_sha == sha or (other_dh and hash_distance(dh, other_dh) <= config.DUPLICATE_HASH_BITS):
            duplicate_of = other_case
            flags.append(f"Same or nearly the same photo was uploaded with case {other_case}.")
            break
    if exif.get("taken") and delivered_date:
        try:
            taken = dt.datetime.strptime(exif["taken"][:10], "%Y:%m:%d").date()
            if taken < dt.date.fromisoformat(delivered_date) - dt.timedelta(days=1):
                flags.append(f"Camera date ({taken.isoformat()}) is before the delivery date ({delivered_date}).")
        except ValueError:
            pass
    if exif.get("software") and EDITING_SOFTWARE.search(exif["software"]):
        flags.append(f"Photo was saved by editing software ({exif['software']}).")
    if not exif:
        notes.append("No camera metadata (common for photos sent through messaging apps).")
    return {"duplicate_of": duplicate_of, "flags": flags, "notes": notes, "exif": exif}


def face_blur_available() -> bool:
    return config.FACE_MODEL.is_file()


def blur_faces(img):
    """Blur faces (OpenCV YuNet detector) before an image is sent to an external API."""
    h, w = img.shape[:2]
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # silence a harmless warning per call
    det = cv2.FaceDetectorYN.create(str(config.FACE_MODEL), "", (w, h), score_threshold=0.6)
    _, faces = det.detect(img)
    out = img.copy()
    for f in faces if faces is not None else []:
        x, y, fw, fh = (int(v) for v in f[:4])
        pad = int(0.25 * max(fw, fh))  # cover hair and chin too
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + fw + pad), min(h, y + fh + pad)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), sigmaX=max(x1 - x0, y1 - y0) / 5)
    return out
