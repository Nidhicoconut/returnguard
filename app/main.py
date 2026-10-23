"""FastAPI backend: upload a return, run the agent, review and correct decisions."""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, config, db, llm, vision

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("returnguard")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="ReturnGuard", version="1.0", lifespan=lifespan)


def _with_events(case: dict) -> dict:
    case["image_url"] = f"/media/{case['image_file']}"
    case["events"] = db.events_for(case["id"])
    return case


@app.get("/api/health")
def health():
    return {"ok": True, "vision_backend": vision.active_backend(), "llm_provider": llm.provider(),
            "return_window_days": config.RETURN_WINDOW_DAYS}


@app.get("/api/orders")
def orders():
    return db.list_orders()


@app.post("/api/cases")
async def create_case(order_id: str = Form(...), reason: str = Form(...), image: UploadFile = File(...)):
    order_id, reason = order_id.strip(), reason.strip()
    if not order_id or not reason:
        raise HTTPException(422, "Order ID and a reason are both required.")
    if len(reason) > 1000:
        raise HTTPException(422, "Reason is too long (1000 characters max).")
    if image.content_type not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP photo.")
    data = await image.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is larger than 8 MB.")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(422, "That file could not be read as an image.")

    filename = f"{uuid.uuid4().hex}.jpg"  # never trust the client's filename
    cv2.imwrite(str(config.UPLOAD_DIR / filename), arr)
    case_id = db.create_case(order_id, reason, filename)
    db.add_event(case_id, "customer", "submitted", {"order_id": order_id, "reason": reason})
    try:
        case = agent.run_case(case_id)
    except Exception:
        log.exception("Agent failed for case %s", case_id)
        db.update_case(case_id, status="Needs human review", decision="escalate", decided_by="system",
                       summary="Automatic processing failed; a human will review this case.")
        db.add_event(case_id, "system", "agent_error", None)
        case = db.get_case(case_id)
    return _with_events(case)


@app.get("/api/cases")
def cases():
    out = db.list_cases()
    for c in out:
        c["image_url"] = f"/media/{c['image_file']}"
    return out


@app.get("/api/cases/{case_id}")
def case_detail(case_id: int):
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found.")
    return _with_events(case)


class Review(BaseModel):
    action: Literal["approve", "reject"]
    note: str = ""
    vision_feedback: Literal["correct", "false_positive", "missed_defect"] | None = None
    staff: str = "store staff"


@app.post("/api/cases/{case_id}/review")
def review(case_id: int, body: Review):
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found.")
    status = "Approved" if body.action == "approve" else "Rejected"
    overridden = case["decision"] is not None and case["decision"] != body.action
    db.update_case(case_id, status=status, decision=body.action, decided_by=body.staff)
    db.add_event(case_id, body.staff, "review", {"action": body.action, "note": body.note,
                                                 "overrode_agent": overridden, "vision_feedback": body.vision_feedback})
    return _with_events(db.get_case(case_id))


config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # must exist before it can be mounted
app.mount("/media", StaticFiles(directory=config.UPLOAD_DIR), name="media")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
