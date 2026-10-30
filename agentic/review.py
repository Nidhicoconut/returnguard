"""Human review: task creation, an append-only queue (in-memory or JSONL), and audited outcomes.

* Only the system's ESCALATE_TO_HUMAN decisions become tasks. Audit-sample spot checks are tasks too
  (flagged ``would_have_auto_approved``), so humans keep calibrating the automatic path.
* Only a human can decline: the machine has no reject path.
* Nothing is ever edited or deleted: tasks and outcomes are appended as events and the current state
  is rebuilt by replaying them. A task can be resolved once; a second resolution is refused, never
  silently overwritten.
* Persistence happens BEFORE the in-memory state changes, so memory is never ahead of disk.
* No database is touched. The JSONL file is the only storage.
"""
import json
import os
import threading
from pathlib import Path
from typing import Optional, Protocol, Sequence

from agentic.models import (
    CaseInput, CaseReport, EvidenceBundle, JudgeReport, ReviewAction, ReviewOutcome, ReviewTask,
)
from agentic.schemas import Decision, DecisionResult

_HIGH = {"HIGH_VALUE", "INTEGRITY_FLAG", "SYSTEM_ERROR", "TOOL_FAILURE", "PRODUCT_MISMATCH"}
_SPOT_CHECK = {"AUDIT_SAMPLE", "AUTO_DISABLED"}

CHECKLIST = {
    "TOOL_FAILURE": "Re-run the analysis or inspect the photos manually: some tools failed.",
    "ORDER_NOT_FOUND": "Verify the order id with the customer and the order system.",
    "NON_RETURNABLE_CATEGORY": "Apply the category rules (R4) manually; this product type is never auto-decided.",
    "OUT_OF_WINDOW": "Decide whether a goodwill exception applies (R1).",
    "HIGH_VALUE": "Confirm the claim carefully; the order value exceeds the auto-approval limit (R7).",
    "INTEGRITY_FLAG": "Check the flagged photo(s) for reuse or edited metadata before deciding.",
    "PRODUCT_MISMATCH": "Confirm the photo shows the ordered product.",
    "CLAIM_UNCLEAR": "Read the customer's message and clarify which defect is claimed.",
    "UNTESTABLE_CLAIM": "The model cannot judge this defect type here: assess the photos yourself.",
    "QUALITY_UNKNOWN": "Photo quality was not assessed: look at the photos directly.",
    "JUDGE_UNAVAILABLE": "The automated consistency check did not run: compare claim and photos yourself.",
    "JUDGE_DOWNGRADED": "The automated consistency check was unreliable: compare claim and photos yourself.",
    "CLAIM_CONTRADICTED": "Compare claim and photos; the system thinks they disagree (it can be wrong).",
    "LOCATION_MISMATCH": "Check where the defect is versus where the customer said.",
    "UNMENTIONED_DEFECT": "Photos show damage the customer did not mention: check for pre-existing or other damage.",
    "EVIDENCE_EXHAUSTED": "The customer already supplied extra evidence the maximum number of times.",
    "LOW_QUALITY": "Photos were too unclear to analyse automatically.",
    "NO_PHOTO": "No photo was provided.",
    "INSUFFICIENT_EVIDENCE": "Evidence was not strong enough for the system to decide.",
    "PRODUCT_UNVERIFIED": "The product in the photos could not be verified automatically.",
    "REGION_UNCONFIRMED": "The detected defect was not confirmed by a second check: look at the marked region.",
    "LOW_DETECTION_SCORE": "Detection confidence is low: look at the marked region.",
    "AUTO_DISABLED": "Automatic approval is off; this case would otherwise have been approved.",
    "AUDIT_SAMPLE": "Spot check: the system would have approved this. Confirm it was right.",
    "DEFAULT_DENY": "The case did not meet every condition for automatic approval.",
}


class ReviewError(Exception):
    """Invalid review operation (unknown/already-resolved task, missing reviewer or note...)."""


def build_review_task(*, case: CaseInput, decision: DecisionResult, bundle: Optional[EvidenceBundle],
                      judge: Optional[JudgeReport], report: Optional[CaseReport], now: str,
                      evidence_round: int = 0) -> ReviewTask:
    codes = list(decision.reason_codes)
    spot = bool(codes) and set(codes) <= _SPOT_CHECK
    priority = "high" if _HIGH & set(codes) else "low" if codes == ["AUDIT_SAMPLE"] else "normal"
    checklist = [CHECKLIST[c] for c in codes if c in CHECKLIST] or ["Review the case manually."]
    return ReviewTask(case_id=case.case_id, created_at=now, priority=priority, machine_decision=decision.decision,
                      reason_codes=codes, clause_ids=list(decision.clause_ids), would_have_auto_approved=spot,
                      checklist=checklist, evidence_round=evidence_round, images=list(case.images),
                      evidence=bundle, judge=judge, report=report)


class ReviewQueue(Protocol):
    def enqueue(self, task: ReviewTask) -> ReviewTask: ...
    def get(self, task_id: str) -> Optional[ReviewTask]: ...
    def list_open(self, priority: Optional[str] = None) -> list[ReviewTask]: ...
    def resolve(self, task_id: str, *, reviewer: str, action: str, note: str = "", now: str = "") -> ReviewOutcome: ...
    def outcomes(self, case_id: Optional[str] = None) -> list[ReviewOutcome]: ...


_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


class InMemoryReviewQueue:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: dict[str, ReviewTask] = {}
        self._outcomes: list[ReviewOutcome] = []
        self._seq = 0

    # persistence hook (JSONL subclass overrides); must raise on failure
    def _persist(self, event: dict) -> None:
        return None

    def enqueue(self, task: ReviewTask) -> ReviewTask:
        with self._lock:
            if task.status != "open":
                raise ReviewError("only open tasks can be enqueued")
            if task.task_id and task.task_id in self._tasks:
                raise ReviewError(f"task {task.task_id} already exists")
            stored = task.model_copy(update={"task_id": task.task_id or f"RT-{self._seq + 1:05d}"})
            self._persist({"type": "task", "task": stored.model_dump(mode="json")})
            self._seq += 1
            self._tasks[stored.task_id] = stored
            return stored

    def get(self, task_id: str) -> Optional[ReviewTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_open(self, priority: Optional[str] = None) -> list[ReviewTask]:
        with self._lock:
            rows = [t for t in self._tasks.values() if t.status == "open" and (priority is None or t.priority == priority)]
        return sorted(rows, key=lambda t: (_PRIORITY_RANK[t.priority], t.task_id))

    def resolve(self, task_id: str, *, reviewer: str, action: str, note: str = "", now: str = "") -> ReviewOutcome:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ReviewError(f"unknown task {task_id!r}")
            if task.status != "open":
                raise ReviewError(f"task {task_id} is already resolved")
            if not (reviewer or "").strip():
                raise ReviewError("a reviewer name is required")
            if action not in ReviewAction.ALL:
                raise ReviewError(f"action must be one of {ReviewAction.ALL}, got {action!r}")
            if action == ReviewAction.DECLINE and len((note or "").strip()) < 3:
                raise ReviewError("declining requires a note explaining why")
            approved = action in (ReviewAction.APPROVE_REFUND, ReviewAction.APPROVE_REPLACEMENT)
            outcome = ReviewOutcome(
                task_id=task_id, case_id=task.case_id, reviewer=reviewer.strip(), action=action, note=(note or "").strip(),
                decided_at=now or task.created_at, machine_decision=task.machine_decision,
                machine_reason_codes=list(task.reason_codes), overrides_machine=task.would_have_auto_approved and not approved)
            self._persist({"type": "outcome", "outcome": outcome.model_dump(mode="json")})
            self._tasks[task_id] = task.model_copy(update={"status": "resolved"})
            self._outcomes.append(outcome)
            return outcome

    def outcomes(self, case_id: Optional[str] = None) -> list[ReviewOutcome]:
        with self._lock:
            return [o for o in self._outcomes if case_id is None or o.case_id == case_id]


class JsonlReviewQueue(InMemoryReviewQueue):
    """Append-only event log, one JSON object per line. Rebuilt by replay on start-up."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.load_warnings: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._replay()

    def _replay(self) -> None:
        if not self.path.exists():
            return
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                if ev["type"] == "task":
                    t = ReviewTask.model_validate(ev["task"])
                    self._tasks[t.task_id] = t
                    self._seq = max(self._seq, int(t.task_id.rsplit("-", 1)[-1]))
                elif ev["type"] == "outcome":
                    o = ReviewOutcome.model_validate(ev["outcome"])
                    self._outcomes.append(o)
                    if o.task_id in self._tasks:
                        self._tasks[o.task_id] = self._tasks[o.task_id].model_copy(update={"status": "resolved"})
                else:
                    raise ValueError(f"unknown event type {ev['type']!r}")
            except Exception as exc:  # noqa: BLE001 - never lose the rest of the log to one bad line
                self.load_warnings.append(f"line {n}: skipped ({type(exc).__name__}: {exc})")

    def _persist(self, event: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
