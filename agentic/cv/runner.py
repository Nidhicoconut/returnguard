"""ToolRunner: the single gate every tool call passes through.

For each call it assigns ``call_id`` / ``step_no``, times it, enforces a timeout, retries once on
an exception, validates the output against its schema, and appends a ``ToolCall`` to the ledger.
It NEVER raises for a tool failure: the failure is returned as a ``ToolResult`` with ``error`` set,
so a flaky or crashing model can only ever lead to a safer (more cautious) outcome.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel

from agentic.schemas import BBox, EvidenceRef, ToolCall


@dataclass
class ToolResult:
    call: ToolCall
    output: Any = None          # validated schema object, or list of them; None on failure
    image_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.call.error is None

    @property
    def call_id(self) -> str:
        return self.call.call_id

    def ref(self, bbox: Optional[BBox] = None) -> EvidenceRef:
        """Evidence pointer for reports/findings. Only valid for a call tied to an image."""
        if self.image_id is None:
            raise ValueError("this tool call is not tied to an image")
        return EvidenceRef(image_id=self.image_id, tool_call_id=self.call.call_id, bbox=bbox)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


class ToolRunner:
    def __init__(self, timeout_s: float = 30.0, max_retries: int = 1, clock: Callable[[], float] = time.monotonic,
                 id_prefix: str = ""):
        self.id_prefix = id_prefix  # e.g. "r1_" for the second evidence round, keeping call ids unique across rounds
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._clock = clock
        self._step = 0
        self.ledger: list[ToolCall] = []

    def call(
        self,
        tool: str,
        fn: Callable[[], Any],
        *,
        args: dict,
        expect: type[BaseModel],
        many: bool = False,
        image_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> ToolResult:
        """Run ``fn`` (a zero-arg closure over the real arguments). ``args`` is the JSON-safe
        description stored in the audit record. ``expect`` is the schema the output must satisfy;
        ``many=True`` means a list of ``expect``. ``timeout_s`` / ``max_retries`` override the runner
        defaults for this one call (e.g. a slower LLM call)."""
        timeout = self.timeout_s if timeout_s is None else timeout_s
        retries = self.max_retries if max_retries is None else max_retries
        self._step += 1
        call_id, step_no = f"{self.id_prefix}c{self._step}", self._step
        safe_args = _jsonable(args)
        t0 = self._clock()
        output: Any = None
        error: Optional[str] = None

        for _attempt in range(retries + 1):
            try:
                raw = self._run_with_timeout(fn, timeout)
                output = self._validate(raw, expect, many, image_id)
                error = None
                break
            except FutureTimeout:
                error = f"TimeoutError: {tool} exceeded {timeout}s"
                output = None
                break  # do not retry a hang: it would only stack up stuck workers
            except Exception as exc:  # noqa: BLE001 - tool failures must never escape
                error = f"{type(exc).__name__}: {exc}"
                output = None

        latency_ms = int((self._clock() - t0) * 1000)
        record = ToolCall(
            call_id=call_id, step_no=step_no, tool=tool, args=safe_args,
            result={} if error else self._result_dict(output, many),
            latency_ms=latency_ms, error=error,
        )
        self.ledger.append(record)
        return ToolResult(call=record, output=output, image_id=image_id)

    # ------------------------------------------------------------------ internals
    def _run_with_timeout(self, fn: Callable[[], Any], timeout: float) -> Any:
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # never block on a hung worker

    @staticmethod
    def _validate(raw: Any, expect: type[BaseModel], many: bool, image_id: Optional[str]) -> Any:
        if many:
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"expected a list of {expect.__name__}, got {type(raw).__name__}")
            out: Any = [expect.model_validate(item) for item in raw]
            items = out
        else:
            if raw is None:
                raise TypeError(f"expected {expect.__name__}, got None")
            out = expect.model_validate(raw)
            items = [out]
        if image_id is not None:  # guard against a tool answering about the wrong photo
            for it in items:
                got = getattr(it, "image_id", image_id)
                if got != image_id:
                    raise ValueError(f"output image_id {got!r} does not match requested {image_id!r}")
        return out

    @staticmethod
    def _result_dict(output: Any, many: bool) -> dict:
        if many:
            return {"items": _jsonable(output)}
        return _jsonable(output)

    def ledger_json(self) -> str:
        return json.dumps([c.model_dump(mode="json") for c in self.ledger], indent=2)
