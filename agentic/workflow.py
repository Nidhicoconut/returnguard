"""LangGraph orchestration of the whole agentic pipeline.

    parse_claim -> fetch_order -> fetch_policy -> run_cv -> aggregate -> judge -> decide
                                                                                   |
                         REQUEST_MORE_EVIDENCE + provider -> collect_evidence -> (next round) parse_claim
                         anything else -------------------------------------------> finalize

Every node is a thin wrapper over a pure function from another module. Safety properties:

* ``run()`` NEVER raises. Any failure becomes ESCALATE_TO_HUMAN with reason SYSTEM_ERROR.
* Each node is individually guarded; the first failure routes to ``failsafe`` and skips the rest.
* Only ``policy_engine.decide`` produces a Decision. The judge contributes a validated verdict enum.
* Each evidence round gets its own ToolRunner with a unique call-id prefix (round 1 -> ``r1_c1``), so the
  full audit trail across rounds has globally unique ids.
* A review task is created for every ESCALATE_TO_HUMAN, even when the workflow itself failed.
* Start-up misconfiguration (bad CV backend, missing LLM key) raises immediately; it is not hidden.
"""
import datetime as dt
from typing import Callable, Optional, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from agentic.claim_parser import analyze_claim
from agentic.config import AgenticConfig
from agentic.cv import CVToolbox, ToolRunner, build_toolbox, run_cv_plan
from agentic.evidence import build_evidence
from agentic.judge import ConsistencyJudge, RulesJudge, build_judges, run_judge
from agentic.models import (
    CaseInput, ClaimInfo, CVCaseReport, EvidenceBundle, ImageInput, JudgeReport, OrderLookupResult, WorkflowResult,
)
from agentic.policy_engine import decide, system_error_result
from agentic.report import build_report, validate_report
from agentic.review import ReviewQueue, build_review_task
from agentic.schemas import Decision, DecisionResult, PolicyClause, ToolCall
from agentic.tools_core import OrderStore, PolicyStore, fetch_order, fetch_policy

# provider(case, decision) -> new photos (or None if the customer has not / will not respond)
EvidenceProvider = Callable[[CaseInput, DecisionResult], Optional[Sequence[ImageInput]]]
MAX_GRAPH_STEPS = 400


class WorkflowState(TypedDict, total=False):
    case: CaseInput
    round: int
    runner: ToolRunner
    provider: Optional[EvidenceProvider]
    claim: ClaimInfo
    order: Optional[OrderLookupResult]
    clauses: list[PolicyClause]
    cv: CVCaseReport
    cv_skipped: bool
    bundle: EvidenceBundle
    judge: JudgeReport
    decision: DecisionResult
    ledgers: list[ToolCall]       # calls from finished evidence rounds
    warnings: list[str]
    provider_done: bool
    system_error: str
    result: WorkflowResult


class AgenticWorkflow:
    def __init__(self, *, order_store: OrderStore, config: Optional[AgenticConfig] = None,
                 toolbox: Optional[CVToolbox] = None, policy_store: Optional[PolicyStore] = None,
                 judge: Optional[ConsistencyJudge] = None, fallback_judge: Optional[ConsistencyJudge] = None,
                 review_queue: Optional[ReviewQueue] = None,
                 clock: Optional[Callable[[], dt.datetime]] = None, today: Optional[Callable[[], dt.date]] = None):
        self.config = config or AgenticConfig.from_env()
        self.order_store = order_store
        self.toolbox = toolbox or build_toolbox(self.config)            # raises CVBackendError on bad setup
        self.policy_store = policy_store or PolicyStore()
        if judge is None:
            judge, fallback_judge = build_judges(self.config)            # raises JudgeConfigError on bad setup
        self.judge, self.fallback_judge = judge, fallback_judge
        self.review_queue = review_queue
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._today = today or dt.date.today
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ public
    def run(self, case: CaseInput, *, evidence_provider: Optional[EvidenceProvider] = None,
            evidence_round: int = 0) -> WorkflowResult:
        """Process one case. Never raises."""
        state: WorkflowState = {"case": case, "round": evidence_round, "provider": evidence_provider, "ledgers": [],
                                "warnings": [], "runner": self._new_runner(evidence_round)}
        try:
            final = self._graph.invoke(state, config={"recursion_limit": MAX_GRAPH_STEPS})
            result = final.get("result")
            if result is None:
                raise RuntimeError("workflow finished without a result")
            return result
        except Exception as exc:  # noqa: BLE001 - last resort: a human gets the case
            return self._failsafe_result(state, f"graph: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ graph
    def _new_runner(self, rnd: int) -> ToolRunner:
        return ToolRunner(timeout_s=self.config.cv_timeout_s, max_retries=self.config.cv_max_retries,
                          id_prefix="" if rnd == 0 else f"r{rnd}_")

    def _build_graph(self):
        g = StateGraph(WorkflowState)
        pipeline = [("parse_claim", self._parse_claim), ("fetch_order", self._fetch_order),
                    ("fetch_policy", self._fetch_policy), ("run_cv", self._run_cv), ("aggregate", self._aggregate),
                    ("judge", self._judge), ("decide", self._decide)]
        for name, fn in pipeline:
            g.add_node(name, self._guard(name, fn))
        g.add_node("collect_evidence", self._guard("collect_evidence", self._collect_evidence))
        g.add_node("finalize", self._guard("finalize", self._finalize))
        g.add_node("failsafe", self._failsafe)
        g.set_entry_point("parse_claim")

        names = [n for n, _ in pipeline]
        for cur, nxt in zip(names, names[1:]):
            g.add_conditional_edges(cur, self._then(nxt), {nxt: nxt, "failsafe": "failsafe"})
        g.add_conditional_edges("decide", self._route_after_decide,
                                {"collect_evidence": "collect_evidence", "finalize": "finalize", "failsafe": "failsafe"})
        g.add_conditional_edges("collect_evidence", self._route_after_collect,
                                {"parse_claim": "parse_claim", "finalize": "finalize", "failsafe": "failsafe"})
        g.add_conditional_edges("finalize", lambda s: "failsafe" if s.get("system_error") else "end",
                                {"end": END, "failsafe": "failsafe"})
        g.add_edge("failsafe", END)
        return g.compile()

    @staticmethod
    def _guard(name: str, fn: Callable[[WorkflowState], dict]):
        def node(state: WorkflowState) -> dict:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001
                return {"system_error": f"{name}: {type(exc).__name__}: {exc}"}
        return node

    @staticmethod
    def _then(nxt: str):
        return lambda s: "failsafe" if s.get("system_error") else nxt

    @staticmethod
    def _route_after_decide(s: WorkflowState) -> str:
        if s.get("system_error"):
            return "failsafe"
        d = s["decision"].decision
        return "collect_evidence" if d is Decision.REQUEST_MORE_EVIDENCE and s.get("provider") else "finalize"

    @staticmethod
    def _route_after_collect(s: WorkflowState) -> str:
        if s.get("system_error"):
            return "failsafe"
        return "finalize" if s.get("provider_done") else "parse_claim"

    # ------------------------------------------------------------------ nodes (thin wrappers)
    def _parse_claim(self, s: WorkflowState) -> dict:
        return {"claim": analyze_claim(s["case"].claim_text)}

    def _fetch_order(self, s: WorkflowState) -> dict:
        res = fetch_order(s["case"].order_id, self.order_store, s["runner"], self.config, self._today())
        return {"order": res.output}  # None when the tool failed; the failure is in the ledger

    def _fetch_policy(self, s: WorkflowState) -> dict:
        res = fetch_policy(self.policy_store, s["runner"])
        return {"clauses": list(res.output.clauses) if res.ok else []}

    def _run_cv(self, s: WorkflowState) -> dict:
        case, order = s["case"], s.get("order")
        if order is None or order.status != "found" or not order.order.window_ok:
            # nothing the photos could say changes the outcome: a human decides. Skip the CV cost.
            return {"cv": CVCaseReport(case_id=case.case_id), "cv_skipped": True}
        cv = run_cv_plan(case, order.order.sku, self.toolbox, s["runner"], self.config)
        return {"cv": cv, "cv_skipped": False}

    def _aggregate(self, s: WorkflowState) -> dict:
        bundle = build_evidence(case=s["case"], claim=s["claim"], order=s.get("order"), clauses=s.get("clauses", []),
                                cv=s["cv"], config=self.config, ledger=s["runner"].ledger)
        return {"bundle": bundle}

    def _judge(self, s: WorkflowState) -> dict:
        if s.get("cv_skipped"):  # no photo evidence to weigh: cheap offline judge, no LLM call
            rep = run_judge(s["bundle"], RulesJudge(), s["runner"])
        else:
            rep = run_judge(s["bundle"], self.judge, s["runner"], self.fallback_judge,
                            timeout_s=self.config.judge_timeout_s, max_retries=self.config.judge_max_retries)
        return {"judge": rep}

    def _decide(self, s: WorkflowState) -> dict:
        return {"decision": decide(s["bundle"], s["judge"], config=self.config, evidence_round=s["round"])}

    def _collect_evidence(self, s: WorkflowState) -> dict:
        case, rnd, warnings = s["case"], s["round"], list(s.get("warnings", []))
        try:
            offered = s["provider"](case, s["decision"]) or []
        except Exception as exc:  # noqa: BLE001 - a flaky provider must not crash the case
            warnings.append(f"evidence provider failed: {type(exc).__name__}: {exc}")
            return {"provider_done": True, "warnings": warnings}
        seen = {i.image_id for i in case.images}
        fresh = []
        for img in offered:
            if img.image_id in seen:
                warnings.append(f"ignored new image with duplicate id {img.image_id!r}")
                continue
            seen.add(img.image_id)
            fresh.append(img)
        if not fresh or rnd >= self.config.max_evidence_rounds:
            return {"provider_done": True, "warnings": warnings}
        runner = s["runner"]
        return {"provider_done": False, "warnings": warnings, "round": rnd + 1,
                "case": case.model_copy(update={"images": list(case.images) + fresh}),
                "ledgers": list(s.get("ledgers", [])) + list(runner.ledger), "runner": self._new_runner(rnd + 1)}

    def _finalize(self, s: WorkflowState) -> dict:
        case, bundle, judge, decision = s["case"], s["bundle"], s["judge"], s["decision"]
        warnings = list(s.get("warnings", []))
        report = build_report(case=case, decision=decision, bundle=bundle, judge=judge, now=self._now(), config=self.config,
                              evidence_round=s["round"], cv_skipped=s.get("cv_skipped", False))
        warnings += [f"report validation: {p}" for p in validate_report(report, bundle.ledger)]
        task, enqueued = self._review(case, decision, bundle, judge, report, s["round"], warnings)
        ledger = list(s.get("ledgers", [])) + list(s["runner"].ledger)
        return {"result": WorkflowResult(
            case_id=case.case_id, decision=decision, rounds_used=s["round"], evidence=bundle, judge=judge, report=report,
            review_task=task, review_enqueued=enqueued, warnings=warnings, ledger=ledger)}

    # ------------------------------------------------------------------ helpers
    def _now(self) -> str:
        return self._clock().isoformat()

    def _review(self, case, decision, bundle, judge, report, rnd, warnings):
        if decision.decision is not Decision.ESCALATE_TO_HUMAN:
            return None, False
        task = build_review_task(case=case, decision=decision, bundle=bundle, judge=judge, report=report,
                                 now=self._now(), evidence_round=rnd)
        if self.review_queue is None:
            return task, False
        try:
            return self.review_queue.enqueue(task), True
        except Exception as exc:  # noqa: BLE001 - never drop an escalated case silently
            warnings.append(f"REVIEW ENQUEUE FAILED ({type(exc).__name__}: {exc}); the case still needs a human")
            return task, False

    def _failsafe(self, s: WorkflowState) -> dict:
        return {"result": self._failsafe_result(s, s.get("system_error", "unknown error"))}

    def _failsafe_result(self, s: WorkflowState, error: str) -> WorkflowResult:
        """Build the safest possible result from whatever partial state exists. Cannot raise."""
        case = s.get("case") or CaseInput(case_id="UNKNOWN", order_id="UNKNOWN")
        decision = system_error_result(error)
        bundle, judge = s.get("bundle"), s.get("judge")
        warnings = list(s.get("warnings", [])) + [f"system error: {error}"]
        try:
            ledger = list(s.get("ledgers", [])) + (list(s["runner"].ledger) if s.get("runner") else [])
        except Exception:  # noqa: BLE001
            ledger = []
        report = task = None
        enqueued = False
        try:
            report = build_report(case=case, decision=decision, bundle=bundle, judge=judge, now=self._now(), config=self.config,
                                  evidence_round=s.get("round", 0), warnings=warnings, cv_skipped=s.get("cv_skipped", False))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"report could not be built: {type(exc).__name__}: {exc}")
        try:
            task, enqueued = self._review(case, decision, bundle, judge, report, s.get("round", 0), warnings)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"review task could not be created: {type(exc).__name__}: {exc}")
        return WorkflowResult(case_id=case.case_id, decision=decision, rounds_used=s.get("round", 0), evidence=bundle,
                              judge=judge, report=report, review_task=task, review_enqueued=enqueued,
                              system_error=error, warnings=warnings, ledger=ledger)
