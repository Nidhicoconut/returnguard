"""Offline deterministic judge. Used in tests, as the LLM judge's fallback, and as the default backend.

Contradiction needs positive grounds: a usable image the detector actually scanned, a claim the
detector can test, and no support for the claimed defect. Anything murky is INSUFFICIENT.
Either way the verdict is only an input to the policy engine, which still sends it to a human.
"""
from agentic.models import EvidenceBundle, JudgeDraft, ObservedDefect
from agentic.schemas import ConsistencyResult, EvidenceRef, Finding, Verdict


def _finding(text: str, refs: list[EvidenceRef]) -> Finding:
    return Finding(text=text, refs=refs)


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


class RulesJudge:
    name = "rules"

    def judge(self, bundle: EvidenceBundle) -> JudgeDraft:
        return JudgeDraft(result=self._judge(bundle))

    # ------------------------------------------------------------------
    def _judge(self, b: EvidenceBundle) -> ConsistencyResult:
        f = b.features
        claimed = f.claimed_defect
        ins = lambda why, findings=(): ConsistencyResult(verdict=Verdict.INSUFFICIENT, rationale=why, findings=list(findings))

        if claimed is None:
            return ins("The claim does not name exactly one defect type, so it cannot be checked against the photos.")
        if not f.claim_testable:
            return ins(f"The detector cannot test '{claimed.value}' for this product category.")
        if f.n_scanned_usable_images < 1:
            return ins("No clear photo was successfully analysed by the defect detector.")

        name = claimed.value.replace("_", " ")
        support = sorted([d for d in b.defects if d.defect_class == claimed and d.status in ("confirmed", "unverified")],
                         key=lambda d: -d.score)
        if support:
            best = support[0]
            if f.location_match is False:
                return ins(f"A {name} was detected, but not where the customer said ({b.claim.parsed.location!r} vs "
                           f"{best.part_location!r}).", [self._det_finding(best, name)])
            tail = "confirmed by a region check" if best.status == "confirmed" else "not independently confirmed by a region check"
            findings = [self._det_finding(best, name)]
            if best.region_call_id:
                findings.append(_finding(f"Region check {'confirms' if best.region_confirms else 'does not confirm'} the {name} "
                                         f"(confidence {_pct(best.region_confidence or 0)}).",
                                         [EvidenceRef(image_id=best.image_id, tool_call_id=best.region_call_id, bbox=best.bbox)]))
            return ConsistencyResult(verdict=Verdict.SUPPORTED, rationale=f"The claimed {name} is visible in the photos ({tail}).",
                                     findings=findings)

        disputed = [d for d in b.defects if d.defect_class == claimed and d.status == "disputed"]
        if disputed:
            return ins(f"The detector flagged a {name} but the region check disagrees; the evidence conflicts.",
                       [self._det_finding(disputed[0], name)])
        if f.claimed_defect_weak:
            weak = next(d for d in b.defects if d.defect_class == claimed and d.status == "weak")
            return ins(f"Only a low-confidence {name} detection ({_pct(weak.score)}) was found.", [self._det_finding(weak, name)])

        # positive grounds for a mismatch
        scanned = [i for i in b.images if i.usable and i.detector_ok and i.detector_call_id]
        refs = [EvidenceRef(image_id=i.image_id, tool_call_id=i.detector_call_id) for i in scanned]
        others = sorted([d for d in b.defects if d.status in ("confirmed", "unverified") and d.defect_class != claimed],
                        key=lambda d: -d.score)
        if others:
            o = others[0]
            oname = o.defect_class.value.replace("_", " ")
            return ConsistencyResult(
                verdict=Verdict.CONTRADICTED, rationale=f"The photos show a {oname}, not the claimed {name}.",
                findings=[self._det_finding(o, oname), _finding(f"No {name} was detected in the clear photo(s).", refs)])
        return ConsistencyResult(
            verdict=Verdict.CONTRADICTED,
            rationale=f"No {name} was detected in {len(scanned)} clear photo(s). (Detector recall is limited, so a human reviews this.)",
            findings=[_finding(f"No {name} detected in {len(scanned)} usable image(s).", refs)])

    @staticmethod
    def _det_finding(d: ObservedDefect, name: str):
        where = f" on the {d.part_location}" if d.part_location else ""
        return _finding(f"{name.capitalize()} detected{where} with score {_pct(d.score)} ({d.status}).",
                        [EvidenceRef(image_id=d.image_id, tool_call_id=d.detection_call_id, bbox=d.bbox)])
