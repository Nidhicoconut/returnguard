"""Rule-based claim parser (FR-2). Pure, offline, deterministic.

Customer text is treated as DATA: it is only pattern-matched, never interpreted as instructions.
Anything ambiguous resolves to "unknown" (which later steps treat as insufficient evidence).
"""
import re

from agentic.models import ClaimInfo
from agentic.schemas import DefectClass, ParsedClaim, Resolution, Severity, Timing

MAX_CLAIM_CHARS = 2000

_DEFECT_PATTERNS: dict[DefectClass, str] = {
    DefectClass.stain_discoloration: r"stains?|stained|spots?|discolou?r\w*|dirt\w*|faded|fading|bleed\w*|bled",
    DefectClass.tear_hole: r"holes?|tears?|tearing|torn|rips?|ripped|ripping|cuts?|punctur\w+|splits?",
    DefectClass.crack_break: r"crack\w*|brok(?:e|en)|breaks?|shatter\w*|snapped|chipped",
    DefectClass.dent_deformation: r"dents?|dented|bent|deform\w+|warped|crushed",
    DefectClass.scratch_scuff: r"scratch\w*|scuff\w*|scrap(?:e|ed|es)",
    DefectClass.missing_part: r"missing|not included|absent|(?:fell|came|falling) off",
}
_NEGATOR = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bnone\b|n't\b)(?:\s+\w+){0,3}\s*$")

_LOC_BASE = (r"sleeves?|collar|neckline|neck|hem|cuffs?|pockets?|buttons?|zip(?:per)?|seams?|shoulders?|waist|chest|"
             r"soles?|heels?|toe|laces?|straps?|hinges?|screen|lid|handles?|rim|base|corners?|edges?|case|body|legs?|knees?|hood")
_LOC = re.compile(rf"\b(?:(?:left|right|upper|lower|inner|outer|top|bottom|front|back)\s+)?(?:{_LOC_BASE})\b")

_SEV = {
    Severity.minor: re.compile(r"\b(?:small|tiny|minor|slight\w*|little|barely)\b"),
    Severity.moderate: re.compile(r"\b(?:medium|moderate|noticeable)\b"),
    Severity.severe: re.compile(r"\b(?:big|large|huge|deep|major|severe\w*|badly|completely|massive)\b"),
}
_ON_ARRIVAL = re.compile(r"\b(?:on arrival|arrived|out of the box|when i opened|as soon as|upon opening|from the start|delivered)\b")
_AFTER_USE = re.compile(r"\b(?:after (?:a |one |two |three |few |several )?(?:\w+ )?(?:days?|weeks?|months?|wash\w*|use|using|wearing)|"
                        r"washed|after i (?:used|wore|washed)|while (?:using|wearing)|worn|wore)\b")
_REPLACE = re.compile(r"\b(?:replace\w*|exchange\w*|swap)\b")
_REFUND = re.compile(r"\b(?:refund\w*|money back|reimburse\w*)\b")


def _clean(text: str) -> tuple[str, bool]:
    text = (text or "")
    truncated = len(text) > MAX_CLAIM_CHARS
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text[:MAX_CLAIM_CHARS])
    return text.strip(), truncated


def extract_defect_mentions(text: str) -> list[DefectClass]:
    """Defects the customer names, in order of first appearance. Negated mentions ("no stain") are ignored."""
    low = re.sub(r"\bnot only\b", "also", text.lower())
    found: list[tuple[int, DefectClass]] = []
    for defect, pat in _DEFECT_PATTERNS.items():
        for m in re.finditer(rf"\b(?:{pat})\b", low):
            window = low[max(0, m.start() - 40):m.start()]
            if _NEGATOR.search(window):
                continue
            found.append((m.start(), defect))
            break  # first non-negated mention of this class is enough
    found.sort(key=lambda t: t[0])
    return [d for _, d in found]


def _one(patterns: dict, low: str, default):
    hits = [k for k, rx in patterns.items() if rx.search(low)]
    return hits[0] if len(hits) == 1 else default


def analyze_claim(text: str) -> ClaimInfo:
    clean, truncated = _clean(text)
    low = clean.lower()
    mentions = extract_defect_mentions(clean)
    loc = _LOC.search(low)
    arrival, after = bool(_ON_ARRIVAL.search(low)), bool(_AFTER_USE.search(low))
    timing = Timing.on_arrival if arrival and not after else Timing.after_use if after and not arrival else Timing.unknown
    replace, refund = bool(_REPLACE.search(low)), bool(_REFUND.search(low))
    resolution = Resolution.replacement if replace and not refund else Resolution.refund if refund and not replace else Resolution.unspecified
    parsed = ParsedClaim(
        defect_type=mentions[0] if len(mentions) == 1 else None,  # several different defects => unclear
        location=loc.group(0) if loc else "",
        severity=_one(_SEV, low, Severity.unknown),
        timing=timing,
        requested_resolution=resolution,
    )
    return ClaimInfo(parsed=parsed, mentioned_defects=mentions, claim_text=clean, truncated=truncated)


def parse_claim(text: str) -> ParsedClaim:
    return analyze_claim(text).parsed
