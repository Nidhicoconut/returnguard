"""Settings for the agentic layer. Self-contained: does not import ``app/``.

Every value can be overridden with an ``RG_*`` environment variable. Invalid values raise
``ValueError`` at load time (a misconfigured safety setting must not be silently ignored).
"""
import os
from dataclasses import dataclass
from typing import Mapping, Optional

from agentic.schemas import Category

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _bool(name: str, raw: str) -> bool:
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE | _FALSE)}, got {raw!r}")


def _float(name: str, raw: str, lo: float, hi: float) -> float:
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _int(name: str, raw: str, lo: int, hi: int) -> int:
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _choice(name: str, raw: str, options: set[str]) -> str:
    v = raw.strip().lower()
    if v not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}, got {raw!r}")
    return v


def _categories(name: str, raw: str) -> tuple[str, ...]:
    valid = {c.value for c in Category}
    items = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    bad = [i for i in items if i not in valid]
    if bad or not items:
        raise ValueError(f"{name} must be a comma list of {sorted(valid)}, got {raw!r}")
    return items


@dataclass(frozen=True)
class AgenticConfig:
    # --- CV integration
    cv_backend: str = "mock"                       # mock | real
    cv_real_module: str = "agentic.cv.real_template"  # module exposing build_toolbox()
    cv_timeout_s: float = 30.0                     # per tool call
    cv_max_retries: int = 1                        # retries after an exception (not after a timeout)
    cv_skip_on_bad_quality: bool = True            # skip T4/T5/T9 when T1 says the photo is unusable
    cv_max_regions_per_image: int = 5              # cap on T5/T9 follow-ups per image
    cv_supported_categories: tuple[str, ...] = ("apparel_footwear",)  # categories the detector was trained for
    detection_min_score: float = 0.50              # detections below this are "weak" and get no follow-up tools
    region_min_confidence: float = 0.50            # T9 must be at least this sure to count as "confirmed"
    severity_minor_max_area_pct: float = 2.0       # estimated severity buckets, % of image covered
    severity_moderate_max_area_pct: float = 8.0

    # --- Order / policy
    return_window_days: int = 30                   # R1

    # --- Consistency judge
    judge_backend: str = "rules"                   # rules | llm
    judge_provider: str = "anthropic"              # anthropic | openai (any OpenAI-compatible endpoint)
    judge_model: str = "claude-sonnet-5-5"
    judge_timeout_s: float = 60.0
    judge_max_retries: int = 1

    # --- Policy / safeguards (used by later increments)
    auto_approve_enabled: bool = True              # kill switch: False => never AUTO_APPROVE
    audit_sample_rate: float = 0.10                # share of auto-approvals routed to a human spot check
    high_value_limit_inr: float = 10000.0          # R7
    max_evidence_rounds: int = 2                   # REQUEST_MORE_EVIDENCE rounds before escalating
    auto_approve_min_score: float = 0.70           # best claimed-defect detection must reach this to auto-approve

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AgenticConfig":
        e = os.environ if env is None else env
        d = cls()

        def get(key: str, default, parse):
            raw = e.get(key)
            return default if raw is None or raw.strip() == "" else parse(key, raw)

        return cls(
            cv_backend=get("RG_CV_BACKEND", d.cv_backend, lambda k, r: r.strip().lower()),
            cv_real_module=get("RG_CV_REAL_MODULE", d.cv_real_module, lambda k, r: r.strip()),
            cv_timeout_s=get("RG_CV_TIMEOUT_S", d.cv_timeout_s, lambda k, r: _float(k, r, 0.1, 600)),
            cv_max_retries=get("RG_CV_MAX_RETRIES", d.cv_max_retries, lambda k, r: _int(k, r, 0, 5)),
            cv_skip_on_bad_quality=get("RG_CV_SKIP_ON_BAD_QUALITY", d.cv_skip_on_bad_quality, _bool),
            cv_max_regions_per_image=get("RG_CV_MAX_REGIONS", d.cv_max_regions_per_image, lambda k, r: _int(k, r, 1, 50)),
            cv_supported_categories=get("RG_CV_SUPPORTED_CATEGORIES", d.cv_supported_categories, _categories),
            detection_min_score=get("RG_DETECTION_MIN_SCORE", d.detection_min_score, lambda k, r: _float(k, r, 0, 1)),
            region_min_confidence=get("RG_REGION_MIN_CONFIDENCE", d.region_min_confidence, lambda k, r: _float(k, r, 0, 1)),
            severity_minor_max_area_pct=get("RG_SEVERITY_MINOR_MAX_PCT", d.severity_minor_max_area_pct, lambda k, r: _float(k, r, 0, 100)),
            severity_moderate_max_area_pct=get("RG_SEVERITY_MODERATE_MAX_PCT", d.severity_moderate_max_area_pct, lambda k, r: _float(k, r, 0, 100)),
            return_window_days=get("RG_RETURN_WINDOW_DAYS", d.return_window_days, lambda k, r: _int(k, r, 0, 3650)),
            judge_backend=get("RG_JUDGE_BACKEND", d.judge_backend, lambda k, r: _choice(k, r, {"rules", "llm"})),
            judge_provider=get("RG_JUDGE_PROVIDER", d.judge_provider, lambda k, r: _choice(k, r, {"anthropic", "openai"})),
            judge_model=get("RG_JUDGE_MODEL", d.judge_model, lambda k, r: r.strip()),
            judge_timeout_s=get("RG_JUDGE_TIMEOUT_S", d.judge_timeout_s, lambda k, r: _float(k, r, 0.1, 600)),
            judge_max_retries=get("RG_JUDGE_MAX_RETRIES", d.judge_max_retries, lambda k, r: _int(k, r, 0, 5)),
            auto_approve_enabled=get("RG_AUTO_APPROVE_ENABLED", d.auto_approve_enabled, _bool),
            audit_sample_rate=get("RG_AUDIT_SAMPLE_RATE", d.audit_sample_rate, lambda k, r: _float(k, r, 0, 1)),
            high_value_limit_inr=get("RG_HIGH_VALUE_LIMIT_INR", d.high_value_limit_inr, lambda k, r: _float(k, r, 0, 1e9)),
            max_evidence_rounds=get("RG_MAX_EVIDENCE_ROUNDS", d.max_evidence_rounds, lambda k, r: _int(k, r, 0, 10)),
            auto_approve_min_score=get("RG_AUTO_APPROVE_MIN_SCORE", d.auto_approve_min_score, lambda k, r: _float(k, r, 0, 1)),
        )
