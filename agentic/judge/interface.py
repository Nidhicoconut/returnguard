"""Judge contract. A judge ASSESSES whether the evidence is consistent with the claim.
It returns a verdict + cited findings. It never sees, emits or influences a ``Decision``."""
from typing import Protocol, runtime_checkable

from agentic.models import EvidenceBundle, JudgeDraft


class JudgeConfigError(RuntimeError):
    """The requested judge cannot be built (e.g. missing API key). Fail loudly at start-up."""


@runtime_checkable
class ConsistencyJudge(Protocol):
    name: str  # "rules" | "llm"

    def judge(self, bundle: EvidenceBundle) -> JudgeDraft: ...
