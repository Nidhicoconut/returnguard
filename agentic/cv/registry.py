"""Choose the CV backend. ``mock`` (default) or ``real`` (your friend's model).

The ``real`` backend is any importable module that exposes ``build_toolbox() -> CVToolbox``.
Set ``RG_CV_REAL_MODULE`` (default ``agentic.cv.real_template``) and ``RG_CV_BACKEND=real``.
The workflow never changes: only this switch does.
"""
import importlib
from typing import Mapping, Optional

from agentic.config import AgenticConfig
from agentic.cv.interfaces import CVToolbox
from agentic.cv.mock import MockScenario, build_mock_toolbox


class CVBackendError(RuntimeError):
    """The requested CV backend cannot be built. Callers should escalate, not guess."""


def build_toolbox(config: Optional[AgenticConfig] = None, *,
                  mock_scenarios: Optional[Mapping[str, MockScenario]] = None,
                  mock_default: MockScenario = MockScenario.CLEAN) -> CVToolbox:
    cfg = config or AgenticConfig.from_env()
    backend = cfg.cv_backend
    if backend == "mock":
        return build_mock_toolbox(mock_scenarios, mock_default)
    if backend == "real":
        try:
            module = importlib.import_module(cfg.cv_real_module)
            toolbox = module.build_toolbox()
        except NotImplementedError as exc:
            raise CVBackendError(f"real CV backend is not implemented yet: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise CVBackendError(f"could not build real CV backend from {cfg.cv_real_module!r}: {exc}") from exc
        if not isinstance(toolbox, CVToolbox) or toolbox.detector is None:
            raise CVBackendError("real CV backend must return a CVToolbox with at least a detector")
        return toolbox
    raise ValueError(f"unknown CV backend {backend!r} (expected 'mock' or 'real')")
