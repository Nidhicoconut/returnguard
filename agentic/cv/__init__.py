"""CV integration layer: interfaces, runner, mocks, registry."""
from agentic.cv.interfaces import (
    CVToolbox, DefectDetector, IntegrityChecker, ProductMatcher, QualityChecker, RegionVerifier, Segmenter,
)
from agentic.cv.mock import MockScenario, build_mock_toolbox
from agentic.cv.plan import run_cv_plan
from agentic.cv.registry import CVBackendError, build_toolbox
from agentic.cv.runner import ToolResult, ToolRunner

__all__ = [
    "CVToolbox", "DefectDetector", "IntegrityChecker", "ProductMatcher", "QualityChecker", "RegionVerifier",
    "Segmenter", "MockScenario", "build_mock_toolbox", "run_cv_plan", "CVBackendError", "build_toolbox",
    "ToolResult", "ToolRunner",
]
