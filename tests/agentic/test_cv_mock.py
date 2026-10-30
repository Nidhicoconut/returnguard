import pytest

from agentic.config import AgenticConfig
from agentic.cv import CVBackendError, CVToolbox, MockScenario, build_toolbox, build_mock_toolbox
from agentic.cv.interfaces import (
    DefectDetector, IntegrityChecker, ProductMatcher, QualityChecker, RegionVerifier, Segmenter,
)
from agentic.cv.mock import ScenarioResolver
from agentic.models import CaseInput, ImageInput
from agentic.schemas import DefectClass

SKU = "SKU-TSHIRT-1"


def img(i="img1", path="/x/photo.jpg"):
    return ImageInput(image_id=i, path=path)


def test_mock_toolbox_satisfies_every_protocol():
    tb = build_mock_toolbox()
    assert isinstance(tb.detector, DefectDetector) and isinstance(tb.quality, QualityChecker)
    assert isinstance(tb.integrity, IntegrityChecker) and isinstance(tb.product_matcher, ProductMatcher)
    assert isinstance(tb.segmenter, Segmenter) and isinstance(tb.region_verifier, RegionVerifier)
    assert tb.name == "mock" and len(tb.available()) == 6


@pytest.mark.parametrize("path,expected", [
    ("/u/case7__hole.jpg", MockScenario.HOLE),
    ("/u/stain_on_sleeve.png", MockScenario.STAIN_ON_SLEEVE),
    ("/u/BLURRY-shot.jpg", MockScenario.BLURRY),
    ("/u/whole_shirt.jpg", MockScenario.CLEAN),        # "hole" inside "whole" must not match
    ("/u/random.jpg", MockScenario.CLEAN),
])
def test_scenario_resolution_by_filename(path, expected):
    assert ScenarioResolver().resolve(img(path=path)) is expected


def test_explicit_mapping_beats_filename():
    r = ScenarioResolver({"img1": MockScenario.WRONG_PRODUCT})
    assert r.resolve(img(path="/u/hole.jpg")) is MockScenario.WRONG_PRODUCT


def test_clean_scenario_finds_nothing_and_is_deterministic():
    tb = build_mock_toolbox()
    assert tb.detector.detect(img()) == []
    assert tb.detector.detect(img()) == tb.detector.detect(img())
    assert tb.quality.check(img()).ok and tb.integrity.check(img(), [img()]).duplicate_of is None
    assert tb.product_matcher.match(img(), SKU).match is True


def test_defect_scenarios_return_schema_valid_detections():
    tb = build_mock_toolbox()
    (stain,) = tb.detector.detect(img(path="a_stain_on_sleeve.jpg"))
    (hole,) = tb.detector.detect(img(path="a_hole.jpg"))
    assert stain.defect_class is DefectClass.stain_discoloration and stain.score > 0.8
    assert hole.defect_class is DefectClass.tear_hole
    seg = tb.segmenter.segment(img(path="a_stain_on_sleeve.jpg"), stain.bbox)
    assert seg.part_location == "sleeve" and seg.area_pct == pytest.approx(2.7)
    rc = tb.region_verifier.verify(img(path="a_stain_on_sleeve.jpg"), stain.bbox, stain.defect_class)
    assert rc.confirms and rc.image_id == "img1"
    # asking about a defect class that is not there -> does not confirm
    assert not tb.region_verifier.verify(img(path="a_stain_on_sleeve.jpg"), stain.bbox, DefectClass.tear_hole).confirms


def test_failure_scenarios():
    tb = build_mock_toolbox()
    assert not tb.quality.check(img(path="x_blurry.jpg")).ok
    assert tb.integrity.check(img(path="x_duplicate_image.jpg"), []).duplicate_of
    pm = tb.product_matcher.match(img(path="x_wrong_product.jpg"), SKU)
    assert pm.match is False and pm.top_k[0][0] != SKU
    with pytest.raises(RuntimeError):
        tb.detector.detect(img(path="x_detector_crash.jpg"))


def test_mock_never_touches_the_filesystem():
    # path does not exist; mocks must not try to open it
    assert build_mock_toolbox().detector.detect(img(path="/definitely/not/here/hole.jpg"))


# ---------------------------------------------------------------- registry
def test_registry_defaults_to_mock():
    assert build_toolbox(AgenticConfig()).name == "mock"


def test_registry_passes_scenarios_to_mock():
    tb = build_toolbox(AgenticConfig(), mock_scenarios={"img1": MockScenario.HOLE})
    assert tb.detector.detect(img())


def test_registry_real_template_fails_safely():
    with pytest.raises(CVBackendError, match="not implemented"):
        build_toolbox(AgenticConfig(cv_backend="real"))


def test_registry_unknown_backend_and_bad_module():
    with pytest.raises(ValueError):
        build_toolbox(AgenticConfig(cv_backend="magic"))
    with pytest.raises(CVBackendError):
        build_toolbox(AgenticConfig(cv_backend="real", cv_real_module="no.such.module"))


def test_registry_loads_a_custom_real_module(tmp_path, monkeypatch):
    (tmp_path / "fake_real_cv.py").write_text(
        "from agentic.cv.interfaces import CVToolbox\n"
        "class D:\n    def detect(self, image):\n        return []\n"
        "def build_toolbox():\n    return CVToolbox(detector=D(), name='fake-real')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    tb = build_toolbox(AgenticConfig(cv_backend="real", cv_real_module="fake_real_cv"))
    assert tb.name == "fake-real" and tb.available() == ["detector"]


def test_registry_rejects_toolbox_without_detector(tmp_path, monkeypatch):
    (tmp_path / "nodetector_cv.py").write_text(
        "from agentic.cv.interfaces import CVToolbox\ndef build_toolbox():\n    return CVToolbox()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(CVBackendError, match="detector"):
        build_toolbox(AgenticConfig(cv_backend="real", cv_real_module="nodetector_cv"))


def test_models_validate_input():
    with pytest.raises(ValueError):
        ImageInput(image_id=" ", path="a.jpg")
    with pytest.raises(ValueError):
        CaseInput(case_id="c", order_id="o", images=[img("a"), img("a")])
