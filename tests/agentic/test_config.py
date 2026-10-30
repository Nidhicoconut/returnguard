import pytest

from agentic.config import AgenticConfig


def test_defaults_match_approved_decisions():
    c = AgenticConfig.from_env({})
    assert c.audit_sample_rate == 0.10
    assert c.cv_backend == "mock"
    assert c.auto_approve_enabled is True
    assert c.max_evidence_rounds == 2


def test_env_overrides():
    c = AgenticConfig.from_env({"RG_AUDIT_SAMPLE_RATE": "0.25", "RG_AUTO_APPROVE_ENABLED": "off",
                                "RG_CV_BACKEND": "REAL", "RG_CV_MAX_RETRIES": "0"})
    assert c.audit_sample_rate == 0.25 and c.auto_approve_enabled is False
    assert c.cv_backend == "real" and c.cv_max_retries == 0


def test_blank_values_use_defaults():
    assert AgenticConfig.from_env({"RG_AUDIT_SAMPLE_RATE": "  "}).audit_sample_rate == 0.10


@pytest.mark.parametrize("env", [
    {"RG_AUDIT_SAMPLE_RATE": "1.5"}, {"RG_AUDIT_SAMPLE_RATE": "-0.1"}, {"RG_AUDIT_SAMPLE_RATE": "lots"},
    {"RG_AUTO_APPROVE_ENABLED": "maybe"}, {"RG_CV_TIMEOUT_S": "0"}, {"RG_CV_MAX_RETRIES": "99"},
])
def test_invalid_settings_fail_loudly(env):
    with pytest.raises(ValueError):
        AgenticConfig.from_env(env)


def test_config_is_frozen():
    with pytest.raises(Exception):
        AgenticConfig().audit_sample_rate = 1.0  # type: ignore[misc]


def test_new_settings_defaults_and_overrides():
    c = AgenticConfig.from_env({})
    assert (c.return_window_days, c.judge_backend, c.judge_model) == (30, "rules", "claude-sonnet-5-5")
    assert c.cv_supported_categories == ("apparel_footwear",) and c.region_min_confidence == 0.5
    c = AgenticConfig.from_env({"RG_JUDGE_BACKEND": "LLM", "RG_JUDGE_PROVIDER": "openai", "RG_RETURN_WINDOW_DAYS": "14",
                                "RG_CV_SUPPORTED_CATEGORIES": "apparel_footwear, electronics"})
    assert (c.judge_backend, c.judge_provider, c.return_window_days) == ("llm", "openai", 14)
    assert c.cv_supported_categories == ("apparel_footwear", "electronics")


@pytest.mark.parametrize("env", [{"RG_JUDGE_BACKEND": "magic"}, {"RG_JUDGE_PROVIDER": "gemini"},
                                 {"RG_CV_SUPPORTED_CATEGORIES": "toys"}, {"RG_RETURN_WINDOW_DAYS": "-1"}])
def test_new_settings_reject_bad_values(env):
    with pytest.raises(ValueError):
        AgenticConfig.from_env(env)


def test_auto_approve_min_score_setting():
    assert AgenticConfig.from_env({}).auto_approve_min_score == 0.70
    assert AgenticConfig.from_env({"RG_AUTO_APPROVE_MIN_SCORE": "0.8"}).auto_approve_min_score == 0.8
    with pytest.raises(ValueError):
        AgenticConfig.from_env({"RG_AUTO_APPROVE_MIN_SCORE": "1.5"})
