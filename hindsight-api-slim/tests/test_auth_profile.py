import pytest

from hindsight_api.extensions import Extension
from hindsight_api.extensions.auth_profile import get_auth_profile_info, validate_auth_profile


class ProfileTenant(Extension):
    auth_profile = "example"
    auth_profile_component = "tenant"
    required_auth_profile_components = frozenset({"tenant", "operation_validator"})


class ProfileValidator(Extension):
    auth_profile = "example"
    auth_profile_component = "operation_validator"
    required_auth_profile_components = frozenset({"tenant", "operation_validator"})


def test_auth_profile_disabled_is_compatible(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_AUTH_PROFILE", raising=False)

    info = get_auth_profile_info(None, None)

    assert info.auth_profile == "disabled"
    assert info.auth_profile_ready is True
    assert info.missing_components == ()
    validate_auth_profile(None, None)


def test_auth_profile_requires_a_loaded_profile_extension(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_AUTH_PROFILE", "example")

    with pytest.raises(ValueError, match="no loaded extension declares that profile"):
        validate_auth_profile(None, None)


def test_auth_profile_reports_missing_required_components(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_AUTH_PROFILE", "example")

    info = get_auth_profile_info(ProfileTenant({}), None)

    assert info.auth_profile_ready is False
    assert info.missing_components == ("operation_validator",)
    with pytest.raises(ValueError, match="operation_validator"):
        validate_auth_profile(ProfileTenant({}), None)


def test_auth_profile_accepts_complete_group(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_AUTH_PROFILE", "example")

    info = get_auth_profile_info(ProfileTenant({}), ProfileValidator({}))

    assert info.auth_profile_ready is True
    assert info.missing_components == ()
    validate_auth_profile(ProfileTenant({}), ProfileValidator({}))
