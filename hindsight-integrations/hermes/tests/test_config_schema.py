from hindsight_hermes.config_schema import CONFIG_SCHEMA


def test_dashboard_config_schema_exposes_compatible_per_user_bank_field():
    field = next(f for f in CONFIG_SCHEMA["fields"] if f.get("key") == "bank_id_by_user")
    assert field["kind"] == "text"  # test stub deliberately has no KIND_JSON
    assert field["default"] == "{}"
