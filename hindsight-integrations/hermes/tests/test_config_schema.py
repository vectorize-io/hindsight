from hindsight_hermes.config_schema import CONFIG_SCHEMA


def test_dashboard_config_schema_exposes_per_user_bank_json_field():
    field = next(f for f in CONFIG_SCHEMA["fields"] if f.get("key") == "bank_id_by_user")
    assert field["kind"] == "json"
    assert field["default"] == {}
