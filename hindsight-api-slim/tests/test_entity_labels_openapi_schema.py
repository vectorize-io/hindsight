"""Schema tests for the typed ``entity_labels`` request field (issue #3107).

``DryRunExtractRequest`` and ``BankTemplateConfig`` used to type ``entity_labels`` as a bare
``list`` / ``list[dict]``, so the ``LabelGroup`` shape never appeared in the OpenAPI document and
callers couldn't discover it. These are pure-Pydantic tests (no DB / no LLM): they assert the field
now references the ``LabelGroup`` schema, still accepts the legacy ``free_values``/``multi_value``
shape, and round-trips back to plain dicts for the storage path.
"""

from hindsight_api.api.http import BankTemplateConfig, DryRunExtractRequest
from hindsight_api.engine.retain.entity_labels import LabelGroup


def _label_ref_published(model) -> bool:
    """True if the model's JSON schema references a LabelGroup component for entity_labels."""
    schema = model.model_json_schema()
    prop = schema["properties"]["entity_labels"]
    # anyOf[array-of-$ref, null]; find the array branch and confirm its items are a $ref (not a bare object).
    refs = [
        branch["items"]["$ref"]
        for branch in prop["anyOf"]
        if branch.get("type") == "array" and "$ref" in branch.get("items", {})
    ]
    return any("LabelGroup" in ref for ref in refs)


def test_dry_run_entity_labels_publishes_label_group_schema():
    assert _label_ref_published(DryRunExtractRequest)


def test_bank_template_entity_labels_publishes_label_group_schema():
    assert _label_ref_published(BankTemplateConfig)


def test_dry_run_accepts_new_type_shape():
    req = DryRunExtractRequest(
        content="x",
        entity_labels=[{"key": "pedagogy", "type": "value", "values": [{"value": "scaffolding"}]}],
    )
    assert req.entity_labels is not None
    assert isinstance(req.entity_labels[0], LabelGroup)
    assert req.entity_labels[0].key == "pedagogy"
    assert req.entity_labels[0].type == "value"


def test_dry_run_migrates_legacy_free_values_shape():
    """Legacy free_values=True must still map to type='text' (backward compat with parse_entity_labels)."""
    req = DryRunExtractRequest(
        content="x",
        entity_labels=[{"key": "note", "free_values": True}],
    )
    assert req.entity_labels[0].type == "text"


def test_dry_run_migrates_legacy_multi_value_shape():
    req = DryRunExtractRequest(
        content="x",
        entity_labels=[{"key": "interest", "multi_value": True, "values": [{"value": "active"}]}],
    )
    assert req.entity_labels[0].type == "multi-values"


def test_entity_labels_round_trips_to_dicts_for_storage():
    """BankTemplateConfig.get_config_updates() feeds the stored config; it must emit plain dicts
    (not LabelGroup instances) so the persistence path is unchanged."""
    cfg = BankTemplateConfig(entity_labels=[{"key": "pedagogy", "type": "value", "values": [{"value": "scaffolding"}]}])
    updates = cfg.get_config_updates()
    assert isinstance(updates["entity_labels"], list)
    assert all(isinstance(item, dict) for item in updates["entity_labels"])
    assert updates["entity_labels"][0]["key"] == "pedagogy"


def test_entity_labels_none_by_default():
    assert DryRunExtractRequest(content="x").entity_labels is None
    assert BankTemplateConfig().entity_labels is None
