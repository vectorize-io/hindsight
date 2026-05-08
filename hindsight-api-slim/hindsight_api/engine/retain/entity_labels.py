"""
Entity labels models and helpers for retain pipeline.

Defines a controlled vocabulary of key:value classification labels
(e.g., 'pedagogy:scaffolding', 'interest:active') that are extracted
at retain time and stored as entities.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model


class LabelValue(BaseModel):
    """A single allowed value for a label group."""

    value: str
    description: str = ""


class MapField(BaseModel):
    """A field within a map-type entity label group. Supports recursion via type='map'."""

    type: Literal["text", "value", "multi-values", "map"] = "text"
    description: str = ""
    values: list[LabelValue] = []
    fields: dict[str, "MapField"] = {}


MapField.model_rebuild()


class LabelGroup(BaseModel):
    """A label group (dimension) with its type and allowed values."""

    key: str
    description: str = ""
    type: Literal["value", "multi-values", "text", "map"] = "value"
    optional: bool = True
    tag: bool = False
    values: list[LabelValue] = []
    fields: dict[str, MapField] = {}


class EntityLabelsConfig(BaseModel):
    """Entity labels configuration for a bank (controlled vocabulary)."""

    attributes: list[LabelGroup] = []


def parse_entity_labels(raw: dict | list | None) -> EntityLabelsConfig | None:
    """
    Parse raw entity labels config into EntityLabelsConfig.

    Accepts:
    - None → returns None
    - list → list of attribute dicts (each may use legacy free_values/multi_value or new type field)
    - dict → {attributes: [...]}

    Legacy migration (backward-compat):
    - free_values=True          → type="text"
    - multi_value=True          → type="multi-values"
    - neither / free_values=False → type="value"

    Args:
        raw: Raw entity labels config from bank config

    Returns:
        EntityLabelsConfig or None if raw is None/empty
    """
    if raw is None:
        return None

    if isinstance(raw, list):
        if not raw:
            return None
        attributes = [LabelGroup.model_validate(_migrate_label_group(a)) for a in raw]
        return EntityLabelsConfig(attributes=attributes)

    if isinstance(raw, dict):
        attrs_raw = raw.get("attributes", [])
        if not attrs_raw:
            return None
        attributes = [LabelGroup.model_validate(_migrate_label_group(a)) for a in attrs_raw]
        return EntityLabelsConfig(attributes=attributes)

    return None


def _migrate_label_group(raw: dict) -> dict:
    """Migrate legacy free_values/multi_value fields to the new type field."""
    if not isinstance(raw, dict) or "type" in raw:
        return raw
    patched = dict(raw)
    if patched.get("free_values"):
        patched["type"] = "text"
    elif patched.get("multi_value"):
        patched["type"] = "multi-values"
    else:
        patched["type"] = "value"
    # Remove legacy keys so Pydantic doesn't error on unknown fields
    patched.pop("free_values", None)
    patched.pop("multi_value", None)
    return patched


def _build_map_fields_model(fields: dict[str, MapField], model_name: str) -> type[BaseModel] | None:
    """
    Build a dynamic Pydantic model for a set of map fields (recursive).

    Each field becomes a typed Pydantic field based on its type:
    - text         → str | None
    - value        → Literal[...] | None
    - multi-values → list[Literal[...]]
    - map          → list[NestedModel] (recursive)

    Returns:
        Dynamic Pydantic model class, or None if no fields defined
    """
    if not fields:
        return None
    model_fields: dict[str, Any] = {}
    for field_name, map_field in fields.items():
        description = map_field.description or field_name

        if map_field.type == "map":
            nested = _build_map_fields_model(map_field.fields, model_name + field_name.capitalize())
            if nested is not None:
                model_fields[field_name] = (
                    list[nested],  # type: ignore[valid-type]
                    Field(default_factory=list, description=description),
                )
        elif map_field.type == "text":
            model_fields[field_name] = (str | None, Field(default=None, description=description))
        else:
            # value / multi-values — enum-constrained
            if not map_field.values:
                model_fields[field_name] = (str | None, Field(default=None, description=description))
                continue
            values = tuple(v.value for v in map_field.values if v.value)
            if not values:
                model_fields[field_name] = (str | None, Field(default=None, description=description))
                continue
            literal_type = Literal[values]  # type: ignore[valid-type]
            if map_field.type == "multi-values":
                model_fields[field_name] = (
                    list[literal_type],  # type: ignore[valid-type]
                    Field(default_factory=list, description=description),
                )
            else:
                model_fields[field_name] = (
                    literal_type | None,  # type: ignore[valid-type]
                    Field(default=None, description=description),
                )

    if not model_fields:
        return None
    return create_model(model_name, **model_fields)


def _build_map_entity_model(group: LabelGroup) -> type[BaseModel] | None:
    """
    Build a dynamic Pydantic model for a map-type entity label group.

    Delegates to ``_build_map_fields_model`` which handles recursion.
    """
    # Capitalize group key for the model name (e.g., "person" → "Person")
    model_name = group.key.capitalize() + "Entity"
    return _build_map_fields_model(group.fields, model_name)


def build_labels_model(labels_cfg: EntityLabelsConfig) -> type[BaseModel] | None:
    """
    Build a dynamic Pydantic model for structured label extraction.

    Each LabelGroup becomes a typed field based on its type:
    - type="text"                        → str | None  (always optional)
    - type="value",   optional=True      → Literal["v1","v2"] | None
    - type="value",   optional=False     → Literal["v1","v2"]  (required)
    - type="multi-values"                → list[Literal["v1","v2"]]
    - type="map"                         → list[MapModel]  (structured entity)

    Args:
        labels_cfg: Parsed EntityLabelsConfig

    Returns:
        Dynamic Pydantic model class, or None if no groups defined
    """
    fields: dict = {}
    for group in labels_cfg.attributes:
        if not group.key:
            continue
        description = group.description or group.key

        if group.type == "map":
            map_model = _build_map_entity_model(group)
            if map_model is not None:
                fields[group.key] = (
                    list[map_model],  # type: ignore[valid-type]
                    Field(default_factory=list, description=description),
                )
        elif group.type == "text":
            # Free-form: any string value accepted, always optional
            fields[group.key] = (str | None, Field(default=None, description=description))
        else:
            # Enum-constrained: must have defined values
            if not group.values:
                continue
            values = tuple(v.value for v in group.values if v.value)
            if not values:
                continue
            # Literal[("v1", "v2")] is equivalent to Literal["v1", "v2"] in Python 3.11+
            literal_type = Literal[values]  # type: ignore[valid-type]
            if group.type == "multi-values":
                fields[group.key] = (
                    list[literal_type],  # type: ignore[valid-type]
                    Field(default_factory=list, description=description),
                )
            elif group.optional:
                fields[group.key] = (
                    literal_type | None,  # type: ignore[valid-type]
                    Field(default=None, description=description),
                )
            else:
                fields[group.key] = (
                    literal_type,  # type: ignore[valid-type]
                    Field(description=description),
                )

    if not fields:
        return None

    return create_model("Labels", **fields)


def _is_map_label_entity(text_lower: str, prefix: str, fields: dict[str, MapField]) -> bool:
    """Recursively check if text matches a map field path (e.g. 'person:address:city:...')."""
    for field_name, map_field in fields.items():
        field_prefix = f"{prefix}{field_name.lower()}:"
        if map_field.type == "map" and map_field.fields:
            if _is_map_label_entity(text_lower, field_prefix, map_field.fields):
                return True
        elif text_lower.startswith(field_prefix):
            return True
    return False


def is_label_entity(text: str, labels_cfg: EntityLabelsConfig, labels_lookup: set[str]) -> bool:
    """
    Return True if entity text belongs to any configured label group.

    For enum groups: checks the pre-built lookup set.
    For text groups: checks that the text starts with a known key prefix.
    For map groups: recursively checks ``key:field:...:value`` patterns.
    """
    if text.lower() in labels_lookup:
        return True
    for group in labels_cfg.attributes:
        if group.type == "text" and group.key and text.lower().startswith(f"{group.key.lower()}:"):
            return True
        if group.type == "map" and group.key and group.fields:
            prefix = f"{group.key.lower()}:"
            if _is_map_label_entity(text.lower(), prefix, group.fields):
                return True
    return False


def build_labels_lookup(labels_cfg: EntityLabelsConfig | list | None) -> set[str]:
    """
    Build a set of valid 'key:value' label strings (lowercase) for fast lookup.

    Accepts either EntityLabelsConfig or raw list/None for backwards compatibility.

    Args:
        labels_cfg: EntityLabelsConfig, raw list of attribute dicts, or None

    Returns:
        Set of lowercase 'key:value' strings
    """
    if labels_cfg is None:
        return set()

    # Accept raw list/dict for backwards compatibility
    if not isinstance(labels_cfg, EntityLabelsConfig):
        parsed = parse_entity_labels(labels_cfg)
        if parsed is None:
            return set()
        labels_cfg = parsed

    valid = set()
    for group in labels_cfg.attributes:
        if group.type in ("text", "map"):
            continue  # text: no fixed vocabulary; map: uses three-level key:field:value strings
        for v in group.values:
            if group.key and v.value:
                valid.add(f"{group.key}:{v.value}".lower())
    return valid
