"""A schema we send a provider must be one the provider can accept.

Pydantic renders a ``Field(discriminator=...)`` union as ``oneOf`` plus a
``discriminator`` block. Nothing we talk to takes that: OpenAI's strict subset
accepts ``anyOf`` and rejects ``oneOf``, and the Gemini SDK's ``types.Schema``
forbids both keys outright — it raises while building the request, so the call
never reaches the network and no amount of retrying helps.

That is why the mental-model delta call spent its life in text mode with a
hand-written JSON parser (#3901). These tests pin the rewrite that lets it send
a schema like every other pipeline call, and pin the property that makes the
rewrite safe to apply everywhere: for a model with no tagged union it changes
nothing at all.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

import pytest
from pydantic import BaseModel, Field

from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
from hindsight_api.engine.structured_output import (
    has_tagged_union,
    provider_json_schema,
    strict_json_schema,
)


class _Flat(BaseModel):
    """A model shaped like the ones already in the pipeline: no union."""

    name: str
    count: int = 0


class _Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    lives: int = 9


class _Dog(BaseModel):
    kind: Literal["dog"] = "dog"
    good: bool = True


class _Pets(BaseModel):
    pets: list[Annotated[Union[_Cat, _Dog], Field(discriminator="kind")]] = Field(default_factory=list)


def _keys_in(schema: dict) -> str:
    return json.dumps(schema)


class TestUnionsAreTransportable:
    @pytest.mark.parametrize("model", [DeltaOperationList, _Pets])
    @pytest.mark.parametrize("serialize", [provider_json_schema, strict_json_schema])
    def test_no_oneof_or_discriminator_survives(self, model: type[BaseModel], serialize):
        """Both serializers must produce a schema every provider can accept."""
        rendered = _keys_in(serialize(model))
        assert "oneOf" not in rendered
        assert '"discriminator"' not in rendered
        assert "anyOf" in rendered

    def test_variants_are_preserved(self):
        """Rewriting the union must not drop any operation type."""
        schema = provider_json_schema(DeltaOperationList)
        variants = schema["properties"]["operations"]["items"]["anyOf"]
        assert len(variants) == 8, "every delta operation must remain reachable"

    @pytest.mark.parametrize("serialize", [provider_json_schema, strict_json_schema])
    def test_the_discriminator_tag_is_required_on_every_variant(self, serialize):
        """Dropping the ``discriminator`` block makes the tag load-bearing.

        Pydantic leaves ``op`` out of ``required`` because each variant defaults it,
        and ``anyOf`` alone gives a reader nothing else to tell eight near-identical
        object shapes apart. A grammar-constrained model omitting ``op`` would emit
        an operation matching no variant — the exact validation failure this path
        exists to prevent.
        """
        for name, definition in serialize(DeltaOperationList)["$defs"].items():
            assert "op" in definition.get("required", []), f"{name} does not require its discriminator"

    def test_pydantic_still_parses_what_the_schema_describes(self):
        """The rewrite is a serialization concern; validation is unchanged, so a
        payload the schema permits must still round-trip through the model."""
        payload = {"operations": [{"op": "add_section", "heading": "Tools", "blocks": ["- Linear"]}]}
        assert DeltaOperationList.model_validate(payload).operations[0].heading == "Tools"


class TestNonUnionModelsAreUntouched:
    """The safety property behind applying this serializer to every provider."""

    @pytest.mark.parametrize("model", [_Flat, _Cat])
    def test_union_safe_schema_is_identical_for_models_without_unions(self, model: type[BaseModel]):
        assert provider_json_schema(model) == model.model_json_schema()

    @pytest.mark.parametrize("model", [_Flat, _Cat])
    def test_has_tagged_union_is_false(self, model: type[BaseModel]):
        assert has_tagged_union(model) is False

    @pytest.mark.parametrize("model", [DeltaOperationList, _Pets])
    def test_has_tagged_union_is_true(self, model: type[BaseModel]):
        assert has_tagged_union(model) is True


class TestGeminiAcceptsTheSchema:
    """The failure this guards is raised by the SDK, before any network call, so
    it is reproducible offline — which is the only reason it can be a unit test."""

    def test_sdk_rejects_the_raw_pydantic_union(self):
        """If this ever starts passing, the SDK grew support and the branch in
        ``gemini_llm`` that serializes the schema by hand can be removed."""
        from google.genai import _transformers as gemini_transformers
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            gemini_transformers.t_schema(None, DeltaOperationList)

    def test_sdk_accepts_the_rewritten_schema(self):
        from google.genai import _transformers as gemini_transformers

        assert gemini_transformers.t_schema(None, provider_json_schema(DeltaOperationList)) is not None

    def test_request_carries_no_field_the_backend_rejects(self):
        """The SDK accepting a schema does not mean the backend will.

        Every op model sets ``extra="forbid"``, so pydantic emits
        ``additionalProperties``. The pydantic-class path drops it; the dict path
        maps it to ``Schema.additional_properties``, and Vertex answers
        ``400 INVALID_ARGUMENT: Unknown name "additional_properties"``. The SDK
        builds that request happily, so asserting on ``t_schema`` alone missed it
        — this asserts on what actually goes over the wire.
        """
        import json

        from google.genai import _transformers as gemini_transformers

        from hindsight_api.engine.providers.gemini_llm import _gemini_dict_schema

        schema = _gemini_dict_schema(DeltaOperationList)
        assert "additionalProperties" not in json.dumps(schema)

        serialized = json.dumps(gemini_transformers.t_schema(None, schema).model_dump(exclude_none=True))
        assert "additional_properties" not in serialized
        # The rewrite must still be intact underneath the strip.
        assert "any_of" in serialized
