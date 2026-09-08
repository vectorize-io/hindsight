"""Responses drop null fields where it is wire-compatible to do so.

`create_app` installs `ExcludeNoneRoute`, which enables `response_model_exclude_none`
for every route whose response model can afford it. A model that must keep its nulls —
one with a required-and-nullable field (an omitted key would break strict generated
clients), or an `OpenRowModel` row that shipped as a bare dict before #4218 — keeps
emitting them. These tests lock in that classification and the resulting serialization.
"""

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from hindsight_api.api.http import (
    DocumentResponse,
    EntityGraphResponse,
    ExcludeNoneRoute,
    GraphDataResponse,
    ListDocumentsResponse,
    ListMemoryUnitsResponse,
    OperationResponse,
    RecallResponse,
    ReflectResponse,
    RetainResponse,
    WebhookDeliveryResponse,
    WebhookResponse,
    _response_model_must_keep_nulls,
)


class _RequiredNullable(BaseModel):
    value: str | None  # required (no default) AND nullable


class _OptionalNullable(BaseModel):
    value: str | None = None  # optional (has default)


class _NestsRequiredNullable(BaseModel):
    items: list[_RequiredNullable]


def test_required_nullable_detection() -> None:
    # Direct required-nullable field.
    assert _response_model_must_keep_nulls(_RequiredNullable) is True
    # Optional (has default) is fine to drop.
    assert _response_model_must_keep_nulls(_OptionalNullable) is False
    # Detection recurses through nested models and generic containers.
    assert _response_model_must_keep_nulls(_NestsRequiredNullable) is True
    assert _response_model_must_keep_nulls(list[_RequiredNullable]) is True
    assert _response_model_must_keep_nulls(_RequiredNullable | None) is True


def test_high_traffic_responses_are_cleaned() -> None:
    # These have only optional (defaulted) nullable fields -> safe to drop nulls.
    for model in (RecallResponse, RetainResponse, ReflectResponse):
        assert _response_model_must_keep_nulls(model) is False


def test_required_nullable_responses_are_preserved() -> None:
    # These carry a required-nullable field (e.g. error_message, content_hash) that
    # strict clients expect present -> must keep emitting nulls.
    for model in (DocumentResponse, OperationResponse, WebhookResponse, WebhookDeliveryResponse):
        assert _response_model_must_keep_nulls(model) is True


def _make_route(response_model: type[BaseModel]) -> ExcludeNoneRoute:
    return ExcludeNoneRoute("/_t", endpoint=lambda: None, response_model=response_model)


def test_typed_row_responses_are_preserved() -> None:
    # #4218 typed these rows, which used to be `dict[str, Any]`. `exclude_none` never
    # reached inside a dict, so the rows' nulls were always on the wire; dropping them now
    # would be a `KeyError` for anyone indexing a row.
    for model in (ListDocumentsResponse, ListMemoryUnitsResponse, GraphDataResponse, EntityGraphResponse):
        assert _response_model_must_keep_nulls(model) is True
        assert _make_route(model).response_model_exclude_none is False


def test_route_class_sets_exclude_none_per_model() -> None:
    assert _make_route(RecallResponse).response_model_exclude_none is True
    assert _make_route(DocumentResponse).response_model_exclude_none is False


def test_explicit_decorator_flag_is_respected() -> None:
    # An explicit response_model_exclude_none on the decorator is not overridden.
    route = ExcludeNoneRoute(
        "/_t", endpoint=lambda: None, response_model=DocumentResponse, response_model_exclude_none=True
    )
    assert route.response_model_exclude_none is True


def test_cleaned_response_omits_null_keys() -> None:
    resp = RecallResponse(results=[], trace=None, entities=None, chunks=None, source_facts=None)
    cleaned = jsonable_encoder(resp, exclude_none=True)
    assert cleaned == {"results": []}
    assert "trace" not in cleaned and "entities" not in cleaned
