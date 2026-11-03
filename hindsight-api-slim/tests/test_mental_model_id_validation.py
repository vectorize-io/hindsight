"""Mental model ids and page max_tokens are checked on the way in (#6235).

An id with "/" splits the URL path, so the model could not be read or deleted over
HTTP, and any id outside the documented form made the whole bank's export fail.
A page max_tokens past int32 overflowed the column with a 500.
"""

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import CreateMentalModelRequest, CreatePageRequest, UpdateNodeRequest
from hindsight_api.engine.bank_templates import BankTemplateMentalModel
from hindsight_api.engine.memory_engine import MemoryEngine

BAD_IDS = ["team/standup", "ok\n", "a\tb", "x" * 193]


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_create_request_rejects_bad_id(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        CreateMentalModelRequest(id=bad_id, name="n", source_query="q")


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_template_rejects_bad_id(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        BankTemplateMentalModel(id=bad_id, name="n", source_query="q")


@pytest.mark.parametrize(
    "good_id", ["team-standup", "MyModel", "team_notes", "Team Notes", "a?b#c", "界" * 64, None, ""]
)
def test_create_request_accepts_good_or_missing_id(good_id: str | None) -> None:
    assert CreateMentalModelRequest(id=good_id, name="n", source_query="q").id == good_id


@pytest.mark.parametrize("model", [CreatePageRequest, UpdateNodeRequest])
def test_page_max_tokens_is_bounded(model: type) -> None:
    extra = {"name": "P", "source_query": "q"}
    for bad in (255, 8193, 1099511627776):
        with pytest.raises(ValidationError):
            model(**extra, max_tokens=bad)
    assert model(**extra, max_tokens=256).max_tokens == 256
    assert model(**extra).max_tokens is None


async def test_engine_rejects_bad_id(memory: MemoryEngine, request_context) -> None:
    # The MCP create_mental_model tool hands a caller-chosen id straight to the engine.
    with pytest.raises(ValueError, match="must not contain '/'"):
        await memory.create_mental_model(
            bank_id="unused",
            name="n",
            source_query="q",
            content="",
            mental_model_id="team/standup",
            request_context=request_context,
        )
