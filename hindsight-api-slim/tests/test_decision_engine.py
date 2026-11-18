"""The decision engine's listwise questions: reflect's "is this enough?", the page-impact gate,
and (with pruning) the rank-and-cut reflect prunes its evidence with.

The HTTP round trip is faked; what is asserted is the requests the engine builds and the
verdicts it reads from the answers. The engine asks them in every provider's wire format;
which providers reflect itself runs on is MemoryEngine's call (test_reflect_decision_model_switch).
"""

import base64
import json as jsonlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.decision_model import DecisionModel
from hindsight_api.engine.remote_retry import RetryPolicy
from hindsight_api.engine.token_encoding import count_tokens


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status = 200

    async def json(self, content_type=None):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Answers a rank question from `ranking` and a cut question from `cut_level`, in System One.

    `ranking` maps an option key ("c0", "c1", ...) to the probability the model would give it,
    so a pool ranked in several rounds is answered per round exactly as the real API would.
    """

    # Set to refuse a body the way Jev does when it counts it over its window. Jev counts
    # mixed-script text at twice our tokenizer (#5715), so the window here does too.
    window: int | None = None

    def __init__(self, ranking: dict[str, float], cut_level: float = 0.0):
        self.ranking = ranking
        self.cut_level = cut_level
        self.posted: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self.refused: list[dict] = []
        # Jev's refusal; a test swaps it for another server's (#6522).
        self.refusal = UpstreamHTTPError(400, '{"detail":{"error_type":"max_tokens_exceeded"}}', {}, "")

    def get(self):
        return self

    def answer(self, question: dict) -> dict:
        if question["type"] == "choice":
            keys = list(question["criteria"])
            return {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {key: self.ranking.get(key, 0.0) for key in keys},
                "confidence": 0.9,
            }
        return {
            "type": "score",
            "score": self.cut_level,
            "legend": dict(enumerate(question["criteria"])),
            "confidence": 0.9,
        }

    @asynccontextmanager
    async def _post(self, url, headers=None, json: Any = None):
        self.urls.append(url)
        self.posted.append(json)
        question_id, question = next(iter(json["questions"].items()))
        yield _FakeResponse(
            {"answers": {question_id: self.answer(question)}, "usage": {"input_tokens": 1, "output_tokens": 1}}
        )

    def post(self, url, headers=None, json: Any = None):
        self.headers.append(headers or {})
        if self.window is not None and 2 * count_tokens(jsonlib.dumps(json)) > self.window:
            self.refused.append(json)
            raise self.refusal
        return self._post(url, headers=headers, json=json)

    @property
    def rank_requests(self) -> list[dict]:
        return [body for body in self.posted if next(iter(body["questions"].values()))["type"] == "choice"]

    @property
    def cut_requests(self) -> list[dict]:
        return [body for body in self.posted if next(iter(body["questions"].values()))["type"] == "score"]


def _engine(session: _FakeSession, *, provider: str = "typesafe", vision: bool = False, **kwargs) -> DecisionModel:
    engine = DecisionModel(
        provider=provider,
        api_key="k",
        model="jev-latest",
        base_url=kwargs.pop("base_url", "https://api.typesafe.ai"),
        timeout=5.0,
        max_concurrent=4,
        max_request_tokens=kwargs.pop("max_request_tokens", 26_000),
        vision=vision,
        **kwargs,
    )
    engine._session = session  # type: ignore[assignment]
    return engine


class _SufficiencySession(_FakeSession):
    """Answers the sufficiency Score with a fixed expected score and per-level probabilities."""

    def __init__(self, score: float, probabilities: dict[str, float] | None):
        super().__init__({})
        self.score = score
        self.probabilities = probabilities

    @asynccontextmanager
    async def _post(self, url, headers=None, json: Any = None):
        self.posted.append(json)
        answer = {"type": "score", "score": self.score, "confidence": 0.5}
        if self.probabilities is not None:
            answer["probabilities"] = self.probabilities
        yield _FakeResponse({"answers": {"sufficient": answer}})


class _RateLimitedOnceSession(_SufficiencySession):
    """Refuses the first request with a 429, as a shared endpoint under load does."""

    def post(self, url, headers=None, json: Any = None):
        if not self.refused:
            self.refused.append(json)
            raise UpstreamHTTPError(429, "Too Many Requests", {}, url)
        return super().post(url, headers=headers, json=json)


class TestSufficiency:
    @pytest.mark.asyncio
    async def test_a_rate_limit_is_retried_not_read_as_not_enough(self):
        """A 429 used to fail the verdict, and reflect then ran the agent turns it exists to save."""
        session = _RateLimitedOnceSession(2.0, {"0": 0.0, "1": 0.1, "2": 0.9})
        engine = _engine(session, retry_policy=RetryPolicy(initial_backoff=0.01, max_backoff=0.01))

        assert await engine.evidence_is_sufficient("q", "evidence") is True
        assert len(session.refused) == 1 and len(session.posted) == 1

    @staticmethod
    async def _verdict(score: float, probabilities: dict[str, float] | None) -> bool:
        return await _engine(_SufficiencySession(score, probabilities)).evidence_is_sufficient("q", "evidence")

    @pytest.mark.asyncio
    async def test_the_likeliest_level_decides_not_the_rounded_score(self):
        """Measured on sde-bench: expected 1.43 rounds to "partly", but "fully" was likeliest."""
        assert await self._verdict(1.43, {"0": 0.10, "1": 0.36, "2": 0.54}) is True

    @pytest.mark.asyncio
    async def test_partly_is_not_enough(self):
        assert await self._verdict(1.13, {"0": 0.02, "1": 0.84, "2": 0.14}) is False

    @pytest.mark.asyncio
    async def test_without_probabilities_the_score_is_rounded(self):
        assert await self._verdict(1.6, None) is True
        assert await self._verdict(1.4, None) is False

    @pytest.mark.asyncio
    async def test_evidence_is_cut_shorter_until_it_fits_the_window(self):
        session = _SufficiencySession(2.0, None)
        session.window = 32_000

        assert await _engine(session).evidence_is_sufficient("q", "evidence words " * 20_000) is True
        assert len(session.refused) == 1 and len(session.posted) == 1


def _picture(label: str, size: tuple[int, int], fmt: str = "PNG", progressive: bool = False):
    import io

    from PIL import Image

    from hindsight_api.engine.decision_rerank_interface import EvidenceImage

    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format=fmt, **({"progressive": True} if progressive else {}))
    media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return EvidenceImage(label=label, media_type=media_type, data=out.getvalue())


class TestSufficiencyPictures:
    """With HINDSIGHT_API_DECISION_MODEL_VISION, "is this enough?" sees the pictures the evidence names.

    An answer only a picture holds otherwise read as "not enough", and bought reflect another round.
    """

    @staticmethod
    async def _ask(vision: bool, pictures: list, provider: str = "typesafe") -> tuple[dict, dict, list[dict]]:
        session = _SufficiencySession(2.0, {"0": 0.0, "1": 0.1, "2": 0.9})
        engine = _engine(session, provider=provider, vision=vision)
        calls: list[dict] = []
        recorder = type("R", (), {"record_llm_call": lambda self, **kw: calls.append(kw)})()
        with patch("hindsight_api.engine.decision_model.get_span_recorder", return_value=recorder):
            assert await engine.evidence_is_sufficient("door code?", "door-sign: picture", pictures=pictures)
        return session.posted[0], session.headers[0], calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["typesafe", "hindsight"])
    async def test_the_pictures_system_one_takes_go_as_assets_named_after_their_label(self, provider):
        sign = _picture("Picture of door-sign.png, page 1", (900, 400))
        scan = _picture("Picture of scan.pdf, page 2", (1200, 1500), fmt="JPEG")
        ignored = [
            _picture("tiny", (64, 64)),  # under 280 px a side
            _picture("huge", (2000, 500)),  # over 1568 px a side
            _picture("webp", (600, 600), fmt="WEBP"),  # PNG and JPEG only
            _picture("progressive", (600, 600), fmt="JPEG", progressive=True),  # baseline JPEG only
        ]
        body, headers, calls = await self._ask(True, [sign, *ignored, scan], provider)

        assert headers["TypeSafe-Danger-Zone-Images"] == "true"
        assert body["state"].endswith(
            "Pictures:\nPicture of door-sign.png, page 1: [[picture-1]]\nPicture of scan.pdf, page 2: [[picture-2]]"
        )
        assert {ref: (a["type"], a["format"]) for ref, a in body["assets"].items()} == {
            "[[picture-1]]": ("image", "png"),
            "[[picture-2]]": ("image", "jpg"),
        }
        assert base64.b64decode(body["assets"]["[[picture-1]]"]["data_base64"]) == sign.data
        # The trace keeps which pictures went, never their bytes.
        traced = calls[0]["structured_input"]
        assert traced["assets"] == ["[[picture-1]]", "[[picture-2]]"]

    @pytest.mark.asyncio
    async def test_without_vision_or_any_picture_system_one_takes_the_request_is_text_only(self):
        for vision, pictures in ((False, [_picture("sign", (900, 400))]), (True, [_picture("tiny", (64, 64))])):
            body, headers, _ = await self._ask(vision, pictures)
            assert "assets" not in body and "Pictures:" not in body["state"]
            assert "TypeSafe-Danger-Zone-Images" not in headers

    @pytest.mark.asyncio
    async def test_a_model_that_takes_no_images_is_never_shown_pictures(self):
        """OpenAI's Decisions API has no assets (DecisionModel.takes_images)."""
        session = _OpenAISession({}, cut_level=2.0)
        engine = _engine(session, provider="openai", vision=True, base_url="https://api.openai.com")
        assert await engine.evidence_is_sufficient("q", "e", pictures=[_picture("sign", (900, 400))])
        assert "assets" not in session.posted[0] and "Pictures:" not in session.posted[0]["input"]
        assert "TypeSafe-Danger-Zone-Images" not in session.headers[0]


def _enveloped(post):
    @asynccontextmanager
    async def wrapped(url, headers=None, json: Any = None):
        async with post(url, headers=headers, json=json) as response:
            yield _FakeResponse({"result": await response.json()})

    return wrapped


class _PageImpactSession(_FakeSession):
    """Answers the page-impact choice with a fixed answer and records each request."""

    def __init__(self, answer: dict):
        super().__init__({})
        self.fixed = answer

    @asynccontextmanager
    async def _post(self, url, headers=None, json: Any = None):
        self.posted.append(json)
        yield _FakeResponse({"answers": {"impact": {"type": "choice", **self.fixed}}})


@dataclass(frozen=True)
class _PageImpactCall:
    verdict: bool
    request: dict


class TestPageImpact:
    @staticmethod
    async def _ask(answer: dict, page: str = "# Page\n\nDana is on call.") -> _PageImpactCall:
        session = _PageImpactSession(answer)
        verdict = await _engine(session).could_change_page("Who is on call?", page, "Dana left the company.")
        return _PageImpactCall(verdict=verdict, request=session.posted[0])

    @pytest.mark.asyncio
    async def test_no_drops_the_memory(self):
        assert (await self._ask({"choice": "no", "probabilities": {"yes": 0.1, "no": 0.9}})).verdict is False

    @pytest.mark.asyncio
    async def test_yes_keeps_it(self):
        assert (await self._ask({"choice": "yes"})).verdict is True

    @pytest.mark.asyncio
    async def test_without_a_choice_the_likeliest_option_decides(self):
        assert (await self._ask({"probabilities": {"yes": 0.2, "no": 0.8}})).verdict is False

    @pytest.mark.asyncio
    async def test_an_empty_answer_keeps_it(self):
        """A wrong "no" is permanent, a wrong "yes" costs one LLM call: doubt keeps."""
        assert (await self._ask({})).verdict is True

    @pytest.mark.asyncio
    async def test_the_state_carries_topic_memory_and_page(self):
        body = (await self._ask({"choice": "yes"})).request
        state = body["state"]
        assert "Who is on call?" in state and "Dana left the company." in state and "Dana is on call." in state
        criteria = set(body["questions"]["impact"]["criteria"])
        assert {"yes", "no"} <= criteria and len(criteria) > 2

    @pytest.mark.asyncio
    async def test_a_hedged_answer_keeps_it(self):
        """Every option but "no" keeps the memory, so a doubt split across them never drops it."""
        assert (await self._ask({"choice": "maybe"})).verdict is True
        assert (await self._ask({"choice": "no"})).verdict is False

    @pytest.mark.asyncio
    async def test_an_oversized_page_keeps_its_head(self):
        page = "HEAD " + "filler " * 40_000 + "TAIL"
        body = (await self._ask({"choice": "yes"}, page=page)).request
        assert "HEAD" in body["state"] and "TAIL" not in body["state"]
        assert count_tokens(body["state"]) <= DecisionModel.MAX_QUESTION_TOKENS


class _OpenAISession(_FakeSession):
    """Answers in the Decisions API's shape: a list of named answers."""

    @asynccontextmanager
    async def _post(self, url, headers=None, json: Any = None):
        self.urls.append(url)
        self.posted.append(json)
        answers = []
        for question in json["questions"]:
            if question["type"] == "choice":
                values = [option["value"] for option in question["choices"]]
                probabilities = [{"value": v, "probability": self.ranking.get(v, 0.0)} for v in values]
                best = max(values, key=lambda v: self.ranking.get(v, 0.0))
                answers.append(
                    {"type": "choice", "name": question["name"], "choice": best, "probabilities": probabilities}
                )
            else:
                answers.append({"type": "score", "name": question["name"], "score": self.cut_level})
        yield _FakeResponse({"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}})


class TestWindow:
    """Reflect sizes its questions to the configured window (#6659), counted in its tokenizer."""

    @pytest.mark.asyncio
    async def test_a_configured_window_is_taken_at_the_first_request(self):
        size = 20
        docs = [f"{i} " + "long document words " * 100 for i in range(size)]  # ~6k tokens
        ranking = {f"c{i}": 1.0 / (i + 1) for i in range(size)}

        default = _FakeSession(ranking)
        default.window = 8_000
        await _engine(default, max_request_tokens=None).rank("q", docs)
        assert default.refused, "at the 26k default the first request overflows this window"

        sized = _FakeSession(ranking)
        sized.window = 8_000
        scores = await _engine(sized, max_request_tokens=3_500).rank("q", docs)
        assert not sized.refused
        assert len(set(scores)) == size


class TestEveryProvider:
    """The engine asks reflect's questions in every provider's wire format."""

    @pytest.mark.asyncio
    async def test_openai_ranks_cuts_judges_and_gates(self):
        session = _OpenAISession({"c0": 0.2, "c1": 0.7, "c2": 0.1, "no": 0.9, "yes": 0.1}, cut_level=2.0)
        engine = _engine(session, provider="openai", base_url="https://api.openai.com")

        scores = await engine.rank("q", ["a", "b", "c"], prune=True)
        assert scores[1] > scores[0] > scores[2] > 0.0, "level 2 keeps the first three"
        assert await engine.evidence_is_sufficient("q", "evidence") is True
        assert await engine.could_change_page("topic", "page", "memory") is False
        assert set(session.urls) == {"https://api.openai.com/v1/decisions"}

    @pytest.mark.asyncio
    async def test_cloudflare_ranks_in_its_envelope(self):
        session = _FakeSession({"c0": 0.2, "c1": 0.7})
        session._post = _enveloped(session._post)  # type: ignore[method-assign]
        engine = _engine(session, provider="cloudflare", base_url="https://cf.example/ai/run")

        scores = await engine.rank("q", ["a", "b"])
        assert scores[1] > scores[0]
        assert session.urls == ["https://cf.example/ai/run/@cf/cloudflare/jev-latest"]
