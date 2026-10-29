"""`hindsight-admin cloud login` and the hindsight decision-model provider it feeds."""

import asyncio
import json
import os
import stat
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from hindsight_api.config import DecisionModelConfig, HindsightConfig
from hindsight_api.engine.providers import cloud_auth


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_auth, "HOME", tmp_path)
    return tmp_path


def _config(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return HindsightConfig.from_env()


def _save(home, key="hsk_saved_1234"):
    (home / cloud_auth.CREDENTIAL_FILE).write_text(json.dumps({"api_key": key, "org_id": "org-a", "org_name": "Acme"}))


def test_provider_defaults_with_saved_login(home, monkeypatch):
    _save(home)
    config = _config(monkeypatch, {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight"})
    config.validate()
    expected = DecisionModelConfig(
        provider="hindsight",
        api_key="hsk_saved_1234",
        model="hs-decide-1-flash",
        base_url="https://inference.hindsight.vectorize.io",
        timeout=60.0,
        max_concurrent=24,
        max_request_tokens=54_000,
        tokenizer=None,
    )
    assert config.decision_model_config("consolidation") == expected
    assert config.decision_model_config("reflect") == expected


def test_env_key_wins_over_saved_login(home, monkeypatch):
    _save(home)
    config = _config(
        monkeypatch,
        {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight", "HINDSIGHT_API_DECISION_MODEL_API_KEY": "hsk_env"},
    )
    assert config.decision_model_config("reflect").api_key == "hsk_env"


def test_explicit_request_ceiling_wins_over_provider_default(home, monkeypatch):
    _save(home)
    config = _config(
        monkeypatch,
        {
            "HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight",
            "HINDSIGHT_API_DECISION_MODEL_MAX_REQUEST_TOKENS": "20000",
        },
    )
    assert config.decision_model_config("reflect").max_request_tokens == 20_000


def test_recall_keeps_its_reranker_unless_overridden(home, monkeypatch):
    _save(home)
    config = _config(monkeypatch, {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight"})
    assert config.decision_model_config("recall") is None
    monkeypatch.setenv("HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER", "hindsight")
    assert HindsightConfig.from_env().decision_model_config("recall").model == "hs-decide-1-flash"


def test_no_key_anywhere_names_the_login_command(home, monkeypatch):
    with pytest.raises(ValueError, match="hindsight-admin cloud login"):
        _config(monkeypatch, {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight"}).validate()


def _embeddings(monkeypatch, env):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    for name, value in {"HINDSIGHT_API_EMBEDDINGS_PROVIDER": "hindsight", **env}.items():
        monkeypatch.setenv(name, value)
    clear_config_cache()
    try:
        return create_embeddings_from_env()
    finally:
        clear_config_cache()


def _reranker(monkeypatch, env):
    from hindsight_api.engine.cross_encoder import create_cross_encoder

    config = _config(monkeypatch, {"HINDSIGHT_API_RERANKER_PROVIDER": "hindsight", **env})
    return create_cross_encoder(config.reranker_chain()[0])


def test_embeddings_and_reranker_use_hindsight_inference_with_the_saved_login(home, monkeypatch):
    _save(home)
    embeddings = _embeddings(monkeypatch, {})
    assert (embeddings.api_key, embeddings.model, embeddings.base_url) == (
        "hsk_saved_1234",
        "hs-embed-1",
        "https://inference.hindsight.vectorize.io/v1",
    )
    rerank = _reranker(monkeypatch, {})._http_client
    assert (rerank.api_key, rerank.model, rerank.rerank_url) == (
        "hsk_saved_1234",
        "hs-rerank-1",
        "https://inference.hindsight.vectorize.io/v1/rerank",
    )


def test_embeddings_and_reranker_keys_win_over_saved_login(home, monkeypatch):
    _save(home)
    assert _embeddings(monkeypatch, {"HINDSIGHT_API_EMBEDDINGS_HINDSIGHT_API_KEY": "hsk_emb"}).api_key == "hsk_emb"
    reranker = _reranker(monkeypatch, {"HINDSIGHT_API_RERANKER_HINDSIGHT_API_KEY": "hsk_rr"})
    assert reranker._http_client.api_key == "hsk_rr"


def test_embeddings_and_reranker_without_a_key_name_the_login_command(home, monkeypatch):
    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_HINDSIGHT_API_KEY.*hindsight-admin cloud login"):
        _embeddings(monkeypatch, {})
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_HINDSIGHT_API_KEY.*hindsight-admin cloud login"):
        _reranker(monkeypatch, {})


async def _browser(url: str, *, state: str | None = None, key: str = "hsk_new_key_abcd") -> int:
    """Stand in for Cloud: read the connect link and hit the CLI's callback."""
    query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert query["source"] == "hindsight_cli" and query["host"]
    callback = f"http://127.0.0.1:{query['port']}/callback"
    params = {"state": state or query["state"], "key": key, "org_id": "org-a", "org_name": "Acme"}
    async with aiohttp.ClientSession() as session, session.get(callback, params=params) as response:
        return response.status


async def test_login_saves_the_key_privately(home):
    links: list[str] = []
    task = asyncio.create_task(
        cloud_auth.login(home, cloud_url="http://cloud.test", open_browser=None, out=links.append, timeout=10)
    )
    while not links:
        await asyncio.sleep(0.01)
    url = links[0].split("\n\n")[1].strip()
    assert url.startswith("http://cloud.test/connect/desktop?")
    statuses = [await _browser(url, state="forged"), await _browser(url)]
    credential = await task
    assert statuses == [400, 200]  # a wrong state is refused and does not end the login
    assert credential.api_key == "hsk_new_key_abcd" and credential.org_name == "Acme"
    path = home / cloud_auth.CREDENTIAL_FILE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert cloud_auth.saved_api_key(home) == "hsk_new_key_abcd"
    status = "\n".join(cloud_auth.status_lines(home))
    assert "hsk_…abcd" in status
    assert f"Signed in: {credential.created_at.isoformat()}" in status


async def test_login_refuses_a_callback_without_a_key(home):
    links: list[str] = []
    task = asyncio.create_task(
        cloud_auth.login(home, cloud_url="http://cloud.test", open_browser=None, out=links.append, timeout=10)
    )
    while not links:
        await asyncio.sleep(0.01)
    await _browser(links[0].split("\n\n")[1].strip(), key="")
    with pytest.raises(cloud_auth.CloudAuthError, match="no API key"):
        await task
    assert not (home / cloud_auth.CREDENTIAL_FILE).exists()


def test_from_file_status_and_logout(home, tmp_path_factory, capsys):
    other = tmp_path_factory.mktemp("other")
    _save(other, key="hsk_copied_9999")
    at = ["--home", str(home)]
    assert cloud_auth.main([*at, "login", "--from-file", str(other / cloud_auth.CREDENTIAL_FILE)]) == 0
    assert cloud_auth.saved_api_key(home) == "hsk_copied_9999"
    assert cloud_auth.main([*at, "status"]) == 0
    assert "Acme" in capsys.readouterr().out
    assert cloud_auth.main([*at, "logout"]) == 0
    assert cloud_auth.saved_api_key(home) is None
    assert cloud_auth.main([*at, "login", "--from-file", str(other / "missing.json")]) == 1


def test_login_shows_the_new_key_and_how_to_use_it(home):
    credential = cloud_auth.CloudCredential(api_key="hsk_full_key_1234", org_name="Acme", cloud_url="http://cloud.test")
    text = "\n".join(cloud_auth.success_lines(credential, home / cloud_auth.CREDENTIAL_FILE))
    assert "Signed in to Acme" in text
    assert "hsk_full_key_1234" in text  # the whole key, once, like `claude setup-token`
    assert "HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight" in text
    assert "HINDSIGHT_API_DECISION_MODEL_API_KEY=hsk_full_key_1234" in text
    assert "HINDSIGHT_API_EMBEDDINGS_HINDSIGHT_API_KEY=hsk_full_key_1234" in text
    assert "HINDSIGHT_API_RERANKER_HINDSIGHT_API_KEY=hsk_full_key_1234" in text
    assert "http://cloud.test/api-keys" in text
    assert "\033[" not in text  # no colors when stdout is not a terminal


async def test_signed_in_page_names_the_org_safely():
    port = cloud_auth._free_port()
    wait = await cloud_auth._start_callback_listener(port, "s", timeout=10)
    params = {"state": "s", "key": "hsk_k", "org_name": "<script>x</script> Co"}
    async with (
        aiohttp.ClientSession() as session,
        session.get(f"http://127.0.0.1:{port}/callback", params=params) as response,
    ):
        page = await response.text()
    await wait()
    assert "Signed in to &lt;script&gt;x&lt;/script&gt; Co" in page
    assert "<script>" not in page
    assert 'src="data:image/png;base64,' in page  # no second request: the listener is gone by then
