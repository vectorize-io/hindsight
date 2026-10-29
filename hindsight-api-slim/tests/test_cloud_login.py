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
    monkeypatch.setenv("HINDSIGHT_API_CLOUD_HOME", str(tmp_path))
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
        model="hs-decision-1",
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
    assert HindsightConfig.from_env().decision_model_config("recall").model == "hs-decision-1"


def test_no_key_anywhere_names_the_login_command(home, monkeypatch):
    with pytest.raises(ValueError, match="hindsight-admin cloud login"):
        _config(monkeypatch, {"HINDSIGHT_API_DECISION_MODEL_PROVIDER": "hindsight"}).validate()


async def _browser(url: str, *, state: str | None = None, key: str = "hsk_new_key_abcd") -> int:
    """Stand in for Cloud: read the connect link and hit the CLI's callback."""
    query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert query["source"] == "hindsight_cli" and query["host"]
    callback = f"http://127.0.0.1:{query['port']}/callback"
    params = {"state": state or query["state"], "key": key, "org_id": "org-a", "org_name": "Acme"}
    async with aiohttp.ClientSession() as session, session.get(callback, params=params) as response:
        return response.status


async def test_login_saves_the_key_privately(home):
    statuses: list[int] = []

    async def run():
        links: list[str] = []
        task = asyncio.create_task(
            cloud_auth.login(home, cloud_url="http://cloud.test", open_browser=None, out=links.append, timeout=10)
        )
        while not links:
            await asyncio.sleep(0.01)
        url = links[0].split("\n\n")[1].strip()
        assert url.startswith("http://cloud.test/connect/desktop?")
        statuses.append(await _browser(url, state="forged"))
        statuses.append(await _browser(url))
        return await task

    credential = await run()
    assert statuses == [400, 200]  # a wrong state is refused and does not end the login
    assert credential.api_key == "hsk_new_key_abcd" and credential.org_name == "Acme"
    path = home / cloud_auth.CREDENTIAL_FILE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert cloud_auth.saved_api_key(home) == "hsk_new_key_abcd"
    assert "hsk_…abcd" in "\n".join(cloud_auth.status_lines(home))


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
