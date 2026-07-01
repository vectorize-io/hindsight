"""Integration test for Supabase-backed organization authz.

The test starts the local stack from hindsight-control-plane/supabase with the
Supabase CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
API_PATH = REPO_ROOT / "hindsight-api-slim"
SUPABASE_PROJECT_PATH = REPO_ROOT / "hindsight-control-plane" / "supabase"


@dataclass(frozen=True)
class SupabaseEnv:
    url: str
    anon_key: str
    service_key: str
    managed_by_test: bool = False


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)


def _parse_supabase_status_env(output: str) -> SupabaseEnv:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return SupabaseEnv(
        url=values["API_URL"].rstrip("/"),
        anon_key=values["ANON_KEY"],
        service_key=values["SERVICE_ROLE_KEY"],
        managed_by_test=True,
    )


@pytest.fixture(scope="module")
def supabase_env():
    if not shutil.which("supabase"):
        pytest.fail("Supabase CLI is not installed", pytrace=False)
    if not (SUPABASE_PROJECT_PATH / "config.toml").exists():
        pytest.fail("Supabase CLI config is not present", pytrace=False)

    start = _run(["supabase", "start"], cwd=SUPABASE_PROJECT_PATH)
    if start.returncode != 0:
        pytest.fail(f"Supabase local stack could not start: {start.stderr or start.stdout}", pytrace=False)
    reset = _run(["supabase", "db", "reset"], cwd=SUPABASE_PROJECT_PATH)
    if reset.returncode != 0:
        _run(["supabase", "stop"], cwd=SUPABASE_PROJECT_PATH)
        pytest.fail(f"Supabase local database could not be reset: {reset.stderr or reset.stdout}", pytrace=False)
    status = _run(["supabase", "status", "-o", "env"], cwd=SUPABASE_PROJECT_PATH)
    if status.returncode != 0:
        _run(["supabase", "stop"], cwd=SUPABASE_PROJECT_PATH)
        pytest.fail(f"Supabase status failed: {status.stderr or status.stdout}", pytrace=False)
    env = _parse_supabase_status_env(status.stdout)
    _wait_for_supabase(env)
    try:
        yield env
    finally:
        if env.managed_by_test:
            _run(["supabase", "stop"], cwd=SUPABASE_PROJECT_PATH)


def _wait_for_supabase(env: SupabaseEnv) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{env.url}/auth/v1/settings", headers={"apikey": env.anon_key}, timeout=3)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Supabase local stack did not become ready")


@pytest.fixture
def supabase_client(supabase_env: SupabaseEnv):
    client = httpx.Client(timeout=10)
    created: dict[str, list[str]] = {"orgs": [], "users": []}
    try:
        yield client, created
    finally:
        headers = _service_headers(supabase_env)
        for org_id in created["orgs"]:
            client.delete(f"{supabase_env.url}/rest/v1/organizations", params={"id": f"eq.{org_id}"}, headers=headers)
        for user_id in created["users"]:
            client.delete(f"{supabase_env.url}/auth/v1/admin/users/{user_id}", headers=headers)


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_supabase_org_resolver_with_real_supabase(supabase_env: SupabaseEnv, supabase_client) -> None:
    client, created = supabase_client
    email = f"hindsight-it-{uuid.uuid4().hex}@example.com"
    password = f"Password-{uuid.uuid4().hex}!"
    org_id = f"org_{uuid.uuid4().hex[:16]}"
    api_key = f"hs_{uuid.uuid4().hex}{uuid.uuid4().hex}"
    api_key_id = str(uuid.uuid4())

    user = _create_auth_user(client, supabase_env, email, password)
    created["users"].append(user["id"])
    _insert(client, supabase_env, "organizations", {"id": org_id, "name": "Integration Org", "config": {"llm_model": "mock"}})
    created["orgs"].append(org_id)
    _insert(
        client,
        supabase_env,
        "organization_members",
        {"org_id": org_id, "user_id": user["id"], "email": email, "role": "admin"},
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_keys",
        {
            "id": api_key_id,
            "org_id": org_id,
            "created_by_user_id": user["id"],
            "name": "Integration key",
            "key_hash": hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            "role": "member",
            "allowed_operations": ["recall"],
        },
    )
    _insert(client, supabase_env, "hindsight_api_key_bank_scopes", {"api_key_id": api_key_id, "bank_id": "bank_a"})

    jwt_token = _sign_in(client, supabase_env, email, password)
    _assert_authz_tables_are_not_publicly_accessible(client, supabase_env, jwt_token)
    resolved = _resolve_with_backend_process(
        supabase_env=supabase_env,
        jwt_token=jwt_token,
        org_id=org_id,
        api_key=api_key,
    )
    assert resolved["jwt_policy"] == {
        "org_id": org_id,
        "user_id": user["id"],
        "role": "admin",
        "allowed_bank_ids": None,
        "tenant_config": {"llm_model": "mock"},
    }
    assert resolved["key_policy"] == {
        "org_id": org_id,
        "api_key_id": api_key_id,
        "allowed_bank_ids": ["bank_a"],
        "allowed_operations": ["recall"],
    }


def _resolve_with_backend_process(
    *,
    supabase_env: SupabaseEnv,
    jwt_token: str,
    org_id: str,
    api_key: str,
) -> dict[str, Any]:
    script = r"""
import asyncio
import json
import os

from hindsight_api.extensions.builtin.supabase_org import SupabasePolicyResolver
from hindsight_api.models import RequestContext


async def main():
    resolver = SupabasePolicyResolver(
        {
            "supabase_url": os.environ["TEST_SUPABASE_URL"],
            "supabase_service_key": os.environ["TEST_SUPABASE_SERVICE_KEY"],
            "policy_cache_ttl_seconds": "0",
        }
    )
    await resolver.on_startup()
    try:
        jwt_policy = await resolver.resolve(
            RequestContext(api_key=os.environ["TEST_SUPABASE_JWT"], selected_org_id=os.environ["TEST_ORG_ID"])
        )
        key_policy = await resolver.resolve(RequestContext(api_key=os.environ["TEST_HINDSIGHT_API_KEY"]))
        print(
            json.dumps(
                {
                    "jwt_policy": {
                        "org_id": jwt_policy.org_id,
                        "user_id": jwt_policy.user_id,
                        "role": jwt_policy.role,
                        "allowed_bank_ids": sorted(jwt_policy.allowed_bank_ids)
                        if jwt_policy.allowed_bank_ids is not None
                        else None,
                        "tenant_config": jwt_policy.tenant_config,
                    },
                    "key_policy": {
                        "org_id": key_policy.org_id,
                        "api_key_id": key_policy.api_key_id,
                        "allowed_bank_ids": sorted(key_policy.allowed_bank_ids)
                        if key_policy.allowed_bank_ids is not None
                        else None,
                        "allowed_operations": sorted(key_policy.allowed_operations)
                        if key_policy.allowed_operations is not None
                        else None,
                    },
                }
            )
        )
    finally:
        await resolver.on_shutdown()


asyncio.run(main())
"""
    env = os.environ.copy()
    env.update(
        {
            "TEST_SUPABASE_URL": supabase_env.url,
            "TEST_SUPABASE_SERVICE_KEY": supabase_env.service_key,
            "TEST_SUPABASE_JWT": jwt_token,
            "TEST_ORG_ID": org_id,
            "TEST_HINDSIGHT_API_KEY": api_key,
        }
    )
    result = subprocess.run(
        ["uv", "run", "--directory", str(API_PATH), "python", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, f"resolver subprocess failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _service_headers(env: SupabaseEnv) -> dict[str, str]:
    return {"apikey": env.service_key, "Authorization": f"Bearer {env.service_key}"}


def _anon_headers(env: SupabaseEnv) -> dict[str, str]:
    return {"apikey": env.anon_key}


def _user_headers(env: SupabaseEnv, jwt_token: str) -> dict[str, str]:
    return {"apikey": env.anon_key, "Authorization": f"Bearer {jwt_token}"}


def _assert_authz_tables_are_not_publicly_accessible(
    client: httpx.Client,
    env: SupabaseEnv,
    jwt_token: str,
) -> None:
    for table in [
        "organizations",
        "organization_members",
        "organization_invites",
        "hindsight_api_keys",
        "hindsight_api_key_bank_scopes",
    ]:
        for headers in [_anon_headers(env), _user_headers(env, jwt_token)]:
            response = client.get(f"{env.url}/rest/v1/{table}", headers=headers)
            assert response.status_code == 200, response.text
            assert response.json() == []

    response = client.post(
        f"{env.url}/rest/v1/hindsight_api_keys",
        headers={**_user_headers(env, jwt_token), "Content-Type": "application/json"},
        json={
            "org_id": "direct_write_attempt",
            "name": "Direct write attempt",
            "key_hash": hashlib.sha256(b"direct-write").hexdigest(),
            "role": "member",
        },
    )
    assert response.status_code in {401, 403}, response.text


def _insert(client: httpx.Client, env: SupabaseEnv, table: str, body: dict[str, Any]) -> None:
    response = client.post(
        f"{env.url}/rest/v1/{table}",
        headers={**_service_headers(env), "Content-Type": "application/json"},
        json=body,
    )
    assert response.status_code in {200, 201}, response.text


def _create_auth_user(client: httpx.Client, env: SupabaseEnv, email: str, password: str) -> dict[str, Any]:
    response = client.post(
        f"{env.url}/auth/v1/admin/users",
        headers={**_service_headers(env), "Content-Type": "application/json"},
        json={"email": email, "password": password, "email_confirm": True},
    )
    assert response.status_code in {200, 201}, response.text
    data = response.json()
    assert data.get("id")
    return data


def _sign_in(client: httpx.Client, env: SupabaseEnv, email: str, password: str) -> str:
    response = client.post(
        f"{env.url}/auth/v1/token",
        params={"grant_type": "password"},
        headers={**_anon_headers(env), "Content-Type": "application/json"},
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    token = response.json().get("access_token")
    assert token
    return str(token)
