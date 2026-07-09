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
            "permission_mode": "scoped",
            "allowed_operations": ["recall"],
        },
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_key_operation_scopes",
        {"api_key_id": api_key_id, "operation": "recall", "bank_scope_mode": "selected"},
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_key_operation_bank_scopes",
        {
            "api_key_id": api_key_id,
            "operation": "recall",
            "bank_id": "bank_a",
            "bank_internal_id": "bank_a",
        },
    )

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
        "tenant_config": {"llm_model": "mock"},
    }
    assert resolved["key_policy"] == {
        "org_id": org_id,
        "api_key_id": api_key_id,
        "allowed_operations": ["recall"],
        "operation_bank_scope_modes": {"recall": "selected"},
        "operation_bank_internal_ids": {"recall": ["bank_a"]},
    }


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_supabase_member_and_bank_lifecycle_uses_tombstones_for_audit_facts(
    supabase_env: SupabaseEnv, supabase_client
) -> None:
    client, created = supabase_client
    org_id = f"org_{uuid.uuid4().hex[:16]}"
    actor_user_id = str(uuid.uuid4())
    member_user_id = str(uuid.uuid4())
    api_key_id = str(uuid.uuid4())
    headers = {**_service_headers(supabase_env), "Content-Type": "application/json"}

    _insert(client, supabase_env, "organizations", {"id": org_id, "name": "Lifecycle Org"})
    created["orgs"].append(org_id)
    _insert(
        client,
        supabase_env,
        "organization_members",
        {"org_id": org_id, "user_id": member_user_id, "role": "member"},
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_keys",
        {
            "id": api_key_id,
            "org_id": org_id,
            "created_by_user_id": member_user_id,
            "name": "Lifecycle key",
            "key_hash": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
            "permission_mode": "scoped",
            "allowed_operations": ["recall"],
        },
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_key_operation_scopes",
        {"api_key_id": api_key_id, "operation": "recall", "bank_scope_mode": "selected"},
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_key_operation_bank_scopes",
        {
            "api_key_id": api_key_id,
            "operation": "recall",
            "bank_id": "bank_deleted",
            "bank_internal_id": "internal_deleted",
        },
    )
    _insert(
        client,
        supabase_env,
        "hindsight_api_key_created_banks",
        {
            "api_key_id": api_key_id,
            "bank_id": "bank_deleted",
            "bank_internal_id": "internal_deleted",
        },
    )

    remove_response = client.post(
        f"{supabase_env.url}/rest/v1/rpc/remove_organization_member",
        headers=headers,
        json={
            "p_org_id": org_id,
            "p_user_id": member_user_id,
            "p_removed_by_user_id": actor_user_id,
        },
    )
    assert remove_response.status_code in {200, 204}, remove_response.text
    memberships = client.get(
        f"{supabase_env.url}/rest/v1/organization_members",
        params={"org_id": f"eq.{org_id}", "user_id": f"eq.{member_user_id}"},
        headers=headers,
    ).json()
    assert len(memberships) == 1
    assert memberships[0]["removed_at"] is not None
    assert memberships[0]["removed_by_user_id"] == actor_user_id
    api_keys = client.get(
        f"{supabase_env.url}/rest/v1/hindsight_api_keys",
        params={"id": f"eq.{api_key_id}"},
        headers=headers,
    ).json()
    assert api_keys[0]["revoked_at"] is not None

    # A new membership period must coexist with the retained historical one.
    _insert(
        client,
        supabase_env,
        "organization_members",
        {"org_id": org_id, "user_id": member_user_id, "role": "member"},
    )

    bank_delete_response = client.post(
        f"{supabase_env.url}/rest/v1/rpc/delete_hindsight_bank_references",
        headers=headers,
        json={"p_bank_internal_id": "internal_deleted"},
    )
    assert bank_delete_response.status_code in {200, 204}, bank_delete_response.text
    bank_scopes = client.get(
        f"{supabase_env.url}/rest/v1/hindsight_api_key_operation_bank_scopes",
        params={"bank_internal_id": "eq.internal_deleted"},
        headers=headers,
    ).json()
    created_banks = client.get(
        f"{supabase_env.url}/rest/v1/hindsight_api_key_created_banks",
        params={"bank_internal_id": "eq.internal_deleted"},
        headers=headers,
    ).json()
    assert bank_scopes == []
    assert len(created_banks) == 1
    assert created_banks[0]["deleted_at"] is not None

    membership_periods = client.get(
        f"{supabase_env.url}/rest/v1/organization_members",
        params={"org_id": f"eq.{org_id}", "user_id": f"eq.{member_user_id}"},
        headers=headers,
    ).json()
    assert len(membership_periods) == 2
    assert sum(period["removed_at"] is None for period in membership_periods) == 1


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
            RequestContext(api_key=os.environ["TEST_SUPABASE_JWT"], selected_tenant_id=os.environ["TEST_ORG_ID"])
        )
        key_policy = await resolver.resolve(RequestContext(api_key=os.environ["TEST_HINDSIGHT_API_KEY"]))
        print(
            json.dumps(
                {
                    "jwt_policy": {
                        "org_id": jwt_policy.org_id,
                        "user_id": jwt_policy.user_id,
                        "role": jwt_policy.role,
                        "tenant_config": jwt_policy.tenant_config,
                    },
                    "key_policy": {
                        "org_id": key_policy.org_id,
                        "api_key_id": key_policy.api_key_id,
                        "allowed_operations": sorted(key_policy.allowed_operations)
                        if key_policy.allowed_operations is not None
                        else None,
                        "operation_bank_scope_modes": key_policy.operation_bank_scope_modes,
                        "operation_bank_internal_ids": {
                            operation: sorted(bank_internal_ids)
                            for operation, bank_internal_ids in (
                                key_policy.operation_bank_internal_ids or {}
                            ).items()
                        },
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
        "hindsight_api_key_operation_scopes",
        "hindsight_api_key_operation_bank_scopes",
        "hindsight_api_key_created_banks",
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
            "permission_mode": "scoped",
            "allowed_operations": [],
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
