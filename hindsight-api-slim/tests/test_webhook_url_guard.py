"""Unit tests for webhook SSRF hardening (hindsight_api.webhooks.url_guard).

These are deterministic and DB-free: they cover the destination-URL validation,
the resolve-and-validate logic, and the guarded aiohttp client against a real
loopback server. The security guarantees under test:

- private/loopback/link-local/metadata destinations are blocked by default,
- an operator allowlist re-permits specific hosts / IP ranges,
- DNS names are resolved and every resolved address is checked (so a name that
  resolves to an internal address is blocked) — at connect time, by the
  connector's only resolver,
- the client connects only to a validated IP while preserving the original
  Host header (virtual host) so a rebind cannot swap in an internal address.
"""

import asyncio
import ipaddress
import socket

import pytest
from aiohttp import TCPConnector, web

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.webhooks.url_guard import (
    GuardedResolver,
    GuardedWebhookClient,
    WebhookURLError,
    _ip_is_blocked,
    parse_allowlist,
    resolve_and_validate,
    validate_url_syntax,
)
from tests.aiohttp_stub import stub_server


class TestIpClassification:
    @pytest.mark.parametrize(
        "ip",
        [
            "169.254.169.254",  # cloud metadata (link-local)
            "127.0.0.1",
            "10.0.0.5",
            "172.16.9.9",
            "192.168.1.1",
            "100.64.0.1",  # CGNAT
            "0.0.0.0",
            "::1",
            "fc00::1",  # ULA
            "fe80::1",  # link-local v6
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
        ],
    )
    def test_blocked_ranges(self, ip):
        assert _ip_is_blocked(ipaddress.ip_address(ip)) is True

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1::1"])
    def test_public_allowed(self, ip):
        assert _ip_is_blocked(ipaddress.ip_address(ip)) is False


class TestValidateUrlSyntax:
    def test_rejects_non_http_scheme(self):
        for bad in ["file:///etc/passwd", "gopher://x/y", "ftp://host/f"]:
            with pytest.raises(WebhookURLError):
                validate_url_syntax(bad, parse_allowlist([]))

    def test_rejects_missing_host(self):
        for bad in ["http:///nohost", "http://", "example.com/hook", ""]:
            with pytest.raises(WebhookURLError):
                validate_url_syntax(bad, parse_allowlist([]))

    def test_rejects_malformed(self):
        for bad in ["http://h:abc/", "http://[::1/", "http://h\n.com/"]:
            with pytest.raises(WebhookURLError):
                validate_url_syntax(bad, parse_allowlist([]))

    def test_rejects_internal_ip_literal(self):
        for bad in [
            "http://127.0.0.1/x",
            "http://169.254.169.254/latest/meta-data/",
            "https://10.1.2.3/hook",
            "http://[::1]/x",
            "http://[::ffff:127.0.0.1]/x",
            "http://2130706433/x",  # legacy numeric 127.0.0.1
            "http://127.1/x",
        ]:
            with pytest.raises(WebhookURLError):
                validate_url_syntax(bad, parse_allowlist([]))

    def test_allows_public_dns_name(self):
        # DNS names are not resolved at syntax time — deferred to delivery.
        validate_url_syntax("https://example.com/hook", parse_allowlist([]))
        validate_url_syntax("HTTPS://Example.COM:8443/hook?x=1", parse_allowlist([]))

    def test_allowlist_permits_internal_ip_literal(self):
        validate_url_syntax("http://127.0.0.1:8080/x", parse_allowlist(["127.0.0.1"]))

    def test_allowlist_cidr_permits_range(self):
        validate_url_syntax("http://10.1.2.3/x", parse_allowlist(["10.0.0.0/8"]))


@pytest.mark.asyncio
class TestResolveAndValidate:
    async def test_literal_loopback_blocked(self):
        with pytest.raises(WebhookURLError):
            await resolve_and_validate("127.0.0.1", 80, parse_allowlist([]))

    async def test_literal_loopback_allowlisted(self):
        assert await resolve_and_validate("127.0.0.1", 80, parse_allowlist(["127.0.0.1"])) == ["127.0.0.1"]

    async def test_dns_name_resolving_to_loopback_blocked(self):
        # localhost resolves to a loopback address -> blocked by default.
        with pytest.raises(WebhookURLError):
            await resolve_and_validate("localhost", 80, parse_allowlist([]))

    async def test_dns_name_allowlisted_by_name(self):
        ips = await resolve_and_validate("localhost", 80, parse_allowlist(["localhost"]))
        assert all(ipaddress.ip_address(ip).is_loopback for ip in ips)

    async def test_any_blocked_address_rejects_the_whole_answer(self, monkeypatch):
        # Split-horizon / partial rebind: one public + one internal record.
        _fake_dns(monkeypatch, {"rebind.test": ["93.184.216.34", "169.254.169.254"]})
        with pytest.raises(WebhookURLError, match="169.254.169.254"):
            await resolve_and_validate("rebind.test", 80, parse_allowlist([]))

    async def test_resolution_failure_is_a_url_error(self, monkeypatch):
        _fake_dns(monkeypatch, {})
        with pytest.raises(WebhookURLError, match="did not resolve"):
            await resolve_and_validate("nowhere.test", 80, parse_allowlist([]))


def _fake_dns(monkeypatch, table: dict[str, list[str]]) -> list[str]:
    """Answer the running loop's getaddrinfo from ``table``; return the log of looked-up names.

    Must be called from inside the test's event loop. Only the guard resolves
    through ``loop.getaddrinfo`` here (its connector has no other resolver), so
    this stands in for DNS without touching the connect path.
    """
    lookups: list[str] = []

    async def fake_getaddrinfo(host, port, *, family=0, type=0, proto=0, flags=0):
        lookups.append(host)
        if host not in table:
            raise socket.gaierror(socket.EAI_NONAME, "not found")
        infos = []
        for ip in table[host]:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            if family not in (0, fam):
                continue
            sockaddr = (ip, port, 0, 0) if fam == socket.AF_INET6 else (ip, port)
            infos.append((fam, socket.SOCK_STREAM, 6, "", sockaddr))
        return infos

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    return lookups


@pytest.mark.asyncio
class TestGuardedResolver:
    async def test_returns_only_validated_addresses_with_families(self, monkeypatch):
        _fake_dns(monkeypatch, {"hook.test": ["93.184.216.34", "2606:2800:220:1::1"]})
        results = await GuardedResolver(parse_allowlist([])).resolve("hook.test", 443, family=socket.AF_UNSPEC)
        assert [(r["host"], r["family"], r["hostname"], r["port"]) for r in results] == [
            ("93.184.216.34", socket.AF_INET, "hook.test", 443),
            ("2606:2800:220:1::1", socket.AF_INET6, "hook.test", 443),
        ]

    async def test_rejects_private_answer(self, monkeypatch):
        _fake_dns(monkeypatch, {"internal.test": ["10.0.0.7"]})
        with pytest.raises(WebhookURLError):
            await GuardedResolver(parse_allowlist([])).resolve("internal.test", 80)


async def _echo_host(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    return web.Response(
        text=f"INTERNAL_SECRET host={request.headers.get('Host', '')} method={request.method} "
        f"query={request.query_string} body={body.decode()}"
    )


async def _send(client: GuardedWebhookClient, url: str, method: str = "GET", body: bytes | None = None):
    try:
        return await client.request(method, url, headers={}, params=None, body=body, timeout_seconds=5)
    finally:
        await client.close()


@pytest.mark.asyncio
class TestGuardedWebhookClient:
    async def test_blocks_loopback_literal_by_default(self):
        async with stub_server(_echo_host) as base:
            with pytest.raises(WebhookURLError):
                await _send(GuardedWebhookClient(parse_allowlist([])), f"{base}/internal")

    async def test_blocks_legacy_numeric_loopback(self):
        async with stub_server(_echo_host) as base:
            port = base.rsplit(":", 1)[1]
            with pytest.raises(WebhookURLError):
                await _send(GuardedWebhookClient(parse_allowlist([])), f"http://2130706433:{port}/internal")

    async def test_hostname_resolving_to_private_ip_refused_at_connect(self, monkeypatch):
        """A DNS name that passes the syntax check but resolves internally never connects.

        Goes through the real aiohttp session/connector: the refusal comes from the
        connector's resolver, and the loopback server must see no request.
        """
        seen: list[str] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            seen.append(request.path)
            return web.Response(text="INTERNAL_SECRET")

        async with stub_server(handler) as base:
            port = base.rsplit(":", 1)[1]
            url = f"http://innocent.test:{port}/internal"
            validate_url_syntax(url, parse_allowlist([]))  # accepted at registration
            lookups = _fake_dns(monkeypatch, {"innocent.test": ["127.0.0.1"]})
            with pytest.raises(WebhookURLError, match="disallowed address 127.0.0.1"):
                await _send(GuardedWebhookClient(parse_allowlist([])), url)
        assert lookups == ["innocent.test"]
        assert seen == []

    async def test_rebinding_is_rechecked_on_every_connection(self, monkeypatch):
        """No DNS cache: a name that later rebinds to an internal address is refused."""
        async with stub_server(_echo_host) as base:
            port = base.rsplit(":", 1)[1]
            table = {"rebind.test": ["127.0.0.1"]}
            lookups = _fake_dns(monkeypatch, table)
            client = GuardedWebhookClient(parse_allowlist(["127.0.0.0/8"]))
            url = f"http://rebind.test:{port}/x"
            first = await client.request(
                "GET", url, headers={"Connection": "close"}, params=None, body=None, timeout_seconds=5
            )
            assert first.status_code == 200
            table["rebind.test"] = ["169.254.169.254"]
            with pytest.raises(WebhookURLError):
                await client.request("GET", url, headers={}, params=None, body=None, timeout_seconds=5)
            await client.close()
        assert lookups == ["rebind.test", "rebind.test"]

    async def test_allowlisted_loopback_succeeds_and_preserves_host(self):
        async with stub_server(_echo_host) as base:
            port = base.rsplit(":", 1)[1]
            resp = await _send(GuardedWebhookClient(parse_allowlist(["127.0.0.1"])), f"{base}/internal")
        assert resp.status_code == 200
        # Host header carried the original authority (virtual host preserved).
        assert f"host=127.0.0.1:{port}" in resp.body

    async def test_pins_dns_name_and_falls_back_across_addresses(self, monkeypatch):
        # The name resolves to an unreachable IPv6 address first, then 127.0.0.1;
        # the connector must try each validated address rather than pin only one.
        async with stub_server(_echo_host) as base:
            port = base.rsplit(":", 1)[1]
            _fake_dns(monkeypatch, {"dual.test": ["::1", "127.0.0.1"]})
            resp = await _send(
                GuardedWebhookClient(parse_allowlist(["127.0.0.1", "::1"])), f"http://dual.test:{port}/internal"
            )
        assert resp.status_code == 200
        assert f"host=dual.test:{port}" in resp.body

    async def test_localhost_allowlisted_by_name(self):
        async with stub_server(_echo_host) as base:
            port = base.rsplit(":", 1)[1]
            resp = await _send(GuardedWebhookClient(parse_allowlist(["localhost"])), f"http://localhost:{port}/i")
        assert f"host=localhost:{port}" in resp.body

    async def test_post_sends_raw_body_and_params(self):
        async with stub_server(_echo_host) as base:
            client = GuardedWebhookClient(parse_allowlist(["127.0.0.1"]))
            try:
                resp = await client.request(
                    "POST", base, headers={}, params={"k": "v"}, body=b'{"a":1}', timeout_seconds=5
                )
            finally:
                await client.close()
        assert 'method=POST query=k=v body={"a":1}' in resp.body

    async def test_redirect_is_not_followed(self):
        async def redirect(request: web.Request) -> web.StreamResponse:
            raise web.HTTPFound("http://169.254.169.254/latest/meta-data/")

        async with stub_server(redirect) as base:
            with pytest.raises(UpstreamHTTPError) as exc_info:
                await _send(GuardedWebhookClient(parse_allowlist(["127.0.0.1"])), base)
        assert exc_info.value.status_code == 302

    async def test_non_2xx_carries_status_and_body(self):
        async def fail(request: web.Request) -> web.StreamResponse:
            return web.Response(status=503, text="try later")

        async with stub_server(fail) as base:
            with pytest.raises(UpstreamHTTPError) as exc_info:
                await _send(GuardedWebhookClient(parse_allowlist(["127.0.0.1"])), base)
        assert exc_info.value.status_code == 503
        assert exc_info.value.body == "try later"

    async def test_rejects_non_http_scheme(self):
        with pytest.raises(WebhookURLError):
            await _send(GuardedWebhookClient(parse_allowlist([])), "ftp://example.com/x")

    async def test_connector_has_guarded_resolver_and_no_dns_cache(self):
        connector = GuardedWebhookClient(parse_allowlist([]))._build_connector()
        try:
            assert isinstance(connector, TCPConnector)
            assert isinstance(connector._resolver, GuardedResolver)
            assert connector.use_dns_cache is False
        finally:
            await connector.close()


def _delivery_row_with_body():
    """A minimal async_operations row as returned to WebhookDeliveryResponse."""
    return {
        "operation_id": "11111111-1111-1111-1111-111111111111",
        "status": "completed",
        "retry_count": 0,
        "next_retry_at": None,
        "error_message": None,
        "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00",
        "task_payload": '{"url": "https://example.com/hook", "event_type": "retain.completed", "webhook_id": "w1"}',
        "result_metadata": '{"last_status_code": 200, "last_response_body": "INTERNAL_SECRET_BODY"}',
    }


class TestDeliveryResponseBodyGating:
    """The delivery-history response must not leak the raw upstream body by default."""

    def test_body_withheld_by_default(self):
        from hindsight_api.api.http import WebhookDeliveryResponse

        resp = WebhookDeliveryResponse.from_async_operation_row(_delivery_row_with_body())
        assert resp.last_response_body is None
        # Status is still surfaced — it's the useful, non-sensitive debug signal.
        assert resp.last_response_status == 200

    def test_body_returned_when_opted_in(self):
        from hindsight_api.api.http import WebhookDeliveryResponse

        resp = WebhookDeliveryResponse.from_async_operation_row(_delivery_row_with_body(), expose_response_body=True)
        assert resp.last_response_body == "INTERNAL_SECRET_BODY"
        assert resp.last_response_status == 200


def test_expose_response_body_is_static_not_bank_configurable():
    """The exfil-gating flag must not be tenant/bank-overridable (would let a
    tenant re-enable exfiltration for itself)."""
    from hindsight_api.config import HindsightConfig

    configurable = HindsightConfig.get_configurable_fields()
    assert "webhook_expose_response_body" not in configurable
    assert "webhook_allowed_hosts" not in configurable


def test_parse_allowlist_splits_hosts_and_networks():
    al = parse_allowlist(["127.0.0.1", "internal.svc", "10.0.0.0/8", "  ", ""])
    assert al.allows_host("internal.svc")
    assert al.allows_host("INTERNAL.SVC")  # case-insensitive
    assert al.allows_ip(ipaddress.ip_address("10.5.5.5"))
    assert al.allows_ip(ipaddress.ip_address("127.0.0.1"))
    assert not al.allows_ip(ipaddress.ip_address("192.168.0.1"))
