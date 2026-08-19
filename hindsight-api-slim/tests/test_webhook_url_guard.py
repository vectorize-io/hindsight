"""Unit tests for webhook SSRF hardening (hindsight_api.webhooks.url_guard).

These are deterministic and DB-free: they cover the destination-URL validation,
the resolve-and-pin logic, and the guarded transport's connection pinning
against a real loopback server. The security guarantees under test:

- private/loopback/link-local/metadata destinations are blocked by default,
- an operator allowlist re-permits specific hosts / IP ranges,
- DNS names are resolved and every resolved address is checked (so a name that
  resolves to an internal address is blocked),
- the transport connects only to a validated IP while preserving the original
  Host header (virtual host) so a rebind cannot swap in an internal address.
"""

import ipaddress
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from hindsight_api.webhooks.url_guard import (
    GuardedAsyncTransport,
    WebhookURLError,
    _ip_is_blocked,
    parse_allowlist,
    resolve_and_validate,
    validate_url_syntax,
)


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
        with pytest.raises(WebhookURLError):
            validate_url_syntax("http:///nohost", parse_allowlist([]))

    def test_rejects_internal_ip_literal(self):
        for bad in [
            "http://127.0.0.1/x",
            "http://169.254.169.254/latest/meta-data/",
            "https://10.1.2.3/hook",
            "http://[::1]/x",
        ]:
            with pytest.raises(WebhookURLError):
                validate_url_syntax(bad, parse_allowlist([]))

    def test_allows_public_dns_name(self):
        # DNS names are not resolved at syntax time — deferred to delivery.
        validate_url_syntax("https://example.com/hook", parse_allowlist([]))

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


@pytest.fixture
def loopback_server():
    """A loopback-only HTTP server standing in for an internal endpoint."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"INTERNAL_SECRET host=" + self.headers.get("Host", "").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


@pytest.mark.asyncio
class TestGuardedTransport:
    async def test_blocks_loopback_by_default(self, loopback_server):
        transport = GuardedAsyncTransport(parse_allowlist([]))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(WebhookURLError):
                await client.get(f"http://127.0.0.1:{loopback_server}/internal")

    async def test_allowlisted_loopback_succeeds_and_preserves_host(self, loopback_server):
        transport = GuardedAsyncTransport(parse_allowlist(["127.0.0.1"]))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get(f"http://127.0.0.1:{loopback_server}/internal")
        assert resp.status_code == 200
        # Host header carried the original authority (virtual host preserved).
        assert f"127.0.0.1:{loopback_server}" in resp.text

    async def test_pins_dns_name_and_falls_back_across_addresses(self, loopback_server):
        # localhost may resolve to ::1 first (unreachable here) then 127.0.0.1;
        # the transport must try each validated address rather than pin only one.
        transport = GuardedAsyncTransport(parse_allowlist(["localhost"]))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get(f"http://localhost:{loopback_server}/internal")
        assert resp.status_code == 200
        assert f"localhost:{loopback_server}" in resp.text

    async def test_rejects_non_http_scheme(self):
        transport = GuardedAsyncTransport(parse_allowlist([]))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(WebhookURLError):
                await client.get("ftp://example.com/x")


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
