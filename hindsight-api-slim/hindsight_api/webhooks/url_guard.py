"""SSRF hardening for outbound webhook delivery.

A webhook destination URL (and its method/headers/params) is fully
caller-controlled. Without restriction the delivery worker can be pointed at
the cloud metadata service (``169.254.169.254``), loopback admin endpoints, or
private-network hosts, and its response is stored where the delivery-history
API returns it — server-side request forgery with response exfiltration
(CWE-918 / CWE-200).

This module rejects such destinations. It is deny-by-default for
private/loopback/link-local/reserved ranges; operators re-permit specific
internal hosts (e.g. ``127.0.0.1`` for local testing, or an internal receiver)
via an explicit allowlist. At delivery time the host is resolved by
:class:`GuardedResolver` — the only resolver the delivery connector has — and
the connection is made only to the addresses it validated, so a DNS name cannot
be rebound to an internal address between validation and connect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Mapping
from dataclasses import dataclass

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from yarl import URL

from ..engine.aiohttp_session import LoopLocalSession, UpstreamHTTPError, per_phase_timeout

_ALLOWED_SCHEMES = ("http", "https")

# Ranges to block explicitly because ``ipaddress`` classification for them is
# absent or version-dependent (e.g. CGNAT 100.64.0.0/10 only counts as private
# from Python 3.13). Keeping them here makes the block deterministic across
# interpreter versions.
_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 carrier-grade NAT / shared address space
)

# ipaddress network/address base types differ for v4/v6; alias for annotations.
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class WebhookURLError(ValueError):
    """A webhook destination URL targets a disallowed (private/internal) address."""


@dataclass(frozen=True)
class Allowlist:
    """Operator-configured exceptions to the private-range block.

    Entries are matched two ways: exact hostname (so a DNS name is permitted
    without pinning it to an IP) and IP/CIDR membership (so a resolved address
    can be permitted). An empty allowlist means "public destinations only".
    """

    hostnames: frozenset[str]
    networks: tuple[_IPNetwork, ...]

    def allows_host(self, host: str) -> bool:
        return host.lower() in self.hostnames

    def allows_ip(self, ip: _IPAddress) -> bool:
        return any(ip in net for net in self.networks)


def parse_allowlist(entries: list[str]) -> Allowlist:
    """Split allowlist entries into hostnames and IP/CIDR networks.

    An entry that parses as an IP or CIDR becomes a network match; anything else
    is treated as a hostname. Blank entries are ignored.
    """
    hostnames: set[str] = set()
    networks: list[_IPNetwork] = []
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
            continue
        except ValueError:
            hostnames.add(entry.lower())
    return Allowlist(frozenset(hostnames), tuple(networks))


def _as_ip(host: str) -> _IPAddress | None:
    """Return the parsed IP if ``host`` is an IP literal, else None."""
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def _ip_is_blocked(ip: _IPAddress) -> bool:
    """True if the address is in a range unsafe for caller-supplied fetches."""
    # An IPv4-mapped IPv6 address (::ffff:127.0.0.1) must be judged on the
    # embedded v4 address, otherwise loopback/private checks miss it.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if any(ip in net for net in _EXTRA_BLOCKED_NETWORKS):
        return True
    return (
        ip.is_private  # RFC1918, ULA, etc.
        or ip.is_loopback
        or ip.is_link_local  # 169.254.0.0/16 — includes the cloud metadata IP
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified  # 0.0.0.0 / ::
    )


def _validate_host_literal(host: str, allowlist: Allowlist) -> None:
    """Reject an IP-literal host in a blocked range (no DNS).

    Also catches the legacy numeric IPv4 spellings (``2130706433``, ``127.1``,
    ``0x7f.1``) that ``ipaddress`` refuses but the OS resolver maps onto an
    address, so they cannot smuggle a loopback/private target past the check.
    """
    if allowlist.allows_host(host):
        return
    literal = _as_ip(host)
    if literal is None:
        try:
            literal = ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return  # a DNS name — validated when it is resolved
    if _ip_is_blocked(literal) and not allowlist.allows_ip(literal):
        raise WebhookURLError(
            f"Webhook URL host '{host}' targets a private/loopback/link-local address, "
            "which is not an allowed destination"
        )


def _parse_target(url: str) -> URL:
    """Parse ``url`` and check scheme and host presence (no DNS)."""
    # yarl silently strips control characters (WHATWG behaviour) where httpx
    # rejected them; keep rejecting so the accepted set does not widen.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise WebhookURLError("Invalid webhook URL: contains non-printable characters")
    try:
        parsed = URL(url)
        host = parsed.host
        parsed.port  # noqa: B018 — force port validation (yarl parses it lazily)
    except (ValueError, TypeError) as exc:
        raise WebhookURLError(f"Invalid webhook URL: {exc}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise WebhookURLError(f"Webhook URL scheme must be http or https, got '{parsed.scheme or 'empty'}'")
    if not host:
        raise WebhookURLError("Webhook URL must include a host")
    return parsed


def validate_url_syntax(url: str, allowlist: Allowlist) -> None:
    """Registration-time check (no DNS).

    Rejects non-http(s) schemes, missing hosts, and IP-literal hosts that point
    at a blocked range. Hosts that are DNS names are accepted here and fully
    validated (resolved + pinned) at delivery time, so this never blocks on the
    network in the request path.

    Raises:
        WebhookURLError: if the URL is structurally unsafe.
    """
    parsed = _parse_target(url)
    assert parsed.host is not None  # guaranteed by _parse_target
    _validate_host_literal(parsed.host, allowlist)


@dataclass(frozen=True)
class _ResolvedAddress:
    family: int
    ip: str


async def _resolve(host: str, port: int, family: int = socket.AF_UNSPEC) -> list[_ResolvedAddress]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebhookURLError(f"Webhook host '{host}' did not resolve: {exc}") from exc
    addresses: list[_ResolvedAddress] = []
    for info in infos:
        address = _ResolvedAddress(family=info[0], ip=str(info[4][0]))
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise WebhookURLError(f"Webhook host '{host}' did not resolve to any address")
    return addresses


async def _resolve_validated(
    host: str, port: int, allowlist: Allowlist, family: int = socket.AF_UNSPEC
) -> list[_ResolvedAddress]:
    """Resolve a DNS name and reject the whole answer if any address is blocked."""
    addresses = await _resolve(host, port, family)
    if allowlist.allows_host(host):
        return addresses
    for address in addresses:
        ip = ipaddress.ip_address(address.ip)
        if _ip_is_blocked(ip) and not allowlist.allows_ip(ip):
            raise WebhookURLError(f"Webhook host '{host}' resolved to disallowed address {address.ip}")
    return addresses


async def resolve_and_validate(host: str, port: int, allowlist: Allowlist) -> list[str]:
    """Resolve ``host`` and return the validated IPs a connection may use.

    Rejects if *any* resolved address is blocked (rather than silently keeping
    the good ones) so a split-horizon / partially-rebound DNS answer can't slip
    an internal address through. The full validated list is returned (not just
    the first) so the connector can fall back across A/AAAA records while still
    only ever connecting to an address we checked.

    Raises:
        WebhookURLError: on a disallowed literal, a disallowed resolved address,
        or a resolution failure.
    """
    literal = _as_ip(host)
    if literal is not None:
        if _ip_is_blocked(literal) and not allowlist.allows_ip(literal):
            raise WebhookURLError(f"Webhook host '{host}' targets a disallowed address")
        return [host]
    return [address.ip for address in await _resolve_validated(host, port, allowlist)]


class GuardedResolver(AbstractResolver):
    """aiohttp resolver that only ever yields validated addresses.

    It is the sole resolver of the webhook delivery connector, so every DNS
    name a delivery connects to is resolved *and checked* here, at connect
    time. aiohttp then connects to the returned addresses in turn (dual-stack
    fallback) while keeping the original hostname for the ``Host`` header and
    TLS SNI / certificate verification — it never re-resolves on its own.
    """

    def __init__(self, allowlist: Allowlist) -> None:
        self._allowlist = allowlist

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        addresses = await _resolve_validated(host, port, self._allowlist, family)
        return [
            ResolveResult(
                hostname=host,
                host=address.ip,
                port=port,
                family=address.family,
                proto=0,
                flags=socket.AI_NUMERICHOST | socket.AI_NUMERICSERV,
            )
            for address in addresses
        ]

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class WebhookResponse:
    """A 2xx webhook response, body already read."""

    status_code: int
    body: str


_DEFAULT_TIMEOUT_SECONDS = 30.0


class GuardedWebhookClient:
    """The only HTTP client webhook delivery uses; every request is SSRF-checked.

    * IP-literal hosts never reach a resolver in aiohttp, so they are checked
      here before the request is sent.
    * DNS names are resolved and checked by :class:`GuardedResolver` at connect
      time. The connector's DNS cache is off, so every new connection re-runs
      the check (a rebind cannot reuse a stale validated answer); pooled
      connections are keyed by host/port/TLS and were validated when opened.
    * Redirects are not followed (a 3xx is a failed delivery), and proxy
      environment variables are ignored, so the validated destination is the
      only one ever contacted.
    """

    def __init__(self, allowlist: Allowlist) -> None:
        self._allowlist = allowlist
        self._sessions = LoopLocalSession(
            timeout=per_phase_timeout(_DEFAULT_TIMEOUT_SECONDS),
            connector_factory=self._build_connector,
        )

    def _build_connector(self) -> aiohttp.BaseConnector:
        return aiohttp.TCPConnector(resolver=GuardedResolver(self._allowlist), use_dns_cache=False)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str] | None,
        body: bytes | None,
        timeout_seconds: float,
    ) -> WebhookResponse:
        """Send one request and return the 2xx response.

        Raises:
            WebhookURLError: the destination is disallowed (never valid on retry).
            UpstreamHTTPError: a non-2xx response (3xx included), body attached.
            aiohttp.ClientError / asyncio.TimeoutError: transport failures.
        """
        parsed = _parse_target(url)
        assert parsed.host is not None  # guaranteed by _parse_target
        _validate_host_literal(parsed.host, self._allowlist)

        session = self._sessions.get()
        async with session.request(
            method,
            parsed,
            headers=dict(headers),
            params=dict(params) if params else None,
            data=body,
            timeout=per_phase_timeout(timeout_seconds),
            allow_redirects=False,
        ) as response:
            text = await response.text(errors="replace")
            if not 200 <= response.status < 300:
                raise UpstreamHTTPError(response.status, text, response.headers, str(response.url))
            return WebhookResponse(status_code=response.status, body=text)

    async def close(self) -> None:
        await self._sessions.close()
