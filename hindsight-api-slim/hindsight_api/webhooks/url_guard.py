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
via an explicit allowlist. At delivery time the host is resolved and the
connection is pinned to the validated IP so a DNS name cannot be rebound to an
internal address between validation and connect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass

import httpx

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


def validate_url_syntax(url: str, allowlist: Allowlist) -> None:
    """Registration-time check (no DNS).

    Rejects non-http(s) schemes, missing hosts, and IP-literal hosts that point
    at a blocked range. Hosts that are DNS names are accepted here and fully
    validated (resolved + pinned) at delivery time, so this never blocks on the
    network in the request path.

    Raises:
        WebhookURLError: if the URL is structurally unsafe.
    """
    try:
        parsed = httpx.URL(url)
    except Exception as exc:  # httpx.InvalidURL and friends
        raise WebhookURLError(f"Invalid webhook URL: {exc}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise WebhookURLError(f"Webhook URL scheme must be http or https, got '{parsed.scheme or 'empty'}'")
    host = parsed.host
    if not host:
        raise WebhookURLError("Webhook URL must include a host")
    if allowlist.allows_host(host):
        return
    literal = _as_ip(host)
    if literal is not None and _ip_is_blocked(literal) and not allowlist.allows_ip(literal):
        raise WebhookURLError(
            f"Webhook URL host '{host}' targets a private/loopback/link-local address, "
            "which is not an allowed destination"
        )


async def _resolve(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebhookURLError(f"Webhook host '{host}' did not resolve: {exc}") from exc
    ips = [info[4][0] for info in infos]
    if not ips:
        raise WebhookURLError(f"Webhook host '{host}' did not resolve to any address")
    return ips


async def resolve_and_validate(host: str, port: int, allowlist: Allowlist) -> list[str]:
    """Resolve ``host`` and return the validated IPs to pin the connection to.

    Rejects if *any* resolved address is blocked (rather than silently keeping
    the good ones) so a split-horizon / partially-rebound DNS answer can't slip
    an internal address through. The full validated list is returned (not just
    the first) so the caller can preserve httpx's fallback across A/AAAA records
    while still only ever connecting to an address we checked.

    Raises:
        WebhookURLError: on a disallowed literal, a disallowed resolved address,
        or a resolution failure.
    """
    literal = _as_ip(host)
    if literal is not None:
        if _ip_is_blocked(literal) and not allowlist.allows_ip(literal):
            raise WebhookURLError(f"Webhook host '{host}' targets a disallowed address")
        return [host]

    ips = await _resolve(host, port)
    if allowlist.allows_host(host):
        return ips
    for ip_str in ips:
        ip = ipaddress.ip_address(ip_str)
        if _ip_is_blocked(ip) and not allowlist.allows_ip(ip):
            raise WebhookURLError(f"Webhook host '{host}' resolved to disallowed address {ip_str}")
    return ips


class GuardedAsyncTransport(httpx.AsyncHTTPTransport):
    """httpx transport that validates and pins every outbound webhook request.

    All webhook delivery traffic flows through one instance of this transport,
    so there is no code path that can reach ``httpx`` without the SSRF check.
    On each request it resolves the host, rejects disallowed destinations, and
    rewrites the connection target to the validated IP while preserving the
    original ``Host`` header and TLS SNI (via the ``sni_hostname`` extension) so
    virtual hosting and certificate verification keep working.
    """

    def __init__(self, allowlist: Allowlist, **kwargs) -> None:
        super().__init__(**kwargs)
        self._allowlist = allowlist

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.scheme not in _ALLOWED_SCHEMES:
            raise WebhookURLError(f"Webhook URL scheme must be http or https, got '{url.scheme or 'empty'}'")
        host = url.host
        if not host:
            raise WebhookURLError("Webhook URL must include a host")
        port = url.port or (443 if url.scheme == "https" else 80)

        validated_ips = await resolve_and_validate(host, port, self._allowlist)
        if validated_ips == [host]:
            # Host is already a validated IP literal — nothing to pin/rewrite.
            return await super().handle_async_request(request)

        # Connect to a validated IP but keep the original identity: the Host
        # header (already set by httpx from the original URL) is left untouched,
        # and sni_hostname carries the real name for TLS. Try each validated
        # address in turn so a dual-stack host whose first record is unreachable
        # still connects — without ever re-resolving to an unchecked address.
        base_extensions = {**request.extensions, "sni_hostname": host}
        last_error: Exception | None = None
        for ip in validated_ips:
            request.url = url.copy_with(host=ip)
            request.extensions = dict(base_extensions)
            try:
                return await super().handle_async_request(request)
            except httpx.ConnectError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise last_error
