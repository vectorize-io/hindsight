"""A webhook is delivered to somewhere allowed, signed so the receiver can trust it.

Webhooks are the one place Hindsight makes an outbound request to a URL a caller
chose, which makes them two features at once.

The first is delivery: something happened, and a payload describing it reaches
the endpoint. Asserted here against a real receiver, because the server's own
delivery log only proves it *tried*.

The second is that the caller's URL is untrusted input. A server that fetches
whatever it is handed is an SSRF primitive — point a webhook at `169.254.169.254`
or a service on the internal network and Hindsight becomes the attacker's HTTP
client, from inside the perimeter. That is GHSA-ggrr-69wp-fj54, and the guard
that fixed it is asserted here as a security property rather than a config
detail.

The signature is what makes the delivery worth acting on: an unsigned webhook is
indistinguishable from one anybody could forge at the receiver.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac

import pytest
from hindsight_client_api.exceptions import BadRequestException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SECRET = "s3cret"


@pytest.fixture(autouse=True)
def _extraction(llm):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())


async def _deliver(client, stubs, bank: str, stub_url: str, settled) -> None:
    """Register a webhook, cause an event, wait for it to land at the receiver."""
    await client.webhooks.create_webhook(bank, {"url": f"{stub_url}/webhook", "secret": SECRET})
    await client.aretain(bank_id=bank, content="Alice moved to Berlin.")
    await settled(bank)

    for _ in range(30):
        if stubs.webhooks:
            return
        await asyncio.sleep(1)
    raise AssertionError("no webhook delivery arrived at the receiver")


async def test_a_delivery_reaches_the_endpoint(client, stubs, bank_id, stub_server, settled):
    """At the receiver, not merely in the server's own log — that would only show
    an attempt."""
    await _deliver(client, stubs, bank_id, stub_server.url, settled)

    assert len(stubs.webhooks) == 1


async def test_the_payload_says_what_happened(client, stubs, bank_id, stub_server, settled):
    """A receiver has to be able to act without calling back for context: which
    bank, which event, which operation, and when."""
    await _deliver(client, stubs, bank_id, stub_server.url, settled)
    body = stubs.webhooks[0].body

    assert body["bank_id"] == bank_id
    assert body["event"] == "consolidation.completed"
    assert body["operation_id"]
    assert body["timestamp"]
    assert "data" in body


async def test_the_delivery_is_signed_with_the_shared_secret(client, stubs, bank_id, stub_server, settled):
    """Recomputed here the way a receiver would, rather than merely asserting a
    header exists. A signature over the wrong bytes, or with the wrong secret, is
    a header that looks right and verifies nowhere.
    """
    await _deliver(client, stubs, bank_id, stub_server.url, settled)
    received = stubs.webhooks[0]

    import json as _json

    raw = _json.dumps(received.body, separators=(",", ":")).encode()
    expected = hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()

    signature = received.headers["x-hindsight-signature"]
    assert signature.startswith("sha256=")
    assert hmac.compare_digest(signature.removeprefix("sha256="), expected)


async def test_the_event_type_travels_in_a_header_too(client, stubs, bank_id, stub_server, settled):
    """So a receiver can route without parsing the body first."""
    await _deliver(client, stubs, bank_id, stub_server.url, settled)

    assert stubs.webhooks[0].headers["x-hindsight-event"] == "consolidation.completed"


async def test_the_server_records_the_delivery_as_completed(client, stubs, bank_id, stub_server, settled):
    """The other side of the same event: whoever is debugging a receiver that
    saw nothing needs to know whether Hindsight sent it and what came back."""
    await _deliver(client, stubs, bank_id, stub_server.url, settled)

    webhooks = await client.webhooks.list_webhooks(bank_id)
    deliveries = await client.webhooks.list_webhook_deliveries(bank_id, webhooks.items[0].id)

    assert len(deliveries.items) == 1
    delivery = deliveries.items[0]
    assert delivery.status == "completed"
    assert delivery.last_response_status == 200
    assert delivery.last_error is None


async def test_a_loopback_destination_is_refused(client, bank_id):
    """The SSRF guard (GHSA-ggrr-69wp-fj54).

    The caller chooses this URL, so an unguarded webhook turns Hindsight into an
    HTTP client the attacker aims — at link-local metadata endpoints, at services
    reachable only from inside the network. Refused at registration, where the
    operator finds out, rather than at delivery time where it is a log line.

    Note the test server allowlists exactly the stub's host so the tests above can
    receive anything at all; this address is private and *not* on that list, so
    the guard is genuinely still under test.
    """
    with pytest.raises(BadRequestException):
        await client.webhooks.create_webhook(bank_id, {"url": "http://10.0.0.1/webhook"})


async def test_a_link_local_metadata_address_is_refused(client, bank_id):
    """The specific address that makes SSRF profitable on every major cloud."""
    with pytest.raises(BadRequestException):
        await client.webhooks.create_webhook(bank_id, {"url": "http://169.254.169.254/latest/meta-data/"})
