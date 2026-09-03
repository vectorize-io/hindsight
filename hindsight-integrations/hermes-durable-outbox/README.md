# Hermes durable Hindsight retain outbox — proposal

**Status:** proposal/reference pattern; not a released integration.

This document proposes a client-side durability pattern for Hermes Agent (or another
Hindsight client) when the Hindsight service can be temporarily unavailable. It is
intentionally an integration concern: it does **not** change Hindsight's server,
database, extraction, embedding, reranking, consolidation, recall, reflect, or API
semantics.

The corresponding Hermes implementation and tests are in the companion proposal:
[NousResearch/hermes-agent#95755](https://github.com/NousResearch/hermes-agent/pull/95755),
with the design discussion in
[Hindsight#3821](https://github.com/vectorize-io/hindsight/issues/3821).

## Problem

A process-local in-memory retain queue is insufficient when:

- Hindsight is offline during a retain;
- the client process exits or restarts before delivery;
- more than one client worker can attempt replay;
- a request is accepted remotely but the client crashes before recording success.

The client needs to protect the request payload locally without pretending that the
client-side queue is part of Hindsight's memory engine.

## Proposed contract

A compatible integration should:

1. Serialize the complete retain request before attempting network delivery.
2. Store it in a profile-scoped durable store before handing work to a worker.
3. Claim rows atomically so concurrent workers cannot deliver the same row concurrently.
4. Prevent a later row for the same document from being claimed while an earlier row
   is pending or in flight.
5. Keep failed rows pending and retry with bounded backoff.
6. Release stale claims and replay pending rows on client startup.
7. Fence acknowledgements and reschedules with a per-claim lease token so a stale
   worker cannot mutate a row after another worker has reclaimed it.
8. Acknowledge/delete a row only after the retain request is accepted by Hindsight.
9. Persist a stable operation identifier for asynchronous retains when the SDK/API
   supports one, and pass that identifier on retries.
10. Route fresh delivery and replay through one ordered writer/coordinator rather than
    letting a replay loop dispatch concurrently with normal retains.
11. Treat operation-status `404` as "no longer pending" only when the provider's
   documented operation lifecycle makes that interpretation safe.

SQLite is one suitable client-side implementation. It should use parameterized SQL,
WAL/busy-timeout settings, a unique local deduplication key, claim fencing, and
restrictive file permissions where supported. The exact local schema is an
implementation detail and must not be added to Hindsight's server database.

## Semantics and limits

This is delivery protection, not universal exactly-once delivery:

- Hindsight remains the source of truth after it accepts a retain.
- A crash after server acceptance and before local acknowledgement can still produce a
  duplicate unless Hindsight honors the supplied operation identifier idempotently.
- Synchronous retains and SDKs without operation-ID support remain at-least-once in
  that narrow crash window.
- A client must not delete a queued payload merely because a network call returned an
  ambiguous exception.
- The outbox must not alter recall ranking, memory extraction, embedding, reranking,
  or any server-side operation.

## Prefetch/read-after-write behavior

If an integration prefetches the next recall in the background, local queue drain is
not sufficient for asynchronous retains: `aretain_batch` can return when Hindsight has
accepted the request but before the resulting memory is recall-visible.

The recommended ordering is:

1. wait for the local retain dispatch to finish, within a bounded deadline;
2. poll the server-side operation status for tracked asynchronous retains;
3. recall only after completion, or proceed with an explicitly logged timeout;
4. discard unresolved operation IDs from the short-lived prefetch wait set so a broken
   status endpoint cannot impose unbounded latency on every later turn.

The operation-status check should avoid expensive first-use SDK imports on the prefetch
thread. In the Hermes `0.9.2` client, importing the generated exception/model graph
lazily on that thread was measured at roughly 4.4 seconds; checking the documented
status/class shape avoids that startup delay while preserving 404 handling.

## Validation matrix

A client integration proposal should test these cases separately:

- persist and reopen a queued retain across process boundaries;
- deduplicate the same local request;
- fail delivery and verify the row remains pending with retry metadata;
- recover the service and verify replay/acknowledgement;
- claim concurrently from multiple connections and verify one claimant;
- release a stale claim after simulated process loss;
- preserve the same operation identifier across replay;
- forward that identifier only when the installed SDK exposes the parameter;
- stop writer, replay, prefetch, client, and local-store resources in safe order;
- verify the prefetch ordering with pending and completed server operations;
- verify that a transient status error remains pending;
- verify that an operation-not-found response follows the documented completion rule;
- verify local database and backup permissions where the platform enforces them.

## Version compatibility

Server and SDK versions are separate release streams. A future implementation should
verify both independently:

```bash
gh api repos/vectorize-io/hindsight/releases/latest --jq '.tag_name'
python -m pip index versions hindsight-client
curl -fsS http://<hindsight-host>:<port>/version
```

The SDK shape must be inspected rather than inferred from matching version numbers:

```python
import inspect
from hindsight_client import Hindsight

assert "operation_id" in inspect.signature(Hindsight.aretain_batch).parameters
```

If the SDK changes its operation-status or idempotency contract, update the client
integration and its tests first. Do not modify Hindsight server behavior merely to make
a client queue work.

## Maintainer questions

- Should Hindsight document a recommended client-side outbox pattern for integrations?
- What exact operation-ID/idempotency guarantees should the SDK document?
- Should a small reusable helper live under `hindsight-integrations`, or should each
  client own its queue implementation?
- What is the canonical interpretation of an operation-status `404` after retention?
- Which retry and acknowledgement guidance should be common across official clients?

This proposal intentionally leaves those choices with Hindsight maintainers. It is a
reference contract and review artifact, not a request for a Hindsight server migration.
