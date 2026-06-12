"""Federation — outbound integration to a shared Graphiti world graph (C-track).

Owns the HTTP client, circuit breaker, and the outbox worker that drains
``graphiti_outbox`` rows into Graphiti ``/triplet`` calls. Channel A of C4
(re-consolidation triggered by an ``invalidated_edges`` response) lives in
the worker too — deep-dive 4 §1.2 established that channel A needs zero new
infrastructure (it reuses the same response field that the forwarder already
parses), so the worker's response handling covers it in the same code path.

Channel B (cross-Agent push via overlay post-processing) and channel C
(polling fallback for mixed-traffic deployments) are out of scope for the
worker — they live in the overlay / inbound endpoint (deep-dive 4 §1.2-1.3).
"""
