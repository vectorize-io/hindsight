"""Governed ingestion pipeline.

The connector emits jobs; the worker processes them. The Drive connector never
embeds directly — embedding/indexing happen here, behind governance and audit.
v0.1 proves the *flow* (states + audit + status); the embed/index steps are
delegated to the memory/vector layer and stubbed so no real vectors are written.
"""
