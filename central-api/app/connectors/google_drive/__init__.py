"""Read-only Google Drive connector v0.1.

Read-only by policy: minimal scopes (drive.metadata.readonly, drive.readonly).
No write/delete/admin scopes, no file write-back, no deletion from Drive. The
connector discovers metadata, snapshots permissions, and *emits ingestion jobs*
— it never embeds directly.
"""
