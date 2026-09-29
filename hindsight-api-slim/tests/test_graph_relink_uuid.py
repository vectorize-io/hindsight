"""relink_pass ids: str on PostgreSQL (id::text), already UUID on the Oracle backend."""

import uuid


def test_as_uuid_accepts_str_and_uuid():
    from hindsight_api.engine.memories.pg.graph import _as_uuid

    u = uuid.uuid4()
    assert _as_uuid(u) is u
    assert _as_uuid(str(u)) == u
