"""Fail-closed retrieval governance tests."""

from datetime import timedelta

from app.db.ids import utcnow
from app.governance.retrieval_policy import Principal, evaluate_retrieval

CONNECTED = {"status": "connected"}
DOC = {"enabled": True, "trashed": False}
READER_PERM = [{"ptype": "user", "role": "reader", "email_address": "a@x.com"}]


def _member(email="a@x.com", domain=None):
    return Principal(user_id="u", email=email, domain=domain, is_workspace_member=True)


def test_allow_when_user_permitted():
    d = evaluate_retrieval(principal=_member(), connector=CONNECTED, document=DOC,
                           permissions=READER_PERM)
    assert d.allowed and d.reason == "permitted"


def test_allow_anyone_and_domain():
    anyone = [{"ptype": "anyone", "role": "reader"}]
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED, document=DOC,
                              permissions=anyone).allowed
    domain = [{"ptype": "domain", "role": "reader", "domain": "x.com"}]
    assert evaluate_retrieval(principal=_member(domain="x.com"), connector=CONNECTED,
                              document=DOC, permissions=domain).allowed


def test_deny_non_member():
    p = Principal(user_id="u", email="a@x.com", is_workspace_member=False)
    assert evaluate_retrieval(principal=p, connector=CONNECTED, document=DOC,
                              permissions=READER_PERM).reason == "not_workspace_member"


def test_deny_missing_connector():
    assert evaluate_retrieval(principal=_member(), connector=None, document=DOC,
                              permissions=READER_PERM).reason == "missing_connector"


def test_deny_inactive_connector():
    assert evaluate_retrieval(principal=_member(), connector={"status": "disconnected"},
                              document=DOC, permissions=READER_PERM).reason == "inactive_connector"


def test_deny_missing_document():
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED, document=None,
                              permissions=READER_PERM).reason == "missing_source_metadata"


def test_deny_disabled_and_trashed():
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED,
                              document={"enabled": False, "trashed": False},
                              permissions=READER_PERM).reason == "document_disabled"
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED,
                              document={"enabled": True, "trashed": True},
                              permissions=READER_PERM).reason == "source_trashed"


def test_deny_unknown_permission_when_no_snapshot():
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED, document=DOC,
                              permissions=None).reason == "unknown_permission"


def test_deny_when_not_in_permissions():
    perms = [{"ptype": "user", "role": "reader", "email_address": "other@x.com"}]
    assert evaluate_retrieval(principal=_member(), connector=CONNECTED, document=DOC,
                              permissions=perms).reason == "not_permitted_on_source"


def test_deny_expired_permission():
    perms = [{"ptype": "user", "role": "reader", "email_address": "a@x.com",
              "expiration_time": utcnow() - timedelta(days=1)}]
    assert not evaluate_retrieval(principal=_member(), connector=CONNECTED, document=DOC,
                                  permissions=perms).allowed
