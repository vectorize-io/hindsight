"""Unit tests for pg0 URL parsing (`parse_pg0_url`)."""

from hindsight_api.pg0 import Pg0Url, parse_pg0_url


def test_bare_pg0():
    assert parse_pg0_url("pg0") == Pg0Url(is_pg0=True, instance_name="hindsight")


def test_named_instance():
    assert parse_pg0_url("pg0://mydb") == Pg0Url(is_pg0=True, instance_name="mydb")


def test_named_instance_with_port():
    assert parse_pg0_url("pg0://mydb:5544") == Pg0Url(is_pg0=True, instance_name="mydb", port=5544)


def test_empty_instance_falls_back_to_default():
    assert parse_pg0_url("pg0://").instance_name == "hindsight"
    assert parse_pg0_url("pg0://:5544") == Pg0Url(is_pg0=True, instance_name="hindsight", port=5544)


def test_non_pg0_url_passthrough():
    parsed = parse_pg0_url("postgresql://user:pwd@localhost:5432/db")
    assert parsed == Pg0Url(is_pg0=False)


def test_credentials_user_and_password():
    assert parse_pg0_url("pg0://alice:s3cret@mydb:5544") == Pg0Url(
        is_pg0=True,
        instance_name="mydb",
        port=5544,
        username="alice",
        password="s3cret",
    )


def test_credentials_user_only():
    assert parse_pg0_url("pg0://alice@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password=None
    )


def test_credentials_without_port():
    assert parse_pg0_url("pg0://alice:s3cret@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password="s3cret"
    )


def test_password_may_contain_at_sign():
    # rsplit on the last "@" keeps an "@" inside the password intact.
    assert parse_pg0_url("pg0://alice:p@ss@mydb:5544") == Pg0Url(
        is_pg0=True,
        instance_name="mydb",
        port=5544,
        username="alice",
        password="p@ss",
    )


def test_empty_password_after_colon_is_empty_string():
    # "user:" explicitly sets an empty password (distinct from omitting it).
    assert parse_pg0_url("pg0://alice:@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password=""
    )
