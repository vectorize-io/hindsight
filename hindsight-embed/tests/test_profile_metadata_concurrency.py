"""Concurrent profile commands must preserve each other's discovery metadata."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from hindsight_embed.profile_manager import ProfileManager


@pytest.mark.parametrize("first_operation", ["create", "delete"])
def test_concurrent_profile_changes_preserve_both_updates(tmp_path, monkeypatch, first_operation):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    first = ProfileManager()
    second = ProfileManager()
    if first_operation == "delete":
        first.create_profile("alpha", 9100, {"KEY": "first"})
    saving = Event()
    release = Event()
    second_started = Event()
    original_save = first._save_metadata

    def pause_first_save(metadata):
        saving.set()
        assert release.wait(5), "second command did not release the first writer"
        original_save(metadata)

    monkeypatch.setattr(first, "_save_metadata", pause_first_save)

    def first_command():
        if first_operation == "create":
            first.create_profile("alpha", 9100, {"KEY": "first"})
        else:
            first.delete_profile("alpha")

    def second_command():
        second_started.set()
        second.create_profile("beta", 9101, {"KEY": "second"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first_command)
        try:
            assert saving.wait(5)
            second_future = pool.submit(second_command)
            assert second_started.wait(5)
            # Before the fix the second writer publishes while the first owns
            # stale metadata. A transaction lock instead makes it wait here.
            try:
                second_future.result(timeout=0.3)
            except TimeoutError:
                pass
        finally:
            release.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    expected = {"alpha", "beta"} if first_operation == "create" else {"beta"}
    assert set(first._load_metadata().profiles) == expected
    assert second.load_profile_config("beta")["KEY"] == "second"
