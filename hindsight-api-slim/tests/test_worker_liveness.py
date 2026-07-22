"""Tests for hindsight_api.worker.liveness heartbeat-staleness logic."""

from hindsight_api.worker.liveness import Heartbeat, is_alive


def test_is_alive_fresh_heartbeat():
    """A heartbeat younger than the threshold is alive."""
    assert is_alive(age_seconds=0.0, threshold_seconds=30.0) is True
    assert is_alive(age_seconds=29.999, threshold_seconds=30.0) is True


def test_is_alive_stale_heartbeat():
    """A heartbeat at or beyond the threshold is not alive.

    This is the case a genuinely wedged event loop hits: the heartbeat never
    catches up, so the livenessProbe eventually fails and K8s restarts the pod.
    """
    assert is_alive(age_seconds=30.0, threshold_seconds=30.0) is False
    assert is_alive(age_seconds=45.0, threshold_seconds=30.0) is False


def test_is_alive_tolerates_multi_second_block():
    """A several-second loop block (sync SigV4 signing during Bedrock
    inference) stays alive under the default 30s threshold — the whole point
    of the thread-based liveness server."""
    assert is_alive(age_seconds=8.0, threshold_seconds=30.0) is True


def test_heartbeat_beat_resets_age():
    """beat() resets the age toward zero; a fresh Heartbeat is nearly zero age."""
    hb = Heartbeat()
    assert hb.age() < 1.0
    hb.beat()
    assert hb.age() < 1.0
