from hindsight_api.engine.remote_retry import jittered_backoff


def test_jittered_backoff_grows_caps_and_jitters():
    assert 0.8 <= jittered_backoff(0, 1.0, 10.0) <= 1.2
    assert 1.6 <= jittered_backoff(1, 1.0, 10.0) <= 2.4
    assert 8.0 <= jittered_backoff(10, 1.0, 10.0) <= 12.0  # capped at max_backoff, then jittered
