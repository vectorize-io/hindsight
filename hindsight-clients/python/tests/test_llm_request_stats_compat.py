"""New clients can read token stats from servers without reasoning usage."""

from hindsight_client_api.models.llm_request_token_sums import LLMRequestTokenSums


def test_token_stats_accept_older_response_without_reasoning_field():
    stats = LLMRequestTokenSums.from_dict({"input": 100, "output": 20, "cached": 10, "total": 120})

    assert stats is not None
    assert stats.input == 100
    assert stats.output == 20
    assert stats.thoughts is None
    assert stats.total == 120


def test_token_stats_keep_reported_reasoning_field():
    stats = LLMRequestTokenSums.from_dict(
        {"input": 100, "output": 20, "cached": 10, "thoughts": 60, "total": 120}
    )

    assert stats is not None
    assert stats.thoughts == 60
