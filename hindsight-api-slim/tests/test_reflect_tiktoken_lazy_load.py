import importlib
import sys
from unittest.mock import MagicMock, patch


def _drop_reflect_modules() -> None:
    for name in list(sys.modules):
        if name == "hindsight_api.engine.reflect" or name.startswith("hindsight_api.engine.reflect."):
            sys.modules.pop(name)


def test_reflect_import_does_not_load_tiktoken_encoding():
    _drop_reflect_modules()

    with patch("tiktoken.get_encoding") as get_encoding:
        reflect = importlib.import_module("hindsight_api.engine.reflect")

    get_encoding.assert_not_called()
    assert reflect.run_reflect_agent is not None


def test_reflect_token_counting_loads_tiktoken_encoding_when_used():
    _drop_reflect_modules()
    fake_encoding = MagicMock()
    fake_encoding.encode.side_effect = lambda text: text.split()

    with patch("tiktoken.get_encoding", return_value=fake_encoding) as get_encoding:
        agent = importlib.import_module("hindsight_api.engine.reflect.agent")
        prompts = importlib.import_module("hindsight_api.engine.reflect.prompts")

        count = agent._count_messages_tokens([{"role": "user", "content": "one two"}])
        final_prompt = prompts.build_final_prompt(
            query="What happened?",
            context_history=[{"tool": "recall", "output": {"answer": "three four"}}],
            bank_profile={"name": "test"},
            max_context_tokens=1000,
        )

    assert count == 2
    assert "three four" in final_prompt
    get_encoding.assert_called_once_with("cl100k_base")
