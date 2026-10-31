"""Every OpenAI SDK client Hindsight builds carries the hindsight User-Agent (#5942).

Each client is constructed in its own module, so the one that forgets the header is the one
nobody tested: embeddings shipped with the SDK's default ``AsyncOpenAI/Python`` identity while
every LLM transport sent ``hindsight/openai-compatible/<version>``. Scan the whole package so a
new construction site fails here instead of in a gateway's traffic report.
"""

import re
from pathlib import Path

import hindsight_api

_CONSTRUCTION = re.compile(r"\b(?:Async)?(?:Azure)?OpenAI\(")


def test_every_openai_client_module_applies_the_user_agent_helper():
    package = Path(hindsight_api.__file__).parent
    sites = [p for p in package.rglob("*.py") if _CONSTRUCTION.search(p.read_text())]
    assert sites, "no OpenAI client construction found; the pattern has gone stale"
    missing = [str(p.relative_to(package)) for p in sites if "with_openai_compatible_user_agent(" not in p.read_text()]
    assert missing == []
