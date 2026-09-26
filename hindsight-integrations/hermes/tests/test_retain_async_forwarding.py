"""The hindsight_retain tool handler must forward the provider's configured
retain_async mode into aretain_batch as a CALL argument (retain_async is a
call-level arg, never an item key) — otherwise tool retains silently drop the
async/sync choice and diverge from the auto-retain path
(NousResearch/hermes-agent#60648)."""

import pytest


@pytest.mark.parametrize("retain_async", [True, False])
def test_retain_tool_forwards_configured_retain_async(provider, retain_async):
    instance, fake = provider({"retain_async": retain_async})

    instance.handle_tool_call("hindsight_retain", {"content": "remember this"})

    assert fake.retains[0]["retain_async"] is retain_async
    instance.shutdown()
