"""The process-global metrics collector must be restored, not leaked (#3780).

``create_metrics_collector()`` replaces a module global. Before this, the API
lifespan installed a real ``MetricsCollector`` and never put the previous one
back, so a single app start poisoned the rest of the process: unlike
``NoOpMetricsCollector``, the real collector inspects the values it is handed
(``if cached_input_tokens > 0``), and the provider tests feed it bare
``MagicMock`` usage objects. Whichever test files happened to run after an
app-starting test in the same xdist worker then failed with "'>' not supported
between instances of 'MagicMock' and 'int'".
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.api.http import create_app
from hindsight_api.metrics import (
    MetricsCollector,
    NoOpMetricsCollector,
    create_metrics_collector,
    get_metrics_collector,
    initialize_metrics,
    reset_metrics_collector,
)


@pytest.fixture(autouse=True)
def _metrics_initialized():
    """``MetricsCollector()`` needs a meter; the app's lifespan sets one up itself."""
    initialize_metrics(service_name="hindsight-api-test", service_version="0.0.0")


def test_reset_restores_the_given_collector():
    sentinel = NoOpMetricsCollector()
    reset_metrics_collector(sentinel)

    create_metrics_collector()
    assert isinstance(get_metrics_collector(), MetricsCollector)

    reset_metrics_collector(sentinel)
    assert get_metrics_collector() is sentinel


def test_reset_without_argument_falls_back_to_noop():
    create_metrics_collector()
    assert isinstance(get_metrics_collector(), MetricsCollector)

    reset_metrics_collector()
    assert isinstance(get_metrics_collector(), NoOpMetricsCollector)


@pytest.mark.asyncio
async def test_app_lifespan_restores_the_collector_it_replaced():
    """Starting and stopping the app must leave the global collector as it found it."""
    # A stub engine: the lifespan only needs the attributes it touches, and
    # initialize_memory=False keeps it away from the database entirely. A backend
    # that reports no worker-poller support keeps the poller from starting
    # whatever HINDSIGHT_API_WORKER_ENABLED says.
    memory = MagicMock()
    memory._pool = None
    memory._backend.supports_worker_poller = False
    memory.tenant_extension = None
    memory._tenant_extension = None
    memory.close = AsyncMock()

    before = NoOpMetricsCollector()
    reset_metrics_collector(before)

    app = create_app(memory, initialize_memory=False)
    async with app.router.lifespan_context(app):
        # The app really does install its own collector — otherwise this test
        # would pass for the wrong reason.
        assert isinstance(get_metrics_collector(), MetricsCollector)

    assert get_metrics_collector() is before
