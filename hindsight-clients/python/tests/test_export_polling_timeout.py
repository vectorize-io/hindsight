"""Export completion deadlines over real HTTP; no generated-client methods are mocked."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from aiohttp import web

from hindsight_client import Hindsight
from hindsight_client_api.models.document_export_submit_response import DocumentExportSubmitResponse
from hindsight_client_api.models.operation_status_response import OperationStatusResponse

OPERATION_ID = "123e4567-e89b-12d3-a456-426614174000"
DOWNLOAD_URL = "/v1/default/files/download/banks/test-bank/export.zip"
ARCHIVE = b"PK\x03\x04synthetic archive"
EXPORT_METHODS = ("aexport_documents", "aexport_bank")


@dataclass
class ExportServer:
    base_url: str = ""
    status: str = "completed"
    hold: Literal["submission", "poll", "download"] | None = None
    submissions: int = 0
    polls: int = 0
    downloads: int = 0
    submitted: asyncio.Event = field(default_factory=asyncio.Event)
    polled: asyncio.Event = field(default_factory=asyncio.Event)
    downloading: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    responded: asyncio.Event = field(default_factory=asyncio.Event)

    async def submit(self, request: web.Request) -> web.Response:
        self.submissions += 1
        self.submitted.set()
        if self.hold == "submission":
            await self.release.wait()
        response = DocumentExportSubmitResponse(operation_id=OPERATION_ID)
        return web.json_response(response.model_dump(exclude_none=True), status=202)

    async def poll(self, request: web.Request) -> web.Response:
        self.polls += 1
        self.polled.set()
        if self.hold == "poll":
            await self.release.wait()
        response = OperationStatusResponse(
            operation_id=OPERATION_ID,
            status=self.status,
            result_metadata={"download_url": DOWNLOAD_URL} if self.status == "completed" else None,
            error_message="synthetic export failure" if self.status in ("failed", "cancelled") else None,
        )
        self.responded.set()
        return web.json_response(response.model_dump(exclude_none=True))

    async def download(self, request: web.Request) -> web.Response:
        self.downloads += 1
        self.downloading.set()
        if self.hold == "download":
            await self.release.wait()
        return web.Response(body=ARCHIVE, content_type="application/zip")


@pytest.fixture
async def export_server() -> AsyncIterator[ExportServer]:
    state = ExportServer()
    app = web.Application()
    app.router.add_post("/v1/default/banks/test-bank/document-transfer/export", state.submit)
    app.router.add_post("/v1/default/banks/test-bank/transfer/export", state.submit)
    app.router.add_get(f"/v1/default/banks/test-bank/operations/{OPERATION_ID}", state.poll)
    app.router.add_get(DOWNLOAD_URL, state.download)
    runner = web.AppRunner(app)
    await runner.setup()
    # Keep the OS-assigned listener, avoiding a free-port selection/rebind race.
    server = await asyncio.get_running_loop().create_server(runner.server, "127.0.0.1", 0)
    state.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        yield state
    finally:
        state.release.set()
        server.close()
        await server.wait_closed()
        await runner.cleanup()


async def clean_up_task(task: asyncio.Task[bytes], client: Hindsight, server: ExportServer) -> None:
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, asyncio.TimeoutError, TimeoutError, RuntimeError):
        await task
    server.release.set()
    await client.aclose()


@pytest.mark.parametrize("method", EXPORT_METHODS)
@pytest.mark.parametrize("late_status", ("processing", "completed"))
async def test_export_timeout_bounds_pending_status_request(
    export_server: ExportServer, method: str, late_status: str
) -> None:
    export_server.hold = "poll"
    export_server.status = late_status
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=0.05, poll_interval=0))
    try:
        await asyncio.wait_for(export_server.polled.wait(), timeout=2)
        # The peer cannot finish its response until this test releases it. A
        # one-second guard gives the 50ms client deadline ample scheduler slack.
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "export timeout did not interrupt the pending status request"
        with pytest.raises(TimeoutError):
            task.result()
        export_server.release.set()
        await asyncio.wait_for(export_server.responded.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(export_server.downloading.wait(), timeout=0.1)
        assert export_server.downloads == 0
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_export_timeout_bounds_poll_interval(export_server: ExportServer, method: str) -> None:
    export_server.status = "processing"
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=0.05, poll_interval=5))
    try:
        await asyncio.wait_for(export_server.responded.wait(), timeout=2)
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "export timeout waited for the full polling interval"
        with pytest.raises(TimeoutError):
            task.result()
        assert export_server.polls == 1
        assert export_server.downloads == 0
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_caller_cancellation_does_not_download_late_export(export_server: ExportServer, method: str) -> None:
    export_server.hold = "poll"
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=60, poll_interval=0))
    try:
        await asyncio.wait_for(export_server.polled.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        export_server.release.set()
        await asyncio.wait_for(export_server.responded.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(export_server.downloading.wait(), timeout=0.1)
        assert export_server.downloads == 0
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
@pytest.mark.parametrize("status", ("completed", "failed", "cancelled"))
async def test_export_terminal_status_is_preserved(export_server: ExportServer, method: str, status: str) -> None:
    export_server.status = status
    client = Hindsight(base_url=export_server.base_url)
    try:
        call = getattr(client, method)("test-bank", timeout=5, poll_interval=0)
        if status == "completed":
            assert await call == ARCHIVE
            assert export_server.downloads == 1
        else:
            with pytest.raises(RuntimeError, match=f"{status}: synthetic export failure"):
                await call
            assert export_server.downloads == 0
        assert export_server.submissions == 1
        assert export_server.polls == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize("method", EXPORT_METHODS)
@pytest.mark.parametrize("stage", ("submission", "download"))
async def test_export_timeout_only_bounds_completion_polling(
    export_server: ExportServer, method: str, stage: Literal["submission", "download"]
) -> None:
    export_server.hold = stage
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=0.05, poll_interval=0))
    try:
        entered = export_server.submitted if stage == "submission" else export_server.downloading
        await asyncio.wait_for(entered.wait(), timeout=2)
        # These stages use the independently configured per-request timeout;
        # tightening the polling deadline must not silently change that scope.
        done, _ = await asyncio.wait({task}, timeout=0.15)
        assert task not in done
        export_server.release.set()
        assert await asyncio.wait_for(task, timeout=2) == ARCHIVE
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", ("export_documents", "export_bank"))
async def test_sync_export_wrapper_uses_same_http_flow(export_server: ExportServer, method: str) -> None:
    def export() -> bytes:
        with Hindsight(base_url=export_server.base_url) as client:
            return getattr(client, method)("test-bank", timeout=5, poll_interval=0)

    assert await asyncio.wait_for(asyncio.to_thread(export), timeout=2) == ARCHIVE
    assert export_server.submissions == 1
    assert export_server.polls == 1
    assert export_server.downloads == 1


@pytest.mark.parametrize("method", EXPORT_METHODS)
@pytest.mark.parametrize("timeout", (0, -1))
async def test_nonpositive_export_timeout_does_not_poll(
    export_server: ExportServer, method: str, timeout: float
) -> None:
    client = Hindsight(base_url=export_server.base_url)
    try:
        with pytest.raises(TimeoutError, match="did not complete within"):
            await getattr(client, method)("test-bank", timeout=timeout)
        assert export_server.submissions == 1
        assert export_server.polls == 0
        assert export_server.downloads == 0
    finally:
        await client.aclose()


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_shorter_request_timeout_is_not_export_deadline(export_server: ExportServer, method: str) -> None:
    export_server.hold = "poll"
    client = Hindsight(base_url=export_server.base_url, timeout=0.05)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=60))
    try:
        await asyncio.wait_for(export_server.polled.wait(), timeout=2)
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done
        with pytest.raises(asyncio.TimeoutError) as error:
            task.result()
        assert "did not complete within" not in str(error.value)
        assert export_server.downloads == 0
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_caller_cancellation_between_polls(export_server: ExportServer, method: str) -> None:
    export_server.status = "processing"
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=60, poll_interval=60))
    try:
        await asyncio.wait_for(export_server.responded.wait(), timeout=2)
        # Give the already-sent status response time to settle, while confirming
        # the export is still pending before cancelling the long interval.
        done, _ = await asyncio.wait({task}, timeout=0.1)
        assert task not in done
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert export_server.polls == 1
        assert export_server.downloads == 0
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_client_is_reusable_after_export_timeout(export_server: ExportServer, method: str) -> None:
    export_server.hold = "poll"
    client = Hindsight(base_url=export_server.base_url)
    task = asyncio.create_task(getattr(client, method)("test-bank", timeout=0.05))
    try:
        await asyncio.wait_for(export_server.polled.wait(), timeout=2)
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done
        with pytest.raises(TimeoutError):
            task.result()
        export_server.release.set()
        await asyncio.wait_for(export_server.responded.wait(), timeout=2)
        export_server.hold = None
        assert await getattr(client, method)("test-bank", timeout=5) == ARCHIVE
        assert export_server.downloads == 1
    finally:
        await clean_up_task(task, client, export_server)


@pytest.mark.parametrize("method", EXPORT_METHODS)
async def test_caller_cancellation_wins_when_status_completes(export_server: ExportServer, method: str) -> None:
    client = Hindsight(base_url=export_server.base_url)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    cancellation_injected = asyncio.Event()
    cancel_results: list[bool] = []
    selected_children: list[asyncio.Task[Any]] = []
    export_task: asyncio.Task[bytes] | None = None

    def cancel_export_on_status_completion(status_task: asyncio.Task[Any]) -> None:
        if not status_task.cancelled() and export_task is not None:
            cancellation_injected.set()
            cancel_results.append(export_task.cancel())

    def task_factory(
        event_loop: asyncio.AbstractEventLoop, coroutine: Coroutine[Any, Any, Any], **kwargs: Any
    ) -> asyncio.Task[Any]:
        if previous_factory is None:
            task = asyncio.Task(coroutine, loop=event_loop, **kwargs)
        else:
            task = previous_factory(event_loop, coroutine, **kwargs)
        if (
            export_task is not None
            and asyncio.current_task(loop=event_loop) is export_task
            and export_server.submitted.is_set()
            and not selected_children
        ):
            # Select the export's first owned child after submission, without
            # depending on any private helper or generated coroutine name. Its
            # HTTP status response is real; only scheduling cancellation here
            # makes the completion/cancellation boundary deterministic.
            selected_children.append(task)
            task.add_done_callback(cancel_export_on_status_completion)
        return task

    loop.set_task_factory(task_factory)
    try:
        export_task = asyncio.create_task(getattr(client, method)("test-bank", timeout=5))
        caller_cancelled = False
        try:
            await asyncio.wait_for(export_task, timeout=2)
        except asyncio.CancelledError:
            caller_cancelled = True
        assert cancellation_injected.is_set(), "status-completion cancellation was not exercised"
        assert caller_cancelled, "export swallowed cancellation at status completion"
        assert cancel_results == [True]
        assert export_server.polls == 1
        assert export_server.responded.is_set()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(export_server.downloading.wait(), timeout=0.1)
        assert export_server.downloads == 0
    finally:
        loop.set_task_factory(previous_factory)
        if export_task is not None:
            await clean_up_task(export_task, client, export_server)
        else:
            await client.aclose()
