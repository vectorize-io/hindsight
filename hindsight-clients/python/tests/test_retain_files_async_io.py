"""File preparation must leave the asyncio loop available for unrelated work."""

import asyncio
import os
import threading
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named-pipe filesystem proof requires POSIX")
async def test_file_upload_yields_while_a_real_filesystem_read_waits(tmp_path: Path) -> None:
    fifo = tmp_path / "notes.txt"
    os.mkfifo(fifo)
    reader_connected = threading.Event()
    heartbeat = threading.Event()
    observed_heartbeat: list[bool] = []
    payload = b"memory from a delayed filesystem source"

    def writer() -> None:
        with fifo.open("wb") as stream:
            reader_connected.set()
            observed_heartbeat.append(heartbeat.wait(timeout=1))
            stream.write(payload)

    async def tick() -> None:
        while not reader_connected.is_set():
            await asyncio.sleep(0)
        heartbeat.set()

    async def retain(request: web.Request) -> web.Response:
        reader = await request.multipart()
        received_files = []
        async for part in reader:
            if part.filename:
                received_files.append((part.filename, bytes(await part.read())))
        assert received_files == [("notes.txt", payload)]
        return web.json_response({"operation_ids": []})

    app = web.Application()
    app.router.add_post("/v1/default/banks/test/files/retain", retain)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")).rstrip("/"))
        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        ticker = asyncio.create_task(tick())
        try:
            await asyncio.wait_for(client.aretain_files("test", [fifo]), timeout=3)
            await ticker
            assert observed_heartbeat == [True], "the file read prevented unrelated loop work"
        finally:
            heartbeat.set()
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            await client.aclose()
            await asyncio.to_thread(thread.join, 2)


async def test_unreadable_file_preserves_its_filesystem_error(tmp_path: Path) -> None:
    client = Hindsight(base_url="http://127.0.0.1:1")
    try:
        with pytest.raises(FileNotFoundError):
            await client.aretain_files("test", [tmp_path / "missing.txt"])
    finally:
        await client.aclose()
