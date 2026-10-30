"""A transaction whose query timed out must fail, not hang, when its connection drops.

asyncpg <= 0.31 left a timed-out query's cancellation pending until the server
answered it. If the connection died first, nothing ever resolved it, and every
later query on that connection awaited forever — including the ROLLBACK that
``transaction()`` sends on the way out. That is how the worker's claim loop
wedged silently while /health stayed green (#6067); asyncpg 0.32 fixed it.

This drives our ``PostgresConnection.transaction()`` through a local proxy that
swallows the server's replies, then cuts the connection, so a lock-file
downgrade below 0.32 fails here instead of in production.
"""

import asyncio
from urllib.parse import urlparse

import asyncpg
import pytest

from hindsight_api.engine.db.postgresql import PostgresConnection


class _BlackholeProxy:
    """TCP proxy to Postgres that can stop forwarding replies, then drop the client."""

    def __init__(self, pg_host: str, pg_port: int) -> None:
        self._pg_host = pg_host
        self._pg_port = pg_port
        self._client: asyncio.StreamWriter | None = None
        self.swallow_replies = False

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    def drop_client(self) -> None:
        assert self._client is not None
        self._client.transport.abort()

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, replies: bool) -> None:
        try:
            while data := await reader.read(65536):
                if replies and self.swallow_replies:
                    continue
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _handle(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        # Only the first client is the session; asyncpg opens a second, short-lived
        # connection to send the cancel request for the timed-out query.
        if self._client is None:
            self._client = client_writer
        server_reader, server_writer = await asyncio.open_connection(self._pg_host, self._pg_port)
        await asyncio.gather(
            self._pipe(client_reader, server_writer, replies=False),
            self._pipe(server_reader, client_writer, replies=True),
        )


@pytest.mark.asyncio
async def test_timed_out_query_on_dropped_connection_fails_the_transaction(pg0_db_url):
    url = urlparse(pg0_db_url)
    proxy = _BlackholeProxy(url.hostname or "localhost", url.port or 5432)
    proxy_port = await proxy.start()
    raw = await asyncpg.connect(pg0_db_url, host="127.0.0.1", port=proxy_port, ssl=False)
    conn = PostgresConnection(raw)

    async def claim() -> None:
        async with conn.transaction():
            proxy.swallow_replies = True
            asyncio.get_running_loop().call_later(1.0, proxy.drop_client)
            await conn.execute("SELECT pg_sleep(5)", timeout=0.3)

    task = asyncio.create_task(claim())
    try:
        # Before the fix the ROLLBACK never returns; bound the wait so that hang
        # fails the test instead of stalling the run.
        done, _ = await asyncio.wait({task}, timeout=10)
        assert task in done, "transaction is stuck in ROLLBACK on a dropped connection"
        with pytest.raises((asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError)):
            task.result()
        assert raw.is_closed()
    finally:
        task.cancel()
        raw.terminate()
        await proxy.close()
