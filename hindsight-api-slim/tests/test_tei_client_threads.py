"""The TEI client is not shared between threads.

`encode` runs through `run_in_executor`, so several threads use this provider at once. On a
free-threaded build they run genuinely in parallel, and httpcore's sync pool has an unguarded
check-then-use on shared connection state:

    keepalive_expired = self._expire_at is not None and now > self._expire_at

Another thread can null `_expire_at` between the two halves, and the comparison raises
`'>' not supported between instances of 'float' and 'NoneType'` — seen as a 500 out of recall.
"""

from __future__ import annotations

import threading

from hindsight_api.engine.embeddings import RemoteTEIEmbeddings


def test_each_thread_gets_its_own_client():
    tei = RemoteTEIEmbeddings(base_url="http://tei.invalid")
    seen: dict[int, int] = {}
    barrier = threading.Barrier(4)

    def grab(i: int) -> None:
        barrier.wait()
        seen[i] = id(tei._client_for_thread())

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen.values())) == 4, f"threads shared a client: {seen}"


def test_the_same_thread_reuses_its_client():
    """Per thread, not per call: a client per request would open a pool per request."""
    tei = RemoteTEIEmbeddings(base_url="http://tei.invalid")
    assert tei._client_for_thread() is tei._client_for_thread()
