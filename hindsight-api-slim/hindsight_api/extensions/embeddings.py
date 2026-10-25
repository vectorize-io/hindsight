"""Embeddings extension: supply the embeddings model the engine stores and queries with.

The server's embeddings settings (``HINDSIGHT_API_EMBEDDINGS_*``) build one model at
startup, and every vector the engine writes or searches comes from it. This
extension supplies that object instead, so a deployment can:

- **Use a provider core does not ship**, or a client with its own auth, pooling or
  telemetry.
- **Choose the model per tenant.** Return an :class:`~hindsight_api.engine.embeddings.Embeddings`
  that looks at the current tenant schema
  (:func:`hindsight_api.engine.memory_engine.get_current_schema`) on each call and
  delegates to the model that tenant's vectors were made with -- for example to
  put new tenants on a new model while existing ones keep theirs.

It is asked once, when the engine is built. ``build_default`` builds the model the
server's own settings describe; call it to wrap or fall back to that model, or ignore
it to replace it outright (it is never built unless called). ``self.context`` is
already set when it is asked, but the engine has not started yet: its database pool
does not exist until ``initialize()``.

Things an implementation must get right:

- **One dimension.** ``dimension`` is read at startup to size the vector columns of
  every schema. A routing implementation must only route to models of that
  dimension; every vector is checked against it, so a model of another dimension
  fails each call rather than storing a vector of the wrong size.
- **A stored vector is only comparable with vectors from the same model.** A routing
  implementation must give a tenant the same model for its whole life (or re-embed
  it), and should refuse a call it cannot route rather than fall back to a default:
  a vector from the wrong model is not an error anywhere, it just makes that
  tenant's recall worse.
- **Initialization.** The engine calls ``initialize()`` on the returned object once
  at startup; a wrapper initializes the models it holds.
- **Optional capabilities pass through.** Callers read ``batch_size``,
  ``max_concurrent_requests``, ``tokenizer_encoding``, ``query_prefix`` and
  ``passage_prefix`` from the object when present. A wrapper that delegates should
  expose the delegate's.

Load it with::

    HINDSIGHT_API_EMBEDDINGS_EXTENSION=mypackage.embeddings:MyEmbeddings
    HINDSIGHT_API_EMBEDDINGS_MY_OPTION=value   # -> config["my_option"]

The configuration prefix ``HINDSIGHT_API_EMBEDDINGS_`` is shared with the server's
own embeddings settings, so ``config`` holds those too (``provider``, ``tei_url``,
...). Give the extension's own options names the server does not use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api.engine.embeddings import Embeddings


class EmbeddingsExtension(Extension, ABC):
    """Supplies the :class:`Embeddings` the memory engine uses for every vector."""

    @abstractmethod
    def create_embeddings(self, build_default: Callable[[], Embeddings]) -> Embeddings:
        """Return the embeddings model the engine will use.

        Args:
            build_default: Builds the model described by the server's
                ``HINDSIGHT_API_EMBEDDINGS_*`` settings. Each call builds a new one.

        Raising fails engine startup.
        """
