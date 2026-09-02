"""
Memory System for AI Agents.

Temporal + Semantic Memory Architecture using PostgreSQL with pgvector.

**Importing this package is cheap.** The public names below resolve on first ACCESS, via the
module ``__getattr__`` of PEP 562, rather than being pulled in at import time. That is a load-time
property other things depend on, not a preference:

``hindsight-api --workers N`` was killing and respawning its workers forever. uvicorn's
multiprocess supervisor uses **spawn**, so every child rebuilds ``__main__`` by re-running
``sys.argv[0]`` — pip's console-script wrapper — whose top line is
``from hindsight_api.main import main``. That import reached this module, and this module used to
pull in the engine, both model wrappers and the whole search stack. So each child re-imported the
entire application BEFORE uvicorn's child bootstrap had even started, missed the supervisor's 5 s
healthcheck, and was SIGKILLed — silently, since SIGKILL leaves no traceback — over and over.
Measured in a dev pod: 12-16 child deaths per 70 s with one worker ever reaching startup.

So: adding an eager top-level import here is not a neutral edit. It puts the cost back on every
spawned worker, and the failure it causes does not look like an import problem — it looks like the
pod being fine.

``apply_default_thread_limits()`` stays eager on purpose, and is the one thing that must: it caps
the native ML thread pools by setting environment variables that OpenBLAS/OpenMP/MKL read only at
load time, so it has to run before anything pulls in numpy/torch/onnxruntime. Laziness below makes
that guarantee stronger, not weaker — those libraries now load later than they used to.
"""

from typing import TYPE_CHECKING

# Cap native ML thread pools (OpenBLAS/OpenMP/MKL) before any import pulls in
# numpy/torch/onnxruntime — they read these env vars only at load time. See
# hindsight_api/_thread_limits.py for the rationale.
from ._thread_limits import apply_default_thread_limits

apply_default_thread_limits()

__version__ = "0.9.2"

# name -> (module, attribute). Kept as data so `__all__`, `__dir__` and the resolver cannot drift.
_LAZY_EXPORTS: "dict[str, tuple[str, str]]" = {
    "HindsightConfig": (".config", "HindsightConfig"),
    "get_config": (".config", "get_config"),
    "CrossEncoderModel": (".engine.cross_encoder", "CrossEncoderModel"),
    "LocalSTCrossEncoder": (".engine.cross_encoder", "LocalSTCrossEncoder"),
    "RemoteTEICrossEncoder": (".engine.cross_encoder", "RemoteTEICrossEncoder"),
    "Embeddings": (".engine.embeddings", "Embeddings"),
    "LocalSTEmbeddings": (".engine.embeddings", "LocalSTEmbeddings"),
    "RemoteTEIEmbeddings": (".engine.embeddings", "RemoteTEIEmbeddings"),
    "LLMConfig": (".engine.llm_wrapper", "LLMConfig"),
    "MemoryEngine": (".engine.memory_engine", "MemoryEngine"),
    "EntryPoint": (".engine.search.trace", "EntryPoint"),
    "NodeVisit": (".engine.search.trace", "NodeVisit"),
    "QueryInfo": (".engine.search.trace", "QueryInfo"),
    "SearchPhaseMetrics": (".engine.search.trace", "SearchPhaseMetrics"),
    "SearchSummary": (".engine.search.trace", "SearchSummary"),
    "SearchTrace": (".engine.search.trace", "SearchTrace"),
    "WeightComponents": (".engine.search.trace", "WeightComponents"),
    "SearchTracer": (".engine.search.tracer", "SearchTracer"),
    "RequestContext": (".models", "RequestContext"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve a public name on first access, then cache it in the module globals."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    # Bind it so every later access is a plain global lookup rather than another __getattr__.
    globals()[name] = value
    return value


def __dir__() -> "list[str]":
    return sorted([*__all__, "__version__"])


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs, never at runtime
    # `noqa: F401` throughout: these are re-exports, and `__all__` is COMPUTED from
    # `_LAZY_EXPORTS` above so the resolver and the export list cannot drift. A linter cannot
    # see a computed `__all__`, so it reads every one of these as unused.
    from .config import HindsightConfig, get_config  # noqa: F401
    from .engine.cross_encoder import CrossEncoderModel, LocalSTCrossEncoder, RemoteTEICrossEncoder  # noqa: F401
    from .engine.embeddings import Embeddings, LocalSTEmbeddings, RemoteTEIEmbeddings  # noqa: F401
    from .engine.llm_wrapper import LLMConfig  # noqa: F401
    from .engine.memory_engine import MemoryEngine  # noqa: F401
    from .engine.search.trace import (  # noqa: F401
        EntryPoint,
        NodeVisit,
        QueryInfo,
        SearchPhaseMetrics,
        SearchSummary,
        SearchTrace,
        WeightComponents,
    )
    from .engine.search.tracer import SearchTracer  # noqa: F401
    from .models import RequestContext  # noqa: F401
