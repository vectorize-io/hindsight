"""The tenant schema and the bank the current operation runs for.

Both are context variables, so each async task sees its own values. They live
here rather than in ``engine.memory_engine`` so that config, metrics, extensions
and the engine's own modules can read them without importing the engine.
"""

import contextvars

from .config import get_config

# Note: default is None, actual default comes from config via get_current_schema()
_current_schema: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_schema", default=None)

# Set by the engine wherever it learns the bank (recall/retain/batch/reflect/task execution) so
# downstream provider calls can attribute spend per bank — e.g. tagging the OpenAI `user`
# field for cost gateways. None outside a bank-scoped operation.
_current_bank_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_bank_id", default=None)


def get_current_schema() -> str:
    """Get the current schema from context (falls back to config default)."""
    schema = _current_schema.get()
    if schema is None:
        return get_config().database_schema
    return schema


def get_current_bank_id() -> str | None:
    """Get the bank id of the in-flight operation, or None outside a bank-scoped context."""
    return _current_bank_id.get()
