"""Extension-declared, bank-scoped tables.

An extension may provision its own tables in the tenant schema (e.g. audit
receipts, per-bank policy state). Those tables are invisible to core, so they
silently fall out of the per-tenant data-lifecycle operations core owns:

* **Backup / restore** — ``hindsight-admin backup``/``restore`` copies a fixed
  set of core tables and ``TRUNCATE ... CASCADE``\\ s them on restore. An
  extension table absent from that set is dropped from the backup *and* — if it
  carries a FK to ``banks`` — wiped by the cascade with no way to restore it.
* **Bank teardown** — :meth:`MemoryEngine.delete_bank` clears a bank by
  deleting the core rows and letting ``banks``' FK cascade handle the rest. An
  extension table that scopes by ``bank_id`` without a cascading FK leaks
  orphaned rows when the bank is deleted.

An extension declares its bank-scoped tables via
:meth:`TenantExtension.extra_bank_tables`; core consults that list in the
operations above. The extension still owns the DDL (creation lives in its
provisioning path) — this descriptor only tells core which tables to sweep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Unquoted SQL identifiers only. ``name`` and ``bank_id_column`` are
# interpolated into SQL (schema-qualified via ``fq_table``), so they must be
# validated to a safe identifier shape rather than trusted verbatim.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class BankScopedTable:
    """A bank-scoped table an extension owns and core should sweep.

    Args:
        name: Unqualified table name. Schema-qualified at use via ``fq_table``.
        bank_id_column: Column holding the bank id, used to scope a per-bank
            delete. Defaults to ``"bank_id"``.
        include_in_backup: Include the table in ``hindsight-admin``
            backup/restore. Defaults to ``True`` — a bank-scoped table almost
            always wants restore coverage; opt out only for regenerable or
            transient state.
        delete_with_bank: Delete the table's rows for a bank during a full
            :meth:`MemoryEngine.delete_bank`. Defaults to ``True``. Set
            ``False`` to retain rows that should outlive the bank (e.g. audit
            receipts a compliance regime requires kept).
    """

    name: str
    bank_id_column: str = "bank_id"
    include_in_backup: bool = True
    delete_with_bank: bool = True

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.match(self.name):
            raise ValueError(f"BankScopedTable.name {self.name!r} is not a valid SQL identifier")
        if not _IDENTIFIER_RE.match(self.bank_id_column):
            raise ValueError(f"BankScopedTable.bank_id_column {self.bank_id_column!r} is not a valid SQL identifier")
