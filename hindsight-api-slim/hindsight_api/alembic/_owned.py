"""Let a deployment own a maintenance routine.

The cross-schema discovery routines (``mental_models_with_cron``,
``banks_needing_consolidation``, ``schemas_with_expired_rows``,
``schemas_with_expired_operations``) are installed by migrations and called by
name from the maintenance loop. That late binding is deliberate: an operator can
``CREATE OR REPLACE`` one of them with an implementation that suits their
installation — typically a registry table maintained by a row trigger, so
discovery costs O(rows-of-interest) instead of O(schemas).

The catch is that the replacement is silently undone. Any later migration that
reinstalls the routine wins, because these are all ``CREATE OR REPLACE`` against
the same name, and the operator only finds out from the resulting load. Nothing
in the schema records that the routine was deliberately replaced.

``HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES`` is that record. It is a
comma-separated list of routine names the deployment has taken ownership of;
migrations skip installing anything named in it and leave the existing
definition alone.

    HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES=mental_models_with_cron,banks_needing_consolidation

Read from the process environment at migration time, so it applies to whatever
runs the migration — the CLI, a Job, or an application startup path — without
threading a new option through each of them.

Empty by default: an installation that has not replaced anything is unaffected,
and every routine installs exactly as before.

Two things this deliberately does NOT do:

- It does not stamp the choice into the database. Ownership is a property of the
  deployment, not of the schema, and a deployment that stops overriding a routine
  should get the stock one back on its next migration without a data fix-up.
- It does not verify that a replacement exists. Naming a routine here and then
  not installing one leaves whatever was there before, which for a fresh database
  is nothing — the maintenance loop will then fail on a missing function rather
  than run a scan the operator did not want. That is the safer direction, and it
  fails loudly.
"""

from __future__ import annotations

import logging
import os

from alembic import op

logger = logging.getLogger(__name__)

ENV_EXTERNALLY_OWNED_ROUTINES = "HINDSIGHT_API_EXTERNALLY_OWNED_ROUTINES"


def _owned() -> frozenset[str]:
    raw = os.environ.get(ENV_EXTERNALLY_OWNED_ROUTINES, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def externally_owned(name: str) -> bool:
    """True when the deployment owns ``name`` and migrations must not install it.

    ``name`` is the bare routine name, unqualified by schema — ownership is about
    which implementation runs, not where it lives.
    """
    owned = name in _owned()
    if owned:
        # Logged rather than silent: a skipped install is exactly the kind of
        # thing someone will be trying to account for later, and the migration
        # output is where they will look first.
        logger.info(
            "%s names %r; leaving the existing definition in place.",
            ENV_EXTERNALLY_OWNED_ROUTINES,
            name,
        )
    return owned


def execute_unless_owned(name: str, sql: str) -> None:
    """Run ``sql`` (which installs or drops ``name``) unless the deployment owns it.

    Call-site shape mirrors ``op.execute`` so guarding an existing migration is a
    one-line change and the SQL keeps its original indentation — the routine body
    ends up in ``pg_proc.prosrc`` verbatim, so reformatting it would show up as a
    spurious difference for anyone diffing installed routines against the source.
    """
    if externally_owned(name):
        return
    op.execute(sql)
