"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

from hindsight_api.alembic._dialect import run_for_dialect

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, Sequence[str], None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def _pg_upgrade() -> None:
    """PostgreSQL upgrade. Set to ``None`` below if this migration is Oracle-only."""
    ${upgrades if upgrades else "pass"}


def _pg_downgrade() -> None:
    ${downgrades if downgrades else "pass"}


def _oracle_upgrade() -> None:
    """Oracle upgrade. Set to ``None`` below if this migration is Postgres-only."""
    pass


def _oracle_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
