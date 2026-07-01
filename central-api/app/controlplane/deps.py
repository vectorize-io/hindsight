"""Principal resolution + membership guards for control-plane routes.

Bridges the verified ``RequestContext`` to a control-plane ``users`` row. In dev
the context resolves to a stable dev user (bootstrapped on first use) so the
contract is exercisable without a real IdP. Membership is always checked against
the DB — never trusted from the request body.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import DEFAULT_ACTOR, ContextDep, RequestContext
from app.db import repositories as repo
from app.db import tables as t
from app.db.engine import get_session
from app.db.ids import utcnow

DEV_EMAIL = "dev@collabmind.local"


async def ensure_principal(context: RequestContext, session: AsyncSession) -> dict:
    """Return the users row for the caller, creating the dev user if needed."""
    user = await repo.get_user(session, context.actor_id)
    if user:
        return user
    if context.actor_id == DEFAULT_ACTOR or context.auth_method == "dev_default":
        # Bootstrap the dev principal so dev/test flows have a real user row.
        await session.execute(t.users.insert().values(
            id=context.actor_id, email=DEV_EMAIL, display_name="Dev Operator",
            is_operator=True, created_at=utcnow(),
        ))
        return await repo.get_user(session, context.actor_id)  # type: ignore[return-value]
    raise HTTPException(status_code=401, detail="unknown principal")


async def get_principal(
    context: ContextDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> tuple[dict, RequestContext, AsyncSession]:
    user = await ensure_principal(context, session)
    return user, context, session


PrincipalDep = Annotated[tuple[dict, RequestContext, AsyncSession], Depends(get_principal)]


async def require_member(session: AsyncSession, *, workspace_id: str, user_id: str) -> None:
    if not await repo.is_workspace_member(session, workspace_id=workspace_id, user_id=user_id):
        raise HTTPException(status_code=403, detail="not a workspace member")
