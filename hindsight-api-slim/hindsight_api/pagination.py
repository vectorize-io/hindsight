"""Page-size caps shared by every list endpoint.

A ``limit`` above the cap is trimmed to the cap rather than rejected, so a client
asking for 100,000 rows gets a full page of the cap and keeps working.
"""

from pydantic import AfterValidator

MAX_PAGE_LIMIT = 1000
MAX_GRAPH_LIMIT = 5000


def trim_to(cap: int) -> AfterValidator:
    """Validator that lowers a ``limit`` above ``cap`` to ``cap``."""
    return AfterValidator(lambda value: min(value, cap))


# Put after the Query in the annotation: ``Annotated[int, Query(ge=0), TRIM_TO_PAGE]``.
# Nesting an Annotated alias inside another one breaks FastAPI's OpenAPI output.
TRIM_TO_PAGE = trim_to(MAX_PAGE_LIMIT)
TRIM_TO_GRAPH = trim_to(MAX_GRAPH_LIMIT)
