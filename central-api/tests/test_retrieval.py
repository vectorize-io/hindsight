"""Tests for retrieval service."""

import pytest


@pytest.mark.asyncio
async def test_retrieval_service_search_modes():
    """Test search modes."""
    from app.retrieval.service import RetrievalService
    from sqlalchemy.ext.asyncio import AsyncSession

    svc = RetrievalService(AsyncSession(), "test")
    assert svc.tenant_id == "test"


def test_retrieval_service_index_code():
    """Test code indexing stub."""
    import asyncio
    from app.retrieval.service import RetrievalService
    from sqlalchemy.ext.asyncio import AsyncSession

    svc = RetrievalService(AsyncSession(), "test")
    result = asyncio.run(svc.index_code("src/main.py", "def main(): pass", "python"))
    assert result["path"] == "src/main.py"
    assert result["language"] == "python"
    assert result["status"] == "indexed"


def test_retrieval_service_index_document():
    """Test document indexing stub."""
    import asyncio
    from app.retrieval.service import RetrievalService
    from sqlalchemy.ext.asyncio import AsyncSession

    svc = RetrievalService(AsyncSession(), "test")
    result = asyncio.run(svc.index_document("README.md", "# Project"))
    assert result["path"] == "README.md"
    assert result["status"] == "indexed"
