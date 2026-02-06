"""Tests for the search_docs MCP tool."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from hindsight_api.mcp_tools import (
    DocsSource,
    _clean_text,
    _search_vectorize_pipeline,
    _register_search_docs,
)


@pytest.fixture
def mock_vectorize_config():
    """Create a mock config with Vectorize settings."""
    config = MagicMock()
    config.vectorize_org_id = "test-org"
    config.vectorize_api_token = "test-token"
    config.vectorize_core_pipeline_id = "core-pipeline"
    config.vectorize_cloud_pipeline_id = "cloud-pipeline"
    config.vectorize_api_base_url = "https://api.vectorize.io"
    return config


@pytest.fixture
def mock_vectorize_response():
    """Create a mock Vectorize API response."""
    return {
        "documents": [
            {
                "source": "https://docs.example.com/page1",
                "similarity": 0.9,
                "text": "<p>Result 1 with <b>HTML</b></p>",
            },
            {
                "source": "https://docs.example.com/page2",
                "similarity": 0.8,
                "text": "Result 2 plain text",
            },
        ]
    }


class TestDocsSource:
    """Test DocsSource enum."""

    def test_docs_source_values(self):
        """Test DocsSource enum has expected values."""
        assert DocsSource.CORE.value == "core"
        assert DocsSource.CLOUD.value == "cloud"
        assert DocsSource.ALL.value == "all"

    def test_docs_source_from_string(self):
        """Test DocsSource can be created from string."""
        assert DocsSource("core") == DocsSource.CORE
        assert DocsSource("cloud") == DocsSource.CLOUD
        assert DocsSource("all") == DocsSource.ALL

    def test_docs_source_invalid_raises(self):
        """Test invalid source raises ValueError."""
        with pytest.raises(ValueError):
            DocsSource("invalid")


class TestCleanText:
    """Test _clean_text helper function."""

    def test_removes_html_tags(self):
        """Test HTML tags are removed."""
        html = "<p>Hello <b>world</b></p>"
        assert _clean_text(html) == "Hello world"

    def test_collapses_whitespace(self):
        """Test multiple whitespace is collapsed."""
        text = "Hello    world\n\ntest"
        assert _clean_text(text) == "Hello world test"

    def test_strips_whitespace(self):
        """Test leading/trailing whitespace is stripped."""
        text = "  hello world  "
        assert _clean_text(text) == "hello world"

    def test_handles_complex_html(self):
        """Test complex HTML with attributes is cleaned."""
        html = '<div class="test"><span id="foo">Content</span></div>'
        assert _clean_text(html) == "Content"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _clean_text("") == ""

    def test_no_html(self):
        """Test plain text passes through."""
        text = "Plain text content"
        assert _clean_text(text) == "Plain text content"


class TestSearchVectorizePipeline:
    """Test _search_vectorize_pipeline function."""

    @pytest.mark.asyncio
    async def test_search_pipeline_success(self):
        """Test successful pipeline search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "documents": [
                {"source": "https://docs.example.com/page1", "similarity": 0.9, "text": "Result 1"},
                {"source": "https://docs.example.com/page2", "similarity": 0.8, "text": "Result 2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("hindsight_api.mcp_tools.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await _search_vectorize_pipeline(
                api_base_url="https://api.vectorize.io",
                org_id="test-org",
                pipeline_id="test-pipeline",
                api_token="test-token",
                query="test query",
                num_results=5,
            )

            assert "documents" in result
            assert len(result["documents"]) == 2
            assert result["documents"][0]["similarity"] == 0.9

            # Verify the request was made correctly
            mock_instance.post.assert_called_once()
            call_args = mock_instance.post.call_args
            assert "https://api.vectorize.io/v1/org/test-org/pipelines/test-pipeline/retrieval" in str(call_args)

    @pytest.mark.asyncio
    async def test_search_pipeline_with_custom_num_results(self):
        """Test pipeline search with custom num_results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"documents": []}
        mock_response.raise_for_status = MagicMock()

        with patch("hindsight_api.mcp_tools.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            await _search_vectorize_pipeline(
                api_base_url="https://api.vectorize.io",
                org_id="test-org",
                pipeline_id="test-pipeline",
                api_token="test-token",
                query="test query",
                num_results=10,
            )

            # Verify numResults was passed correctly
            call_args = mock_instance.post.call_args
            assert call_args.kwargs["json"]["numResults"] == 10


class TestSearchDocsTool:
    """Tests for the search_docs MCP tool function."""

    @pytest.mark.asyncio
    async def test_search_docs_tool_registration(self):
        """Test that search_docs tool is properly registered."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        assert "search_docs" in tools

    @pytest.mark.asyncio
    async def test_search_docs_returns_error_when_not_configured(self):
        """Test search_docs returns error when Vectorize not configured."""
        from fastmcp import FastMCP

        mock_config = MagicMock()
        mock_config.vectorize_org_id = None
        mock_config.vectorize_api_token = None

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_config):
            result = await search_docs_tool.fn(query="test query")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not configured" in parsed["error"]
        assert parsed["results"] == []

    @pytest.mark.asyncio
    async def test_search_docs_searches_core_only(self, mock_vectorize_config, mock_vectorize_response):
        """Test search_docs with source='core' only searches core pipeline."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_vectorize_response

                result = await search_docs_tool.fn(query="test query", source="core", num_results=3)

                # Should only call once for core pipeline
                assert mock_search.call_count == 1
                call_args = mock_search.call_args
                assert call_args.kwargs["pipeline_id"] == "core-pipeline"

        parsed = json.loads(result)
        assert len(parsed["results"]) == 2
        assert all(r["source"] == "hindsight-core" for r in parsed["results"])

    @pytest.mark.asyncio
    async def test_search_docs_searches_cloud_only(self, mock_vectorize_config, mock_vectorize_response):
        """Test search_docs with source='cloud' only searches cloud pipeline."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_vectorize_response

                result = await search_docs_tool.fn(query="test query", source="cloud", num_results=3)

                # Should only call once for cloud pipeline
                assert mock_search.call_count == 1
                call_args = mock_search.call_args
                assert call_args.kwargs["pipeline_id"] == "cloud-pipeline"

        parsed = json.loads(result)
        assert len(parsed["results"]) == 2
        assert all(r["source"] == "hindsight-cloud" for r in parsed["results"])

    @pytest.mark.asyncio
    async def test_search_docs_searches_all_sources(self, mock_vectorize_config, mock_vectorize_response):
        """Test search_docs with source='all' searches both pipelines."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_vectorize_response

                result = await search_docs_tool.fn(query="test query", source="all", num_results=3)

                # Should call twice - once for core, once for cloud
                assert mock_search.call_count == 2

        parsed = json.loads(result)
        # 2 results from core + 2 from cloud = 4 total
        assert len(parsed["results"]) == 4

    @pytest.mark.asyncio
    async def test_search_docs_results_sorted_by_similarity(self, mock_vectorize_config):
        """Test that results are sorted by similarity score."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        # Return different similarity scores for core and cloud
        core_response = {"documents": [{"source": "core1", "similarity": 0.5, "text": "Core result"}]}
        cloud_response = {"documents": [{"source": "cloud1", "similarity": 0.9, "text": "Cloud result"}]}

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.side_effect = [core_response, cloud_response]

                result = await search_docs_tool.fn(query="test query", source="all")

        parsed = json.loads(result)
        # Cloud result (0.9) should come before core result (0.5)
        assert parsed["results"][0]["similarity"] == 0.9
        assert parsed["results"][1]["similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_search_docs_handles_pipeline_error(self, mock_vectorize_config):
        """Test that search_docs handles pipeline errors gracefully."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.side_effect = Exception("API error")

                result = await search_docs_tool.fn(query="test query", source="core")

        parsed = json.loads(result)
        # Should have error entry but not crash
        assert len(parsed["results"]) == 1
        assert "error" in parsed["results"][0]

    @pytest.mark.asyncio
    async def test_search_docs_cleans_html_from_results(self, mock_vectorize_config, mock_vectorize_response):
        """Test that HTML is stripped from result text."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_vectorize_response

                result = await search_docs_tool.fn(query="test query", source="core")

        parsed = json.loads(result)
        # First result had HTML tags - they should be stripped
        assert "<p>" not in parsed["results"][0]["text"]
        assert "<b>" not in parsed["results"][0]["text"]
        assert "Result 1 with HTML" in parsed["results"][0]["text"]

    @pytest.mark.asyncio
    async def test_search_docs_invalid_source_defaults_to_all(self, mock_vectorize_config, mock_vectorize_response):
        """Test that invalid source defaults to 'all'."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        _register_search_docs(mcp)

        tools = mcp._tool_manager._tools
        search_docs_tool = tools["search_docs"]

        with patch("hindsight_api.mcp_tools.get_config", return_value=mock_vectorize_config):
            with patch("hindsight_api.mcp_tools._search_vectorize_pipeline", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_vectorize_response

                # Pass invalid source - should default to 'all'
                result = await search_docs_tool.fn(query="test query", source="invalid")

                # Should call both pipelines (all)
                assert mock_search.call_count == 2
