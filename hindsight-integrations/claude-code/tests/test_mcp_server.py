"""Regression tests for mcp_server.py.

The MCP server runs as a stdio subprocess; the Claude Code MCP client buffers
up to 16 MB looking for a JSON-RPC message boundary (newline) and disconnects
if a single response exceeds that ceiling. The /mental-models LIST endpoint
defaults to detail=full, which returns synthesized content + reflect_response
for every page in the bank — at realistic agent scales (tens of pages with
~100 KB content each) this exceeds 16 MB in a single response and triggers a
deterministic disconnect during agent_knowledge_list_pages. The fix is to
request detail=metadata, which the API supports specifically for this use case
(see the upstream PR that added the parameter: "reduces payload for agent
boot flows and MCP clients where context budget is limited").
"""

import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _read_mcp_server_source() -> str:
    return open(os.path.join(SCRIPTS_DIR, "mcp_server.py"), encoding="utf-8").read()


class TestDefaultBankUsesProjectCwd:
    """The MCP server must derive its default bank from Claude's project cwd."""

    def test_default_bank_prefers_launcher_project_cwd(self):
        src = _read_mcp_server_source()
        assert "HINDSIGHT_MCP_PROJECT_CWD" in src
        assert 'os.environ.get("HINDSIGHT_MCP_PROJECT_CWD", os.getcwd())' in src
        assert '_hook_input = {"cwd": _project_cwd, "session_id": ""}' in src


class TestListPagesUsesMetadataProjection:
    """`agent_knowledge_list_pages` must request the API's metadata projection.

    The /mental-models endpoint accepts a `detail` query parameter with values
    `metadata` / `content` / `full`, defaulting to `full`. The full projection
    returns synthesized content + reflect_response for every page in the bank,
    which has caused real disconnects (>16 MB single JSON-RPC response). The
    docstring promises "IDs and names only", so the request must pin
    `detail=metadata`.
    """

    def test_list_pages_request_uses_detail_metadata(self):
        src = _read_mcp_server_source()
        assert "/mental-models?detail=metadata" in src, (
            "list_pages must request detail=metadata; the API defaults to full"
        )
        list_pages_def = src.find("def agent_knowledge_list_pages")
        next_def = src.find("def agent_knowledge_get_page")
        assert list_pages_def > 0 and next_def > list_pages_def
        list_pages_body = src[list_pages_def:next_def]
        assert "detail=metadata" in list_pages_body


class TestDisabledKnowledgeToolsKeepsServerAlive:
    """When `enableKnowledgeTools` is false the server must NOT exit at startup.

    `.mcp.json` registers this server unconditionally, so Claude Code expects a
    live process. If the server exits immediately (the old behavior), Claude
    Code treats it as a crashed server and surfaces a `-32000` reconnect error
    on every prompt (issue #1995). The disabled path must instead run an empty
    MCP server (`mcp.run`) so the process stays alive with no tools registered.
    """

    def test_disabled_branch_runs_empty_server_not_bare_exit(self):
        src = _read_mcp_server_source()
        gate = src.find('if not _config.get("enableKnowledgeTools")')
        assert gate > 0, "expected the enableKnowledgeTools startup gate"
        # The disabled branch ends at the next top-level statement (`try:`).
        branch = src[gate : src.find("\ntry:", gate)]
        assert "mcp.run(transport=\"stdio\")" in branch, (
            "disabled path must run an empty MCP server so the process stays "
            "alive — exiting triggers a -32000 reconnect loop (issue #1995)"
        )


class TestGetPageUsesContentProjection:
    """`agent_knowledge_get_page` must request the API's `content` projection.

    `detail=full` additionally returns `reflect_response`, the internal trace
    metadata used to build a page. Measured on real banks, reflect_response is
    70-95% of the response bytes while the synthesized `content` field is
    typically 1-2%. At realistic page sizes (200-280 KB at full) the response
    overflows the MCP host's per-tool-result token cap and the result spills to
    disk, where the agent cannot consume it inline as part of its startup
    knowledge load. The docstring promises only "the full synthesized content",
    so the request must pin `detail=content`.
    """

    def test_get_page_request_uses_detail_content(self):
        src = _read_mcp_server_source()
        # Check the URL pattern in the request line; comments may legitimately
        # mention detail=full when explaining the difference.
        assert "/mental-models/{page_id}?detail=content" in src, (
            "get_page must request detail=content; detail=full includes "
            "reflect_response which dwarfs the actual content"
        )
