"""Regression tests for mcp_server.py, ported from the Claude Code plugin.

These guard three failures that already happened once on the Claude Code side
and would recur here, because `mcp_server.py` is a near-verbatim port:

  - `agent_knowledge_list_pages` requesting the API's default `detail=full`
    projection, which returns synthesized content + reflect_response for every
    page and can exceed the MCP client's 16 MB JSON-RPC message ceiling.
  - `agent_knowledge_get_page` doing the same, where reflect_response is
    70-95% of the bytes and the content field is 1-2%.
  - The `enableKnowledgeTools=false` path exiting instead of running an empty
    server, which the host reads as a crash and retries forever (-32000).

They assert on source text rather than behaviour because `mcp_server` imports
the `mcp` package, which is installed into the MCP server's own virtualenv by
run_mcp.sh and is deliberately not a test dependency of this integration.
"""

import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _read_mcp_server_source() -> str:
    return open(os.path.join(SCRIPTS_DIR, "mcp_server.py"), encoding="utf-8").read()


class TestDefaultBankUsesProjectCwd:
    """The MCP server must derive its default bank from the session's project cwd.

    `run_mcp.sh` cds into the plugin data dir before exec (so a project-local
    unreadable `.env` can't break FastMCP's settings loader), which destroys
    the cwd signal `derive_bank_id` needs. The launcher stashes it in
    HINDSIGHT_MCP_PROJECT_CWD first; without that, every project would share
    one bank named after the plugin data directory.
    """

    def test_default_bank_prefers_launcher_project_cwd(self):
        src = _read_mcp_server_source()
        assert 'os.environ.get("HINDSIGHT_MCP_PROJECT_CWD", os.getcwd())' in src
        assert '_hook_input = {"cwd": _project_cwd, "session_id": ""}' in src


class TestListPagesUsesMetadataProjection:
    def test_list_pages_request_uses_detail_metadata(self):
        src = _read_mcp_server_source()
        assert "/mental-models?detail=metadata" in src, (
            "list_pages must request detail=metadata; the API defaults to full"
        )
        list_pages_def = src.find("def agent_knowledge_list_pages")
        next_def = src.find("def agent_knowledge_get_page")
        assert list_pages_def > 0 and next_def > list_pages_def
        assert "detail=metadata" in src[list_pages_def:next_def]


class TestGetPageUsesContentProjection:
    def test_get_page_request_uses_detail_content(self):
        src = _read_mcp_server_source()
        assert "/mental-models/{_encode_page(page_id)}?detail=content" in src, (
            "get_page must request detail=content; detail=full includes "
            "reflect_response which dwarfs the actual content"
        )


class TestDisabledKnowledgeToolsKeepsServerAlive:
    """Devin CLI loads mcp_config.json entries unconditionally at session start.

    Same contract as Claude Code's `.mcp.json`: the host expects a live
    process. Exiting on the disabled path is read as a crashed server and
    surfaces a reconnect error on every prompt.
    """

    def test_disabled_branch_runs_empty_server_not_bare_exit(self):
        src = _read_mcp_server_source()
        gate = src.find('if not _config.get("enableKnowledgeTools")')
        assert gate > 0, "expected the enableKnowledgeTools startup gate"
        branch = src[gate : src.find("\ntry:", gate)]
        assert 'mcp.run(transport="stdio")' in branch, (
            "disabled path must run an empty MCP server so the process stays alive — exiting triggers a reconnect loop"
        )


class TestPageIdsAreUrlEncoded:
    def test_no_tool_interpolates_a_raw_page_id_into_a_path(self):
        """page_id is an agent-supplied tool argument, so it can contain anything.

        An unescaped `/` re-routes the request and `?`/`#` truncate the path,
        leaving such a page unreachable through every tool.
        """
        src = _read_mcp_server_source()

        assert "mental-models/{page_id}" not in src, "found a raw {page_id} in a URL path — wrap it in _encode_page()"
        assert "def _encode_page(" in src


class TestIngestFileDocIdKeepsTheExtension:
    """`README.md` and `README.txt` must not both become `readme`.

    `retain` replaces by `document_id`, so a collision means ingesting the
    second file silently overwrites the first. The derivation is a single
    expression in a module that cannot be imported here, so this pins it at
    source level — which is exactly the regression to guard: someone
    re-introducing the extension strip.
    """

    def test_doc_id_does_not_strip_the_extension(self):
        src = _read_mcp_server_source()
        assert 'doc_id = os.path.basename(file_path).lower().replace(" ", "-")' in src
        assert 'rsplit(".", 1)[0]' not in src, "stripping the extension collides README.md with README.txt"
