"""Tests for execution adapters."""

import pytest

from app.execution.adapters.base import get_adapter, get_registry
from app.execution.adapters.docker import DockerAdapter
from app.execution.adapters.github import GitHubAdapter
from app.execution.adapters.ssh import SSHAdapter


@pytest.mark.asyncio
async def test_docker_adapter_registration():
    """Test Docker adapter is registered."""
    adapter_class = get_adapter("docker")
    assert adapter_class is DockerAdapter


@pytest.mark.asyncio
async def test_ssh_adapter_registration():
    """Test SSH adapter is registered."""
    adapter_class = get_adapter("ssh")
    assert adapter_class is SSHAdapter


@pytest.mark.asyncio
async def test_github_adapter_registration():
    """Test GitHub adapter is registered."""
    adapter_class = get_adapter("github")
    assert adapter_class is GitHubAdapter


@pytest.mark.asyncio
async def test_docker_adapter_execute():
    """Test Docker adapter execution."""
    adapter = DockerAdapter()
    result = await adapter.execute({
        "action_type": "docker",
        "target": "alpine:latest",
        "params": {"cmd": "echo", "args": ["hello"]},
    })
    assert "exit_code" in result
    assert "output" in result
    assert "result" in result


@pytest.mark.asyncio
async def test_ssh_adapter_execute():
    """Test SSH adapter execution."""
    adapter = SSHAdapter()
    result = await adapter.execute({
        "action_type": "ssh",
        "target": "localhost",
        "params": {"cmd": "echo hello"},
    })
    assert "exit_code" in result
    assert "output" in result
    assert "result" in result


@pytest.mark.asyncio
async def test_github_adapter_no_token():
    """Test GitHub adapter fails without token."""
    adapter = GitHubAdapter()
    result = await adapter.execute({
        "action_type": "github",
        "target": "owner/repo",
        "params": {"action": "list-issues"},
    })
    assert result["exit_code"] == 1
    assert "error" in result["result"]


def test_registry_list():
    """Test registry lists all adapters."""
    registry = get_registry()
    adapters = registry.list()
    assert "docker" in adapters
    assert "ssh" in adapters
    assert "github" in adapters
