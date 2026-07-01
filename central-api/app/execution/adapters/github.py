"""GitHub API execution adapter."""

import asyncio
import json
from typing import Any

import httpx

from app.execution.adapters.base import BaseAdapter, register_adapter


class GitHubAdapter(BaseAdapter):
    """Execute GitHub API operations."""

    async def execute(self, execution: dict) -> dict:
        """Execute GitHub API call.
        
        Args:
            execution: {action_type: 'github', target: 'repo/owner', params: {action, token, body}}
        
        Returns:
            {exit_code, output, result}
        """
        repo = execution.get("target", "owner/repo")
        params = execution.get("params", {})
        action = params.get("action", "list-issues")
        token = params.get("token", "")
        body = params.get("body", {})

        if not token:
            return {"exit_code": 1, "output": "", "result": {"error": "missing token"}}

        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        base_url = f"https://api.github.com/repos/{repo}"

        endpoints = {
            "list-issues": f"{base_url}/issues",
            "create-issue": (f"{base_url}/issues", "POST"),
            "list-pulls": f"{base_url}/pulls",
            "create-pull": (f"{base_url}/pulls", "POST"),
        }

        if action not in endpoints:
            return {"exit_code": 1, "output": "", "result": {"error": f"unknown action: {action}"}}

        endpoint = endpoints[action]
        method = "GET"
        url = endpoint

        if isinstance(endpoint, tuple):
            url, method = endpoint

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if method == "GET":
                    resp = await client.get(url, headers=headers)
                else:
                    resp = await client.request(method, url, headers=headers, json=body)

                return {
                    "exit_code": 0 if resp.status_code < 400 else 1,
                    "output": resp.text,
                    "result": {"status": resp.status_code, "data": resp.json() if resp.text else {}},
                }
        except Exception as e:
            return {"exit_code": 1, "output": "", "result": {"error": str(e)}}


register_adapter("github", GitHubAdapter)
