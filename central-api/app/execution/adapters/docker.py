"""Docker container execution adapter."""

import asyncio
import json
from typing import Any

from app.execution.adapters.base import BaseAdapter, register_adapter


class DockerAdapter(BaseAdapter):
    """Execute commands in Docker containers."""

    async def execute(self, execution: dict) -> dict:
        """Execute docker run command.
        
        Args:
            execution: {action_type: 'docker', target: 'image:tag', params: {cmd, args, env}}
        
        Returns:
            {exit_code, output, result}
        """
        image = execution.get("target", "alpine:latest")
        params = execution.get("params", {})
        cmd = params.get("cmd", "echo")
        args = params.get("args", [])
        env_vars = params.get("env", {})

        env_str = " ".join([f"-e {k}={v}" for k, v in env_vars.items()])
        full_cmd = f"docker run --rm {env_str} {image} {cmd} {' '.join(args)}"

        try:
            proc = await asyncio.create_subprocess_shell(
                full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {
                "exit_code": proc.returncode,
                "output": stdout.decode(),
                "result": {"success": proc.returncode == 0, "error": stderr.decode()},
            }
        except asyncio.TimeoutError:
            return {"exit_code": 124, "output": "", "result": {"error": "timeout"}}
        except Exception as e:
            return {"exit_code": 1, "output": "", "result": {"error": str(e)}}


register_adapter("docker", DockerAdapter)
