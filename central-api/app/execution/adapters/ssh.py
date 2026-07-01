"""SSH command execution adapter."""

import asyncio
from typing import Any

from app.execution.adapters.base import BaseAdapter, register_adapter


class SSHAdapter(BaseAdapter):
    """Execute commands over SSH."""

    async def execute(self, execution: dict) -> dict:
        """Execute ssh command.
        
        Args:
            execution: {action_type: 'ssh', target: 'user@host:22', params: {cmd, key_path}}
        
        Returns:
            {exit_code, output, result}
        """
        target = execution.get("target", "localhost")
        params = execution.get("params", {})
        cmd = params.get("cmd", "echo")
        key_path = params.get("key_path", "~/.ssh/id_rsa")

        ssh_cmd = f"ssh -i {key_path} {target} '{cmd}'"

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
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


register_adapter("ssh", SSHAdapter)
