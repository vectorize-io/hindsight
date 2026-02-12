"""
Integration test for base path deployment using Docker Compose.

This test validates that Hindsight works correctly when deployed
behind a reverse proxy with path-based routing using the actual
Docker Compose examples from docker/compose-examples/.

Tests:
1. API with base path (direct, no proxy)
2. Full stack via docker-compose with Nginx reverse proxy
3. Regression: API without base path still works

Requirements:
- Docker and docker-compose installed
- No nginx required on host!
"""
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
API_PATH = REPO_ROOT / "hindsight-api"
COMPOSE_EXAMPLES_PATH = REPO_ROOT / "docker" / "docker-compose" / "nginx"

# Add hindsight-api to path for direct API testing
sys.path.insert(0, str(API_PATH))


def run_command(cmd: list[str], cwd: str | Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def check_docker_available() -> bool:
    """Check if Docker is available."""
    result = run_command(["docker", "info"])
    return result.returncode == 0


def get_docker_compose_command() -> list[str]:
    """Get the docker-compose command (modern or legacy)."""
    import shutil
    # Try modern docker compose plugin first
    if shutil.which("docker"):
        result = run_command(["docker", "compose", "version"])
        if result.returncode == 0:
            return ["docker", "compose"]
    # Fall back to legacy docker-compose
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("docker-compose not available")


def check_docker_compose_available() -> bool:
    """Check if docker-compose is available."""
    try:
        get_docker_compose_command()
        return True
    except RuntimeError:
        return False


class APIServer:
    """Helper to manage API server lifecycle for direct testing."""

    def __init__(self, base_path: str | None = None, port: int = 18888):
        self.base_path = base_path
        self.port = port
        self.process = None
        self.env = os.environ.copy()
        if base_path:
            self.env["HINDSIGHT_API_BASE_PATH"] = base_path

    def start(self):
        """Start the API server."""
        cmd = [
            "uv",
            "run",
            "--directory",
            str(API_PATH),
            "hindsight-api",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
        ]

        log_file = f"/tmp/hindsight-api-{self.port}.log"
        self.log_file = open(log_file, "w")

        self.process = subprocess.Popen(
            cmd, env=self.env, stdout=self.log_file, stderr=subprocess.STDOUT
        )

        # Wait for server to be ready
        base_url = f"http://localhost:{self.port}"
        if self.base_path:
            health_url = f"{base_url}{self.base_path}/health"
        else:
            health_url = f"{base_url}/health"

        for _ in range(60):  # 60 second timeout
            try:
                response = httpx.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(1)

        # Failed to start
        self.log_file.flush()
        with open(log_file) as f:
            print(f"API server failed to start. Logs:\n{f.read()}")
        raise RuntimeError(f"API server failed to start on port {self.port}")

    def stop(self):
        """Stop the API server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DockerComposeStack:
    """Helper to manage docker-compose stack lifecycle."""

    def __init__(self, compose_file: Path, project_name: str = "hindsight-test"):
        self.compose_file = compose_file
        self.project_name = project_name
        self.compose_cmd = get_docker_compose_command()
        self.env = os.environ.copy()
        # Set required env vars for docker-compose
        self.env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "test-key")
        self.env["HINDSIGHT_API_LLM_PROVIDER"] = os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "mock")
        self.env["HINDSIGHT_API_LLM_MODEL"] = os.environ.get("HINDSIGHT_API_LLM_MODEL", "mock-model")

    def start(self, timeout: int = 120):
        """Start the docker-compose stack."""
        print(f"Starting docker-compose stack: {self.compose_file.name}")

        # Pull images first (but don't fail if it doesn't work)
        run_command(
            self.compose_cmd + ["-f", str(self.compose_file), "-p", self.project_name, "pull"],
            cwd=self.compose_file.parent,
            env=self.env,
        )

        # Start services
        result = run_command(
            self.compose_cmd + ["-f", str(self.compose_file), "-p", self.project_name, "up", "-d", "--build"],
            cwd=self.compose_file.parent,
            env=self.env,
        )

        if result.returncode != 0:
            print(f"Failed to start docker-compose:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError("Failed to start docker-compose stack")

        # Wait for services to be healthy
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = run_command(
                self.compose_cmd + ["-f", str(self.compose_file), "-p", self.project_name, "ps", "--format", "json"],
                cwd=self.compose_file.parent,
                env=self.env,
            )

            if result.returncode == 0:
                # Give it a few more seconds to fully initialize
                time.sleep(5)
                return

            time.sleep(2)

        # Timeout - show logs and fail
        self.show_logs()
        raise RuntimeError(f"Docker compose stack failed to start within {timeout}s")

    def show_logs(self):
        """Show docker-compose logs."""
        result = run_command(
            self.compose_cmd + ["-f", str(self.compose_file), "-p", self.project_name, "logs", "--tail=100"],
            cwd=self.compose_file.parent,
            env=self.env,
        )
        print(f"Docker compose logs:\n{result.stdout}\n{result.stderr}")

    def stop(self):
        """Stop and remove the docker-compose stack."""
        print(f"Stopping docker-compose stack: {self.compose_file.name}")
        result = run_command(
            self.compose_cmd + ["-f", str(self.compose_file), "-p", self.project_name, "down", "-v"],
            cwd=self.compose_file.parent,
            env=self.env,
        )

        if result.returncode != 0:
            print(f"Warning: Failed to stop docker-compose:\n{result.stderr}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_api_without_base_path():
    """Regression test: API works at root path (default behavior)."""
    with APIServer(base_path=None, port=18888) as server:
        base_url = f"http://localhost:{server.port}"

        # Health check
        response = httpx.get(f"{base_url}/health")
        assert response.status_code == 200
        assert "status" in response.json()

        # API endpoints
        response = httpx.get(f"{base_url}/v1/default/banks")
        assert response.status_code == 200
        assert "banks" in response.json()

        # OpenAPI docs
        response = httpx.get(f"{base_url}/docs")
        assert response.status_code == 200


def test_api_with_base_path_direct():
    """Test API with base path configuration (direct, no proxy)."""
    base_path = "/hindsight"

    with APIServer(base_path=base_path, port=18889) as server:
        base_url = f"http://localhost:{server.port}"

        # Base path SHOULD work
        response = httpx.get(f"{base_url}{base_path}/health")
        assert response.status_code == 200
        assert "status" in response.json()

        # API endpoints with base path
        response = httpx.get(f"{base_url}{base_path}/v1/default/banks")
        assert response.status_code == 200
        assert "banks" in response.json()

        # OpenAPI docs with base path
        response = httpx.get(f"{base_url}{base_path}/docs")
        assert response.status_code == 200

        # OpenAPI schema should have correct server URL
        response = httpx.get(f"{base_url}{base_path}/openapi.json")
        assert response.status_code == 200
        openapi = response.json()
        assert "servers" in openapi
        assert openapi["servers"][0]["url"] == base_path


@pytest.mark.skipif(
    not check_docker_compose_available(),
    reason="docker-compose not available"
)
def test_reverse_proxy_simple_config():
    """
    Test reverse proxy deployment using docker-compose with Nginx.

    This creates a minimal test setup with:
    - API server running on HOST via uv (not Docker - faster, no image build needed!)
    - Nginx container that proxies to the host API

    This tests the actual reverse proxy scenario without requiring the
    heavy Hindsight Docker image.
    """
    base_path = "/hindsight"
    api_port = 18890

    # Start API on host with base path
    with APIServer(base_path=base_path, port=api_port):
        # Create test docker-compose file (nginx only)
        test_compose = COMPOSE_EXAMPLES_PATH / "test-reverse-proxy.yml"

        # Determine host address for nginx to reach host machine
        # host.docker.internal works on Docker Desktop (Mac/Windows)
        # On Linux, we use host network mode
        import platform
        if platform.system() == "Linux":
            network_mode = "host"
            api_host = "localhost"
            nginx_port = 18080  # With host mode, nginx must listen on 18080 directly
            port_mapping = ""  # No port mapping with host mode
        else:
            network_mode = "bridge"
            api_host = "host.docker.internal"
            nginx_port = 80  # With bridge mode, nginx listens on 80 and is mapped
            port_mapping = """    ports:
      - "18080:80"
"""

        compose_content = f"""version: '3.8'

services:
  nginx:
    image: nginx:alpine
{port_mapping}    volumes:
      - ./test-nginx.conf:/etc/nginx/nginx.conf:ro
    network_mode: {network_mode}
"""

        # Create test nginx config
        nginx_config = f"""events {{
    worker_connections 1024;
}}

http {{
    server {{
        listen {nginx_port};
        server_name localhost;

        location {base_path}/ {{
            proxy_pass http://{api_host}:{api_port};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
"""

        # Write test files
        test_compose.write_text(compose_content)
        test_nginx_conf = COMPOSE_EXAMPLES_PATH / "test-nginx.conf"
        test_nginx_conf.write_text(nginx_config)

        try:
            # Start nginx via docker-compose
            with DockerComposeStack(test_compose, project_name="hindsight-base-path-test"):
                proxy_url = "http://localhost:18080"

                # Give nginx a moment to start
                time.sleep(2)

                # Test through nginx proxy
                response = httpx.get(f"{proxy_url}{base_path}/health", timeout=10.0)
                assert response.status_code == 200
                assert "status" in response.json()

                # API endpoints through proxy
                response = httpx.get(f"{proxy_url}{base_path}/v1/default/banks", timeout=10.0)
                assert response.status_code == 200
                assert "banks" in response.json()

                # OpenAPI docs through proxy
                response = httpx.get(f"{proxy_url}{base_path}/docs", timeout=10.0)
                assert response.status_code == 200

        finally:
            # Cleanup test files
            test_compose.unlink(missing_ok=True)
            test_nginx_conf.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_full_workflow_with_base_path():
    """Test full retain/recall workflow through base path."""
    base_path = "/hindsight"
    bank_id = "integration_test_bank"

    with APIServer(base_path=base_path, port=18891) as server:
        base_url = f"http://localhost:{server.port}{base_path}"

        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            # 1. Get bank profile (creates if needed)
            response = await client.get(f"/v1/default/banks/{bank_id}/profile")
            assert response.status_code == 200

            # 2. Store a memory
            response = await client.post(
                f"/v1/default/banks/{bank_id}/memories",
                json={
                    "items": [
                        {
                            "content": "Hindsight supports deployment under custom base paths for reverse proxy scenarios.",
                            "context": "integration test"
                        }
                    ]
                },
            )
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True

            # 3. Recall the memory
            response = await client.post(
                f"/v1/default/banks/{bank_id}/memories/recall",
                json={"query": "base path deployment"}
            )
            assert response.status_code == 200
            recall_result = response.json()
            assert "results" in recall_result
            # Should find our memory
            assert len(recall_result["results"]) > 0


if __name__ == "__main__":
    """Run tests directly with python."""
    import sys

    print("=" * 70)
    print("Hindsight Base Path Integration Tests")
    print("=" * 70)
    print()

    all_passed = True

    # Test 1: Without base path
    print("Test 1: API without base path (regression test)")
    print("-" * 70)
    try:
        test_api_without_base_path()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        all_passed = False

    # Test 2: With base path (direct)
    print("Test 2: API with base path (direct, no proxy)")
    print("-" * 70)
    try:
        test_api_with_base_path_direct()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        all_passed = False

    # Test 3: Docker compose reverse proxy
    print("Test 3: Reverse proxy via docker-compose")
    print("-" * 70)
    if check_docker_available() and check_docker_compose_available():
        try:
            test_reverse_proxy_simple_config()
            print("✅ PASSED\n")
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False
    else:
        print("⚠️  SKIPPED: Docker or docker-compose not available\n")

    # Test 4: Full workflow
    print("Test 4: Full retain/recall workflow with base path")
    print("-" * 70)
    try:
        asyncio.run(test_full_workflow_with_base_path())
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        all_passed = False

    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
