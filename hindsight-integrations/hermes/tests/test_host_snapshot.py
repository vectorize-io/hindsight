"""Opt-in contract probe with real Hermes imports in a separate process."""

import os
from pathlib import Path
import subprocess
import textwrap

import pytest


def test_real_host_delivers_detached_snapshot(tmp_path):
    host = os.environ.get("HERMES_TEST_SOURCE")
    interpreter = os.environ.get("HERMES_TEST_PYTHON")
    if not host or not interpreter:
        pytest.skip("Set HERMES_TEST_SOURCE and HERMES_TEST_PYTHON to an isolated Hermes snapshot build")
    provider = Path(__file__).resolve().parents[1]
    code = textwrap.dedent("""
        import importlib.util
        import json
        from pathlib import Path
        import sys
        sys.path.insert(0, sys.argv[1])
        from agent.memory_manager import MemoryManager
        root = Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location(
            'real_hindsight_probe', root / '__init__.py', submodule_search_locations=[str(root)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
        instance = plugin.HindsightMemoryProvider()
        instance._observation_scopes = []
        manager = MemoryManager()
        manager.add_provider(instance)
        host_jobs, writer_jobs, calls = [], [], []
        manager._submit_background = lambda fn, **kw: host_jobs.append(fn)
        instance._enqueue_retain = writer_jobs.append
        instance._resolve_retain_target = lambda doc: ('session', 'append')
        instance._retain_items = lambda items, **kw: calls.append((items, kw))
        plugin._event_timestamp = lambda: '2026-10-03T00:00:00Z'
        source = [
            {'role': 'user', 'content': 'q', 'timestamp': 1735812000},
            {'role': 'tool', 'content': 'private result'},
            {'role': 'assistant', 'content': 'a', 'timestamp': '2025-01-02T10:00:10Z'},
        ]
        manager.sync_all('q', 'a', session_id='session', messages=source)
        source[0]['timestamp'] = '2030-01-01T00:00:00Z'
        host_jobs[0]()
        plugin._event_timestamp = lambda: '2026-10-04T00:00:00Z'
        writer_jobs[0]()
        assert len(calls) == 1
        items, kwargs = calls[0]
        assert kwargs['document_id'] == 'session'
        assert [item['timestamp'] for item in items] == [
            '2025-01-02T10:00:00Z', '2025-01-02T10:00:10Z']
        assert [json.loads(item['content'])[0]['content'] for item in items] == ['User: q', 'Assistant: a']
        print('real Hermes snapshot -> provider retain boundary: passed')
    """)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), HERMES_RUNTIME_DIR=str(tmp_path / "runtime"))
    result = subprocess.run(
        [interpreter, "-c", code, host, str(provider)], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
