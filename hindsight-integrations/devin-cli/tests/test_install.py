"""Tests for scripts/install.py and scripts/uninstall.py.

`install.py` is the largest divergence from the Claude Code plugin: because
Devin CLI has no `${DEVIN_PLUGIN_ROOT}` substitution, this script writes
absolute paths straight into the user's `~/.config/devin/` files. That makes it
the one component that can corrupt config it does not own, so the invariants
below are about *not* damaging the user's setup:

  - foreign hooks and MCP servers survive install and uninstall
  - re-running install never accumulates duplicate entries
  - unparseable config is refused, not overwritten
  - an upgraded (relocated) plugin replaces its old entries instead of
    leaving a second, dead copy behind

The last one is not hypothetical: Devin CLI caches plugins under
`.../plugins/cache/<hash>/hindsight-memory/<version>/`, so the install path
changes on every version bump.
"""

import json
import os
import shlex
import subprocess

import pytest

import install
import uninstall

HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "Stop", "SessionEnd")

FOREIGN_HOOK = {"hooks": [{"type": "command", "command": "/opt/other-tool/lint.sh"}]}


@pytest.fixture()
def devin_config(tmp_path, monkeypatch):
    """Point install/uninstall at throwaway config files and a plugin root."""
    config_dir = tmp_path / "config"
    scripts_dir = tmp_path / "plugin" / "scripts"
    scripts_dir.mkdir(parents=True)
    config_path = config_dir / "config.json"
    mcp_path = config_dir / "mcp_config.json"

    for module in (install, uninstall):
        monkeypatch.setattr(module, "CONFIG_PATH", str(config_path), raising=False)
        monkeypatch.setattr(module, "MCP_CONFIG_PATH", str(mcp_path), raising=False)
    # Only install.py has a SCRIPTS_DIR — it is what `_script()` builds commands
    # from. uninstall.py recognises entries via `_OURS_RE`, which is path-agnostic
    # by design, so there is nothing install-path-shaped to patch on that side.
    monkeypatch.setattr(install, "SCRIPTS_DIR", str(scripts_dir))

    class Paths:
        def __init__(self):
            self.config = config_path
            self.mcp = mcp_path
            self.scripts = scripts_dir

        def read_config(self):
            return json.loads(self.config.read_text())

        def read_mcp(self):
            return json.loads(self.mcp.read_text())

    return Paths()


def _all_commands(hooks: dict) -> list:
    return [hook["command"] for entries in hooks.values() for entry in entries for hook in entry["hooks"]]


class TestInstallHooks:
    def test_registers_every_lifecycle_event(self, devin_config):
        install.install_hooks()

        hooks = devin_config.read_config()["hooks"]
        assert set(hooks) == set(HOOK_EVENTS)

    def test_commands_use_absolute_paths(self, devin_config):
        """Relative paths resolve against the project cwd at hook-run time.

        That is the whole reason this script exists instead of a plugin-shipped
        hooks.json, so a relative command here would silently reintroduce the
        bug the README documents.
        """
        install.install_hooks()

        for command in _all_commands(devin_config.read_config()["hooks"]):
            # shlex.split so this holds whichever quoting form shlex.quote() chose
            # for the install path: `python3 <path> || python <path>`.
            script_path = shlex.split(command)[1]
            assert os.path.isabs(script_path), f"hook command is not absolute: {command}"

    def test_each_event_maps_to_its_own_script(self, devin_config):
        install.install_hooks()

        hooks = devin_config.read_config()["hooks"]
        expected = {
            "SessionStart": "session_start.py",
            "UserPromptSubmit": "recall.py",
            "Stop": "retain.py",
            "SessionEnd": "session_end.py",
        }
        for event, script in expected.items():
            assert script in hooks[event][0]["hooks"][0]["command"]

    def test_every_hook_has_a_timeout(self, devin_config):
        """An unbounded hook can hang the session on a wedged daemon."""
        install.install_hooks()

        hooks = devin_config.read_config()["hooks"]
        for entries in hooks.values():
            for entry in entries:
                for hook in entry["hooks"]:
                    assert isinstance(hook.get("timeout"), int)
                    assert hook["timeout"] > 0

    def test_is_idempotent(self, devin_config):
        install.install_hooks()
        first = devin_config.read_config()
        install.install_hooks()
        install.install_hooks()

        assert devin_config.read_config() == first

    def test_preserves_foreign_hooks_on_the_same_event(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [FOREIGN_HOOK]}}))

        install.install_hooks()

        stop_commands = [
            hook["command"] for entry in devin_config.read_config()["hooks"]["Stop"] for hook in entry["hooks"]
        ]
        assert "/opt/other-tool/lint.sh" in stop_commands
        assert any("retain.py" in c for c in stop_commands)

    def test_preserves_unrelated_top_level_config_keys(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"model": "devin-2", "hooks": {}}))

        install.install_hooks()

        assert devin_config.read_config()["model"] == "devin-2"

    def test_refuses_to_overwrite_unparseable_config(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text("{ not json")

        with pytest.raises(SystemExit):
            install.install_hooks()

        assert devin_config.config.read_text() == "{ not json"


class TestInstallMcpServer:
    def test_registers_the_hindsight_server(self, devin_config):
        install.install_mcp_server()

        servers = devin_config.read_mcp()["mcpServers"]
        assert "hindsight" in servers
        assert servers["hindsight"]["command"] == "bash"
        assert servers["hindsight"]["args"][0].endswith("run_mcp.sh")
        assert os.path.isabs(servers["hindsight"]["args"][0])

    def test_preserves_foreign_mcp_servers(self, devin_config):
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": {"other": {"command": "node", "args": ["x.js"]}}}))

        install.install_mcp_server()

        servers = devin_config.read_mcp()["mcpServers"]
        assert servers["other"] == {"command": "node", "args": ["x.js"]}
        assert "hindsight" in servers

    def test_is_idempotent(self, devin_config):
        install.install_mcp_server()
        first = devin_config.read_mcp()
        install.install_mcp_server()

        assert devin_config.read_mcp() == first

    def test_refuses_to_replace_a_foreign_hindsight_server(self, devin_config, capsys):
        """ "hindsight" is a name a user may have configured themselves.

        A warning is not consent: it scrolls past, and by the time it is read
        the value it names is already gone. Declining leaves them a choice.
        """
        foreign = {"command": "node", "args": ["my-own-hindsight.js"]}
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": {"hindsight": foreign}}))

        assert install.install_mcp_server() is False

        err = capsys.readouterr().err
        assert "my-own-hindsight.js" in err and "--force" in err
        assert devin_config.read_mcp()["mcpServers"]["hindsight"] == foreign, (
            "replaced a user's own 'hindsight' MCP server without being asked to"
        )

    def test_force_replaces_a_foreign_hindsight_server(self, devin_config):
        foreign = {"command": "node", "args": ["my-own-hindsight.js"]}
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": {"hindsight": foreign}}))

        assert install.install_mcp_server(force=True) is True

        assert devin_config.read_mcp()["mcpServers"]["hindsight"]["args"][0].endswith("run_mcp.sh")

    def test_replacing_our_own_entry_is_not_announced(self, devin_config, capsys):
        install.install_mcp_server()
        capsys.readouterr()

        install.install_mcp_server()

        assert capsys.readouterr().err == "", "an ordinary upgrade warned about its own entry"

    def test_a_non_object_mcpservers_is_left_alone(self, devin_config, capsys):
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": ["not", "an", "object"]}))

        install.install_mcp_server()

        assert "non-object" in capsys.readouterr().err
        assert devin_config.read_mcp()["mcpServers"] == ["not", "an", "object"]


class TestMalformedHookConfigDoesNotAbortTheInstaller:
    """`config.json` is user-authored JSON: every level can be the wrong shape.

    Each of these used to end the installer on an unhandled traceback, which
    leaves the plugin uninstallable for a config the user can neither diagnose
    from the output nor be told how to fix.
    """

    @pytest.mark.parametrize("hooks", [None, "not-an-object", ["a", "list"], 7])
    def test_a_non_object_hooks_key_is_replaced_not_fatal(self, devin_config, hooks, capsys):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"model": "devin-2", "hooks": hooks}))

        install.install_hooks()

        written = devin_config.read_config()
        assert set(written["hooks"]) == set(HOOK_EVENTS)
        assert written["model"] == "devin-2", "an unrelated key was lost"
        if hooks is not None:
            assert "non-object" in capsys.readouterr().err

    @pytest.mark.parametrize("entries", [None, "nope", 7, {"not": "a list"}])
    def test_a_non_list_event_value_is_replaced_not_fatal(self, devin_config, entries):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"SessionStart": entries}}))

        install.install_hooks()

        session_start = devin_config.read_config()["hooks"]["SessionStart"]
        assert len(session_start) == 1
        assert "session_start.py" in session_start[0]["hooks"][0]["command"]

    @pytest.mark.parametrize("entry", [None, "nope", 7, {"hooks": "not-a-list"}, {"hooks": [None, 7]}])
    def test_a_malformed_entry_is_preserved_rather_than_claimed(self, devin_config, entry):
        """An entry we cannot parse is not ours, so it survives — the safe direction."""
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [entry]}}))

        install.install_hooks()

        stop = devin_config.read_config()["hooks"]["Stop"]
        assert entry in stop, "a hook entry this plugin did not write was dropped"
        assert any(install._is_ours(e) for e in stop)

    def test_uninstall_leaves_a_foreign_hindsight_server_in_place(self, devin_config):
        foreign = {"command": "node", "args": ["my-own-hindsight.js"]}
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": {"hindsight": foreign}}))

        uninstall.uninstall_mcp_server()

        assert devin_config.read_mcp()["mcpServers"]["hindsight"] == foreign, (
            "uninstall deleted an MCP server this plugin never wrote"
        )


class TestConfigWritesAreStagedPerProcess:
    def test_write_json_does_not_use_a_shared_temp_name(self, devin_config, tmp_path):
        """A fixed `<path>.tmp` lets one installer replace the file another is filling."""
        target = tmp_path / "cfg" / "config.json"
        staged = []
        real_replace = install.os.replace

        def _capture(src, dst):
            staged.append(src)
            return real_replace(src, dst)

        install.os.replace = _capture
        try:
            install._write_json(str(target), {"a": 1})
            install._write_json(str(target), {"a": 2})
        finally:
            install.os.replace = real_replace

        assert len(set(staged)) == 2, f"both writes staged at the same path: {staged}"
        assert json.loads(target.read_text()) == {"a": 2}
        assert not (tmp_path / "cfg" / "config.json.tmp").exists()


class TestReinstallAfterPluginMove:
    """Devin CLI's plugin cache path contains the version, so upgrades relocate.

    Re-running install from the new location must not leave the previous
    version's entries behind: those paths no longer exist, so every hook event
    would spawn a failing process for the life of the install.
    """

    def _install_from(self, monkeypatch, scripts_dir: str) -> None:
        monkeypatch.setattr(install, "SCRIPTS_DIR", scripts_dir, raising=False)
        install.install_hooks()

    def test_old_version_hooks_are_replaced_not_duplicated(self, devin_config, tmp_path, monkeypatch):
        old_scripts = tmp_path / "cache" / "hindsight-memory" / "0.1.0" / "scripts"
        new_scripts = tmp_path / "cache" / "hindsight-memory" / "0.2.0" / "scripts"
        old_scripts.mkdir(parents=True)
        new_scripts.mkdir(parents=True)

        self._install_from(monkeypatch, str(old_scripts))
        self._install_from(monkeypatch, str(new_scripts))

        commands = _all_commands(devin_config.read_config()["hooks"])
        stale = [c for c in commands if str(old_scripts) in c]
        assert not stale, (
            f"upgrading left {len(stale)} hook command(s) pointing at the "
            f"removed 0.1.0 install; every event would spawn a failing process"
        )
        assert len(commands) == len(HOOK_EVENTS), (
            "one command per event — the python3/python fallback lives inside "
            "a single command string, so a count above this means duplicates"
        )

    def test_foreign_hooks_still_survive_a_reinstall_after_move(self, devin_config, tmp_path, monkeypatch):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [FOREIGN_HOOK]}}))
        old_scripts = tmp_path / "v1" / "scripts"
        new_scripts = tmp_path / "v2" / "scripts"
        old_scripts.mkdir(parents=True)
        new_scripts.mkdir(parents=True)

        self._install_from(monkeypatch, str(old_scripts))
        self._install_from(monkeypatch, str(new_scripts))

        commands = _all_commands(devin_config.read_config()["hooks"])
        assert "/opt/other-tool/lint.sh" in commands

    def test_shell_metacharacters_in_the_install_path_are_neutralised(self, devin_config, tmp_path, monkeypatch):
        """The plugin cache path is Devin CLI's to choose, not ours.

        A `$` or backtick in it would be expanded by the shell that runs the hook,
        so every lifecycle event would invoke a mangled path and fail for the life
        of the install.
        """
        scripts = tmp_path / "cache" / "a$b`c d" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "recall.py").write_text("import sys\nprint(sys.argv[0])\n")

        self._install_from(monkeypatch, str(scripts))

        # Run it through a real shell: nothing short of that proves the quoting,
        # since shlex.split() parses the string without performing expansion.
        command = devin_config.read_config()["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(scripts / "recall.py")

    def test_a_quoted_install_path_is_still_recognised_as_ours(self, devin_config, tmp_path, monkeypatch):
        """`_OURS_RE` matches the literal `_hook_command()` writes.

        Quoting changes that literal, so a metacharacter path must not make the
        installer stop recognising its own entries — that would duplicate every
        hook on re-run and leave uninstall unable to clean up.
        """
        scripts = tmp_path / "cache" / "a$b c" / "scripts"
        scripts.mkdir(parents=True)

        self._install_from(monkeypatch, str(scripts))
        self._install_from(monkeypatch, str(scripts))

        commands = _all_commands(devin_config.read_config()["hooks"])
        assert len(commands) == len(HOOK_EVENTS)

        uninstall.uninstall_hooks()
        assert devin_config.read_config().get("hooks") == {}


class TestUninstall:
    def test_removes_all_hindsight_hooks(self, devin_config):
        install.install_hooks()

        uninstall.uninstall_hooks()

        assert devin_config.read_config().get("hooks") == {}

    def test_removes_the_mcp_server(self, devin_config):
        install.install_mcp_server()

        uninstall.uninstall_mcp_server()

        assert "hindsight" not in devin_config.read_mcp()["mcpServers"]

    def test_leaves_foreign_hooks_in_place(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [FOREIGN_HOOK]}}))
        install.install_hooks()

        uninstall.uninstall_hooks()

        hooks = devin_config.read_config()["hooks"]
        assert hooks == {"Stop": [FOREIGN_HOOK]}

    def test_leaves_foreign_mcp_servers_in_place(self, devin_config):
        devin_config.mcp.parent.mkdir(parents=True, exist_ok=True)
        devin_config.mcp.write_text(json.dumps({"mcpServers": {"other": {"command": "node", "args": ["x.js"]}}}))
        install.install_mcp_server()

        uninstall.uninstall_mcp_server()

        assert devin_config.read_mcp()["mcpServers"] == {"other": {"command": "node", "args": ["x.js"]}}

    def test_removes_hooks_left_by_an_older_install_path(self, devin_config, tmp_path, monkeypatch):
        """Uninstalling after an upgrade must clean up, not strand old entries."""
        old_scripts = tmp_path / "v1" / "scripts"
        new_scripts = tmp_path / "v2" / "scripts"
        old_scripts.mkdir(parents=True)
        new_scripts.mkdir(parents=True)
        monkeypatch.setattr(install, "SCRIPTS_DIR", str(old_scripts), raising=False)
        install.install_hooks()
        monkeypatch.setattr(install, "SCRIPTS_DIR", str(new_scripts))

        uninstall.uninstall_hooks()

        assert devin_config.read_config().get("hooks") == {}

    def test_is_safe_when_nothing_is_installed(self, devin_config):
        uninstall.uninstall_hooks()
        uninstall.uninstall_mcp_server()  # must not raise


class TestForeignHooksSharingAnEntryWithOursSurvive:
    """A hooks.json entry is a *list* of commands under one matcher.

    Nothing stops a user from putting this plugin's hook in the same entry as
    their own, and both install and uninstall used to key ownership on the whole
    entry: one matching command made the entire entry ours to delete. Reinstall
    or uninstall then took the user's unrelated hooks with it — silently, and
    with no copy left anywhere to restore from.
    """

    def _mixed_entry(self, devin_config, script):
        """An entry holding this plugin's hook next to a third-party one."""
        return {
            "matcher": "*",
            "hooks": [
                {"type": "command", "command": install._hook_command(script)},
                {"type": "command", "command": "/opt/other-tool/lint.sh"},
            ],
        }

    def _commands(self, entries):
        return [h.get("command") for e in entries if isinstance(e, dict) for h in e.get("hooks", [])]

    def test_reinstall_keeps_the_third_party_hook_in_a_shared_entry(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [self._mixed_entry(devin_config, "retain.py")]}}))

        install.install_hooks()

        assert "/opt/other-tool/lint.sh" in self._commands(devin_config.read_config()["hooks"]["Stop"])

    def test_reinstall_still_replaces_our_half_of_a_shared_entry(self, devin_config):
        """Preserving theirs must not turn into duplicating ours."""
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [self._mixed_entry(devin_config, "retain.py")]}}))

        install.install_hooks()

        ours = [c for c in self._commands(devin_config.read_config()["hooks"]["Stop"]) if install._OURS_RE.match(c)]
        assert len(ours) == 1, "reinstall duplicated this plugin's hook instead of replacing it"

    def test_uninstall_keeps_the_third_party_hook_in_a_shared_entry(self, devin_config):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [self._mixed_entry(devin_config, "retain.py")]}}))

        uninstall.uninstall_hooks()

        remaining = devin_config.read_config()["hooks"]["Stop"]
        assert self._commands(remaining) == ["/opt/other-tool/lint.sh"]

    def test_uninstall_writes_the_file_when_only_part_of_an_entry_is_removed(self, devin_config):
        """The entry count is unchanged, so a length check reads this as a no-op.

        That is the bug: the file was never written, and the hook this plugin
        registered went on running after the user uninstalled it.
        """
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [self._mixed_entry(devin_config, "retain.py")]}}))

        uninstall.uninstall_hooks()

        left = self._commands(devin_config.read_config()["hooks"]["Stop"])
        assert not any(install._OURS_RE.match(c) for c in left), "an uninstalled hook is still registered"

    def test_an_entry_that_is_only_ours_is_removed_whole(self, devin_config):
        """The shared-entry fix must not start leaving empty husks behind."""
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(
            json.dumps({"hooks": {"Stop": [{"hooks": [{"command": install._hook_command("retain.py")}]}]}})
        )

        uninstall.uninstall_hooks()

        assert "Stop" not in devin_config.read_config().get("hooks", {})

    def test_a_literal_null_entry_is_not_mistaken_for_the_drop_sentinel(self, devin_config):
        """`None` is a hook entry a user's config can contain.

        _strip_our_hooks() returns unrecognised entries unchanged and a sentinel
        for "delete this one". If that sentinel were None, an entry that *is*
        None would be deleted by the very filter meant to preserve it.
        """
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": [None]}}))

        install.install_hooks()

        assert None in devin_config.read_config()["hooks"]["Stop"]


class TestMalformedHookConfigDoesNotAbortTheUninstaller:
    """uninstall.py guards the same shapes install.py does.

    Fixing one side and not the other is how a contract drifts apart: a config
    the installer now handles would still take the uninstaller down, leaving the
    user with hooks they cannot remove.
    """

    @pytest.mark.parametrize("hooks", [None, "nope", 7, []])
    def test_a_non_object_hooks_key_is_left_alone(self, devin_config, hooks):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": hooks}))

        uninstall.uninstall_hooks()

        assert devin_config.read_config()["hooks"] == hooks

    @pytest.mark.parametrize("entries", [None, "nope", 7, {"a": 1}])
    def test_a_non_list_event_value_is_left_alone(self, devin_config, entries):
        devin_config.config.parent.mkdir(parents=True, exist_ok=True)
        devin_config.config.write_text(json.dumps({"hooks": {"Stop": entries}}))

        uninstall.uninstall_hooks()

        assert devin_config.read_config()["hooks"]["Stop"] == entries
