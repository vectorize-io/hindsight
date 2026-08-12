import { describe, expect, it } from "vitest";
import {
  createInstallerUi,
  fitSelectRow,
  selectKeyAction,
  splitKeys,
  type InstallerUi,
} from "./install-ui";

/** Renderer under a fixed clock and captured output; colors off unless a test opts in. */
function makeUi(command?: string, colors = false): { ui: InstallerUi; lines: string[] } {
  const lines: string[] = [];
  let tick = 0;
  const ui = createInstallerUi({
    home: "/home/u",
    command,
    version: "1.2.3",
    harnessNames: ["claude-code", "codex"],
    auxNames: ["server"],
    colors,
    write: (l) => lines.push(l),
    now: () => (tick += 250),
  });
  return { ui, lines };
}

describe("installer UI renderer", () => {
  it("frames the run, groups messages by harness prefix, and picks symbols from phrasing", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log("detected: claude-code, codex");
    ui.log("claude-code: hooks merged into /home/u/.claude/settings.json");
    // A message without a harness prefix must land inside the currently open group.
    ui.log("skill installed at /home/u/.claude/skills/hindsight-coding-agent");
    ui.log(
      "claude-code: could not run `claude mcp add` — register the tools manually:\n" +
        '  claude mcp add --scope user hindsight -- node "/opt/dist/mcp-server.js"'
    );
    ui.log("codex: hooks merged into /home/u/.codex/hooks.json");
    ui.outro(0);

    const out = lines.join("\n");
    expect(out).toContain("┌  Hindsight coding agents  v1.2.3");
    expect(out).toContain("○ detected: claude-code, codex");
    expect(out).toContain("◇  claude-code");
    expect(out).toContain("◇  codex");
    // $HOME shortened to ~ everywhere.
    expect(out).toContain("✓ hooks merged into ~/.claude/settings.json");
    expect(out).not.toContain("/home/u/");
    // Manual-step phrasing renders as a warning, continuation line stays on the rail.
    expect(out).toContain("▲ could not run `claude mcp add`");
    expect(out).toContain("│    claude mcp add --scope user hindsight");
    // The prefixless skill line sits between the two group headers, i.e. inside claude-code's group.
    const claudeAt = lines.findIndex((l) => l.includes("◇  claude-code"));
    const skillAt = lines.findIndex((l) => l.includes("skill installed"));
    const codexAt = lines.findIndex((l) => l.includes("◇  codex"));
    expect(claudeAt).toBeLessThan(skillAt);
    expect(skillAt).toBeLessThan(codexAt);
    expect(out).toMatch(/└ {2}✓ Installed 2 agents in \d+\.\ds/);
    expect(out).toContain("~/.hindsight/coding-agent.json");
  });

  it("renders the server-setup step as a group but does not count it as an agent", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log(
      "\nWhere should memory live?\n  1) Hindsight Cloud\n  2) Self-hosted\n  3) Local daemon\n"
    );
    ui.log("server: daemon (/home/u/.hindsight/coding-agent.json)");
    ui.log("codex: hooks merged into /home/u/.codex/hooks.json");
    ui.outro(0);
    const out = lines.join("\n");
    // The mode question is an info line with quiet option detail, not a completed action.
    expect(out).toContain("○ Where should memory live?");
    expect(out).toContain("│    1) Hindsight Cloud");
    expect(out).toContain("◇  server");
    expect(out).toContain("✓ daemon (~/.hindsight/coding-agent.json)");
    expect(out).toMatch(/Installed 1 agent in/); // codex only — server is a step, not an agent
  });

  it("adopts run()'s own emoji severity markers instead of double-marking", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log(
      "\n❌ codex: `node:sqlite` is unavailable in the node on PATH.\n   Upgrade to Node 22.5."
    );
    ui.log("⚠️  `uv` is not on PATH. The daemon is fetched and run with it.");
    ui.outro(1);
    const out = lines.join("\n");
    // The ❌ line still opens its harness group once the marker is stripped.
    expect(out).toContain("◇  codex");
    expect(out).toContain("✖ `node:sqlite` is unavailable");
    expect(out).not.toContain("❌");
    expect(out).toContain("▲ `uv` is not on PATH");
    expect(out).not.toContain("⚠️");
  });

  it("never doubles the rail spacer between intro, groups, and outro", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log("claude-code: hooks merged into /home/u/.claude/settings.json");
    ui.outro(0);
    for (let i = 1; i < lines.length; i++) {
      if (lines[i] === "│") expect(lines[i - 1]).not.toBe("│");
    }
  });

  it("counts a single agent without the plural s", () => {
    const { ui, lines } = makeUi("uninstall");
    ui.intro();
    ui.log("codex: hooks + MCP section + skill removed");
    ui.outro(0);
    expect(lines.join("\n")).toMatch(/✓ Uninstalled 1 agent in \d+\.\ds/);
  });

  it("renders guard-path failures as an error and an aborted outro when nothing was wired", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log('unknown harness "nope" — expected "all" or one of: claude-code, codex');
    ui.outro(1);
    const out = lines.join("\n");
    expect(out).toContain('✖ unknown harness "nope"');
    expect(out).toContain("✖ Aborted — nothing was changed.");
    expect(out).not.toContain("Installed");
  });

  it("reports a partial failure honestly once some agents were already wired", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log("codex: hooks merged into /home/u/.codex/hooks.json");
    ui.log("\n❌ not installed: devin-cli — this machine can't run it (see above).");
    ui.outro(1);
    const out = lines.join("\n");
    expect(out).toContain("✖ Completed with errors — see above.");
    expect(out).not.toContain("nothing was changed");
  });

  it("closes a bare usage run with just the frame — no success banner", () => {
    const { ui, lines } = makeUi(undefined);
    ui.intro();
    ui.log(
      "usage: hindsight-coding-agents <install|uninstall> <all|harness...>\n  all  every agent"
    );
    ui.outro(0);
    const out = lines.join("\n");
    expect(out).toContain("○ usage:");
    expect(out).not.toContain("Installed");
    expect(lines.at(-2)).toBe("└");
  });

  it("renders an indented follow-up message as detail lines, not a fresh ✓ item", () => {
    const { ui, lines } = makeUi("install");
    ui.intro();
    ui.log("claude-code: conversation import did not finish — re-run it any time with:");
    ui.log('  node "/opt/dist/deepen.js" --repo "/home/u/w" --conversations "/tmp/c.json"');
    const detail = lines.at(-1)!;
    expect(detail).toContain('node "/opt/dist/deepen.js"');
    expect(detail).not.toContain("✓");
    expect(detail.startsWith("│    ")).toBe(true);
  });

  it("names the overridden config path in the outro when one is set", () => {
    const lines: string[] = [];
    const ui = createInstallerUi({
      home: "/home/u",
      command: "install",
      harnessNames: ["codex"],
      configPath: "/tmp/config.json",
      colors: false,
      write: (l) => lines.push(l),
      now: (() => {
        let t = 0;
        return () => (t += 100);
      })(),
    });
    ui.intro();
    ui.log("codex: hooks merged into /home/u/.codex/hooks.json");
    ui.outro(0);
    expect(lines.join("\n")).toContain("settings live in /tmp/config.json.");
    expect(lines.join("\n")).not.toContain("~/.hindsight");
  });

  it("styles readline prompts onto the rail", () => {
    const { ui } = makeUi("install");
    expect(ui.prompt("Choose [1-3] (default 1): ")).toBe("│  Choose [1-3] (default 1): ");
  });

  it("select keys: arrows wrap, vi keys move, Enter submits, digits shortcut, Esc/q/Ctrl+C cancel", () => {
    expect(selectKeyAction("\x1b[B", 0, 3)).toEqual({ kind: "move", index: 1 });
    expect(selectKeyAction("\x1b[B", 2, 3)).toEqual({ kind: "move", index: 0 }); // wraps down
    expect(selectKeyAction("\x1b[A", 0, 3)).toEqual({ kind: "move", index: 2 }); // wraps up
    expect(selectKeyAction("j", 0, 3)).toEqual({ kind: "move", index: 1 });
    expect(selectKeyAction("k", 1, 3)).toEqual({ kind: "move", index: 0 });
    expect(selectKeyAction("\r", 1, 3)).toEqual({ kind: "submit", index: 1 });
    expect(selectKeyAction("3", 0, 3)).toEqual({ kind: "submit", index: 2 });
    expect(selectKeyAction("9", 0, 3)).toEqual({ kind: "none", index: 0 }); // out of range
    expect(selectKeyAction("\x1b", 1, 3)).toEqual({ kind: "cancel", index: 1 });
    expect(selectKeyAction("q", 1, 3)).toEqual({ kind: "cancel", index: 1 });
    expect(selectKeyAction("\x03", 1, 3)).toEqual({ kind: "cancel", index: 1 });
    expect(selectKeyAction("x", 1, 3)).toEqual({ kind: "none", index: 1 });
  });

  it("splits a burst of keys from one read into individual tokens", () => {
    // Key repeat / paste: ↓ then Enter arriving in a single read must act as two keys.
    expect(splitKeys("\x1b[B\r")).toEqual(["\x1b[B", "\r"]);
    expect(splitKeys("\x1b[A\x1b[A\x1b[B")).toEqual(["\x1b[A", "\x1b[A", "\x1b[B"]);
    expect(splitKeys("2\r")).toEqual(["2", "\r"]);
    expect(splitKeys("\x1b")).toEqual(["\x1b"]); // a lone Esc stays a cancel key
    expect(splitKeys("jk")).toEqual(["j", "k"]);
    expect(splitKeys("\x1b[")).toEqual(["\x1b["]); // truncated CSI at chunk end doesn't crash
  });

  it("fits select rows to the terminal width so a row can never wrap", () => {
    const label = "Local daemon (on-device)";
    const hint = "runs hindsight-embed here; no account, needs uv + an LLM key";
    // Wide terminal: everything fits untouched.
    const wide = fitSelectRow(label, hint, 120);
    expect(wide).toEqual({ label, hint: ` — ${hint}` });
    // Narrow terminal: the hint is truncated with an ellipsis, the label survives.
    const narrow = fitSelectRow(label, hint, 60);
    expect(narrow.label).toBe(label);
    expect(narrow.hint.endsWith("…")).toBe(true);
    expect(narrow.label.length + narrow.hint.length).toBeLessThanOrEqual(60 - 6);
    // Tiny terminal: even the label is cut, and the hint is dropped entirely.
    const tiny = fitSelectRow(label, hint, 20);
    expect(tiny.hint).toBe("");
    expect(tiny.label.endsWith("…")).toBe(true);
    expect(tiny.label.length).toBeLessThanOrEqual(20 - 6);
    // No hint at all stays stable.
    expect(fitSelectRow("Cloud", undefined, 80)).toEqual({ label: "Cloud", hint: "" });
  });

  it("emits ANSI only when colors are on", () => {
    const on = makeUi("install", true);
    on.ui.intro();
    on.ui.log("claude-code: hooks merged into /home/u/.claude/settings.json");
    on.ui.outro(0);
    expect(on.lines.join("\n")).toContain("\x1b[");

    const off = makeUi("install", false);
    off.ui.intro();
    off.ui.log("claude-code: hooks merged into /home/u/.claude/settings.json");
    off.ui.outro(0);
    expect(off.lines.join("\n")).not.toContain("\x1b[");
  });
});
