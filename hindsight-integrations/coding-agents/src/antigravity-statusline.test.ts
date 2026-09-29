import { describe, expect, it } from "vitest";
import { resolveConfig } from "./core/config";
import { buildAntigravityStatusLine } from "./antigravity-statusline";

const stripAnsi = (value: string): string => value.replace(/\x1b\[[0-9;]*m/g, "");

describe("Antigravity status line", () => {
  it("identifies the resolved Hindsight bank for the active workspace", () => {
    const out = buildAntigravityStatusLine(
      { cwd: "/workspace/acme" },
      resolveConfig({ bankId: "team::acme" })
    );
    expect(stripAnsi(out)).toBe("Hindsight · team::acme");
  });

  it("uses workspace.current_dir when cwd is absent and hides disabled banks", () => {
    const cfg = resolveConfig({
      dynamicBankId: false,
      bankId: "default",
      banks: { default: { disabled: true } },
    });
    expect(buildAntigravityStatusLine({ workspace: { current_dir: "/workspace/acme" } }, cfg)).toBe(
      ""
    );
  });

  it("keeps a small indicator while Antigravity has not supplied a workspace", () => {
    expect(stripAnsi(buildAntigravityStatusLine({}, resolveConfig()))).toBe("Hindsight");
  });
});
