import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readLegacyEndpoint } from "./legacy";

const homes: string[] = [];

function homeWith(config: unknown, file = "claude-code.json"): string {
  const home = mkdtempSync(join(tmpdir(), "hindsight-legacy-"));
  homes.push(home);
  if (config !== undefined) {
    mkdirSync(join(home, ".hindsight"), { recursive: true });
    writeFileSync(
      join(home, ".hindsight", file),
      typeof config === "string" ? config : JSON.stringify(config)
    );
  }
  return home;
}

/** Both old plugins that shipped a user config; same keys, different filename. */
function homeWithBoth(claude: unknown, codex: unknown): string {
  const home = homeWith(claude);
  mkdirSync(join(home, ".hindsight"), { recursive: true });
  writeFileSync(join(home, ".hindsight", "codex.json"), JSON.stringify(codex));
  return home;
}

afterEach(() => {
  while (homes.length) rmSync(homes.pop()!, { recursive: true, force: true });
});

describe("readLegacyEndpoint", () => {
  it("is undefined when the old plugin was never configured", () => {
    expect(readLegacyEndpoint(homeWith(undefined))).toBeUndefined();
  });

  it("carries a self-hosted server and its token", () => {
    const e = readLegacyEndpoint(
      homeWith({ hindsightApiUrl: "http://box:8888", hindsightApiToken: "t" })
    );
    expect(e?.serverMode).toBe("self-hosted");
    expect(e?.apiUrl).toBe("http://box:8888");
    expect(e?.apiToken).toBe("t");
  });

  it("recognises the Cloud URL as cloud, not self-hosted", () => {
    const e = readLegacyEndpoint(
      homeWith({ hindsightApiUrl: "https://api.hindsight.vectorize.io" })
    );
    expect(e?.serverMode).toBe("cloud");
  });

  // The old plugin treated an empty URL as "use the local daemon" (daemon.py:get_api_url), so a
  // config that never sets one describes daemon mode rather than an unconfigured user.
  it("reads an absent URL as daemon mode", () => {
    const e = readLegacyEndpoint(homeWith({ retainMode: "full-session" }));
    expect(e?.serverMode).toBe("daemon");
    expect(e?.apiUrl).toBeUndefined();
  });

  it("carries a non-default daemon port, since in daemon mode the port is the endpoint", () => {
    expect(readLegacyEndpoint(homeWith({ apiPort: 9100 }))?.apiPort).toBe(9100);
    expect(readLegacyEndpoint(homeWith({ apiPort: 9077 }))?.apiPort).toBeUndefined();
  });

  // A config we cannot parse is not a decision we can honour — fall through to the normal flow
  // rather than guessing an endpoint.
  it("ignores an unparseable config", () => {
    expect(readLegacyEndpoint(homeWith("{not json"))).toBeUndefined();
  });

  // Behavioural settings are deliberately not translated.
  it("carries no behavioural settings", () => {
    const e = readLegacyEndpoint(
      homeWith({ hindsightApiUrl: "http://box:8888", recallBudget: "high", retainMode: "chunked" })
    );
    expect(Object.keys(e ?? {}).sort()).toEqual(["apiUrl", "harness", "serverMode", "source"]);
  });

  it("reads the Codex plugin's config too", () => {
    const e = readLegacyEndpoint(homeWith({ hindsightApiUrl: "http://cx:8888" }, "codex.json"));
    expect(e?.harness).toBe("codex");
    expect(e?.apiUrl).toBe("http://cx:8888");
  });

  // Installing Codex must not pick up a stale claude-code.json that points somewhere else.
  it("prefers the config of the agent being installed", () => {
    const home = homeWithBoth(
      { hindsightApiUrl: "http://claude:8888" },
      { hindsightApiUrl: "http://codex:8888" }
    );
    expect(readLegacyEndpoint(home, ["codex"])?.apiUrl).toBe("http://codex:8888");
    expect(readLegacyEndpoint(home, ["claude-code"])?.apiUrl).toBe("http://claude:8888");
  });

  // A harness with no legacy plugin still benefits: the common case is one server for both.
  it("falls back to any known legacy config", () => {
    const home = homeWith({ hindsightApiUrl: "http://cx:8888" }, "codex.json");
    expect(readLegacyEndpoint(home, ["cursor-cli"])?.apiUrl).toBe("http://cx:8888");
  });
});
