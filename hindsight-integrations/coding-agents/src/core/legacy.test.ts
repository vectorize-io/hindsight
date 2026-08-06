import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readLegacyEndpoint } from "./legacy";

const homes: string[] = [];

function homeWith(config: unknown): string {
  const home = mkdtempSync(join(tmpdir(), "hindsight-legacy-"));
  homes.push(home);
  if (config !== undefined) {
    mkdirSync(join(home, ".hindsight"), { recursive: true });
    writeFileSync(
      join(home, ".hindsight", "claude-code.json"),
      typeof config === "string" ? config : JSON.stringify(config)
    );
  }
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
    expect(Object.keys(e ?? {}).sort()).toEqual(["apiUrl", "serverMode", "source"]);
  });
});
