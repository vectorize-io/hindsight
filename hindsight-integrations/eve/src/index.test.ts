import { describe, it, expect } from "vitest";
import { once } from "eve/tools/approval";
import {
  resolveHindsightConnection,
  buildHindsightConnectionDefinition,
  defineHindsightConnection,
  HINDSIGHT_CLOUD_MCP_URL,
  DEFAULT_DESCRIPTION,
} from "./index";

const EMPTY_ENV = {} as NodeJS.ProcessEnv;

describe("resolveHindsightConnection", () => {
  it("defaults to Hindsight Cloud with the default description", () => {
    const resolved = resolveHindsightConnection({ apiKey: "hsk_test" }, EMPTY_ENV);
    expect(resolved.url).toBe(HINDSIGHT_CLOUD_MCP_URL);
    expect(resolved.description).toBe(DEFAULT_DESCRIPTION);
    expect(resolved.apiKey).toBe("hsk_test");
    expect(resolved.bankId).toBeNull();
  });

  it("reads url, key, and bank from the environment", () => {
    const resolved = resolveHindsightConnection({}, {
      HINDSIGHT_MCP_URL: "http://localhost:8000/mcp",
      HINDSIGHT_API_KEY: "env_key",
      HINDSIGHT_MCP_BANK_ID: "project-x",
    } as NodeJS.ProcessEnv);
    expect(resolved.url).toBe("http://localhost:8000/mcp");
    expect(resolved.apiKey).toBe("env_key");
    expect(resolved.bankId).toBe("project-x");
  });

  it("prefers explicit options over the environment", () => {
    const resolved = resolveHindsightConnection(
      { url: "http://opt/mcp", apiKey: "opt_key", bankId: "opt_bank" },
      {
        HINDSIGHT_MCP_URL: "http://env/mcp",
        HINDSIGHT_API_KEY: "env_key",
        HINDSIGHT_MCP_BANK_ID: "env_bank",
      } as NodeJS.ProcessEnv
    );
    expect(resolved.url).toBe("http://opt/mcp");
    expect(resolved.apiKey).toBe("opt_key");
    expect(resolved.bankId).toBe("opt_bank");
  });

  it("treats apiKey: null as an explicit no-auth opt-out", () => {
    const resolved = resolveHindsightConnection(
      { url: "http://localhost:8000/mcp", apiKey: null },
      { HINDSIGHT_API_KEY: "env_key" } as NodeJS.ProcessEnv
    );
    expect(resolved.apiKey).toBeNull();
  });

  it("throws when targeting Hindsight Cloud without a key", () => {
    expect(() => resolveHindsightConnection({}, EMPTY_ENV)).toThrow(/API key/);
  });

  it("throws for a Cloud URL with a trailing slash and no key", () => {
    expect(() =>
      resolveHindsightConnection({ url: "https://api.hindsight.vectorize.io/mcp/" }, EMPTY_ENV)
    ).toThrow(/API key/);
  });

  it("throws for a regional Cloud subdomain with no key", () => {
    expect(() =>
      resolveHindsightConnection({ url: "https://api.eu.hindsight.vectorize.io/mcp" }, EMPTY_ENV)
    ).toThrow(/API key/);
  });

  it("does not treat a look-alike host as Cloud", () => {
    // `nothindsight.vectorize.io` must not match the Cloud guard, so a no-auth
    // self-hosted server on a similar domain is allowed.
    const resolved = resolveHindsightConnection(
      { url: "https://nothindsight.vectorize.io/mcp", apiKey: null },
      EMPTY_ENV
    );
    expect(resolved.apiKey).toBeNull();
  });

  it("allows a self-hosted url with no auth", () => {
    const resolved = resolveHindsightConnection(
      { url: "http://localhost:8000/mcp", apiKey: null },
      EMPTY_ENV
    );
    expect(resolved.url).toBe("http://localhost:8000/mcp");
    expect(resolved.apiKey).toBeNull();
  });

  it("ignores empty-string environment values", () => {
    const resolved = resolveHindsightConnection({ apiKey: "k" }, {
      HINDSIGHT_MCP_URL: "",
    } as NodeJS.ProcessEnv);
    expect(resolved.url).toBe(HINDSIGHT_CLOUD_MCP_URL);
  });

  it("passes tool filters through unchanged", () => {
    const resolved = resolveHindsightConnection(
      { apiKey: "k", tools: { allow: ["recall", "retain"] } },
      EMPTY_ENV
    );
    expect(resolved.tools).toEqual({ allow: ["recall", "retain"] });
  });
});

describe("buildHindsightConnectionDefinition", () => {
  it("wires bearer auth whose getToken returns the configured key", async () => {
    const definition = buildHindsightConnectionDefinition(
      resolveHindsightConnection({ url: "http://localhost:8000/mcp", apiKey: "k" }, EMPTY_ENV)
    );
    expect(definition.url).toBe("http://localhost:8000/mcp");
    const auth = definition.auth as { getToken: () => Promise<{ token: string }> };
    expect(await auth.getToken()).toEqual({ token: "k" });
  });

  it("emits no auth when the key is null", () => {
    const definition = buildHindsightConnectionDefinition(
      resolveHindsightConnection({ url: "http://localhost:8000/mcp", apiKey: null }, EMPTY_ENV)
    );
    expect(definition.auth).toBeUndefined();
  });

  it("sets the X-Bank-Id header when a bank is configured", () => {
    const definition = buildHindsightConnectionDefinition(
      resolveHindsightConnection({ apiKey: "k", bankId: "project-x" }, EMPTY_ENV)
    );
    expect(definition.headers).toEqual({ "X-Bank-Id": "project-x" });
  });

  it("passes the approval policy through unchanged", () => {
    const approval = once();
    const definition = buildHindsightConnectionDefinition(
      resolveHindsightConnection({ apiKey: "k", approval }, EMPTY_ENV)
    );
    expect(definition.approval).toBe(approval);
  });

  it("omits approval when none is configured", () => {
    const definition = buildHindsightConnectionDefinition(
      resolveHindsightConnection({ apiKey: "k" }, EMPTY_ENV)
    );
    expect(definition.approval).toBeUndefined();
  });
});

describe("defineHindsightConnection", () => {
  it("builds a connection via the real eve framework without throwing", () => {
    const connection = defineHindsightConnection({
      url: "http://localhost:8000/mcp",
      apiKey: "k",
    });
    expect(connection).toBeDefined();
  });
});
