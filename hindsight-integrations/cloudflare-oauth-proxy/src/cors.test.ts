import { describe, expect, it } from "vitest";
import { applyCors, corsHeaders, preflightResponse, stripCorsHeaders } from "./cors";

describe("corsHeaders", () => {
  it("echoes the allowed origin and includes Vary", () => {
    const headers = corsHeaders("https://claude.ai");
    expect(headers["Access-Control-Allow-Origin"]).toBe("https://claude.ai");
    expect(headers.Vary).toBe("Origin");
  });

  it("uses specific methods, not a wildcard", () => {
    const headers = corsHeaders("https://claude.ai");
    expect(headers["Access-Control-Allow-Methods"]).toBe("GET, POST, OPTIONS");
    expect(headers["Access-Control-Allow-Methods"]).not.toContain("*");
  });

  it("allows the MCP session header", () => {
    const headers = corsHeaders("https://claude.ai");
    expect(headers["Access-Control-Allow-Headers"]).toContain("Mcp-Session-Id");
  });
});

describe("stripCorsHeaders", () => {
  it("removes all CORS headers from a response", () => {
    const response = new Response("hi", {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "X-Leak",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Expose-Headers": "X-Thing",
        "Access-Control-Max-Age": "10",
        "X-Kept": "yes",
      },
    });
    const cleaned = stripCorsHeaders(response);
    expect(cleaned.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(cleaned.headers.get("Access-Control-Allow-Methods")).toBeNull();
    expect(cleaned.headers.get("Access-Control-Allow-Headers")).toBeNull();
    expect(cleaned.headers.get("Access-Control-Allow-Credentials")).toBeNull();
    expect(cleaned.headers.get("Access-Control-Expose-Headers")).toBeNull();
    expect(cleaned.headers.get("Access-Control-Max-Age")).toBeNull();
    expect(cleaned.headers.get("X-Kept")).toBe("yes");
  });
});

describe("applyCors", () => {
  it("adds CORS headers for an allowlisted origin", () => {
    const response = new Response("hi");
    const patched = applyCors(response, "https://claude.ai");
    expect(patched.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
    expect(patched.headers.get("Vary")).toBe("Origin");
  });

  it("supports www.claude.ai", () => {
    const patched = applyCors(new Response(), "https://www.claude.ai");
    expect(patched.headers.get("Access-Control-Allow-Origin")).toBe("https://www.claude.ai");
  });

  it("strips upstream CORS headers when origin is not allowlisted", () => {
    const response = new Response("hi", {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
      },
    });
    const patched = applyCors(response, "https://evil.example.com");
    expect(patched.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(patched.headers.get("Access-Control-Allow-Methods")).toBeNull();
  });

  it("strips upstream CORS headers when no origin header is present", () => {
    const response = new Response("hi", {
      headers: { "Access-Control-Allow-Origin": "*" },
    });
    const patched = applyCors(response, null);
    expect(patched.headers.get("Access-Control-Allow-Origin")).toBeNull();
  });

  it("overrides any pre-existing Access-Control-Allow-Origin with the requesting origin", () => {
    const response = new Response("hi", {
      headers: { "Access-Control-Allow-Origin": "https://attacker.example" },
    });
    const patched = applyCors(response, "https://claude.ai");
    expect(patched.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
  });
});

describe("preflightResponse", () => {
  it("returns 204 with CORS headers for an allowed origin", () => {
    const res = preflightResponse("https://claude.ai");
    expect(res.status).toBe(204);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://claude.ai");
  });

  it("returns 403 for a disallowed origin", () => {
    const res = preflightResponse("https://evil.example.com");
    expect(res.status).toBe(403);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBeNull();
  });

  it("returns 403 when no origin header is present", () => {
    expect(preflightResponse(null).status).toBe(403);
  });
});
