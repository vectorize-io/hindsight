import { describe, expect, it } from "vitest";
import { selectTools } from "./mcp-server";
import { resolveConfig } from "./core/config";
import type { HindsightClient } from "./core/hindsight";

// Plain stub — no SDK, no network. selectTools only needs a client reference to hand to
// buildKnowledgeTools when enabled; it never calls any client method itself.
const stubClient = {} as HindsightClient;

describe("selectTools", () => {
  it("returns [] when cfg.disabled is true — a disabled Hindsight exposes NO tools", () => {
    const cfg = resolveConfig({ disabled: true });
    expect(selectTools(cfg, stubClient, "b")).toEqual([]);
  });

  it("returns the eight hindsight_* tool specs when enabled", () => {
    const cfg = resolveConfig({});
    const tools = selectTools(cfg, stubClient, "b");
    expect(tools.map((t) => t.name).sort()).toEqual(
      [
        "hindsight_sync_status",
        "hindsight_diagnose",
        "hindsight_search_knowledge_pages",
        "hindsight_list_knowledge_pages",
        "hindsight_read_knowledge_page",
        "hindsight_reflect",
        "hindsight_capture_initiative",
        "hindsight_ingest_document",
      ].sort()
    );
  });

  it("propagates the configured harness to diagnostics", async () => {
    const cfg = resolveConfig({ harness: "codex" });
    const tools = selectTools(cfg, stubClient, "b");
    const diagnose = tools.find((tool) => tool.name === "hindsight_diagnose");

    expect(diagnose).toBeDefined();
    const result = await diagnose!.handler({});
    expect(JSON.parse(result.content[0].text)).toMatchObject({ harness: "codex" });
  });
});
