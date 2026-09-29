/**
 * Unit tests for the knowledge-base wrappers' request mapping.
 *
 * Like create_mental_model_mapping, these do NOT require a running server: the
 * generated sdk layer is mocked so we can assert the ergonomic camelCase
 * options land on the snake_case request body — and, for updateKnowledgeNode,
 * that an omitted field is genuinely absent rather than sent as undefined
 * (the server distinguishes "not provided" from `parent_id: null`, which means
 * "move to the root").
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedCreatePage = sdk.createKnowledgePage as jest.MockedFunction<
  typeof sdk.createKnowledgePage
>;
const mockedUpdateNode = sdk.updateKnowledgeNode as jest.MockedFunction<
  typeof sdk.updateKnowledgeNode
>;
const mockedSearch = sdk.searchKnowledgeBase as jest.MockedFunction<typeof sdk.searchKnowledgeBase>;

describe("createKnowledgePage mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedCreatePage.mockReset();
    mockedCreatePage.mockResolvedValue({
      data: { page_id: "kp-1", mental_model_id: "mm-1", operation_id: "op-1" },
    } as any);
  });

  test("maps name, source query, parent and tags", async () => {
    await client.createKnowledgePage("bank", "Deploying the API", "How is the API deployed?", {
      parentId: "kf-1",
      tags: ["ops", "type:runbook"],
      maxTokens: 8192,
    });

    const body = mockedCreatePage.mock.calls[0][0].body as any;
    expect(body.name).toBe("Deploying the API");
    expect(body.source_query).toBe("How is the API deployed?");
    expect(body.parent_id).toBe("kf-1");
    expect(body.tags).toEqual(["ops", "type:runbook"]);
    expect(body.max_tokens).toBe(8192);
  });

  test("omitting trigger sends no trigger (preserves the page defaults)", async () => {
    await client.createKnowledgePage("bank", "Page", "q");

    expect((mockedCreatePage.mock.calls[0][0].body as any).trigger).toBeUndefined();
  });

  test("threads every trigger field through, snake_cased", async () => {
    await client.createKnowledgePage("bank", "Page", "q", {
      trigger: {
        mode: "delta",
        refreshAfterConsolidation: true,
        factTypes: ["observation"],
        excludeMentalModels: true,
        excludeMentalModelIds: ["mm-9"],
        tagsMatch: "any",
        includeChunks: false,
        recallMaxTokens: 4096,
        recallChunksMaxTokens: 1024,
      },
    });

    const trigger = (mockedCreatePage.mock.calls[0][0].body as any).trigger;
    expect(trigger.mode).toBe("delta");
    expect(trigger.refresh_after_consolidation).toBe(true);
    expect(trigger.fact_types).toEqual(["observation"]);
    expect(trigger.exclude_mental_models).toBe(true);
    expect(trigger.exclude_mental_model_ids).toEqual(["mm-9"]);
    expect(trigger.tags_match).toBe("any");
    expect(trigger.include_chunks).toBe(false);
    expect(trigger.recall_max_tokens).toBe(4096);
    expect(trigger.recall_chunks_max_tokens).toBe(1024);
  });
});

describe("updateKnowledgeNode mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedUpdateNode.mockReset();
    mockedUpdateNode.mockResolvedValue({ data: { id: "kp-1", kind: "page", name: "Page" } } as any);
  });

  test("sends only the fields provided", async () => {
    await client.updateKnowledgeNode("bank", "kp-1", { name: "Renamed" });

    const body = mockedUpdateNode.mock.calls[0][0].body as any;
    expect(body).toEqual({ name: "Renamed" });
    expect("parent_id" in body).toBe(false);
  });

  test("an explicit null parentId is sent (move to the root)", async () => {
    await client.updateKnowledgeNode("bank", "kp-1", { parentId: null });

    const body = mockedUpdateNode.mock.calls[0][0].body as any;
    expect("parent_id" in body).toBe(true);
    expect(body.parent_id).toBeNull();
  });

  test("maps page options to snake_case", async () => {
    await client.updateKnowledgeNode("bank", "kp-1", {
      sourceQuery: "New question?",
      tags: [],
      maxTokens: 2048,
    });

    const body = mockedUpdateNode.mock.calls[0][0].body as any;
    expect(body.source_query).toBe("New question?");
    expect(body.tags).toEqual([]);
    expect(body.max_tokens).toBe(2048);
  });

  // The trigger is a PATCH server-side: what isn't sent keeps the page's current value,
  // so a wrapper that dropped the field would leave callers unable to change a refresh
  // policy at all — the endpoint carried no trigger before this.
  test("passes a partial trigger through untouched", async () => {
    await client.updateKnowledgeNode("bank", "kp-1", { trigger: { refresh_cron: "0 3 * * *" } });

    const body = mockedUpdateNode.mock.calls[0][0].body as any;
    expect(body.trigger).toEqual({ refresh_cron: "0 3 * * *" });
  });

  test("omits the trigger entirely when unset", async () => {
    await client.updateKnowledgeNode("bank", "kp-1", { maxTokens: 2048 });

    expect("trigger" in (mockedUpdateNode.mock.calls[0][0].body as any)).toBe(false);
  });
});

describe("searchKnowledgeBase mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedSearch.mockReset();
    mockedSearch.mockResolvedValue({ data: { results: [], total: 0 } } as any);
  });

  test("sends the query as q and omits limit when unset", async () => {
    await client.searchKnowledgeBase("bank", "how do we deploy");

    const query = mockedSearch.mock.calls[0][0].query as any;
    expect(query.q).toBe("how do we deploy");
    expect("limit" in query).toBe(false);
  });

  test("passes limit through when set", async () => {
    await client.searchKnowledgeBase("bank", "deploy", { limit: 5 });

    expect((mockedSearch.mock.calls[0][0].query as any).limit).toBe(5);
  });
});
