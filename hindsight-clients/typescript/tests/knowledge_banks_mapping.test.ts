/**
 * Unit tests for the knowledge-bank wrappers' request mapping.
 *
 * No server needed: the generated sdk layer is mocked so we can assert the
 * camelCase options land on the snake_case request. Mirrors the Python
 * wrapper's test_knowledge_banks_mapping so the two clients stay at parity.
 */

import { HindsightClient, HindsightError } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedSearch = sdk.searchKnowledgeBank as jest.MockedFunction<typeof sdk.searchKnowledgeBank>;
const mockedTexts = sdk.ingestKnowledgeTexts as jest.MockedFunction<
  typeof sdk.ingestKnowledgeTexts
>;
const mockedFiles = sdk.ingestKnowledgeFiles as jest.MockedFunction<
  typeof sdk.ingestKnowledgeFiles
>;
const mockedDocs = sdk.listKnowledgeDocuments as jest.MockedFunction<
  typeof sdk.listKnowledgeDocuments
>;
const mockedConfig = sdk.updateKnowledgeBankConfig as jest.MockedFunction<
  typeof sdk.updateKnowledgeBankConfig
>;

const ingested = { data: { document_ids: [], operation_ids: [] } } as any;
let client: HindsightClient;

beforeEach(() => {
  client = new HindsightClient({ baseUrl: "http://localhost:8888" });
  jest.resetAllMocks();
});

test("searchKnowledgeBank maps every option", async () => {
  mockedSearch.mockResolvedValue({ data: { results: [] } } as any);

  await client.searchKnowledgeBank("kb-1", {
    query: "return window",
    queryImage: "data:image/png;base64,AAAA",
    queryVideo: "data:video/mp4;base64,BBBB",
    budget: "high",
    maxTokens: 2048,
    tags: ["policy"],
    tagsMatch: "all_strict",
    tagGroups: [{ tags: ["x200"], match: "any" }],
    documentIds: ["returns-policy"],
    mediaTypes: ["text", "document"],
    enableReranking: false,
    trace: true,
  });

  const call = mockedSearch.mock.calls[0][0];
  expect(call.path).toEqual({ bank_id: "kb-1" });
  expect(call.body).toEqual({
    query: "return window",
    query_image: "data:image/png;base64,AAAA",
    query_video: "data:video/mp4;base64,BBBB",
    budget: "high",
    max_tokens: 2048,
    tags: ["policy"],
    tags_match: "all_strict",
    tag_groups: [{ tags: ["x200"], match: "any" }],
    document_ids: ["returns-policy"],
    media_types: ["text", "document"],
    enable_reranking: false,
    trace: true,
  });
});

test("ingestKnowledgeTexts maps items and async", async () => {
  mockedTexts.mockResolvedValue(ingested);
  const items = [
    {
      text: "# Returns",
      document_id: "returns",
      filename: "r.md",
      tags: ["policy"],
      strategy: "s",
    },
  ];

  await client.ingestKnowledgeTexts("kb-1", items, { ingestAsync: true });

  const call = mockedTexts.mock.calls[0][0];
  expect(call.path).toEqual({ bank_id: "kb-1" });
  expect(call.body).toEqual({ items, async: true });
});

test("ingestKnowledgeFiles sends per-file options as the request field", async () => {
  mockedFiles.mockResolvedValue(ingested);
  const files = [new Blob(["%PDF-1.4"]), new Blob(["# Quick start"])];

  await client.ingestKnowledgeFiles("kb-1", files, {
    filesOptions: [{ document_id: "manual", tags: ["x200"] }, { document_id: "quick" }],
  });

  const call = mockedFiles.mock.calls[0][0];
  expect(call.path).toEqual({ bank_id: "kb-1" });
  expect(call.body.files).toBe(files);
  expect(JSON.parse(call.body.request as string)).toEqual({
    files: [{ document_id: "manual", tags: ["x200"] }, { document_id: "quick" }],
  });
});

test("ingestKnowledgeFiles without options sends no request field", async () => {
  mockedFiles.mockResolvedValue(ingested);

  await client.ingestKnowledgeFiles("kb-1", [new Blob(["a"])]);

  expect(mockedFiles.mock.calls[0][0].body).not.toHaveProperty("request");
});

test("ingestKnowledgeFiles refuses options that do not match the files", async () => {
  await expect(
    client.ingestKnowledgeFiles("kb-1", [new Blob(["a"])], { filesOptions: [{}, {}] })
  ).rejects.toThrow(HindsightError);
  expect(mockedFiles).not.toHaveBeenCalled();
});

test("listKnowledgeDocuments maps filters", async () => {
  mockedDocs.mockResolvedValue({ data: { items: [], total: 0 } } as any);

  await client.listKnowledgeDocuments("kb-1", {
    q: "x200",
    status: "ready",
    mediaType: "document",
    tags: ["x200"],
    limit: 10,
    offset: 20,
  });

  const call = mockedDocs.mock.calls[0][0];
  expect(call.path).toEqual({ bank_id: "kb-1" });
  expect(call.query).toEqual({
    q: "x200",
    status: "ready",
    media_type: "document",
    tags: ["x200"],
    limit: 10,
    offset: 20,
  });
});

test("updateKnowledgeBankConfig sends updates including clears", async () => {
  mockedConfig.mockResolvedValue({ data: {} } as any);

  await client.updateKnowledgeBankConfig("kb-1", {
    text_passage_tokens: 256,
    disabled_media_types: null,
  });

  expect(mockedConfig.mock.calls[0][0].body).toEqual({
    updates: { text_passage_tokens: 256, disabled_media_types: null },
  });
});
