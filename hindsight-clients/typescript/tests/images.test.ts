import { HindsightClient, sdk } from "../src";

const ok = (data: unknown, headers?: Headers) => ({
  data,
  error: undefined,
  response: new Response(null, { status: 200, headers }),
});

afterEach(() => jest.restoreAllMocks());

test("ordered image multipart metadata, recall map, management and idempotency", async () => {
  const retain = jest
    .spyOn(sdk, "imageRetain")
    .mockResolvedValue(ok({ operation_id: "op" }) as never);
  const recall = jest
    .spyOn(sdk, "recallMemories")
    .mockResolvedValue(ok({ results: [], image_assets: { doc: [{ asset_id: "a/1" }] } }) as never);
  jest.spyOn(sdk, "listImageAssets").mockResolvedValue(ok({ items: [], total: 0 }) as never);
  jest.spyOn(sdk, "deleteImageAsset").mockResolvedValue({
    data: undefined,
    error: undefined,
    response: new Response(null, { status: 204 }),
  } as never);
  jest.spyOn(sdk, "getImageAsset").mockResolvedValue(
    ok(
      new Blob(["bytes"], { type: "image/jpeg" }),
      new Headers({
        "content-type": "image/jpeg",
        "content-length": "5",
        "x-hindsight-image-sha256": "a".repeat(64),
        "x-hindsight-image-width": "8",
        "x-hindsight-image-height": "6",
        "x-hindsight-asset-created-at": "2026-01-01T00:00:00Z",
        "x-hindsight-asset-updated-at": "2026-01-01T00:00:00Z",
      })
    ) as never
  );
  const client = new HindsightClient({ baseUrl: "http://unused" });
  const first = new Blob(["first"], { type: "image/jpeg" });
  const second = new Blob(["second"], { type: "image/png" });
  await client.retainImages(
    "bank",
    [
      { file: first, assetId: "a/1", context: "first" },
      { file: second, assetId: "b/2", context: "second" },
    ],
    { idempotencyKey: "idem" }
  );
  const request = retain.mock.calls[0][0] as any;
  expect(request.body.files).toEqual([first, second]);
  expect(JSON.parse(request.body.request).images).toEqual([
    { asset_id: "a/1", context: "first" },
    { asset_id: "b/2", context: "second" },
  ]);
  expect(request.headers["Idempotency-Key"]).toBe("idem");

  const recalled = await client.recall("bank", "query", { includeImageAssets: true });
  expect((recall.mock.calls[0][0] as any).body.include.image_assets).toBe(true);
  expect(recalled.image_assets?.doc[0].asset_id).toBe("a/1");
  await client.listImageAssets("bank", { documentId: "doc" });
  const downloaded = await client.getImageAsset("bank", "a/1");
  expect(downloaded.descriptor.document_ids).toEqual([]);
  await client.deleteImageAsset("bank", "a/1");
});

test("transfer returns a stream and old services fail explicitly", async () => {
  const client = new HindsightClient({ baseUrl: "http://unused" });
  const stream = new ReadableStream<Uint8Array>();
  (client as any).client.get = jest.fn().mockResolvedValue({
    data: stream,
    error: undefined,
    response: { ok: true, status: 200, body: stream },
  });
  expect(await client.exportDocumentsStream("bank")).toBe(stream);

  jest.spyOn(sdk, "listImageAssets").mockResolvedValue({
    data: undefined,
    error: { detail: "not supported" },
    response: new Response(null, { status: 404 }),
  } as never);
  await expect(client.listImageAssets("bank")).rejects.toEqual(
    expect.objectContaining({ statusCode: 404 })
  );
});

test("image download rejects incomplete descriptor headers", async () => {
  jest.spyOn(sdk, "getImageAsset").mockResolvedValue(
    ok(
      new Blob(["bytes"], { type: "image/jpeg" }),
      new Headers({
        "content-type": "image/jpeg",
        "content-length": "5",
      })
    ) as never
  );
  const client = new HindsightClient({ baseUrl: "http://unused" });
  await expect(client.getImageAsset("bank", "asset")).rejects.toThrow(
    "x-hindsight-asset-created-at"
  );
});
