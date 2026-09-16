/**
 * Unit tests for the bank export/import wrappers' request mapping.
 *
 * Like the other *_mapping tests these need no server: the generated sdk layer is
 * mocked, so we can assert the camelCase options land on the snake_case query the
 * API reads — and that an omitted flag is genuinely absent rather than sent as
 * undefined, because the server's default for each of the three scope flags is
 * what decides how much of a bank travels.
 *
 * The Python wrapper has the same coverage in
 * hindsight-clients/python/tests/test_bank_transfer_wrapper.py; the two wrappers
 * are expected to expose the same surface.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedExport = sdk.exportBankTransfer as jest.MockedFunction<typeof sdk.exportBankTransfer>;
const mockedImport = sdk.importBankTransfer as jest.MockedFunction<typeof sdk.importBankTransfer>;
const mockedStatus = sdk.getOperationStatus as jest.MockedFunction<typeof sdk.getOperationStatus>;

const OPERATION_ID = "029110c8-a2b2-464c-a206-52c99b76cbf1";
const DOWNLOAD_URL = "/v1/default/files/download/banks/my-bank/exports/x/transfer.zip";
const ARCHIVE = new Uint8Array([0x50, 0x4b, 0x03, 0x04]);

describe("exportBank mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedExport.mockReset();
    mockedStatus.mockReset();
    mockedExport.mockResolvedValue({ data: { operation_id: OPERATION_ID } } as any);
    mockedStatus.mockResolvedValue({
      data: { status: "completed", result_metadata: { download_url: DOWNLOAD_URL } },
    } as any);
    // The archive is fetched from the server-provided URL, not through the
    // templated download route — stub that one call.
    (client as any).client.get = jest.fn().mockResolvedValue({ data: ARCHIVE.buffer });
  });

  test("submits, polls and downloads the archive the operation names", async () => {
    const bytes = await client.exportBank("my-bank", { pollIntervalMs: 0 });

    expect(Array.from(bytes)).toEqual(Array.from(ARCHIVE));
    expect(mockedExport.mock.calls[0][0].path).toEqual({ bank_id: "my-bank" });
    expect((client as any).client.get.mock.calls[0][0].url).toBe(DOWNLOAD_URL);
  });

  test("omitted scope flags are not sent, so the server's defaults decide", async () => {
    await client.exportBank("my-bank", { pollIntervalMs: 0 });

    const query = mockedExport.mock.calls[0][0].query as any;
    expect(query.include_data).toBeUndefined();
    expect(query.include_bank_config).toBeUndefined();
    expect(query.include_history).toBeUndefined();
  });

  test("a narrowed scope is forwarded snake_cased", async () => {
    await client.exportBank("my-bank", {
      includeData: true,
      includeBankConfig: false,
      includeHistory: true,
      pollIntervalMs: 0,
    });

    const query = mockedExport.mock.calls[0][0].query as any;
    expect(query.include_data).toBe(true);
    expect(query.include_bank_config).toBe(false);
    expect(query.include_history).toBe(true);
  });

  test("a failed export operation throws rather than returning empty bytes", async () => {
    mockedStatus.mockResolvedValue({
      data: { status: "failed", error_message: "disk full" },
    } as any);

    await expect(client.exportBank("my-bank", { pollIntervalMs: 0 })).rejects.toThrow("disk full");
  });
});

describe("importBank mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedImport.mockReset();
    mockedImport.mockResolvedValue({ data: { operation_id: OPERATION_ID } } as any);
  });

  test("sends the archive as the file body and returns the operation id", async () => {
    const archive = new Blob([ARCHIVE]);

    const operationId = await client.importBank("my-bank", archive, {
      targetBankId: "my-bank-copy",
    });

    expect(operationId).toBe(OPERATION_ID);
    const call = mockedImport.mock.calls[0][0] as any;
    expect(call.path).toEqual({ bank_id: "my-bank" });
    expect(call.query.target_bank_id).toBe("my-bank-copy");
    expect(call.body.file).toBe(archive);
  });

  test("restores a subset when asked, and sends nothing when not", async () => {
    const archive = new Blob([ARCHIVE]);

    await client.importBank("my-bank", archive, { includeBankConfig: false });
    expect((mockedImport.mock.calls[0][0] as any).query.include_bank_config).toBe(false);

    mockedImport.mockClear();
    await client.importBank("my-bank", archive);
    expect((mockedImport.mock.calls[0][0] as any).query.include_bank_config).toBeUndefined();
  });
});
