"use strict";

require("should");
const zapier = require("zapier-platform-core");
const nock = require("nock");

const App = require("../index");

const appTester = zapier.createAppTester(App);
const authData = { apiKey: "hsk_test", apiUrl: "https://api.example.com" };

describe("creates.retain", () => {
  afterEach(() => nock.cleanAll());

  it("POSTs to the memories endpoint with content, tags, and async:false", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/memories", {
        items: [{ content: "hello", tags: ["a", "b"] }],
        async: false,
      })
      .reply(200, { success: true, bank_id: "bank-1", items_count: 1, operation_id: "op-1" });

    const result = await appTester(App.creates.retain.operation.perform, {
      authData,
      inputData: { bank_id: "bank-1", content: "hello", tags: "a, b" },
    });
    result.operation_id.should.eql("op-1");
    result.success.should.be.true();
  });

  it("omits tags when none are given and includes context/timestamp when present", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/memories", {
        items: [{ content: "hi", context: "ctx", timestamp: "2026-01-01T00:00:00Z" }],
        async: false,
      })
      .reply(200, { success: true, bank_id: "bank-1", items_count: 1 });

    await appTester(App.creates.retain.operation.perform, {
      authData,
      inputData: {
        bank_id: "bank-1",
        content: "hi",
        context: "ctx",
        timestamp: "2026-01-01T00:00:00Z",
      },
    });
  });

  it("URL-encodes the bank id", async () => {
    const scope = nock("https://api.example.com")
      .post("/v1/default/banks/user%2F1/memories")
      .reply(200, { success: true });

    await appTester(App.creates.retain.operation.perform, {
      authData,
      inputData: { bank_id: "user/1", content: "hi" },
    });
    scope.isDone().should.be.true();
  });
});
