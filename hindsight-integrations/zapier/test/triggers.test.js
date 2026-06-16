"use strict";

require("should");
const crypto = require("crypto");
const zapier = require("zapier-platform-core");
const nock = require("nock");

const App = require("../index");

const appTester = zapier.createAppTester(App);
const authData = { apiKey: "hsk_test", apiUrl: "https://api.example.com" };

describe("triggers.bankList", () => {
  afterEach(() => nock.cleanAll());

  it("maps banks to { bank_id, name } for the dropdown", async () => {
    nock("https://api.example.com")
      .get("/v1/default/banks")
      .reply(200, { banks: [{ bank_id: "b1", name: "Bank One" }, { bank_id: "b2" }] });

    const banks = await appTester(App.triggers.bankList.operation.perform, { authData });
    banks.should.eql([
      { id: "b1", bank_id: "b1", name: "Bank One" },
      { id: "b2", bank_id: "b2", name: "b2" },
    ]);
  });
});

describe("triggers.retainCompleted (REST hook)", () => {
  afterEach(() => nock.cleanAll());

  it("subscribes with a generated secret and returns { id, bank_id, secret }", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/webhooks", (body) => {
        // body carries the targetUrl, event type, enabled flag, and a hex secret.
        return (
          body.url === "https://hooks.zapier.com/abc" &&
          body.event_types[0] === "retain.completed" &&
          body.enabled === true &&
          typeof body.secret === "string" &&
          body.secret.length >= 32
        );
      })
      .reply(201, { id: "wh-1" });

    const result = await appTester(App.triggers.retainCompleted.operation.performSubscribe, {
      authData,
      inputData: { bank_id: "bank-1" },
      targetUrl: "https://hooks.zapier.com/abc",
    });
    result.id.should.eql("wh-1");
    result.bank_id.should.eql("bank-1");
    result.secret.should.be.a.String();
  });

  it("unsubscribes by deleting the registered webhook", async () => {
    const scope = nock("https://api.example.com")
      .delete("/v1/default/banks/bank-1/webhooks/wh-1")
      .reply(200, { success: true });

    await appTester(App.triggers.retainCompleted.operation.performUnsubscribe, {
      authData,
      subscribeData: { id: "wh-1", bank_id: "bank-1" },
    });
    scope.isDone().should.be.true();
  });

  it("surfaces the inbound payload when there is no secret to verify", async () => {
    const event = {
      event: "retain.completed",
      bank_id: "bank-1",
      operation_id: "op-1",
      status: "completed",
    };
    const result = await appTester(App.triggers.retainCompleted.operation.perform, {
      authData,
      cleanedRequest: event,
    });
    result.should.eql([event]);
  });

  it("accepts a delivery with a valid HMAC signature", async () => {
    const secret = "s3cr3t";
    const raw = '{"event":"retain.completed","bank_id":"bank-1"}';
    const sig = "sha256=" + crypto.createHmac("sha256", secret).update(raw).digest("hex");

    const result = await appTester(App.triggers.retainCompleted.operation.perform, {
      authData,
      subscribeData: { id: "wh-1", bank_id: "bank-1", secret },
      cleanedRequest: { event: "retain.completed", bank_id: "bank-1" },
      rawRequest: { content: raw, headers: { "X-Hindsight-Signature": sig } },
    });
    result[0].event.should.eql("retain.completed");
  });

  it("rejects a delivery with a bad HMAC signature", async () => {
    await appTester(App.triggers.retainCompleted.operation.perform, {
      authData,
      subscribeData: { id: "wh-1", bank_id: "bank-1", secret: "s3cr3t" },
      cleanedRequest: { event: "retain.completed", bank_id: "bank-1" },
      rawRequest: {
        content: '{"event":"retain.completed","bank_id":"bank-1"}',
        headers: { "X-Hindsight-Signature": "sha256=deadbeef" },
      },
    }).should.be.rejectedWith(/signature verification failed/i);
  });

  it("returns a sample from performList", async () => {
    const result = await appTester(App.triggers.retainCompleted.operation.performList, {
      authData,
    });
    result.should.be.an.Array();
    result[0].event.should.eql("retain.completed");
  });
});
