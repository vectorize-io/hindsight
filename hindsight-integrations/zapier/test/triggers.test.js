"use strict";

require("should");
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

  it("subscribes by registering a Hindsight webhook and returns { id, bank_id }", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/webhooks", {
        url: "https://hooks.zapier.com/abc",
        event_types: ["retain.completed"],
        enabled: true,
      })
      .reply(201, { id: "wh-1" });

    const result = await appTester(App.triggers.retainCompleted.operation.performSubscribe, {
      authData,
      inputData: { bank_id: "bank-1" },
      targetUrl: "https://hooks.zapier.com/abc",
    });
    result.should.eql({ id: "wh-1", bank_id: "bank-1" });
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

  it("surfaces the inbound webhook payload from perform", async () => {
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

  it("returns a sample from performList", async () => {
    const result = await appTester(App.triggers.retainCompleted.operation.performList, { authData });
    result.should.be.an.Array();
    result[0].event.should.eql("retain.completed");
  });
});
