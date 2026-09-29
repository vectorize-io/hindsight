"use strict";

require("should");
const zapier = require("zapier-platform-core");
const nock = require("nock");

const App = require("../index");

const appTester = zapier.createAppTester(App);
const authData = { apiKey: "hsk_test", apiUrl: "https://api.example.com" };

describe("searches.recall", () => {
  afterEach(() => nock.cleanAll());

  it("returns the results array (not the envelope) and defaults budget to mid", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/memories/recall", { query: "bands", budget: "mid" })
      .reply(200, { results: [{ id: "1", text: "Tool", type: "world" }] });

    const results = await appTester(App.searches.recall.operation.perform, {
      authData,
      inputData: { bank_id: "bank-1", query: "bands" },
    });
    results.should.be.an.Array();
    results.length.should.eql(1);
    results[0].text.should.eql("Tool");
  });

  it("sends tags and tags_match only when tags are present", async () => {
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/memories/recall", {
        query: "q",
        budget: "high",
        tags: ["x"],
        tags_match: "all",
      })
      .reply(200, { results: [] });

    const results = await appTester(App.searches.recall.operation.perform, {
      authData,
      inputData: { bank_id: "bank-1", query: "q", budget: "high", tags: "x", tags_match: "all" },
    });
    results.should.eql([]);
  });
});

describe("searches.reflect", () => {
  afterEach(() => nock.cleanAll());

  it("surfaces the reflect `text` field as `answer` in a one-element array", async () => {
    // The real reflect response puts the synthesized answer in `text`, not `answer`.
    nock("https://api.example.com")
      .post("/v1/default/banks/bank-1/reflect", { query: "fav band?", budget: "mid" })
      .reply(200, { text: "Tool.", based_on: { memories: [] } });

    const results = await appTester(App.searches.reflect.operation.perform, {
      authData,
      inputData: { bank_id: "bank-1", query: "fav band?" },
    });
    results.should.be.an.Array();
    results.length.should.eql(1);
    results[0].id.should.eql("reflect");
    results[0].answer.should.eql("Tool.");
  });
});
