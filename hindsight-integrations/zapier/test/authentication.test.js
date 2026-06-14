"use strict";

require("should");
const zapier = require("zapier-platform-core");
const nock = require("nock");

const App = require("../index");

const appTester = zapier.createAppTester(App);

describe("authentication", () => {
  afterEach(() => nock.cleanAll());

  const authData = { apiKey: "hsk_test", apiUrl: "https://api.example.com" };

  it("tests the credential against GET /v1/default/banks with a Bearer header", async () => {
    const scope = nock("https://api.example.com", {
      reqheaders: { authorization: "Bearer hsk_test" },
    })
      .get("/v1/default/banks")
      .reply(200, { banks: [] });

    const response = await appTester(App.authentication.test, { authData });
    response.status.should.eql(200);
    scope.isDone().should.be.true();
  });

  it("throws an AuthenticationError on 401", async () => {
    nock("https://api.example.com").get("/v1/default/banks").reply(401, { error: "nope" });

    await appTester(App.authentication.test, { authData }).should.be.rejectedWith(
      /Invalid or unauthorized/
    );
  });

  it("strips a trailing slash from the API URL", async () => {
    const scope = nock("https://api.example.com")
      .get("/v1/default/banks")
      .reply(200, { banks: [] });

    await appTester(App.authentication.test, {
      authData: { apiKey: "hsk_test", apiUrl: "https://api.example.com/" },
    });
    scope.isDone().should.be.true();
  });

  it("builds a connection label from the host", () => {
    const label = App.authentication.connectionLabel({}, { authData });
    label.should.eql("Hindsight (api.example.com)");
  });
});
