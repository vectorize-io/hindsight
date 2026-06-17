"use strict";

const authentication = require("./authentication");
const { addBearerHeader, handleHttpError } = require("./middleware");

const retain = require("./creates/retain");
const recall = require("./searches/recall");
const reflect = require("./searches/reflect");

const bankList = require("./triggers/banks");
const retainCompleted = require("./triggers/retainCompleted");
const consolidationCompleted = require("./triggers/consolidationCompleted");
const memoryDefenseTriggered = require("./triggers/memoryDefenseTriggered");

const App = {
  version: require("./package.json").version,
  platformVersion: require("zapier-platform-core").version,

  // Don't let the platform auto-strip/trim input — our perform functions handle
  // empty/optional fields explicitly, and this keeps behavior predictable.
  flags: { cleanInputData: false },

  authentication,

  // Inject the Bearer header and normalize errors for every request.
  beforeRequest: [addBearerHeader],
  afterResponse: [handleHttpError],

  triggers: {
    [bankList.key]: bankList,
    [retainCompleted.key]: retainCompleted,
    [consolidationCompleted.key]: consolidationCompleted,
    [memoryDefenseTriggered.key]: memoryDefenseTriggered,
  },

  creates: {
    [retain.key]: retain,
  },

  searches: {
    [recall.key]: recall,
    [reflect.key]: reflect,
  },
};

module.exports = App;
