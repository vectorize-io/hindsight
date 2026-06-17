"use strict";

const { baseUrl, enc } = require("../utils");

/**
 * Reflect — get an LLM-synthesized answer grounded in a bank's memories.
 *
 * POST /v1/default/banks/{bank_id}/reflect
 * body: { query, budget } -> { answer, based_on }
 *
 * Modeled as a Zapier *search*. A search must return an array, so the single
 * synthesized answer is wrapped in a one-element array with a stable `id`.
 */
const perform = async (z, bundle) => {
  const response = await z.request({
    method: "POST",
    url: `${baseUrl(bundle)}/v1/default/banks/${enc(bundle.inputData.bank_id)}/reflect`,
    body: { query: bundle.inputData.query, budget: bundle.inputData.budget || "mid" },
  });
  const data = response.data || {};
  // The reflect response carries the synthesized answer in `text` (not `answer`);
  // surface it to Zaps under the friendlier `answer` key.
  return [{ id: "reflect", answer: data.text, based_on: data.based_on }];
};

module.exports = {
  key: "reflect",
  noun: "Answer",
  display: {
    label: "Reflect",
    description:
      "Get an LLM-synthesized answer to a question, grounded in a Hindsight memory bank's memories.",
  },
  operation: {
    inputFields: [
      { key: "bank_id", label: "Bank", required: true, dynamic: "bankList.bank_id.name" },
      {
        key: "query",
        label: "Query",
        type: "string",
        required: true,
        helpText: "The question to answer using the bank's memories.",
      },
      {
        key: "budget",
        label: "Budget",
        type: "string",
        required: false,
        default: "mid",
        choices: { low: "Low (fast)", mid: "Medium", high: "High (thorough)" },
        helpText: "How deep the synthesis should be.",
      },
    ],
    perform,
    sample: {
      id: "reflect",
      answer: "Jon's favorite band is Tool.",
      based_on: { memories: [], mental_models: [] },
    },
    outputFields: [
      { key: "id", label: "ID" },
      { key: "answer", label: "Answer" },
    ],
  },
};
