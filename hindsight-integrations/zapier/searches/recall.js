"use strict";

const { baseUrl, parseTags, enc } = require("../utils");

/**
 * Recall — search a memory bank for relevant memories.
 *
 * POST /v1/default/banks/{bank_id}/memories/recall
 * body: { query, budget, tags?, tags_match?, max_tokens? } -> { results: [...] }
 *
 * Modeled as a Zapier *search* (read-only lookup). A search must return an
 * array, so we return `results` directly.
 */
const perform = async (z, bundle) => {
  const body = {
    query: bundle.inputData.query,
    budget: bundle.inputData.budget || "mid",
  };
  const tags = parseTags(bundle.inputData.tags);
  if (tags.length) {
    body.tags = tags;
    body.tags_match = bundle.inputData.tags_match || "any";
  }
  if (bundle.inputData.max_tokens) body.max_tokens = bundle.inputData.max_tokens;

  const response = await z.request({
    method: "POST",
    url: `${baseUrl(bundle)}/v1/default/banks/${enc(bundle.inputData.bank_id)}/memories/recall`,
    body,
  });
  return response.data.results || [];
};

module.exports = {
  key: "recall",
  noun: "Memory",
  display: {
    label: "Recall Memories",
    description:
      "Search a Hindsight memory bank for memories relevant to a natural-language query.",
  },
  operation: {
    inputFields: [
      { key: "bank_id", label: "Bank", required: true, dynamic: "bankList.bank_id.name" },
      {
        key: "query",
        label: "Query",
        type: "string",
        required: true,
        helpText: "Natural-language query to search memories with.",
      },
      {
        key: "budget",
        label: "Budget",
        type: "string",
        required: false,
        default: "mid",
        choices: { low: "Low (fast)", mid: "Medium", high: "High (thorough)" },
        helpText: "How exhaustive the retrieval should be.",
      },
      {
        key: "tags",
        label: "Tags",
        type: "string",
        required: false,
        helpText: "Comma-separated tags to filter by (leave blank for no filter).",
      },
      {
        key: "tags_match",
        label: "Tags Match",
        type: "string",
        required: false,
        default: "any",
        choices: { any: "Any", all: "All", any_strict: "Any (strict)", all_strict: "All (strict)" },
      },
      {
        key: "max_tokens",
        label: "Max Tokens",
        type: "integer",
        required: false,
        helpText: "Maximum tokens of memories to return.",
      },
    ],
    perform,
    sample: {
      id: "123e4567-e89b-12d3-a456-426614174000",
      text: "Marcus is a marine biologist.",
      type: "world",
      context: "research background",
      tags: ["person:marcus"],
    },
    // Recall results are returned pre-ranked by the server; the response carries
    // no numeric score, and the fact-type field is `type` (not `fact_type`).
    outputFields: [
      { key: "id", label: "ID" },
      { key: "text", label: "Text" },
      { key: "type", label: "Type" },
      { key: "context", label: "Context" },
    ],
  },
};
