"use strict";

const { baseUrl, parseTags, enc } = require("../utils");

/**
 * Retain — store content in a Hindsight memory bank.
 *
 * POST /v1/default/banks/{bank_id}/memories
 * body: { items: [{ content, context?, tags?, timestamp? }], async: false }
 */
const perform = async (z, bundle) => {
  const item = { content: bundle.inputData.content };
  if (bundle.inputData.context) item.context = bundle.inputData.context;
  const tags = parseTags(bundle.inputData.tags);
  if (tags.length) item.tags = tags;
  if (bundle.inputData.timestamp) item.timestamp = bundle.inputData.timestamp;

  const response = await z.request({
    method: "POST",
    url: `${baseUrl(bundle)}/v1/default/banks/${enc(bundle.inputData.bank_id)}/memories`,
    body: { items: [item], async: bundle.inputData.async === true },
  });
  return response.data;
};

module.exports = {
  key: "retain",
  noun: "Memory",
  display: {
    label: "Retain Memory",
    description:
      "Store content in a Hindsight memory bank. Hindsight extracts facts, entities, and relationships from the text.",
  },
  operation: {
    inputFields: [
      {
        key: "bank_id",
        label: "Bank",
        required: true,
        dynamic: "bankList.bank_id.name",
        helpText:
          "The memory bank to store into. A bank is created on first use; you can also type a new bank id.",
      },
      {
        key: "content",
        label: "Content",
        type: "text",
        required: true,
        helpText: "The text to store as a memory.",
      },
      {
        key: "context",
        label: "Context",
        type: "string",
        required: false,
        helpText: "Optional context describing where this content came from.",
      },
      {
        key: "tags",
        label: "Tags",
        type: "string",
        required: false,
        helpText: 'Comma-separated tags, e.g. "user:alex,scope:profile".',
      },
      {
        key: "timestamp",
        label: "Timestamp",
        type: "datetime",
        required: false,
        helpText: "When this content occurred. Defaults to now if left blank.",
      },
      {
        key: "async",
        label: "Process asynchronously",
        type: "boolean",
        required: false,
        default: "false",
        helpText:
          "Return immediately and process in the background. Enable for very large content that might exceed Zapier's action timeout; pair it with the 'Retain Completed' trigger to act on completion.",
      },
    ],
    perform,
    sample: {
      success: true,
      bank_id: "user-123",
      items_count: 1,
      async: false,
      operation_id: "op_abc123",
    },
    outputFields: [
      { key: "success", label: "Success", type: "boolean" },
      { key: "bank_id", label: "Bank ID" },
      { key: "items_count", label: "Items Count", type: "integer" },
      { key: "operation_id", label: "Operation ID" },
    ],
  },
};
