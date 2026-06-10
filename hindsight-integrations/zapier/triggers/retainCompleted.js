"use strict";

const { makeHookTrigger } = require("./_hookFactory");

module.exports = makeHookTrigger({
  key: "retainCompleted",
  noun: "Retain",
  label: "Retain Completed",
  description:
    "Triggers when an asynchronous retain operation finishes processing in a memory bank.",
  eventType: "retain.completed",
  sample: {
    event: "retain.completed",
    bank_id: "user-123",
    operation_id: "op_abc123",
    status: "completed",
    timestamp: "2026-06-10T12:00:00Z",
    data: { document_id: "doc-1", tags: ["user:jon"] },
  },
});
