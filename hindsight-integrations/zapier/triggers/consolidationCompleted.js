"use strict";

const { makeHookTrigger } = require("./_hookFactory");

module.exports = makeHookTrigger({
  key: "consolidationCompleted",
  noun: "Consolidation",
  label: "Consolidation Completed",
  description:
    "Triggers when memory consolidation finishes — observations and mental models have been synthesized.",
  eventType: "consolidation.completed",
  sample: {
    event: "consolidation.completed",
    bank_id: "user-123",
    operation_id: "op_def456",
    status: "completed",
    timestamp: "2026-06-10T12:05:00Z",
    data: {
      observations_created: 2,
      observations_updated: 1,
      observations_deleted: 0,
      error_message: null,
    },
  },
});
