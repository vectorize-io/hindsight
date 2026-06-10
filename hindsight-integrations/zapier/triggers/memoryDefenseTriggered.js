"use strict";

const { makeHookTrigger } = require("./_hookFactory");

module.exports = makeHookTrigger({
  key: "memoryDefenseTriggered",
  noun: "Memory Defense",
  label: "Memory Defense Triggered",
  description:
    "Triggers when Hindsight's memory-defense filter redacts or blocks incoming content (e.g. detected secrets or PII).",
  eventType: "memory_defense.triggered",
  sample: {
    event: "memory_defense.triggered",
    bank_id: "user-123",
    operation_id: "op_ghi789",
    status: "completed",
    timestamp: "2026-06-10T12:10:00Z",
    data: {
      action: "redact",
      detector: "secret_scanner",
      document_id: "doc-2",
      matched_types: ["api_key"],
      message: "Redacted 1 secret before storing.",
    },
  },
});
