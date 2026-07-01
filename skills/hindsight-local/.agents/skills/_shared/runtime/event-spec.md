# OMA L1 Event Spec

This file defines the minimum durable event contract for cross-runtime workflow state.
Events are appended to `.agents/state/sessions/{sid}/events.jsonl`.

## Common Fields

Every event MUST be a single JSON object on one JSONL line:

```json
{
  "eventId": "01HXZK...",
  "ts": "2026-05-25T00:00:00.000Z",
  "sid": "oma-...",
  "kind": "decision.made",
  "writerPid": 12345,
  "vendor": "codex",
  "vendorSid": "runtime-session-id",
  "parentEventId": "01HXZJ...",
  "causalityKey": "workflow-gate",
  "payload": {}
}
```

Required fields:

- `eventId`: unique sortable id generated before append.
- `ts`: ISO timestamp.
- `sid`: OMA session id.
- `kind`: event kind.
- `writerPid`: process id or runtime equivalent.

Optional fields:

- `vendor`: runtime/vendor name.
- `vendorSid`: runtime/vendor session id.
- `parentEventId`: prior event id for causal interpretation.
- `causalityKey`: stable grouping key for related events.
- `payload`: event-specific JSON object.

Readers MUST derive state by sorting valid events by `(ts, eventId)`. Raw file order is an implementation detail only.

## JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://oh-my-agent.dev/schemas/l1-event.schema.json",
  "title": "OMA L1 Event",
  "type": "object",
  "required": ["eventId", "ts", "sid", "kind", "writerPid"],
  "additionalProperties": false,
  "properties": {
    "eventId": { "type": "string", "minLength": 1 },
    "ts": { "type": "string", "format": "date-time" },
    "sid": { "type": "string", "minLength": 1 },
    "kind": {
      "type": "string",
      "enum": [
        "boundary",
        "session.created",
        "workflow.phase",
        "gate.passed",
        "gate.failed",
        "blocker.raised",
        "decision.made",
        "decision.missing",
        "session.ended"
      ]
    },
    "writerPid": { "type": "integer" },
    "vendor": { "type": "string" },
    "vendorSid": { "type": "string" },
    "parentEventId": { "type": "string" },
    "causalityKey": { "type": "string" },
    "payload": { "type": "object" }
  },
  "allOf": [
    {
      "if": { "properties": { "kind": { "const": "decision.made" } } },
      "then": {
        "properties": {
          "payload": {
            "type": "object",
            "required": ["subject", "decision", "rationale"],
            "properties": {
              "subject": { "type": "string", "minLength": 1 },
              "decision": { "type": "string", "minLength": 1 },
              "rationale": { "type": "string", "minLength": 1 }
            }
          }
        }
      }
    },
    {
      "if": { "properties": { "kind": { "const": "session.ended" } } },
      "then": {
        "properties": {
          "payload": {
            "type": "object",
            "required": ["status"],
            "properties": {
              "status": { "enum": ["completed", "failed"] }
            }
          }
        }
      }
    }
  ]
}
```

## Event Kinds

### `boundary`

Durable vendor/session transition mapping.

Required payload fields:

- `reason`
- `toVendor`
- `toVendorSid`

Optional payload fields:

- `fromVendor`
- `fromVendorSid`
- `previousSid`

### `session.created`

Starts an OMA L1 workflow session.

Required payload fields:

- `workflow`
- `category`

### `workflow.phase`

Records a phase transition.

Required payload fields:

- `phase`

### `gate.passed`

Records a passed gate.

Required payload fields:

- `gate`

Optional payload fields:

- `by`
- `evidence`

### `gate.failed`

Records a failed gate.

Required payload fields:

- `gate`
- `reason`

### `blocker.raised`

Records a workflow blocker.

Required payload fields:

- `summary`

Optional payload fields:

- `severity`
- `remediation`

### `decision.made`

Records a critical decision that must survive vendor/session boundaries without AgentMemory.

Required payload fields:

- `subject`: stable verifier key, such as `ultrawork.plan-approved`.
- `decision`: concise decision summary.
- `rationale`: why this decision was made.

Optional payload fields:

- `alternatives`
- `evidence`

### `decision.missing`

Records deterministic verifier failure for a required `decision.made` event.

Required payload fields:

- `workflow`
- `checkpoint`
- `missing`
- `remediation`

### `session.ended`

Records terminal workflow state.

Required payload fields:

- `status`: `completed` or `failed`.

## Emitting Decisions

Call `oma state:emit` **directly** at each required checkpoint — do not wrap it in a shell helper.

Every Bash tool call runs in a fresh shell, so a shell function defined in one call (such as the historical `oma_emit`) does not exist in the next call. The wrapper saved nothing over the underlying command and only produced "command not found" fallbacks, so it has been removed.

Required decision example:

```bash
oma state:emit "decision.made" '{"subject":"ultrawork.plan-approved","decision":"Proceed with the approved plan.","rationale":"PLAN_GATE passed and the user confirmed scope."}'
oma state:verify --workflow ultrawork --checkpoint plan-approved
```
