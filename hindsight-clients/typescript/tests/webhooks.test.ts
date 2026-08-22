import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  WebhookPayloadError,
  WebhookSignatureError,
  constructWebhookEvent,
  parseWebhookEvent,
  verifyWebhookSignature,
} from "../src/webhooks";

type SignatureVector = { secret: string; raw_body: string; signature: string };

const vector: SignatureVector = JSON.parse(
  readFileSync(join(__dirname, "../../testdata/webhook-signatures.json"), "utf8")
);
const encoder = new TextEncoder();

function sign(rawBody: Uint8Array, secret: string): string {
  return `sha256=${createHmac("sha256", secret).update(rawBody).digest("hex")}`;
}

test("verifies the shared signature vector over exact bytes", async () => {
  const rawBody = encoder.encode(vector.raw_body);

  await expect(verifyWebhookSignature(rawBody, vector.signature, vector.secret)).resolves.toBe(
    true
  );
  await expect(
    verifyWebhookSignature(encoder.encode(`${vector.raw_body}\n`), vector.signature, vector.secret)
  ).resolves.toBe(false);
  await expect(
    verifyWebhookSignature(rawBody, vector.signature.toUpperCase(), vector.secret)
  ).resolves.toBe(false);
});

test("constructs a generated known event using a case-insensitive header", async () => {
  const headers = new Headers();
  headers.set("x-hindsight-signature", vector.signature);
  const event = await constructWebhookEvent(
    encoder.encode(vector.raw_body),
    headers,
    vector.secret
  );

  expect(event.kind).toBe("known");
  if (event.kind === "known" && event.event.event === "retain.completed") {
    expect(event.event.data.document_id).toBe("doc-1");
  }
});

test("accepts Node-style header arrays", async () => {
  await expect(
    constructWebhookEvent(
      encoder.encode(vector.raw_body),
      { "x-hindsight-signature": [vector.signature] },
      vector.secret
    )
  ).resolves.toMatchObject({ kind: "known", event: { event: "retain.completed" } });
});

test("verifies before parsing JSON", async () => {
  await expect(
    constructWebhookEvent(
      encoder.encode("{"),
      { "X-Hindsight-Signature": `sha256=${"0".repeat(64)}` },
      "secret"
    )
  ).rejects.toBeInstanceOf(WebhookSignatureError);

  const rawBody = encoder.encode("{");
  await expect(
    constructWebhookEvent(rawBody, { "X-Hindsight-Signature": sign(rawBody, "secret") }, "secret")
  ).rejects.toBeInstanceOf(WebhookPayloadError);
});

test("returns a forward-compatible envelope for an unknown event", () => {
  const event = parseWebhookEvent(
    encoder.encode(
      JSON.stringify({
        event: "future.created",
        bank_id: "bank-1",
        operation_id: "op-2",
        status: "completed",
        timestamp: "2026-08-11T09:30:00Z",
        data: { future: true },
        future_envelope_field: 42,
      })
    )
  );

  expect(event.kind).toBe("unknown");
  if (event.kind === "unknown") {
    expect(event.event.event).toBe("future.created");
    expect(event.event.data.future).toBe(true);
    expect(event.event.future_envelope_field).toBe(42);
  }
});

test("validates known event data", () => {
  expect(() =>
    parseWebhookEvent(
      encoder.encode(
        JSON.stringify({
          event: "memory_defense.triggered",
          bank_id: "bank-1",
          operation_id: "op-3",
          status: "block",
          timestamp: "2026-08-11T09:30:00Z",
          data: {},
        })
      )
    )
  ).toThrow(WebhookPayloadError);
});

test("validates the event timestamp format", () => {
  expect(() =>
    parseWebhookEvent(
      encoder.encode(
        JSON.stringify({
          event: "future.created",
          bank_id: "bank-1",
          operation_id: "op-4",
          status: "completed",
          timestamp: "not-a-date",
          data: {},
        })
      )
    )
  ).toThrow(WebhookPayloadError);
});
