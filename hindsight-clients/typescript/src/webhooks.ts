import type {
  ConsolidationCompletedWebhookEvent,
  MemoryDefenseTriggeredWebhookEvent,
  RetainCompletedWebhookEvent,
  WebhookEvent,
  WebhookEventEnvelope,
} from "../generated/types.gen";
import Ajv, { type ValidateFunction } from "ajv";
import addFormats from "ajv-formats";
import webhookSchema from "../generated/webhook-schema.json";

export const WEBHOOK_SIGNATURE_HEADER = "X-Hindsight-Signature";

export type ConsumedWebhookEvent =
  { kind: "known"; event: WebhookEvent } | { kind: "unknown"; event: WebhookEventEnvelope };
export type WebhookHeaders =
  | { get(name: string): string | null }
  | Readonly<Record<string, string | readonly string[] | undefined>>;

export class WebhookError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebhookError";
  }
}

export class WebhookSignatureError extends WebhookError {
  constructor(message = "webhook signature verification failed") {
    super(message);
    this.name = "WebhookSignatureError";
  }
}

export class WebhookPayloadError extends WebhookError {
  constructor(message: string) {
    super(message);
    this.name = "WebhookPayloadError";
  }
}

function signatureBytes(signature: string): ArrayBuffer | undefined {
  if (!/^sha256=[0-9a-f]{64}$/.test(signature)) return undefined;

  const result = new ArrayBuffer(32);
  const bytes = new Uint8Array(result);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(signature.slice(7 + index * 2, 9 + index * 2), 16);
  }
  return result;
}

function copyToArrayBuffer(value: Uint8Array): ArrayBuffer {
  const result = new ArrayBuffer(value.byteLength);
  new Uint8Array(result).set(value);
  return result;
}

export async function verifyWebhookSignature(
  rawBody: Uint8Array,
  signature: string | null | undefined,
  secret: string | Uint8Array
): Promise<boolean> {
  if (!(rawBody instanceof Uint8Array)) throw new TypeError("rawBody must be a Uint8Array");
  if (typeof signature !== "string") return false;

  const expected = signatureBytes(signature);
  if (!expected) return false;

  const secretBytes = typeof secret === "string" ? new TextEncoder().encode(secret) : secret;
  if (!(secretBytes instanceof Uint8Array)) {
    throw new TypeError("secret must be a string or Uint8Array");
  }

  const key = await globalThis.crypto.subtle.importKey(
    "raw",
    copyToArrayBuffer(secretBytes),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );
  return globalThis.crypto.subtle.verify("HMAC", key, expected, copyToArrayBuffer(rawBody));
}

const ajv = new Ajv({ strict: false, allErrors: true });
addFormats(ajv);
ajv.addSchema(webhookSchema, "hindsight-webhooks");
type WebhookValidators = {
  envelope: ValidateFunction;
  events: Record<string, ValidateFunction>;
};

let cachedValidators: WebhookValidators | undefined;

function getWebhookValidators(): WebhookValidators {
  if (cachedValidators) return cachedValidators;

  const envelope = ajv.getSchema("hindsight-webhooks#/components/schemas/WebhookEventEnvelope");
  const eventMapping = (
    webhookSchema.components.schemas.WebhookEvent as {
      discriminator?: { mapping?: Record<string, string> };
    }
  ).discriminator?.mapping;
  if (!envelope || !eventMapping || Object.keys(eventMapping).length === 0) {
    throw new Error("generated webhook schema has no usable event discriminator");
  }

  const events: Record<string, ValidateFunction> = {};
  for (const [eventName, reference] of Object.entries(eventMapping)) {
    const validator = ajv.getSchema(`hindsight-webhooks${reference}`);
    if (!validator) {
      throw new Error(`generated webhook schema is missing ${eventName}`);
    }
    events[eventName] = validator;
  }

  cachedValidators = { envelope, events };
  return cachedValidators;
}

export function parseWebhookEvent(rawBody: Uint8Array): ConsumedWebhookEvent {
  if (!(rawBody instanceof Uint8Array)) throw new TypeError("rawBody must be a Uint8Array");

  try {
    const { envelope: validateEnvelope, events: validators } = getWebhookValidators();
    const value: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(rawBody));
    if (!validateEnvelope(value)) throw new Error(ajv.errorsText(validateEnvelope.errors));
    const event = value as WebhookEventEnvelope;
    const validateKnown = validators[event.event];
    if (validateKnown) {
      if (!validateKnown(value)) throw new Error(ajv.errorsText(validateKnown.errors));
      return { kind: "known", event: value as WebhookEvent };
    }
    return { kind: "unknown", event };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new WebhookPayloadError(`invalid webhook payload: ${message}`);
  }
}

function getSignature(headers: WebhookHeaders): string | undefined {
  const getter = (headers as { get?: unknown }).get;
  if (typeof getter === "function") {
    const value = (getter as (name: string) => unknown).call(headers, WEBHOOK_SIGNATURE_HEADER);
    return typeof value === "string" ? value : undefined;
  }

  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() === WEBHOOK_SIGNATURE_HEADER.toLowerCase()) {
      if (typeof value === "string") return value;
      return Array.isArray(value) && typeof value[0] === "string" ? value[0] : undefined;
    }
  }
  return undefined;
}

export async function constructWebhookEvent(
  rawBody: Uint8Array,
  headers: WebhookHeaders,
  secret: string | Uint8Array
): Promise<ConsumedWebhookEvent> {
  if (!(await verifyWebhookSignature(rawBody, getSignature(headers), secret))) {
    throw new WebhookSignatureError();
  }
  return parseWebhookEvent(rawBody);
}

export type {
  ConsolidationCompletedWebhookEvent,
  MemoryDefenseTriggeredWebhookEvent,
  RetainCompletedWebhookEvent,
  WebhookEvent,
  WebhookEventEnvelope,
};
