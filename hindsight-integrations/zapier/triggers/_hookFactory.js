"use strict";

const crypto = require("crypto");
const { baseUrl, enc } = require("../utils");

/**
 * Builds a Zapier REST Hook trigger backed by Hindsight's webhook API.
 *
 * On subscribe we register a Hindsight webhook scoped to a single bank and
 * event type, pointed at Zapier's per-Zap `targetUrl`, with a freshly generated
 * HMAC secret. On unsubscribe we delete it. Each inbound delivery's signature is
 * verified before the event reaches the Zap.
 *
 * Signature: Hindsight signs the raw delivery body and sends
 *   `X-Hindsight-Signature: sha256=<hex>` where `<hex>` is
 *   HMAC-SHA256(secret, rawBody). We recompute it over `bundle.rawRequest.content`
 *   (the body is sent byte-for-byte, so this matches) and reject on mismatch.
 *
 * Endpoints:
 *   POST   /v1/default/banks/{bank_id}/webhooks   -> { id, ... }
 *   DELETE /v1/default/banks/{bank_id}/webhooks/{webhook_id}
 */

/** Case-insensitive header lookup (Zapier may lower-case header keys). */
const getHeader = (headers, name) => {
  if (!headers) return undefined;
  const want = name.toLowerCase();
  for (const k of Object.keys(headers)) {
    if (k.toLowerCase() === want) return headers[k];
  }
  return undefined;
};

const makeHookTrigger = ({ key, noun, label, description, eventType, sample }) => {
  const performSubscribe = async (z, bundle) => {
    const secret = crypto.randomBytes(32).toString("hex");
    const response = await z.request({
      method: "POST",
      url: `${baseUrl(bundle)}/v1/default/banks/${enc(bundle.inputData.bank_id)}/webhooks`,
      body: {
        url: bundle.targetUrl,
        event_types: [eventType],
        enabled: true,
        secret,
      },
    });
    // Persisted as `bundle.subscribeData` for unsubscribe + signature checks.
    return { id: response.data.id, bank_id: bundle.inputData.bank_id, secret };
  };

  const performUnsubscribe = async (z, bundle) => {
    const { id, bank_id } = bundle.subscribeData;
    const response = await z.request({
      method: "DELETE",
      url: `${baseUrl(bundle)}/v1/default/banks/${enc(bank_id)}/webhooks/${enc(id)}`,
    });
    return response.data;
  };

  // Inbound delivery — verify the HMAC signature, then surface the parsed event.
  const perform = (z, bundle) => {
    const secret = bundle.subscribeData && bundle.subscribeData.secret;
    if (secret) {
      const raw = (bundle.rawRequest && bundle.rawRequest.content) || "";
      const got = getHeader(
        bundle.rawRequest && bundle.rawRequest.headers,
        "X-Hindsight-Signature"
      );
      const expected = "sha256=" + crypto.createHmac("sha256", secret).update(raw).digest("hex");
      if (got !== expected) {
        throw new z.errors.Error("Webhook signature verification failed.", "SignatureError", 401);
      }
    }
    return [bundle.cleanedRequest];
  };

  // No "list past events" endpoint exists, so the test step returns a sample.
  const performList = () => [sample];

  return {
    key,
    noun,
    display: { label, description },
    operation: {
      type: "hook",
      inputFields: [
        {
          key: "bank_id",
          label: "Bank",
          required: true,
          dynamic: "bankList.bank_id.name",
          helpText: "The memory bank to watch for events.",
        },
      ],
      performSubscribe,
      performUnsubscribe,
      perform,
      performList,
      sample,
    },
  };
};

module.exports = { makeHookTrigger };
