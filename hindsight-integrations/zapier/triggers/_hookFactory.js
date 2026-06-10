"use strict";

const { baseUrl, enc } = require("../utils");

/**
 * Builds a Zapier REST Hook trigger backed by Hindsight's webhook API.
 *
 * On subscribe we register a Hindsight webhook scoped to a single bank and
 * event type, pointed at Zapier's per-Zap `targetUrl`. On unsubscribe we delete
 * it. Inbound deliveries (`{ event, bank_id, operation_id, status, timestamp,
 * data }`) are surfaced directly to the Zap.
 *
 * Security: we rely on Zapier's unguessable `targetUrl` rather than an HMAC
 * secret (deferred to a future version), so no signature verification here.
 *
 * Endpoints:
 *   POST   /v1/default/banks/{bank_id}/webhooks   -> { id, ... }
 *   DELETE /v1/default/banks/{bank_id}/webhooks/{webhook_id}
 */
const makeHookTrigger = ({ key, noun, label, description, eventType, sample }) => {
  const performSubscribe = async (z, bundle) => {
    const response = await z.request({
      method: "POST",
      url: `${baseUrl(bundle)}/v1/default/banks/${enc(bundle.inputData.bank_id)}/webhooks`,
      body: {
        url: bundle.targetUrl,
        event_types: [eventType],
        enabled: true,
      },
    });
    // Persisted as `bundle.subscribeData` for unsubscribe.
    return { id: response.data.id, bank_id: bundle.inputData.bank_id };
  };

  const performUnsubscribe = async (z, bundle) => {
    const { id, bank_id } = bundle.subscribeData;
    const response = await z.request({
      method: "DELETE",
      url: `${baseUrl(bundle)}/v1/default/banks/${enc(bank_id)}/webhooks/${enc(id)}`,
    });
    return response.data;
  };

  // Inbound webhook delivery — Zapier parses the JSON body into cleanedRequest.
  const perform = (z, bundle) => [bundle.cleanedRequest];

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
