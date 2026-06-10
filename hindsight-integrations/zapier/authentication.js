"use strict";

const { baseUrl, DEFAULT_API_URL } = require("./utils");

/**
 * Custom API-key authentication.
 *
 * Mirrors the n8n `hindsightApi` credential: an API key plus an optional base
 * URL (Hindsight Cloud by default, overridable for self-hosted). The key is
 * sent as a Bearer token by the shared `beforeRequest` middleware; the `test`
 * call below exercises the credential against a real authenticated endpoint.
 */

const test = (z, bundle) =>
  z.request({ url: `${baseUrl(bundle)}/v1/default/banks` });

const connectionLabel = (z, bundle) => {
  const host = ((bundle.authData && bundle.authData.apiUrl) || DEFAULT_API_URL).replace(
    /^https?:\/\//,
    ""
  );
  return `Hindsight (${host})`;
};

module.exports = {
  type: "custom",
  test,
  connectionLabel,
  fields: [
    {
      key: "apiKey",
      label: "API Key",
      type: "password",
      required: true,
      helpText:
        "Your Hindsight API key (starts with `hsk_`). Create one at https://ui.hindsight.vectorize.io.",
    },
    {
      key: "apiUrl",
      label: "API URL",
      type: "string",
      required: false,
      default: DEFAULT_API_URL,
      helpText:
        "Hindsight API base URL. Defaults to Hindsight Cloud; change it for a self-hosted instance (e.g. http://localhost:8888).",
    },
  ],
};
