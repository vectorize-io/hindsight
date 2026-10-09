"use strict";

/**
 * Request/response middleware shared by every operation.
 *
 * `addBearerHeader` injects the Hindsight API key and the integration's
 * User-Agent on every outbound request so individual operations don't have to.
 * `handleHttpError` turns non-2xx responses into typed Zapier errors with a
 * useful message.
 */

const { version } = require("./package.json");

// Keep "Zapier" in the value: the platform uses it to tell z.request() traffic
// apart and would otherwise log every request twice.
const USER_AGENT = `hindsight-zapier/${version} Zapier`;

const addBearerHeader = (request, z, bundle) => {
  request.headers = request.headers || {};
  request.headers["user-agent"] = USER_AGENT;
  if (bundle.authData && bundle.authData.apiKey) {
    request.headers.Authorization = `Bearer ${bundle.authData.apiKey}`;
  }
  return request;
};

const handleHttpError = (response, z) => {
  if (response.status === 401 || response.status === 403) {
    throw new z.errors.Error(
      "Invalid or unauthorized Hindsight API key.",
      "AuthenticationError",
      response.status
    );
  }
  if (response.status >= 400) {
    throw new z.errors.Error(
      `Hindsight API error ${response.status}: ${response.content}`,
      "ApiError",
      response.status
    );
  }
  return response;
};

module.exports = { addBearerHeader, handleHttpError };
