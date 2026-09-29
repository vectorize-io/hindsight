"use strict";

/**
 * Request/response middleware shared by every operation.
 *
 * `addBearerHeader` injects the Hindsight API key on every outbound request so
 * individual operations don't have to. `handleHttpError` turns non-2xx
 * responses into typed Zapier errors with a useful message.
 */

const addBearerHeader = (request, z, bundle) => {
  if (bundle.authData && bundle.authData.apiKey) {
    request.headers = request.headers || {};
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
