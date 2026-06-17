"use strict";

const DEFAULT_API_URL = "https://api.hindsight.vectorize.io";

/**
 * Base URL for API calls. Defaults to Hindsight Cloud; trailing slash is
 * stripped so path concatenation never produces a double slash (mirrors the
 * n8n node's URL handling).
 */
const baseUrl = (bundle) =>
  ((bundle.authData && bundle.authData.apiUrl) || DEFAULT_API_URL).replace(/\/$/, "");

/** Parse a comma-separated tag string into a trimmed, non-empty array. */
const parseTags = (raw) =>
  !raw
    ? []
    : String(raw)
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);

module.exports = {
  DEFAULT_API_URL,
  baseUrl,
  parseTags,
  enc: encodeURIComponent,
};
