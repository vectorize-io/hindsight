"use strict";

const { baseUrl } = require("../utils");

/**
 * Hidden trigger that powers the "Bank" dynamic dropdown used by every action
 * and trigger. Referenced as `dynamic: 'bankList.bank_id.name'`.
 *
 * GET /v1/default/banks -> { banks: [{ bank_id, name, ... }] }
 */
const perform = async (z, bundle) => {
  const response = await z.request({ url: `${baseUrl(bundle)}/v1/default/banks` });
  // Zapier requires an `id` on every trigger result (its dedup key); bank_id is
  // unique, so reuse it. The dropdown ref `bankList.bank_id.name` uses bank_id
  // as the value and name as the label.
  return (response.data.banks || []).map((b) => ({
    id: b.bank_id,
    bank_id: b.bank_id,
    name: b.name || b.bank_id,
  }));
};

module.exports = {
  key: "bankList",
  noun: "Bank",
  display: {
    label: "List Banks",
    description: "Internal trigger that populates the Bank dropdown.",
    hidden: true,
  },
  operation: {
    perform,
    canPaginate: false,
    sample: { bank_id: "user-123", name: "User 123" },
  },
};
