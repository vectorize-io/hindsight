/**
 * Edge-case and interaction tests for bank ID derivation.
 *
 * These tests cover scenarios where the two new features (per-user isolation
 * and static shared banks) interact, plus boundary conditions not covered by
 * the main test suite.
 */

import { describe, expect, it } from "vitest";
import { deriveBankId, extractUserFromIssue } from "../src/bank.js";

// ---------------------------------------------------------------------------
// Feature interaction: bankId + user granularity
// ---------------------------------------------------------------------------

describe("feature interaction: static bankId vs user granularity", () => {
  it("static bankId wins over user granularity when dynamicBankId is not true", () => {
    // If someone sets both bankId AND user granularity, static wins.
    // This is intentional — bankId is an explicit override.
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1", userId: "alice@acme.com" },
        {
          bankId: "shared-team-bank",
          bankGranularity: ["company", "agent", "user"],
        }
      )
    ).toBe("shared-team-bank");
  });

  it("dynamicBankId=true restores user granularity even when bankId is set", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1", userId: "alice@acme.com" },
        {
          bankId: "shared-team-bank",
          dynamicBankId: true,
          bankGranularity: ["company", "agent", "user"],
        }
      )
    ).toBe("paperclip::co-1::ag-1::user::alice@acme.com");
  });

  it("dynamicBankId=true with user granularity but no userId omits user segment", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        {
          bankId: "shared-team-bank",
          dynamicBankId: true,
          bankGranularity: ["company", "agent", "user"],
        }
      )
    ).toBe("paperclip::co-1::ag-1");
  });
});

// ---------------------------------------------------------------------------
// Static bankId edge cases
// ---------------------------------------------------------------------------

describe("static bankId edge cases", () => {
  it("bankId with special characters is returned verbatim (no encoding)", () => {
    expect(
      deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankId: "team::project::shared" })
    ).toBe("team::project::shared");
  });

  it("dynamicBankId=false explicitly is same as omitting it", () => {
    const withFalse = deriveBankId(
      { companyId: "co-1", agentId: "ag-1" },
      { bankId: "my-bank", dynamicBankId: false }
    );
    const withOmit = deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankId: "my-bank" });
    expect(withFalse).toBe(withOmit);
    expect(withFalse).toBe("my-bank");
  });

  it("bankId without dynamicBankId set defaults to static", () => {
    // Critical backwards-compat check: if someone sets only bankId,
    // they expect it to be used. dynamicBankId defaults to undefined,
    // which is !== true, so static path should activate.
    expect(deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankId: "spool-farm" })).toBe(
      "spool-farm"
    );
  });

  it("empty string bankId falls through to dynamic", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankId: "", bankGranularity: ["company"] }
      )
    ).toBe("paperclip::co-1");
  });

  it("bankId of only tabs/newlines falls through to dynamic", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankId: "\t\n  \t", bankGranularity: ["agent"] }
      )
    ).toBe("paperclip::ag-1");
  });
});

// ---------------------------------------------------------------------------
// Dynamic derivation edge cases
// ---------------------------------------------------------------------------

describe("dynamic derivation edge cases", () => {
  it("empty granularity array produces only the prefix", () => {
    expect(deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankGranularity: [] })).toBe(
      "paperclip"
    );
  });

  it("user-only granularity with userId", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1", userId: "alice@acme.com" },
        { bankGranularity: ["user"] }
      )
    ).toBe("paperclip::user::alice@acme.com");
  });

  it("user-only granularity without userId produces only prefix", () => {
    expect(
      deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankGranularity: ["user"] })
    ).toBe("paperclip");
  });

  it("duplicate granularity fields are handled (company twice)", () => {
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankGranularity: ["company", "company"] }
      )
    ).toBe("paperclip::co-1::co-1");
  });

  it("no config at all uses default granularity", () => {
    expect(deriveBankId({ companyId: "co-1", agentId: "ag-1" }, {})).toBe("paperclip::co-1::ag-1");
  });
});

// ---------------------------------------------------------------------------
// extractUserFromIssue edge cases
// ---------------------------------------------------------------------------

describe("extractUserFromIssue edge cases", () => {
  it("null originId returns undefined", () => {
    expect(extractUserFromIssue({ originId: null })).toBeUndefined();
  });

  it("null creatorEmail falls through to originId", () => {
    expect(extractUserFromIssue({ creatorEmail: null, originId: "slack::alice@acme.com" })).toBe(
      "alice@acme.com"
    );
  });

  it("empty string creatorEmail falls through to originId", () => {
    // creatorEmail is truthy check, empty string is falsy
    expect(extractUserFromIssue({ creatorEmail: "", originId: "slack::bob@co.io" })).toBe(
      "bob@co.io"
    );
  });

  it("originId with no separators but containing @ is returned", () => {
    expect(extractUserFromIssue({ originId: "user@example.com" })).toBe("user@example.com");
  });

  it("originId with multiple emails picks the last one", () => {
    // Scans backwards — last segment with @ wins
    expect(extractUserFromIssue({ originId: "admin@corp.io::slack::alice@acme.com" })).toBe(
      "alice@acme.com"
    );
  });

  it("both null returns undefined", () => {
    expect(extractUserFromIssue({ originId: null, creatorEmail: null })).toBeUndefined();
  });
});
