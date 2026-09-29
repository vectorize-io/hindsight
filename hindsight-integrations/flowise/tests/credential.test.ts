import { describe, expect, it } from "vitest";

import * as credModule from "../credentials/HindsightApi.credential";

const HindsightApi = (credModule as any).credClass;

describe("HindsightApi credential", () => {
  const cred = new HindsightApi();

  it("declares the expected name and label", () => {
    expect(cred.name).toBe("hindsightApi");
    expect(cred.label).toBe("Hindsight API");
    expect(cred.version).toBe(1.0);
  });

  it("exposes apiUrl and apiKey inputs", () => {
    const names = cred.inputs.map((i: { name: string }) => i.name);
    expect(names).toContain("apiUrl");
    expect(names).toContain("apiKey");
  });

  it("defaults apiUrl to Hindsight Cloud", () => {
    const apiUrl = cred.inputs.find((i: { name: string }) => i.name === "apiUrl");
    expect(apiUrl.default).toBe("https://api.hindsight.vectorize.io");
  });

  it("marks apiKey as a password input", () => {
    const apiKey = cred.inputs.find((i: { name: string }) => i.name === "apiKey");
    expect(apiKey.type).toBe("password");
  });

  it("marks apiKey as optional (self-hosted unauthenticated case)", () => {
    const apiKey = cred.inputs.find((i: { name: string }) => i.name === "apiKey");
    expect(apiKey.optional).toBe(true);
  });
});
