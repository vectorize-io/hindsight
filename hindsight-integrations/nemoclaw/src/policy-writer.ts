import yaml from "js-yaml";
import type { SandboxPolicy } from "./types.js";
import { HINDSIGHT_POLICY_NAME, HINDSIGHT_HOST, OPENCLAW_BINARY } from "./types.js";

const HINDSIGHT_NETWORK_POLICY = {
  name: HINDSIGHT_POLICY_NAME,
  endpoints: [
    {
      host: HINDSIGHT_HOST,
      port: 443,
      protocol: "rest",
      tls: "terminate",
      enforcement: "enforce",
      rules: [
        { allow: { method: "GET", path: "/**" } },
        { allow: { method: "POST", path: "/**" } },
        { allow: { method: "PUT", path: "/**" } },
      ],
    },
  ],
  binaries: [{ path: OPENCLAW_BINARY }],
};

/**
 * Returns true if the policy already has a correct Hindsight network policy entry.
 */
export function hasHindsightPolicy(policy: SandboxPolicy): boolean {
  const np = policy.network_policies?.[HINDSIGHT_POLICY_NAME];
  if (!np) return false;
  return np.endpoints?.some((e) => e.host === HINDSIGHT_HOST) ?? false;
}

/**
 * Merge the Hindsight network policy block into a SandboxPolicy.
 * Idempotent — if the block already exists and is correct, returns policy unchanged.
 */
export function mergeHindsightPolicy(policy: SandboxPolicy): SandboxPolicy {
  const updated: SandboxPolicy = {
    ...policy,
    network_policies: {
      ...(policy.network_policies ?? {}),
      [HINDSIGHT_POLICY_NAME]: HINDSIGHT_NETWORK_POLICY,
    },
  };
  return updated;
}

/**
 * Serialize a SandboxPolicy to a YAML string suitable for `openshell policy set`.
 */
export function serializePolicy(policy: SandboxPolicy): string {
  return yaml.dump(policy, {
    indent: 2,
    lineWidth: -1,
    noRefs: true,
    quotingType: '"',
    forceQuotes: false,
  });
}
