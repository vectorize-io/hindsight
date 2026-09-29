/**
 * Local shim for Flowise types and helpers.
 *
 * Flowise tool nodes live inside the FlowiseAI/Flowise monorepo at
 * `packages/components/nodes/tools/<NodeName>/`, where they import types
 * via deep relative paths (`'../../../src/Interface'` and
 * `'../../../src/utils'`).
 *
 * To keep our source files copy-1:1 ready for upstream submission, we
 * redirect those relative imports to this shim during local builds and
 * tests via the path alias in vitest.config.ts and tsconfig.json. The
 * shim mirrors the subset of the upstream API our nodes use.
 */

export interface ICommonObject {
  [key: string]: unknown;
}

export interface INodeParams {
  label: string;
  name: string;
  type: string;
  default?: unknown;
  description?: string;
  optional?: boolean;
  rows?: number;
  options?: Array<{ label: string; name: string }>;
  placeholder?: string;
  credentialNames?: string[];
}

export interface INodeData {
  inputs?: ICommonObject;
  credential?: string;
}

export interface INode {
  label: string;
  name: string;
  version: number;
  description: string;
  type: string;
  icon: string;
  category: string;
  baseClasses: string[];
  credential?: INodeParams;
  inputs?: INodeParams[];
  init(nodeData: INodeData, _input: string, options: ICommonObject): Promise<unknown>;
}

export interface INodeCredential {
  label: string;
  name: string;
  version: number;
  description?: string;
  inputs: INodeParams[];
}

// Upstream `getBaseClasses` walks the prototype chain to enumerate parent
// class names. For our purposes (a `DynamicStructuredTool` from
// `@langchain/core/tools`) the chain is well-known, so the shim returns the
// expected names. The real upstream implementation handles arbitrary inputs.
export function getBaseClasses(_target: unknown): string[] {
  return ["DynamicStructuredTool", "StructuredTool", "BaseTool", "Tool"];
}

// Stub for credential resolution. Upstream this fetches the user's saved
// credential record from the encrypted store. In tests we let callers
// inject a credential dict via `nodeData.credential` (parsed as JSON) so we
// can drive `init()` without a real store.
export async function getCredentialData(
  credentialId: string,
  _options: ICommonObject
): Promise<ICommonObject> {
  if (!credentialId) return {};
  try {
    return JSON.parse(credentialId) as ICommonObject;
  } catch {
    return {};
  }
}

// Upstream this prefers `nodeData.inputs` over the credential record. The
// shim keeps the same precedence so tests cover the same code path.
export function getCredentialParam(
  paramName: string,
  credentialData: ICommonObject,
  nodeData: INodeData
): string | undefined {
  const fromInputs = nodeData?.inputs?.[paramName];
  if (typeof fromInputs === "string" && fromInputs.length > 0) return fromInputs;
  const fromCred = credentialData[paramName];
  return typeof fromCred === "string" ? fromCred : undefined;
}
