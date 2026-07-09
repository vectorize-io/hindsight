import { OPERATION_UI_GROUPS } from "@/lib/supabase-org/operation-ui";
import operationManifest from "../../../../hindsight-api-slim/hindsight_api/extensions/builtin/supabase_authz_operations.json";

export type ApiKeyOperation = string;
export type OperationAction = "read" | "write";
export type OperationSource = "bank_read" | "bank_write" | "special_bank" | "unscoped";
export type OperationScope = "bank" | "unscoped";

export interface OperationDefinition {
  name: ApiKeyOperation;
  source: OperationSource;
  action: OperationAction;
  scope: OperationScope;
}

export interface OperationSection {
  id: string;
  labelKey: string;
  label: string;
  operations: readonly ApiKeyOperation[];
}

export interface OperationGroup {
  id: string;
  labelKey: string;
  label: string;
  bankScoped: boolean;
  operations: readonly ApiKeyOperation[];
  sections?: readonly OperationSection[];
}

const BANK_READ_OPERATION_NAMES = operationManifest.bank_read;
const BANK_WRITE_OPERATION_NAMES = operationManifest.bank_write;
const SPECIAL_BANK_OPERATION_DEFINITIONS: OperationDefinition[] =
  operationManifest.special_bank.map(({ name, action }) => ({
    name,
    source: "special_bank",
    action: action as OperationAction,
    scope: "bank",
  }));
const UNSCOPED_OPERATION_DEFINITIONS: OperationDefinition[] = operationManifest.unscoped.map(
  ({ name, action }) => ({
    name,
    source: "unscoped",
    action: action as OperationAction,
    scope: "unscoped",
  })
);

function definitionsForSource(
  names: readonly ApiKeyOperation[],
  source: Extract<OperationSource, "bank_read" | "bank_write">,
  action: OperationAction
): OperationDefinition[] {
  return names.map((name) => ({ name, source, action, scope: "bank" }));
}

const operationDefinitions: OperationDefinition[] = [
  ...definitionsForSource(BANK_READ_OPERATION_NAMES, "bank_read", "read"),
  ...definitionsForSource(BANK_WRITE_OPERATION_NAMES, "bank_write", "write"),
  ...SPECIAL_BANK_OPERATION_DEFINITIONS,
  ...UNSCOPED_OPERATION_DEFINITIONS,
];

export const OPERATION_DEFINITIONS = operationDefinitions;
export const BANK_READ_OPERATIONS = operationDefinitions
  .filter((operation) => operation.source === "bank_read")
  .map((operation) => operation.name);
export const BANK_WRITE_OPERATIONS = operationDefinitions
  .filter((operation) => operation.source === "bank_write")
  .map((operation) => operation.name);
export const SPECIAL_BANK_OPERATIONS = operationDefinitions
  .filter((operation) => operation.source === "special_bank")
  .map((operation) => operation.name);
export const UNSCOPED_DATAPLANE_OPERATIONS = operationDefinitions
  .filter((operation) => operation.scope === "unscoped")
  .map((operation) => operation.name);
export const BANK_SCOPED_OPERATIONS = operationDefinitions
  .filter((operation) => operation.scope === "bank")
  .map((operation) => operation.name);
export const API_KEY_OPERATIONS = operationDefinitions.map((operation) => operation.name);

export const OPERATION_ACTIONS = Object.fromEntries(
  operationDefinitions.map((operation) => [operation.name, operation.action])
) as Record<ApiKeyOperation, OperationAction>;

export const OPERATION_GROUPS: OperationGroup[] = OPERATION_UI_GROUPS.map((group) => {
  const sections = group.sections?.map((section) => ({
    id: section.id,
    labelKey: section.labelKey,
    label: section.defaultLabel,
    operations: section.operations,
  }));
  return {
    id: group.id,
    labelKey: group.labelKey,
    label: group.defaultLabel,
    bankScoped: group.bankScoped,
    operations: group.operations,
    sections,
  };
});
