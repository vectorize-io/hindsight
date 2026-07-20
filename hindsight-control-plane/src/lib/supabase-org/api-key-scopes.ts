import { createDataplaneClientForRequest, sdk } from "@/lib/hindsight-client";
import type {
  ApiKeyBankScopeInput,
  ApiKeyOperation,
  ApiKeyOperationScopeInput,
} from "@/lib/supabase-org/store";

interface BankListItem {
  bank_id: string;
  internal_id?: string | null;
}

export interface ApiKeyOperationScopeRequest {
  operation?: string;
  bank_scope_mode?: "all" | "selected";
  bank_ids?: string[] | null;
}

export async function resolveOperationScopes(
  request: Request,
  scopes: ApiKeyOperationScopeRequest[] | null
): Promise<ApiKeyOperationScopeInput[] | null> {
  if (!scopes) return null;
  const banks = await listCurrentBanks(request);
  const bankById = new Map(
    banks
      .filter((bank): bank is BankListItem & { internal_id: string } => Boolean(bank.internal_id))
      .map((bank) => [bank.bank_id, bank])
  );

  return scopes.map((scope) => {
    if (!scope.operation) throw new Error("operation is required");
    const bankScopeMode = scope.bank_scope_mode ?? "all";
    return {
      operation: scope.operation as ApiKeyOperation,
      bank_scope_mode: bankScopeMode,
      bank_scopes:
        bankScopeMode === "selected" ? resolveBankScopes(bankById, scope.bank_ids ?? []) : [],
    };
  });
}

function resolveBankScopes(
  bankById: ReadonlyMap<string, BankListItem & { internal_id: string }>,
  bankIds: string[]
): ApiKeyBankScopeInput[] {
  const uniqueBankIds = Array.from(new Set(bankIds.map((bankId) => bankId.trim()).filter(Boolean)));
  return uniqueBankIds.map((bankId) => {
    const bank = bankById.get(bankId);
    if (!bank) throw new Error(`Selected bank does not exist: ${bankId}`);
    return { bank_id: bank.bank_id, bank_internal_id: bank.internal_id };
  });
}

async function listCurrentBanks(request: Request): Promise<BankListItem[]> {
  const response = await sdk.listBanks({ client: createDataplaneClientForRequest(request) });
  if (response.error || !response.data) throw new Error("Failed to resolve selected banks");
  return (response.data as { banks?: BankListItem[] }).banks ?? [];
}
