"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Copy, KeyRound, Plus, RefreshCw, Save, Trash2, UserPlus } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { withBasePath } from "@/lib/base-path";
import {
  API_KEY_OPERATIONS,
  BANK_SCOPED_OPERATIONS,
  OPERATION_ACTIONS,
  OPERATION_GROUPS,
  UNSCOPED_DATAPLANE_OPERATIONS,
} from "@/lib/supabase-org/operations";
import type { ApiKeyOperation as ApiKeyOperationName } from "@/lib/supabase-org/operations";

type Role = "owner" | "admin" | "member";
type ApiKeyPermissionMode = "scoped" | "full_access";

interface Organization {
  id: string;
  name: string;
  role: Role;
}

interface Member {
  org_id: string;
  user_id: string;
  email?: string;
  role: Role;
}

interface Invite {
  id: string;
  email: string;
  role: Role;
  expires_at: string;
  accepted_at?: string | null;
  revoked_at?: string | null;
}

interface ApiKeySummary {
  id: string;
  name: string;
  permission_mode?: ApiKeyPermissionMode;
  allowed_operations?: string[] | null;
  operation_scopes?: Array<{
    operation: string;
    bank_scope_mode: "all" | "selected";
    scoped_bank_ids?: string[];
  }>;
  owned_banks?: Array<{
    bank_id: string;
    name?: string | null;
  }>;
  revoked_at?: string | null;
  created_at: string;
  can_view_secret?: boolean;
}

interface BankSummary {
  bank_id?: string;
  id?: string;
  name?: string;
}

interface VersionInfo {
  features?: {
    auth_provider?: string;
    profile_match?: boolean;
  };
}

const API_BASE = "/api/auth-profiles/supabase-org";
type BankScopeMode = "all" | "selected";
type BankScopeSelection = { mode: BankScopeMode; bankIds: string[] };
type OperationOverrideMap = Partial<Record<ApiKeyOperationName, BankScopeSelection>>;
const BANK_SCOPED_GROUPS = OPERATION_GROUPS.filter((group) => group.bankScoped);
const COPY: Record<string, string> = {
  organizationSettings: "Organization settings",
  noOrganizationSelected: "No organization selected",
  selectOrganization: "Select organization",
  organizationName: "Organization name",
  newOrganization: "New organization",
  inviteLinkCreated: "Invite link created",
  inviteLinkOneTime:
    "Copy this link now. It is only shown after creation and will not be available after you leave or refresh this page.",
  apiKeys: "API keys",
  keyName: "Key name",
  permissionMode: "Permission mode",
  fullAccess: "Full access",
  fullAccessDescription:
    "Dynamically follows the creator's current organization role across all banks.",
  scopedAccess: "Scoped access",
  scopedAccessDescription: "Choose allowed operations and bank scopes for this key.",
  allAllowedOperations: "All allowed operations",
  allAllowed: "All allowed",
  allCurrentAndFutureBanks: "All current and future banks",
  createBankPermissionDescription:
    "Allows this key to create new banks. While this permission remains enabled, " +
    "the key has full access to banks it created that still exist.",
  createdBanks: "Existing banks created by this key",
  defaultScope: "Default scope",
  noBanksYet: "No banks yet",
  noBanksCreatedByThisKey: "No existing banks were created by this key",
  overrideScope: "Override scope",
  selectedBanksOnly: "Selected banks only",
};

function buildOperationScopes(
  operations: ApiKeyOperationName[],
  groupScopes: Record<string, BankScopeSelection>,
  operationOverrides: OperationOverrideMap,
  excludedBankIds: readonly string[] = []
) {
  return operations.map((operation) => {
    const unscoped = UNSCOPED_DATAPLANE_OPERATIONS.includes(operation);
    const scope = unscoped ? null : scopeForOperation(operation, groupScopes, operationOverrides);
    return {
      operation,
      bank_scope_mode: unscoped ? "all" : scope?.mode,
      bank_ids: scope ? selectedBankIdsForScope(scope, excludedBankIds) : null,
    };
  });
}

function createDefaultGroupScopes(): Record<string, BankScopeSelection> {
  return Object.fromEntries(
    BANK_SCOPED_GROUPS.flatMap((group) =>
      group.sections
        ? group.sections.map((section) => [
            sectionScopeKey(group.id, section.id),
            { mode: "all", bankIds: [] },
          ])
        : [[group.id, { mode: "all", bankIds: [] }]]
    )
  );
}

function groupForOperation(operation: ApiKeyOperationName) {
  return OPERATION_GROUPS.find((group) => group.operations.includes(operation));
}

function sectionForOperation(operation: ApiKeyOperationName) {
  const group = groupForOperation(operation);
  return group?.sections?.find((section) => section.operations.includes(operation));
}

function sectionScopeKey(groupId: string, sectionId: string): string {
  return `${groupId}.${sectionId}`;
}

function scopeKeyForOperation(operation: ApiKeyOperationName): string | null {
  const group = groupForOperation(operation);
  if (!group) return null;
  const section = sectionForOperation(operation);
  return section ? sectionScopeKey(group.id, section.id) : group.id;
}

function scopeForOperation(
  operation: ApiKeyOperationName,
  groupScopes: Record<string, BankScopeSelection>,
  operationOverrides: OperationOverrideMap
): BankScopeSelection {
  const override = operationOverrides[operation];
  if (override) return override;
  const scopeKey = scopeKeyForOperation(operation);
  return (scopeKey && groupScopes[scopeKey]) || { mode: "all", bankIds: [] };
}

function selectedBankIdsForScope(
  scope: BankScopeSelection,
  excludedBankIds: readonly string[] = []
): string[] | null {
  if (scope.mode !== "selected") return null;
  const excluded = new Set(excludedBankIds);
  return scope.bankIds.filter((bankId) => !excluded.has(bankId));
}

function summarizeApiKeyBankScopes(apiKey: ApiKeySummary): string {
  const bankScopedOperations = (apiKey.operation_scopes ?? []).filter(
    (scope) => !UNSCOPED_DATAPLANE_OPERATIONS.includes(scope.operation)
  );
  if (bankScopedOperations.length === 0) return "no bank scopes";
  if (bankScopedOperations.some((scope) => scope.bank_scope_mode === "all")) return "all banks";
  const selectedBankIds = new Set(
    bankScopedOperations.flatMap((scope) => scope.scoped_bank_ids ?? [])
  );
  return `${selectedBankIds.size} banks`;
}

export default function SettingsPage() {
  const t = (key: string) => COPY[key] || key;
  const router = useRouter();
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [selectedOrgId, setSelectedOrgId] = useState("");
  const [members, setMembers] = useState<Member[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeySummary[]>([]);
  const [banks, setBanks] = useState<BankSummary[]>([]);
  const [orgName, setOrgName] = useState("");
  const [newOrgName, setNewOrgName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<Role>("member");
  const [apiKeyName, setApiKeyName] = useState("");
  const [apiKeyPermissionMode, setApiKeyPermissionMode] = useState<ApiKeyPermissionMode>("scoped");
  const [apiKeyOperations, setApiKeyOperations] = useState<ApiKeyOperationName[]>([]);
  const [apiKeyGroupScopes, setApiKeyGroupScopes] = useState<Record<string, BankScopeSelection>>(
    () => createDefaultGroupScopes()
  );
  const [apiKeyOperationOverrides, setApiKeyOperationOverrides] = useState<OperationOverrideMap>(
    {}
  );
  const [editingApiKeyId, setEditingApiKeyId] = useState<string | null>(null);
  const [editApiKeyPermissionMode, setEditApiKeyPermissionMode] =
    useState<ApiKeyPermissionMode>("scoped");
  const [editApiKeyOperations, setEditApiKeyOperations] = useState<ApiKeyOperationName[]>([]);
  const [editApiKeyGroupScopes, setEditApiKeyGroupScopes] = useState<
    Record<string, BankScopeSelection>
  >(() => createDefaultGroupScopes());
  const [editApiKeyOperationOverrides, setEditApiKeyOperationOverrides] =
    useState<OperationOverrideMap>({});
  const [newInviteLink, setNewInviteLink] = useState<string | null>(null);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const currentOrg = useMemo(
    () => organizations.find((organization) => organization.id === selectedOrgId),
    [organizations, selectedOrgId]
  );
  const canAdmin = currentOrg?.role === "owner" || currentOrg?.role === "admin";
  const canOwner = currentOrg?.role === "owner";
  const availableApiKeyOperations = useMemo(
    () => (currentOrg?.role === "member" ? [...BANK_SCOPED_OPERATIONS] : [...API_KEY_OPERATIONS]),
    [currentOrg?.role]
  );

  async function loadAll() {
    setLoading(true);
    try {
      const version = await fetchJson<VersionInfo>("/api/version");
      if (
        version.features?.auth_provider !== "supabase_org" ||
        version.features?.profile_match === false
      ) {
        router.replace("/dashboard");
        return;
      }

      const me = await fetchJson<{
        organizations: Organization[];
        current: { org_id: string; role: Role } | null;
      }>(`${API_BASE}/me`);
      setOrganizations(me.organizations);
      const nextOrgId = me.current?.org_id || me.organizations[0]?.id || "";
      setSelectedOrgId(nextOrgId);
      setOrgName(
        me.organizations.find((organization) => organization.id === nextOrgId)?.name || ""
      );
      const [team, inviteList, keyList, bankList] = await Promise.all([
        fetchJson<{ members: Member[] }>(`${API_BASE}/team`),
        fetchJson<{ invites: Invite[] }>(`${API_BASE}/team/invites`),
        fetchJson<{ api_keys: ApiKeySummary[] }>(`${API_BASE}/api-keys`),
        fetchJson<{ banks: BankSummary[] }>("/api/banks"),
      ]);
      setMembers(team.members);
      setInvites(inviteList.invites);
      setApiKeys(keyList.api_keys);
      setBanks(bankList.banks || []);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load settings");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  useEffect(() => {
    setApiKeyOperations((operations) =>
      operations.length === 0
        ? [...availableApiKeyOperations]
        : operations.filter((operation) => availableApiKeyOperations.includes(operation))
    );
  }, [availableApiKeyOperations]);

  async function createOrg(event: FormEvent) {
    event.preventDefault();
    const response = await fetchJson<{ organization: Organization }>(`${API_BASE}/organizations`, {
      method: "POST",
      body: JSON.stringify({ name: newOrgName }),
    });
    setNewOrgName("");
    setOrganizations((items) => [...items, { ...response.organization, role: "owner" }]);
    await selectOrganization(response.organization.id);
    toast.success("Organization created");
  }

  async function selectOrganization(orgId: string) {
    await fetchJson(`${API_BASE}/auth/select-org`, {
      method: "POST",
      body: JSON.stringify({ org_id: orgId }),
    });
    setSelectedOrgId(orgId);
    setOrgName(organizations.find((organization) => organization.id === orgId)?.name || "");
    await loadAll();
  }

  async function renameOrganization(event: FormEvent) {
    event.preventDefault();
    if (!selectedOrgId) return;
    const response = await fetchJson<{ organization: Organization }>(
      `${API_BASE}/organizations/${encodeURIComponent(selectedOrgId)}`,
      {
        method: "PATCH",
        body: JSON.stringify({ name: orgName }),
      }
    );
    setOrganizations((items) =>
      items.map((item) =>
        item.id === response.organization.id ? { ...item, name: response.organization.name } : item
      )
    );
    setOrgName(response.organization.name);
    toast.success("Organization updated");
  }

  async function inviteMember(event: FormEvent) {
    event.preventDefault();
    const response = await fetchJson<{ invite: { invite_url: string } }>(
      `${API_BASE}/team/invites`,
      {
        method: "POST",
        body: JSON.stringify({ email: inviteEmail, role: inviteRole }),
      }
    );
    setInviteEmail("");
    setNewInviteLink(response.invite.invite_url);
    try {
      await navigator.clipboard.writeText(response.invite.invite_url);
      toast.success("Invite link copied");
    } catch {
      toast.success("Invite link created");
    }
    await loadAll();
  }

  async function updateMember(userId: string, role: Role) {
    await fetchJson(`${API_BASE}/team/members/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    });
    await loadAll();
  }

  async function removeMember(userId: string) {
    await fetchJson(`${API_BASE}/team/members/${encodeURIComponent(userId)}`, { method: "DELETE" });
    await loadAll();
  }

  async function createApiKey(event: FormEvent) {
    event.preventDefault();
    const response = await fetchJson<{ api_key: { key: string } }>(`${API_BASE}/api-keys`, {
      method: "POST",
      body: JSON.stringify({
        name: apiKeyName,
        permission_mode: apiKeyPermissionMode,
        operation_scopes:
          apiKeyPermissionMode === "scoped"
            ? buildOperationScopes(apiKeyOperations, apiKeyGroupScopes, apiKeyOperationOverrides)
            : null,
      }),
    });
    setApiKeyName("");
    setApiKeyPermissionMode("scoped");
    setApiKeyOperations([...availableApiKeyOperations]);
    setApiKeyGroupScopes(createDefaultGroupScopes());
    setApiKeyOperationOverrides({});
    setNewApiKey(response.api_key.key);
    await loadAll();
  }

  function toggleApiKeyOperation(operation: ApiKeyOperationName, checked: boolean) {
    setApiKeyOperations((operations) =>
      checked
        ? Array.from(new Set([...operations, operation]))
        : operations.filter((item) => item !== operation)
    );
  }

  function startEditingApiKey(apiKey: ApiKeySummary) {
    const permissionMode = apiKey.permission_mode ?? "scoped";
    const operations =
      permissionMode === "full_access"
        ? [...availableApiKeyOperations]
        : ((apiKey.allowed_operations ?? []).filter((operation) =>
            availableApiKeyOperations.includes(operation as ApiKeyOperationName)
          ) as ApiKeyOperationName[]);
    const ownedBankIds = new Set((apiKey.owned_banks ?? []).map((bank) => bank.bank_id));
    const excludeOwnedBanks = (bankIds: string[] | undefined) =>
      (bankIds ?? []).filter((bankId) => !ownedBankIds.has(bankId));
    const nextGroupScopes = createDefaultGroupScopes();
    const nextOperationOverrides: OperationOverrideMap = {};
    for (const group of BANK_SCOPED_GROUPS) {
      const scopeUnits = group.sections ?? [
        { id: group.id, label: group.label, operations: group.operations },
      ];
      for (const scopeUnit of scopeUnits) {
        const scopeKey = group.sections ? sectionScopeKey(group.id, scopeUnit.id) : group.id;
        const unitOperations = scopeUnit.operations.filter((operation) =>
          operations.includes(operation)
        );
        const firstScope = apiKey.operation_scopes?.find(
          (scope) => scope.operation === unitOperations[0]
        );
        nextGroupScopes[scopeKey] = {
          mode: firstScope?.bank_scope_mode ?? "all",
          bankIds: excludeOwnedBanks(firstScope?.scoped_bank_ids),
        };
        for (const operation of unitOperations) {
          const scope = apiKey.operation_scopes?.find((item) => item.operation === operation);
          if (!scope || scope.operation === unitOperations[0]) continue;
          const inherited = nextGroupScopes[scopeKey];
          const bankIds = excludeOwnedBanks(scope.scoped_bank_ids);
          const differs =
            scope.bank_scope_mode !== inherited.mode ||
            bankIds.join("\u0000") !== inherited.bankIds.join("\u0000");
          if (differs) {
            nextOperationOverrides[operation] = { mode: scope.bank_scope_mode, bankIds };
          }
        }
      }
    }
    setEditingApiKeyId(apiKey.id);
    setEditApiKeyPermissionMode(permissionMode);
    setEditApiKeyOperations(operations);
    setEditApiKeyGroupScopes(nextGroupScopes);
    setEditApiKeyOperationOverrides(nextOperationOverrides);
  }

  function toggleEditApiKeyOperation(operation: ApiKeyOperationName, checked: boolean) {
    setEditApiKeyOperations((operations) =>
      checked
        ? Array.from(new Set([...operations, operation]))
        : operations.filter((item) => item !== operation)
    );
  }

  async function saveApiKeyPermissions(event: FormEvent) {
    event.preventDefault();
    if (!editingApiKeyId) return;
    const editingApiKey = apiKeys.find((apiKey) => apiKey.id === editingApiKeyId);
    await fetchJson(`${API_BASE}/api-keys/${encodeURIComponent(editingApiKeyId)}`, {
      method: "PATCH",
      body: JSON.stringify({
        permission_mode: editApiKeyPermissionMode,
        operation_scopes:
          editApiKeyPermissionMode === "scoped"
            ? buildOperationScopes(
                editApiKeyOperations,
                editApiKeyGroupScopes,
                editApiKeyOperationOverrides,
                (editingApiKey?.owned_banks ?? []).map((bank) => bank.bank_id)
              )
            : null,
      }),
    });
    setEditingApiKeyId(null);
    await loadAll();
  }

  function renderOperationGroups(
    selectedOperations: ApiKeyOperationName[],
    toggleOperation: (operation: ApiKeyOperationName, checked: boolean) => void,
    setOperations: (operations: ApiKeyOperationName[]) => void,
    groupScopes: Record<string, BankScopeSelection>,
    setGroupScopes: (scopes: Record<string, BankScopeSelection>) => void,
    operationOverrides: OperationOverrideMap,
    setOperationOverrides: (overrides: OperationOverrideMap) => void,
    ownedBanks?: ApiKeySummary["owned_banks"]
  ) {
    const availableSet = new Set<ApiKeyOperationName>(availableApiKeyOperations);
    const ownedBankIds = new Set((ownedBanks ?? []).map((bank) => bank.bank_id));
    const selectableBanks = banks.filter((bank) => {
      const bankId = bank.bank_id || bank.id || "";
      return bankId && !ownedBankIds.has(bankId);
    });
    const updateScope = (
      currentScope: BankScopeSelection,
      nextScope: Partial<BankScopeSelection>
    ): BankScopeSelection => ({
      mode: nextScope.mode ?? currentScope.mode,
      bankIds: nextScope.bankIds ?? currentScope.bankIds,
    });
    const toggleBank = (scope: BankScopeSelection, bankId: string, checked: boolean) =>
      updateScope(scope, {
        bankIds: checked
          ? Array.from(new Set([...scope.bankIds, bankId]))
          : scope.bankIds.filter((item) => item !== bankId),
      });
    const renderBankPicker = (
      scope: BankScopeSelection,
      onScopeChange: (scope: BankScopeSelection) => void
    ) =>
      scope.mode === "selected" ? (
        <div className="grid max-h-32 gap-2 overflow-y-auto pr-1 md:grid-cols-2">
          {selectableBanks.length === 0 ? (
            <span className="text-sm text-muted-foreground">{t("noBanksYet")}</span>
          ) : (
            selectableBanks.map((bank) => {
              const bankId = bank.bank_id || bank.id || "";
              if (!bankId) return null;
              return (
                <label key={bankId} className="flex min-w-0 items-center gap-2 text-sm">
                  <Checkbox
                    checked={scope.bankIds.includes(bankId)}
                    onCheckedChange={(checked) =>
                      onScopeChange(toggleBank(scope, bankId, checked === true))
                    }
                  />
                  <span className="truncate">{bank.name || bankId}</span>
                </label>
              );
            })
          )}
        </div>
      ) : null;
    const renderOperationCards = (
      operations: readonly ApiKeyOperationName[],
      group: (typeof OPERATION_GROUPS)[number],
      scopeKey: string
    ) => (
      <div className="grid gap-2 md:grid-cols-2">
        {operations.map((operation) => (
          <div key={operation} className="min-w-0 rounded border p-2">
            <label className="flex min-w-0 items-center gap-2 text-sm">
              <Checkbox
                checked={selectedOperations.includes(operation)}
                onCheckedChange={(checked) => toggleOperation(operation, checked === true)}
              />
              <span className="min-w-0 flex-1 truncate">{operation}</span>
              <span className="shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground">
                {OPERATION_ACTIONS[operation]}
              </span>
            </label>
            {group.bankScoped && selectedOperations.includes(operation) && (
              <div className="mt-2 space-y-2 pl-6">
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Checkbox
                    checked={Boolean(operationOverrides[operation])}
                    onCheckedChange={(checked) => {
                      const nextOverrides = { ...operationOverrides };
                      if (checked === true) {
                        nextOverrides[operation] = {
                          ...(groupScopes[scopeKey] ?? { mode: "all", bankIds: [] }),
                        };
                      } else {
                        delete nextOverrides[operation];
                      }
                      setOperationOverrides(nextOverrides);
                    }}
                  />
                  <span>{t("overrideScope")}</span>
                </label>
                {operationOverrides[operation] && (
                  <div className="space-y-2">
                    <Select
                      value={operationOverrides[operation]?.mode ?? "all"}
                      onValueChange={(value) =>
                        setOperationOverrides({
                          ...operationOverrides,
                          [operation]: updateScope(
                            operationOverrides[operation] ?? { mode: "all", bankIds: [] },
                            { mode: value as BankScopeMode }
                          ),
                        })
                      }
                    >
                      <SelectTrigger className="h-8 w-36">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">{t("allCurrentAndFutureBanks")}</SelectItem>
                        <SelectItem value="selected">{t("selectedBanksOnly")}</SelectItem>
                      </SelectContent>
                    </Select>
                    {renderBankPicker(
                      operationOverrides[operation] ?? { mode: "all", bankIds: [] },
                      (scope) =>
                        setOperationOverrides({ ...operationOverrides, [operation]: scope })
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    );
    const renderDefaultScope = (scopeKey: string) => (
      <div className="space-y-2 rounded-md bg-muted/30 p-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-muted-foreground">{t("defaultScope")}</span>
          <Select
            value={groupScopes[scopeKey]?.mode ?? "all"}
            onValueChange={(value) =>
              setGroupScopes({
                ...groupScopes,
                [scopeKey]: updateScope(groupScopes[scopeKey] ?? { mode: "all", bankIds: [] }, {
                  mode: value as BankScopeMode,
                }),
              })
            }
          >
            <SelectTrigger className="h-8 w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("allCurrentAndFutureBanks")}</SelectItem>
              <SelectItem value="selected">{t("selectedBanksOnly")}</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="text-xs text-muted-foreground">
          {groupScopes[scopeKey]?.mode === "selected"
            ? "Applies only to the selected banks."
            : "Applies to every current bank and any bank created later."}
        </div>
        {renderBankPicker(groupScopes[scopeKey] ?? { mode: "all", bankIds: [] }, (scope) =>
          setGroupScopes({ ...groupScopes, [scopeKey]: scope })
        )}
      </div>
    );
    return (
      <div className="space-y-3">
        {OPERATION_GROUPS.map((group) => {
          const operations = group.operations.filter((operation) => availableSet.has(operation));
          if (operations.length === 0) return null;
          const selectedCount = operations.filter((operation) =>
            selectedOperations.includes(operation)
          ).length;
          const singleUnscopedOperation = !group.bankScoped && operations.length === 1;
          const operation = operations[0];
          const isCreateBankOperation = singleUnscopedOperation && operation === "create_bank";
          const showOwnedBanks = isCreateBankOperation && ownedBanks !== undefined;
          return (
            <div key={group.id} className="rounded-md border p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                {singleUnscopedOperation ? (
                  <label className="flex min-w-0 items-center gap-2 text-sm font-medium">
                    <Checkbox
                      checked={selectedOperations.includes(operation)}
                      onCheckedChange={(checked) => toggleOperation(operation, checked === true)}
                    />
                    <span>{group.label}</span>
                  </label>
                ) : (
                  <>
                    <span className="text-sm font-medium">
                      {group.label} ({selectedCount}/{operations.length})
                    </span>
                    <div className="flex gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setOperations(Array.from(new Set([...selectedOperations, ...operations])))
                        }
                      >
                        All
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setOperations(
                            selectedOperations.filter(
                              (operation) => !operations.includes(operation)
                            )
                          )
                        }
                      >
                        None
                      </Button>
                    </div>
                  </>
                )}
              </div>
              {isCreateBankOperation ? (
                <div className="mt-2 pl-6 text-xs text-muted-foreground">
                  {t("createBankPermissionDescription")}
                </div>
              ) : null}
              {showOwnedBanks ? (
                <div className="mt-2 space-y-1 pl-6">
                  <div className="text-xs font-medium text-muted-foreground">
                    {t("createdBanks")}
                  </div>
                  {ownedBanks.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {ownedBanks.map((ownedBank) => {
                        const bank = banks.find(
                          (candidate) =>
                            candidate.bank_id === ownedBank.bank_id ||
                            candidate.id === ownedBank.bank_id
                        );
                        return (
                          <span
                            key={ownedBank.bank_id}
                            className="max-w-48 truncate rounded border px-1.5 py-0.5 text-xs text-muted-foreground"
                          >
                            {ownedBank.name || bank?.name || ownedBank.bank_id}
                          </span>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      {t("noBanksCreatedByThisKey")}
                    </div>
                  )}
                </div>
              ) : null}
              {group.bankScoped && !group.sections && (
                <div className="mb-3">{renderDefaultScope(group.id)}</div>
              )}
              {!singleUnscopedOperation && group.sections ? (
                <div className="space-y-3">
                  {group.sections.map((section) => {
                    const sectionOperations = section.operations.filter((operation) =>
                      operations.includes(operation)
                    );
                    if (sectionOperations.length === 0) return null;
                    const scopeKey = sectionScopeKey(group.id, section.id);
                    return (
                      <div key={section.label} className="space-y-2">
                        <div className="text-xs font-medium text-muted-foreground">
                          {section.label}
                        </div>
                        {renderDefaultScope(scopeKey)}
                        {renderOperationCards(sectionOperations, group, scopeKey)}
                      </div>
                    );
                  })}
                </div>
              ) : !singleUnscopedOperation ? (
                renderOperationCards(operations, group, group.id)
              ) : null}
            </div>
          );
        })}
      </div>
    );
  }

  async function revokeApiKey(id: string) {
    await fetchJson(`${API_BASE}/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadAll();
  }

  async function copyApiKey(id: string) {
    const response = await fetchJson<{ api_key: { key: string } }>(
      `${API_BASE}/api-keys/${encodeURIComponent(id)}`
    );
    await navigator.clipboard.writeText(response.api_key.key);
    toast.success("API key copied");
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">{t("organizationSettings")}</h1>
            <p className="text-sm text-muted-foreground">
              {currentOrg?.name || t("noOrganizationSelected")}
            </p>
          </div>
          <Button variant="outline" onClick={() => router.push("/dashboard")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Dashboard
          </Button>
        </div>

        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <Card>
            <CardHeader>
              <h2 className="text-lg font-medium">Organizations</h2>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={selectedOrgId} onValueChange={selectOrganization}>
                <SelectTrigger>
                  <SelectValue placeholder={t("selectOrganization")} />
                </SelectTrigger>
                <SelectContent>
                  {organizations.map((organization) => (
                    <SelectItem key={organization.id} value={organization.id}>
                      {organization.name} ({organization.role})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <form className="flex gap-2" onSubmit={renameOrganization}>
                <Input
                  value={orgName}
                  onChange={(event) => setOrgName(event.target.value)}
                  placeholder={t("organizationName")}
                  disabled={!canOwner}
                />
                <Button
                  type="submit"
                  size="icon"
                  disabled={!canOwner || !orgName.trim() || orgName.trim() === currentOrg?.name}
                >
                  <Save className="h-4 w-4" />
                </Button>
              </form>
              <form className="flex gap-2" onSubmit={createOrg}>
                <Input
                  value={newOrgName}
                  onChange={(event) => setNewOrgName(event.target.value)}
                  placeholder={t("newOrganization")}
                />
                <Button type="submit" size="icon" disabled={!newOrgName.trim()}>
                  <Plus className="h-4 w-4" />
                </Button>
              </form>
              <Button variant="outline" className="w-full" onClick={loadAll} disabled={loading}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium">Team</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                {canAdmin && (
                  <form
                    className="grid gap-2 md:grid-cols-[1fr_150px_auto]"
                    onSubmit={inviteMember}
                  >
                    <Input
                      type="email"
                      value={inviteEmail}
                      onChange={(event) => setInviteEmail(event.target.value)}
                      placeholder="Email"
                    />
                    <Select
                      value={inviteRole}
                      onValueChange={(value) => setInviteRole(value as Role)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="member">member</SelectItem>
                        <SelectItem value="admin">admin</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button type="submit" disabled={!inviteEmail.trim()}>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Invite
                    </Button>
                  </form>
                )}

                {newInviteLink && (
                  <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                    <div className="font-medium">{t("inviteLinkCreated")}</div>
                    <p>{t("inviteLinkOneTime")}</p>
                    <div className="flex items-center gap-2">
                      <code className="min-w-0 flex-1 truncate">{newInviteLink}</code>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => navigator.clipboard.writeText(newInviteLink)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}

                <div className="divide-y rounded-md border">
                  {members.map((member) => (
                    <div
                      key={member.user_id}
                      className="grid gap-3 p-3 md:grid-cols-[1fr_150px_auto] md:items-center"
                    >
                      <div>
                        <div className="font-medium">{member.email || member.user_id}</div>
                        <div className="text-xs text-muted-foreground">{member.user_id}</div>
                      </div>
                      <Select
                        value={member.role}
                        disabled={!canOwner}
                        onValueChange={(value) => updateMember(member.user_id, value as Role)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="owner">owner</SelectItem>
                          <SelectItem value="admin">admin</SelectItem>
                          <SelectItem value="member">member</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!canOwner}
                        onClick={() => removeMember(member.user_id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>

                {invites.length > 0 && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">Invites</h3>
                    {invites.map((invite) => (
                      <div
                        key={invite.id}
                        className="flex items-center justify-between rounded-md border p-3 text-sm"
                      >
                        <span>
                          {invite.email} ({invite.role})
                        </span>
                        <span className="text-muted-foreground">
                          {invite.revoked_at
                            ? "revoked"
                            : invite.accepted_at
                              ? "accepted"
                              : "pending"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium">{t("apiKeys")}</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                <form className="space-y-3" onSubmit={createApiKey}>
                  <div className="grid gap-2 md:grid-cols-[1fr_auto]">
                    <Input
                      value={apiKeyName}
                      onChange={(event) => setApiKeyName(event.target.value)}
                      placeholder={t("keyName")}
                    />
                    <Button
                      type="submit"
                      disabled={
                        !apiKeyName.trim() ||
                        (apiKeyPermissionMode === "scoped" && apiKeyOperations.length === 0)
                      }
                    >
                      <KeyRound className="mr-2 h-4 w-4" />
                      Create
                    </Button>
                  </div>
                  <div className="space-y-2 rounded-md border p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium">{t("permissionMode")}</span>
                      <Select
                        value={apiKeyPermissionMode}
                        onValueChange={(value) =>
                          setApiKeyPermissionMode(value as ApiKeyPermissionMode)
                        }
                      >
                        <SelectTrigger className="h-8 w-40">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="full_access">{t("fullAccess")}</SelectItem>
                          <SelectItem value="scoped">{t("scopedAccess")}</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {apiKeyPermissionMode === "full_access"
                        ? t("fullAccessDescription")
                        : t("scopedAccessDescription")}
                    </p>
                  </div>
                  {apiKeyPermissionMode === "scoped" ? (
                    <div className="rounded-md border p-3">
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <span className="text-sm font-medium">Operations</span>
                        <div className="flex gap-1">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setApiKeyOperations([...availableApiKeyOperations])}
                          >
                            {t("allAllowed")}
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setApiKeyOperations([])}
                          >
                            None
                          </Button>
                        </div>
                      </div>
                      <div className="max-h-72 overflow-y-auto pr-1">
                        {renderOperationGroups(
                          apiKeyOperations,
                          toggleApiKeyOperation,
                          setApiKeyOperations,
                          apiKeyGroupScopes,
                          setApiKeyGroupScopes,
                          apiKeyOperationOverrides,
                          setApiKeyOperationOverrides
                        )}
                      </div>
                    </div>
                  ) : null}
                </form>
                {newApiKey && (
                  <div className="flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                    <code className="min-w-0 flex-1 truncate">{newApiKey}</code>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => navigator.clipboard.writeText(newApiKey)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                )}
                <div className="divide-y rounded-md border">
                  {apiKeys.map((apiKey) => (
                    <div
                      key={apiKey.id}
                      className="grid gap-3 p-3 md:grid-cols-[1fr_160px_auto] md:items-center"
                    >
                      <div>
                        <div className="font-medium">{apiKey.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {(apiKey.permission_mode ?? "scoped") === "full_access"
                            ? `${t("fullAccess")} · follows creator role · all banks`
                            : apiKey.allowed_operations?.length
                              ? `${apiKey.allowed_operations.length} operations · ${summarizeApiKeyBankScopes(apiKey)}`
                              : `${t("allAllowedOperations")} · ${summarizeApiKeyBankScopes(apiKey)}`}
                          {apiKey.owned_banks?.length
                            ? ` · ${apiKey.owned_banks.length} existing owned`
                            : ""}
                        </div>
                        {apiKey.owned_banks?.length ? (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {apiKey.owned_banks.map((bank) => (
                              <span
                                key={bank.bank_id}
                                className="max-w-48 truncate rounded border px-1.5 py-0.5 text-xs text-muted-foreground"
                              >
                                {bank.name || bank.bank_id}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {apiKey.revoked_at ? "revoked" : "active"}
                      </span>
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          disabled={Boolean(apiKey.revoked_at)}
                          onClick={() => startEditingApiKey(apiKey)}
                        >
                          <Save className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          disabled={!apiKey.can_view_secret || Boolean(apiKey.revoked_at)}
                          onClick={() => copyApiKey(apiKey.id)}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          disabled={Boolean(apiKey.revoked_at)}
                          onClick={() => revokeApiKey(apiKey.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                      {editingApiKeyId === apiKey.id && (
                        <form
                          className="space-y-3 rounded-md border p-3 md:col-span-3"
                          onSubmit={saveApiKeyPermissions}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">Permissions</span>
                              <Select
                                value={editApiKeyPermissionMode}
                                onValueChange={(value) =>
                                  setEditApiKeyPermissionMode(value as ApiKeyPermissionMode)
                                }
                              >
                                <SelectTrigger className="h-8 w-40">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="full_access">{t("fullAccess")}</SelectItem>
                                  <SelectItem value="scoped">{t("scopedAccess")}</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                onClick={() => setEditingApiKeyId(null)}
                              >
                                Cancel
                              </Button>
                              <Button
                                type="submit"
                                size="sm"
                                disabled={
                                  editApiKeyPermissionMode === "scoped" &&
                                  editApiKeyOperations.length === 0
                                }
                              >
                                Save
                              </Button>
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            {editApiKeyPermissionMode === "full_access"
                              ? t("fullAccessDescription")
                              : t("scopedAccessDescription")}
                          </p>
                          {editApiKeyPermissionMode === "scoped" ? (
                            <div className="max-h-72 overflow-y-auto pr-1">
                              {renderOperationGroups(
                                editApiKeyOperations,
                                toggleEditApiKeyOperation,
                                setEditApiKeyOperations,
                                editApiKeyGroupScopes,
                                setEditApiKeyGroupScopes,
                                editApiKeyOperationOverrides,
                                setEditApiKeyOperationOverrides,
                                apiKey.owned_banks ?? []
                              )}
                            </div>
                          ) : null}
                        </form>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}

async function fetchJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(withBasePath(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || `Request failed: ${response.status}`);
  return data as T;
}
