import { NextRequest, NextResponse } from "next/server";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth-profiles/supabase-org/constants";
import { getControlPlaneAuthProvider, getOrgCreationPolicy } from "@/lib/auth/provider";
import { sessionCookieOptions } from "@/lib/auth/session";
import {
  API_KEY_OPERATIONS,
  BANK_READ_OPERATIONS,
  BANK_WRITE_OPERATIONS,
  SPECIAL_BANK_OPERATIONS,
  UNSCOPED_DATAPLANE_OPERATIONS,
} from "@/lib/supabase-org/operations";
import type { ApiKeyOperation } from "@/lib/supabase-org/operations";

export type OrganizationRole = "owner" | "admin" | "member";
export type ApiKeyPermissionMode = "scoped" | "full_access";
export type ApiKeyBankScopeMode = "all" | "selected";
export interface ApiKeyBankScopeInput {
  bank_id: string;
  bank_internal_id: string;
}
export interface ApiKeyOperationScopeInput {
  operation: ApiKeyOperation;
  bank_scope_mode?: ApiKeyBankScopeMode;
  bank_scopes?: ApiKeyBankScopeInput[] | null;
}
export interface ApiKeyOperationScopeSummary {
  operation: ApiKeyOperation;
  bank_scope_mode: ApiKeyBankScopeMode;
  scoped_bank_ids?: string[];
  scoped_bank_internal_ids?: string[];
}
export type { ApiKeyOperation };

export interface SupabaseUser {
  id: string;
  email: string;
}

export interface SupabasePasswordSession {
  access_token: string;
  refresh_token?: string;
  expires_in?: number;
  user: SupabaseUser;
}

export interface SupabaseRefreshedSession {
  access_token: string;
  refresh_token?: string;
  expires_in?: number;
  user?: SupabaseUser;
}

export interface OrganizationMembership {
  id?: string;
  org_id: string;
  user_id: string;
  email?: string;
  role: OrganizationRole;
  created_at?: string;
  removed_at?: string | null;
  removed_by_user_id?: string | null;
}

export interface Organization {
  id: string;
  name: string;
  config?: Record<string, unknown>;
}

export interface OrganizationInvite {
  id: string;
  org_id: string;
  email: string;
  role: OrganizationRole;
  expires_at: string;
  accepted_at?: string | null;
  revoked_at?: string | null;
  created_by_user_id: string;
  created_at: string;
}

export interface HindsightApiKeySummary {
  id: string;
  org_id: string;
  created_by_user_id?: string | null;
  name: string;
  permission_mode?: ApiKeyPermissionMode;
  allowed_operations?: ApiKeyOperation[] | null;
  operation_scopes?: ApiKeyOperationScopeSummary[];
  owned_banks?: OwnedBankSummary[];
  expires_at?: string | null;
  revoked_at?: string | null;
  created_at: string;
  can_view_secret?: boolean;
}

export interface OwnedBankSummary {
  bank_id: string;
  bank_internal_id: string;
  created_at?: string | null;
  name?: string | null;
}

export interface CurrentOrgContext {
  user: SupabaseUser;
  selectedOrgId: string;
  membership: OrganizationMembership;
}

const API_KEY_PREFIX = "hs_";
const ORGANIZATION_ROLES: OrganizationRole[] = ["owner", "admin", "member"];
function requireSupabaseOrgProvider(): void {
  if (getControlPlaneAuthProvider() !== "supabase_org") {
    throw new Error(
      "supabase_org control-plane APIs require HINDSIGHT_CP_AUTH_PROVIDER=supabase_org"
    );
  }
}

export async function getAuthenticatedUser(request: Request): Promise<SupabaseUser> {
  requireSupabaseOrgProvider();
  const token = getCookie(request, SUPABASE_ORG_ACCESS_TOKEN_COOKIE);
  if (!token) {
    throw new Error("Missing Supabase access token");
  }
  return getSupabaseUserForToken(token);
}

export async function getAuthenticatedUserWithRefresh(
  request: NextRequest,
  response: NextResponse
): Promise<SupabaseUser> {
  return (await getValidSupabaseSession(request, response)).user;
}

export async function getValidSupabaseSession(
  request: NextRequest,
  response: NextResponse
): Promise<{ accessToken: string; user: SupabaseUser }> {
  requireSupabaseOrgProvider();
  const accessToken = getCookie(request, SUPABASE_ORG_ACCESS_TOKEN_COOKIE);
  if (accessToken) {
    try {
      return { accessToken, user: await getSupabaseUserForToken(accessToken) };
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes("401")) throw error;
    }
  }

  const refreshToken = getCookie(request, SUPABASE_ORG_REFRESH_TOKEN_COOKIE);
  if (!refreshToken) {
    clearSupabaseOrgSessionCookies(response, request);
    throw new Error("Missing Supabase refresh token");
  }

  try {
    const session = await refreshSupabaseSession(refreshToken);
    setSupabaseOrgSessionCookies(response, request, session);
    const user =
      session.user?.id && session.user.email
        ? session.user
        : await getSupabaseUserForToken(session.access_token);
    return { accessToken: session.access_token, user };
  } catch (error) {
    clearSupabaseOrgSessionCookies(response, request);
    throw error;
  }
}

export async function getSupabaseUserForToken(token: string): Promise<SupabaseUser> {
  const response = await fetch(`${supabaseUrl()}/auth/v1/user`, {
    headers: {
      Authorization: `Bearer ${token}`,
      apikey: supabaseAnonOrServiceKey(),
    },
  });
  if (!response.ok) {
    throw new Error(`Supabase user lookup failed: ${response.status}`);
  }
  const data = (await response.json()) as { id?: string; email?: string };
  if (!data.id || !data.email) {
    throw new Error("Supabase user response is missing id or email");
  }
  return { id: data.id, email: data.email };
}

export async function signInWithPassword(
  email: string,
  password: string
): Promise<SupabasePasswordSession> {
  return supabasePasswordAuth("/auth/v1/token?grant_type=password", email, password);
}

export async function refreshSupabaseSession(
  refreshToken: string
): Promise<SupabaseRefreshedSession> {
  const response = await fetch(`${supabaseUrl()}/auth/v1/token?grant_type=refresh_token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: supabaseAnonOrServiceKey(),
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!response.ok) throw new Error(`Supabase session refresh failed: ${response.status}`);
  const data = (await response.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
    user?: { id?: string; email?: string };
  };
  if (!data.access_token)
    throw new Error("Supabase session refresh did not return an access token");
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_in: data.expires_in,
    user:
      data.user?.id && data.user.email ? { id: data.user.id, email: data.user.email } : undefined,
  };
}

export async function updateSupabaseUserPassword(
  accessToken: string,
  newPassword: string
): Promise<void> {
  const response = await fetch(`${supabaseUrl()}/auth/v1/user`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
      apikey: supabaseAnonOrServiceKey(),
    },
    body: JSON.stringify({ password: newPassword }),
  });
  if (!response.ok) throw new Error(`Supabase password update failed: ${response.status}`);
}

export async function signOutSupabaseSession(
  accessToken: string | undefined,
  scope: "local" | "global" | "others" = "local"
): Promise<void> {
  if (!accessToken) return;
  const response = await fetch(`${supabaseUrl()}/auth/v1/logout?scope=${scope}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      apikey: supabaseAnonOrServiceKey(),
    },
  });
  if (!response.ok && response.status !== 401) {
    throw new Error(`Supabase logout failed: ${response.status}`);
  }
}

export async function signUpWithPassword(
  email: string,
  password: string
): Promise<SupabasePasswordSession | null> {
  const response = await fetch(`${supabaseUrl()}/auth/v1/signup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: supabaseAnonOrServiceKey(),
    },
    body: JSON.stringify({ email: normalizeEmail(email), password }),
  });
  if (!response.ok) throw new Error(`Supabase signup failed: ${response.status}`);
  const data = (await response.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
    user?: { id?: string; email?: string };
  };
  if (!data.access_token || !data.user?.id || !data.user.email) return null;
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_in: data.expires_in,
    user: { id: data.user.id, email: data.user.email },
  };
}

export async function ensureInitialOrganizationForSignup(
  user: SupabaseUser,
  organizationName: string | undefined
): Promise<Organization & { role: OrganizationRole }> {
  const organizations = await listOrganizationsForUser(user.id);
  if (organizations[0]) return organizations[0];
  const policy = getOrgCreationPolicy();
  if (policy !== "open" && policy !== "direct_signup_only") {
    throw new Error("Organization creation is disabled");
  }
  const created = await createOrganization(
    user,
    organizationName || `${user.email}'s Organization`
  );
  return { ...created, role: "owner" };
}

export function setSupabaseOrgSessionCookies(
  response: NextResponse,
  request: NextRequest,
  session: SupabasePasswordSession | SupabaseRefreshedSession,
  selectedOrgId?: string
): void {
  response.cookies.set({
    name: SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
    value: session.access_token,
    ...sessionCookieOptions(request),
    maxAge: session.expires_in || 3600,
  });
  if (session.refresh_token) {
    response.cookies.set({
      name: SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
      value: session.refresh_token,
      ...sessionCookieOptions(request),
      maxAge: 30 * 24 * 60 * 60,
    });
  }
  if (selectedOrgId) {
    response.cookies.set({
      name: SUPABASE_ORG_SELECTED_ORG_COOKIE,
      value: selectedOrgId,
      ...sessionCookieOptions(request),
      maxAge: 30 * 24 * 60 * 60,
    });
  }
}

export function clearSupabaseOrgSessionCookies(
  response: NextResponse,
  request: NextRequest,
  options: { keepSelectedOrg?: boolean } = {}
): void {
  for (const name of [
    SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
    SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
    ...(options.keepSelectedOrg ? [] : [SUPABASE_ORG_SELECTED_ORG_COOKIE]),
  ]) {
    response.cookies.set({
      name,
      value: "",
      ...sessionCookieOptions(request),
      maxAge: 0,
    });
  }
}

export function copySupabaseOrgSessionCookies(from: NextResponse, to: NextResponse): void {
  for (const cookie of from.cookies.getAll()) to.cookies.set(cookie);
}

export async function getCurrentOrgContext(request: Request): Promise<CurrentOrgContext> {
  const user = await getAuthenticatedUser(request);
  return getCurrentOrgContextForUser(request, user);
}

export async function getCurrentOrgContextWithRefresh(
  request: NextRequest,
  response: NextResponse
): Promise<CurrentOrgContext> {
  const user = await getAuthenticatedUserWithRefresh(request, response);
  return getCurrentOrgContextForUser(request, user);
}

export async function getCurrentOrgContextForUser(
  request: Request,
  user: SupabaseUser
): Promise<CurrentOrgContext> {
  const selectedOrgId = getCookie(request, SUPABASE_ORG_SELECTED_ORG_COOKIE);
  if (!selectedOrgId) {
    throw new Error("Missing selected organization");
  }
  const memberships = await restGet<OrganizationMembership>("organization_members", {
    org_id: `eq.${selectedOrgId}`,
    user_id: `eq.${user.id}`,
    removed_at: "is.null",
    limit: "1",
  });
  if (memberships.length === 0) {
    throw new Error("User is not a member of the selected organization");
  }
  return { user, selectedOrgId, membership: memberships[0] };
}

export async function listOrganizationsForUser(
  userId: string
): Promise<Array<Organization & { role: OrganizationRole }>> {
  const memberships = await restGet<OrganizationMembership>("organization_members", {
    user_id: `eq.${userId}`,
    removed_at: "is.null",
  });
  if (memberships.length === 0) return [];
  const orgIds = memberships.map((membership) => membership.org_id);
  const organizations = await restGet<Organization>("organizations", {
    id: `in.(${orgIds.join(",")})`,
  });
  const roleByOrg = new Map(memberships.map((membership) => [membership.org_id, membership.role]));
  return organizations.map((organization) => ({
    ...organization,
    role: roleByOrg.get(organization.id) || "member",
  }));
}

export async function createOrganization(user: SupabaseUser, name: string): Promise<Organization> {
  const orgName = normalizeName(name, "organization name");
  const org: Organization = { id: crypto.randomUUID(), name: orgName, config: {} };
  const created = await restPost<Organization>("organizations", org);
  await restPost<OrganizationMembership>("organization_members", {
    org_id: created.id,
    user_id: user.id,
    email: user.email,
    role: "owner",
  });
  return created;
}

export async function updateOrganizationName(
  context: CurrentOrgContext,
  id: string,
  name: string
): Promise<Organization> {
  await requireOrgOwner(context);
  if (context.selectedOrgId !== id) throw new Error("Can only update the selected organization");
  const rows = await restPatch<Organization>(
    "organizations",
    { name: normalizeName(name, "organization name") },
    { id: `eq.${id}` },
    true
  );
  if (!rows[0]) throw new Error("Organization not found");
  return rows[0];
}

export async function requireOrgAdmin(context: CurrentOrgContext): Promise<void> {
  if (context.membership.role !== "owner" && context.membership.role !== "admin") {
    throw new Error("Organization admin role required");
  }
}

export async function requireOrgOwner(context: CurrentOrgContext): Promise<void> {
  if (context.membership.role !== "owner") {
    throw new Error("Organization owner role required");
  }
}

export async function listMembers(orgId: string): Promise<OrganizationMembership[]> {
  return restGet<OrganizationMembership>("organization_members", {
    org_id: `eq.${orgId}`,
    removed_at: "is.null",
  });
}

export async function updateMemberRole(
  context: CurrentOrgContext,
  userId: string,
  role: OrganizationRole
): Promise<OrganizationMembership> {
  await requireOrgOwner(context);
  assertOrganizationRole(role);
  if (userId === context.user.id) throw new Error("Owners cannot change their own role");
  const rows = await restPatch<OrganizationMembership>(
    "organization_members",
    { role },
    {
      org_id: `eq.${context.selectedOrgId}`,
      user_id: `eq.${userId}`,
      removed_at: "is.null",
    },
    true
  );
  if (!rows[0]) throw new Error("Member not found");
  return rows[0];
}

export async function removeMember(context: CurrentOrgContext, userId: string): Promise<void> {
  await requireOrgOwner(context);
  if (userId === context.user.id) throw new Error("Owners cannot remove themselves");
  await restRpc("remove_organization_member", {
    p_org_id: context.selectedOrgId,
    p_user_id: userId,
    p_removed_by_user_id: context.user.id,
  });
}

export async function listInvites(orgId: string): Promise<OrganizationInvite[]> {
  return restGet<OrganizationInvite>("organization_invites", {
    org_id: `eq.${orgId}`,
    order: "created_at.desc",
  });
}

export async function createInvite(
  context: CurrentOrgContext,
  email: string,
  role: OrganizationRole
): Promise<{ id: string; invite_url: string; expires_at: string }> {
  await requireOrgAdmin(context);
  assertOrganizationRole(role);
  if (role === "owner" && context.membership.role !== "owner") {
    throw new Error("Only organization owners can invite owners");
  }
  const inviteEmail = normalizeEmail(email);
  const rawToken =
    crypto.randomUUID().replaceAll("-", "") + crypto.randomUUID().replaceAll("-", "");
  const tokenHash = await sha256Hex(rawToken);
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
  const rows = await restPost<{ id: string; expires_at: string }>("organization_invites", {
    org_id: context.selectedOrgId,
    email: inviteEmail,
    role,
    token_hash: tokenHash,
    expires_at: expiresAt,
    created_by_user_id: context.user.id,
  });
  return {
    id: rows.id,
    invite_url: `${publicBaseUrl()}/login?invite=${rawToken}`,
    expires_at: rows.expires_at,
  };
}

export async function revokeInvite(context: CurrentOrgContext, id: string): Promise<void> {
  await requireOrgAdmin(context);
  await restPatch(
    "organization_invites",
    { revoked_at: new Date().toISOString() },
    { id: `eq.${id}`, org_id: `eq.${context.selectedOrgId}` }
  );
}

export async function acceptInvite(request: Request, token: string): Promise<{ org_id: string }> {
  const user = await getAuthenticatedUser(request);
  return acceptInviteForUser(user, token);
}

export async function acceptInviteForUser(
  user: SupabaseUser,
  token: string
): Promise<{ org_id: string }> {
  const tokenHash = await sha256Hex(token);
  const invites = await restGet<{
    id: string;
    org_id: string;
    email: string;
    role: OrganizationRole;
    expires_at: string;
    accepted_at?: string | null;
    revoked_at?: string | null;
  }>("organization_invites", {
    token_hash: `eq.${tokenHash}`,
    accepted_at: "is.null",
    revoked_at: "is.null",
    limit: "1",
  });
  if (invites.length === 0) throw new Error("Invite not found");
  const invite = invites[0];
  if (new Date(invite.expires_at).getTime() <= Date.now()) {
    throw new Error("Invite has expired");
  }
  if (invite.email.toLowerCase() !== user.email.toLowerCase()) {
    throw new Error("Invite email does not match authenticated user");
  }
  await restPost<OrganizationMembership>("organization_members", {
    org_id: invite.org_id,
    user_id: user.id,
    email: user.email,
    role: invite.role,
  });
  await restPatch(
    "organization_invites",
    { accepted_at: new Date().toISOString() },
    { id: `eq.${invite.id}` }
  );
  return { org_id: invite.org_id };
}

interface HindsightApiKeyRow extends HindsightApiKeySummary {
  encrypted_key?: string | null;
}

interface HindsightApiKeyOperationScope {
  api_key_id: string;
  operation: ApiKeyOperation;
  bank_scope_mode: ApiKeyBankScopeMode;
}

interface HindsightApiKeyOperationBankScope {
  api_key_id: string;
  operation: ApiKeyOperation;
  bank_id: string;
  bank_internal_id?: string | null;
}

interface HindsightApiKeyCreatedBank {
  api_key_id: string;
  bank_id: string;
  bank_internal_id: string;
  created_at?: string | null;
}

export async function listApiKeys(context: CurrentOrgContext): Promise<HindsightApiKeySummary[]> {
  const isAdmin = context.membership.role === "owner" || context.membership.role === "admin";
  const keys = await restGet<HindsightApiKeySummary>("hindsight_api_keys", {
    org_id: `eq.${context.selectedOrgId}`,
    select:
      "id,org_id,created_by_user_id,name,permission_mode,allowed_operations,expires_at,revoked_at,created_at",
    order: "created_at.desc",
  });
  const visibleKeys = keys.filter((key) => isAdmin || key.created_by_user_id === context.user.id);
  const visibleKeyIds = visibleKeys.map((key) => key.id);
  const operationScopes =
    visibleKeyIds.length > 0
      ? await restGet<HindsightApiKeyOperationScope>("hindsight_api_key_operation_scopes", {
          api_key_id: `in.(${visibleKeyIds.join(",")})`,
          select: "api_key_id,operation,bank_scope_mode",
        })
      : [];
  const selectedOperationScopes = operationScopes.filter(
    (scope) => scope.bank_scope_mode === "selected"
  );
  const operationBankScopes =
    selectedOperationScopes.length > 0
      ? await restGet<HindsightApiKeyOperationBankScope>(
          "hindsight_api_key_operation_bank_scopes",
          {
            api_key_id: `in.(${visibleKeyIds.join(",")})`,
            select: "api_key_id,operation,bank_id,bank_internal_id",
          }
        )
      : [];
  const createdBanks =
    visibleKeyIds.length > 0
      ? await restGet<HindsightApiKeyCreatedBank>("hindsight_api_key_created_banks", {
          api_key_id: `in.(${visibleKeyIds.join(",")})`,
          deleted_at: "is.null",
          select: "api_key_id,bank_id,bank_internal_id,created_at",
        })
      : [];
  const operationScopesByKey = new Map<string, ApiKeyOperationScopeSummary[]>();
  const bankScopesByKeyOperation = new Map<string, HindsightApiKeyOperationBankScope[]>();
  for (const scope of operationBankScopes) {
    const key = `${scope.api_key_id}:${scope.operation}`;
    const scopes = bankScopesByKeyOperation.get(key) ?? [];
    scopes.push(scope);
    bankScopesByKeyOperation.set(key, scopes);
  }
  for (const scope of operationScopes) {
    const operationKey = `${scope.api_key_id}:${scope.operation}`;
    const bankScopes = bankScopesByKeyOperation.get(operationKey) ?? [];
    const summaries = operationScopesByKey.get(scope.api_key_id) ?? [];
    summaries.push({
      operation: scope.operation,
      bank_scope_mode: scope.bank_scope_mode,
      scoped_bank_ids:
        scope.bank_scope_mode === "selected" ? bankScopes.map((bank) => bank.bank_id) : undefined,
      scoped_bank_internal_ids:
        scope.bank_scope_mode === "selected"
          ? bankScopes
              .map((bank) => bank.bank_internal_id)
              .filter((bankInternalId): bankInternalId is string => Boolean(bankInternalId))
          : undefined,
    });
    operationScopesByKey.set(scope.api_key_id, summaries);
  }
  for (const key of visibleKeys) {
    const unscoped = (key.allowed_operations ?? []).filter((operation) =>
      UNSCOPED_DATAPLANE_OPERATIONS.includes(operation)
    );
    if (unscoped.length > 0) {
      const summaries = operationScopesByKey.get(key.id) ?? [];
      for (const operation of unscoped) {
        if (!summaries.some((scope) => scope.operation === operation)) {
          summaries.push({ operation, bank_scope_mode: "all" });
        }
      }
      operationScopesByKey.set(key.id, summaries);
    }
  }
  const createdBanksByKey = new Map<string, OwnedBankSummary[]>();
  for (const bank of createdBanks) {
    const banks = createdBanksByKey.get(bank.api_key_id) ?? [];
    banks.push({
      bank_id: bank.bank_id,
      bank_internal_id: bank.bank_internal_id,
      created_at: bank.created_at,
    });
    createdBanksByKey.set(bank.api_key_id, banks);
  }
  return visibleKeys.map((key) => ({
    ...key,
    operation_scopes: operationScopesByKey.get(key.id) ?? [],
    owned_banks: createdBanksByKey.get(key.id) ?? [],
    can_view_secret: true,
  }));
}

export async function createApiKey(
  context: CurrentOrgContext,
  name: string,
  permissionMode: ApiKeyPermissionMode,
  operationScopes: ApiKeyOperationScopeInput[] | null
): Promise<{ id: string; key: string }> {
  const keyName = normalizeName(name, "API key name");
  const normalizedOperationScopes =
    permissionMode === "scoped"
      ? normalizeApiKeyOperationScopes(operationScopes, context.membership.role)
      : [];
  const operations =
    permissionMode === "scoped" ? normalizedOperationScopes.map((scope) => scope.operation) : null;
  const rawKey = `${API_KEY_PREFIX}${crypto.randomUUID().replaceAll("-", "")}${crypto.randomUUID().replaceAll("-", "")}`;
  const keyHash = await sha256Hex(rawKey);
  const encryptedKey = await encryptApiKey(rawKey);
  const row = await restRpcRow<{ id: string }>("create_hindsight_api_key", {
    p_org_id: context.selectedOrgId,
    p_created_by_user_id: context.user.id,
    p_name: keyName,
    p_key_hash: keyHash,
    p_encrypted_key: encryptedKey,
    p_permission_mode: permissionMode,
    p_allowed_operations: operations,
    p_operation_scopes: normalizedOperationScopes,
  });
  return { id: row.id, key: rawKey };
}

export async function revealApiKey(
  context: CurrentOrgContext,
  id: string
): Promise<{ id: string; key: string }> {
  const rows = await restGet<HindsightApiKeyRow>("hindsight_api_keys", {
    id: `eq.${id}`,
    org_id: `eq.${context.selectedOrgId}`,
    select:
      "id,org_id,created_by_user_id,name,permission_mode,allowed_operations,expires_at,revoked_at,created_at,encrypted_key",
    limit: "1",
  });
  const key = rows[0];
  if (!key) throw new Error("API key not found");
  if (!key.encrypted_key) throw new Error("API key secret is not available for this key");
  const isAdmin = context.membership.role === "owner" || context.membership.role === "admin";
  if (!isAdmin && key.created_by_user_id !== context.user.id)
    throw new Error("API key is not owned by this user");
  return { id: key.id, key: await decryptApiKey(key.encrypted_key) };
}

export async function revokeApiKey(context: CurrentOrgContext, id: string): Promise<void> {
  if (context.membership.role !== "owner" && context.membership.role !== "admin") {
    const keys = await restGet<HindsightApiKeySummary>("hindsight_api_keys", {
      id: `eq.${id}`,
      org_id: `eq.${context.selectedOrgId}`,
      select:
        "id,org_id,created_by_user_id,name,permission_mode,allowed_operations,expires_at,revoked_at,created_at",
      limit: "1",
    });
    if (keys[0]?.created_by_user_id !== context.user.id)
      throw new Error("API key is not owned by this user");
  }
  await restPatch(
    "hindsight_api_keys",
    { revoked_at: new Date().toISOString() },
    { id: `eq.${id}`, org_id: `eq.${context.selectedOrgId}` }
  );
}

export async function updateApiKeyPermissions(
  context: CurrentOrgContext,
  id: string,
  permissionMode: ApiKeyPermissionMode,
  operationScopes: ApiKeyOperationScopeInput[] | null
): Promise<void> {
  const rows = await restGet<HindsightApiKeySummary>("hindsight_api_keys", {
    id: `eq.${id}`,
    org_id: `eq.${context.selectedOrgId}`,
    select:
      "id,org_id,created_by_user_id,name,permission_mode,allowed_operations,expires_at,revoked_at,created_at",
    limit: "1",
  });
  const key = rows[0];
  if (!key) throw new Error("API key not found");
  const isAdmin = context.membership.role === "owner" || context.membership.role === "admin";
  if (!isAdmin && key.created_by_user_id !== context.user.id)
    throw new Error("API key is not owned by this user");
  if (key.revoked_at) throw new Error("Cannot update revoked API key");
  const normalizedOperationScopes =
    permissionMode === "scoped"
      ? normalizeApiKeyOperationScopes(operationScopes, context.membership.role)
      : [];
  await restRpc("replace_hindsight_api_key_permissions", {
    p_api_key_id: id,
    p_org_id: context.selectedOrgId,
    p_permission_mode: permissionMode,
    p_allowed_operations:
      permissionMode === "scoped"
        ? normalizedOperationScopes.map((scope) => scope.operation)
        : null,
    p_operation_scopes: normalizedOperationScopes,
  });
}

export function jsonError(message: string, status = 400): NextResponse {
  return NextResponse.json({ error: message }, { status });
}

async function restGet<T>(table: string, params: Record<string, string>): Promise<T[]> {
  const response = await fetch(restUrl(table, params), { headers: serviceHeaders() });
  if (!response.ok) throw new Error(`Supabase query failed: ${response.status}`);
  return (await response.json()) as T[];
}

async function restPost<T>(table: string, body: unknown): Promise<T> {
  const response = await fetch(restUrl(table), {
    method: "POST",
    headers: {
      ...serviceHeaders(),
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Supabase insert failed: ${response.status}`);
  const rows = (await response.json()) as T[];
  if (!rows[0]) throw new Error("Supabase insert returned no row");
  return rows[0];
}

async function restRpc(functionName: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${supabaseUrl()}/rest/v1/rpc/${functionName}`, {
    method: "POST",
    headers: {
      ...serviceHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Supabase RPC failed: ${response.status}`);
  if (response.status === 204) return null;
  return response.json();
}

async function restRpcRow<T>(functionName: string, body: unknown): Promise<T> {
  const rows = (await restRpc(functionName, body)) as T[];
  if (!rows[0]) throw new Error("Supabase RPC returned no row");
  return rows[0];
}

async function restPatch<T>(
  table: string,
  body: unknown,
  params: Record<string, string>,
  returnRepresentation = false
): Promise<T[]> {
  const response = await fetch(restUrl(table, params), {
    method: "PATCH",
    headers: {
      ...serviceHeaders(),
      "Content-Type": "application/json",
      ...(returnRepresentation ? { Prefer: "return=representation" } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Supabase update failed: ${response.status}`);
  if (!returnRepresentation) return [];
  return (await response.json()) as T[];
}

function restUrl(table: string, params: Record<string, string> = {}): string {
  const url = new URL(`${supabaseUrl()}/rest/v1/${table}`);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  return url.toString();
}

function serviceHeaders(): Record<string, string> {
  const serviceKey = supabaseServiceKey();
  return { apikey: serviceKey, Authorization: `Bearer ${serviceKey}` };
}

function supabaseUrl(): string {
  const url = process.env.HINDSIGHT_AUTH_SUPABASE_URL;
  if (!url) throw new Error("HINDSIGHT_AUTH_SUPABASE_URL is required");
  return url.replace(/\/$/, "");
}

function supabaseServiceKey(): string {
  const key = process.env.HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY;
  if (!key) throw new Error("HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY is required");
  return key;
}

function apiKeyEncryptionSecret(): string {
  const key = process.env.HINDSIGHT_AUTH_API_KEY_ENCRYPTION_KEY;
  if (!key) throw new Error("HINDSIGHT_AUTH_API_KEY_ENCRYPTION_KEY is required");
  return key;
}

function supabaseAnonOrServiceKey(): string {
  return process.env.HINDSIGHT_AUTH_SUPABASE_ANON_KEY || supabaseServiceKey();
}

function publicBaseUrl(): string {
  return process.env.HINDSIGHT_AUTH_PUBLIC_BASE_URL || "http://localhost:9999";
}

function getCookie(request: Request, name: string): string | undefined {
  const cookieHeader = request.headers.get("cookie") || "";
  for (const part of cookieHeader.split(";")) {
    const separator = part.indexOf("=");
    if (separator <= 0) continue;
    if (part.slice(0, separator).trim() === name) {
      return decodeURIComponent(part.slice(separator + 1).trim());
    }
  }
  return undefined;
}

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function apiKeyEncryptionKey(): Promise<CryptoKey> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(apiKeyEncryptionSecret())
  );
  return crypto.subtle.importKey("raw", digest, { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
}

function base64UrlDecode(value: string): Uint8Array {
  const padded = value
    .replace(/-/g, "+")
    .replace(/_/g, "/")
    .padEnd(Math.ceil(value.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

function asArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

async function encryptApiKey(value: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    await apiKeyEncryptionKey(),
    new TextEncoder().encode(value)
  );
  return `v1.${base64UrlEncode(iv)}.${base64UrlEncode(new Uint8Array(ciphertext))}`;
}

async function decryptApiKey(value: string): Promise<string> {
  const [version, encodedIv, encodedCiphertext] = value.split(".");
  if (version !== "v1" || !encodedIv || !encodedCiphertext) {
    throw new Error("Unsupported API key secret format");
  }
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: asArrayBuffer(base64UrlDecode(encodedIv)) },
    await apiKeyEncryptionKey(),
    asArrayBuffer(base64UrlDecode(encodedCiphertext))
  );
  return new TextDecoder().decode(plaintext);
}

async function supabasePasswordAuth(
  path: string,
  email: string,
  password: string
): Promise<SupabasePasswordSession> {
  const response = await fetch(`${supabaseUrl()}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: supabaseAnonOrServiceKey(),
    },
    body: JSON.stringify({ email: normalizeEmail(email), password }),
  });
  if (!response.ok) throw new Error(`Supabase password authentication failed: ${response.status}`);
  const data = (await response.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
    user?: { id?: string; email?: string };
  };
  if (!data.access_token || !data.user?.id || !data.user.email) {
    throw new Error("Supabase password authentication did not return a session");
  }
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_in: data.expires_in,
    user: { id: data.user.id, email: data.user.email },
  };
}

export function assertOrganizationRole(role: string): asserts role is OrganizationRole {
  if (!ORGANIZATION_ROLES.includes(role as OrganizationRole)) {
    throw new Error(`Invalid organization role: ${role}`);
  }
}

function normalizeApiKeyOperations(
  operations: string[] | null,
  role: OrganizationRole
): ApiKeyOperation[] {
  const allowedForRole = operationsForRole(role);
  const values = operations ?? allowedForRole;
  const unique = Array.from(new Set(values));
  for (const operation of unique) {
    if (!API_KEY_OPERATIONS.includes(operation as ApiKeyOperation)) {
      throw new Error(`Invalid API key operation: ${operation}`);
    }
    if (!allowedForRole.includes(operation as ApiKeyOperation)) {
      throw new Error(`API key operation exceeds creator permissions: ${operation}`);
    }
  }
  return unique as ApiKeyOperation[];
}

function normalizeApiKeyOperationScopes(
  operationScopes: ApiKeyOperationScopeInput[] | null,
  role: OrganizationRole
): Required<ApiKeyOperationScopeInput>[] {
  const requestedOperations = operationScopes?.map((scope) => scope.operation) ?? null;
  const operations = normalizeApiKeyOperations(requestedOperations, role);
  const scopesByOperation = new Map(
    (operationScopes ?? []).map((scope) => [scope.operation, scope])
  );
  return operations.map((operation) => {
    const requestedScope = scopesByOperation.get(operation);
    const bankScopeMode = UNSCOPED_DATAPLANE_OPERATIONS.includes(operation)
      ? "all"
      : normalizeApiKeyBankScopeMode(requestedScope?.bank_scope_mode ?? "all");
    const bankScopes =
      bankScopeMode === "selected"
        ? normalizeApiKeyBankScopes(requestedScope?.bank_scopes ?? null)
        : [];
    return { operation, bank_scope_mode: bankScopeMode, bank_scopes: bankScopes };
  });
}

function operationsForRole(role: OrganizationRole): ApiKeyOperation[] {
  if (role === "member")
    return [...BANK_READ_OPERATIONS, ...BANK_WRITE_OPERATIONS, ...SPECIAL_BANK_OPERATIONS];
  return API_KEY_OPERATIONS;
}

function normalizeApiKeyBankScopeMode(mode: string): ApiKeyBankScopeMode {
  if (mode !== "all" && mode !== "selected")
    throw new Error(`Invalid API key bank scope mode: ${mode}`);
  return mode;
}

function normalizeApiKeyBankScopes(
  bankScopes: ApiKeyBankScopeInput[] | null
): ApiKeyBankScopeInput[] {
  if (!bankScopes) return [];
  const byInternalId = new Map<string, ApiKeyBankScopeInput>();
  for (const scope of bankScopes) {
    const bankId = normalizeName(scope.bank_id, "bank id");
    const bankInternalId = normalizeName(scope.bank_internal_id, "bank internal id");
    byInternalId.set(bankInternalId, { bank_id: bankId, bank_internal_id: bankInternalId });
  }
  return Array.from(byInternalId.values());
}

function normalizeEmail(email: string): string {
  const value = email.trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(value)) throw new Error("Invalid email");
  return value;
}

function normalizeName(value: string, label: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error(`${label} is required`);
  if (trimmed.length > 200) throw new Error(`${label} is too long`);
  return trimmed;
}
