import { NextRequest, NextResponse } from "next/server";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
  getControlPlaneAuthProvider,
  getOrgCreationPolicy,
} from "@/lib/auth/provider";
import { sessionCookieOptions } from "@/lib/auth/session";

export type OrganizationRole = "owner" | "admin" | "member";
export type ApiKeyOperation = "retain" | "recall" | "reflect";

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

export interface OrganizationMembership {
  org_id: string;
  user_id: string;
  email?: string;
  role: OrganizationRole;
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
  name: string;
  role: OrganizationRole;
  allowed_operations?: ApiKeyOperation[] | null;
  expires_at?: string | null;
  revoked_at?: string | null;
  created_at: string;
}

export interface CurrentOrgContext {
  user: SupabaseUser;
  selectedOrgId: string;
  membership: OrganizationMembership;
}

const API_KEY_PREFIX = "hs_";
const ORGANIZATION_ROLES: OrganizationRole[] = ["owner", "admin", "member"];
const API_KEY_OPERATIONS: ApiKeyOperation[] = ["retain", "recall", "reflect"];

export function requireSupabaseOrgProvider(): void {
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
  session: SupabasePasswordSession,
  selectedOrgId: string
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
  response.cookies.set({
    name: SUPABASE_ORG_SELECTED_ORG_COOKIE,
    value: selectedOrgId,
    ...sessionCookieOptions(request),
    maxAge: 30 * 24 * 60 * 60,
  });
}

export function clearSupabaseOrgSessionCookies(response: NextResponse, request: NextRequest): void {
  for (const name of [
    SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
    SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
    SUPABASE_ORG_SELECTED_ORG_COOKIE,
  ]) {
    response.cookies.set({
      name,
      value: "",
      ...sessionCookieOptions(request),
      maxAge: 0,
    });
  }
}

export async function getCurrentOrgContext(request: Request): Promise<CurrentOrgContext> {
  const user = await getAuthenticatedUser(request);
  const selectedOrgId = getCookie(request, SUPABASE_ORG_SELECTED_ORG_COOKIE);
  if (!selectedOrgId) {
    throw new Error("Missing selected organization");
  }
  const memberships = await restGet<OrganizationMembership>("organization_members", {
    org_id: `eq.${selectedOrgId}`,
    user_id: `eq.${user.id}`,
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
  return restGet<OrganizationMembership>("organization_members", { org_id: `eq.${orgId}` });
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
    { org_id: `eq.${context.selectedOrgId}`, user_id: `eq.${userId}` },
    true
  );
  if (!rows[0]) throw new Error("Member not found");
  return rows[0];
}

export async function removeMember(context: CurrentOrgContext, userId: string): Promise<void> {
  await requireOrgOwner(context);
  if (userId === context.user.id) throw new Error("Owners cannot remove themselves");
  await restDelete("organization_members", {
    org_id: `eq.${context.selectedOrgId}`,
    user_id: `eq.${userId}`,
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

export async function listApiKeys(orgId: string): Promise<HindsightApiKeySummary[]> {
  return restGet<HindsightApiKeySummary>("hindsight_api_keys", {
    org_id: `eq.${orgId}`,
    select: "id,org_id,name,role,allowed_operations,expires_at,revoked_at,created_at",
    order: "created_at.desc",
  });
}

export async function createApiKey(
  context: CurrentOrgContext,
  name: string,
  bankIds: string[] | null,
  allowedOperations: string[] | null
): Promise<{ id: string; key: string }> {
  await requireOrgAdmin(context);
  const keyName = normalizeName(name, "API key name");
  const operations = normalizeApiKeyOperations(allowedOperations);
  const scopedBankIds = normalizeBankIds(bankIds);
  const rawKey = `${API_KEY_PREFIX}${crypto.randomUUID().replaceAll("-", "")}${crypto.randomUUID().replaceAll("-", "")}`;
  const keyHash = await sha256Hex(rawKey);
  const row = await restPost<{ id: string }>("hindsight_api_keys", {
    org_id: context.selectedOrgId,
    created_by_user_id: context.user.id,
    name: keyName,
    key_hash: keyHash,
    role: "admin",
    allowed_operations: operations,
  });
  if (scopedBankIds.length > 0) {
    await restPost(
      "hindsight_api_key_bank_scopes",
      scopedBankIds.map((bankId) => ({ api_key_id: row.id, bank_id: bankId }))
    );
  }
  return { id: row.id, key: rawKey };
}

export async function revokeApiKey(context: CurrentOrgContext, id: string): Promise<void> {
  await requireOrgAdmin(context);
  await restPatch(
    "hindsight_api_keys",
    { revoked_at: new Date().toISOString() },
    { id: `eq.${id}`, org_id: `eq.${context.selectedOrgId}` }
  );
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

async function restPatch<T>(
  table: string,
  body: unknown,
  params: Record<string, string>
): Promise<T[]>;
async function restPatch<T>(
  table: string,
  body: unknown,
  params: Record<string, string>,
  returnRepresentation: true
): Promise<T[]>;
async function restPatch<T>(
  table: string,
  body: unknown,
  params: Record<string, string>,
  returnRepresentation: false
): Promise<T[]>;
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

async function restDelete(table: string, params: Record<string, string>): Promise<void> {
  const response = await fetch(restUrl(table, params), {
    method: "DELETE",
    headers: serviceHeaders(),
  });
  if (!response.ok) throw new Error(`Supabase delete failed: ${response.status}`);
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

function normalizeApiKeyOperations(operations: string[] | null): ApiKeyOperation[] {
  const values = operations ?? API_KEY_OPERATIONS;
  const unique = Array.from(new Set(values));
  for (const operation of unique) {
    if (!API_KEY_OPERATIONS.includes(operation as ApiKeyOperation)) {
      throw new Error(`Invalid API key operation: ${operation}`);
    }
  }
  return unique as ApiKeyOperation[];
}

function normalizeBankIds(bankIds: string[] | null): string[] {
  if (!bankIds) return [];
  return Array.from(new Set(bankIds.map((bankId) => normalizeName(bankId, "bank id"))));
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
