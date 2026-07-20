import type { NextRequest, NextResponse } from "next/server";

import { registerBundledAuthProfiles } from "@/lib/auth-profiles";
import { registerBuiltinAuthProviders } from "@/lib/auth/providers";
import {
  getRegisteredControlPlaneAuthProvider,
  hasRegisteredControlPlaneAuthProvider,
} from "@/lib/auth/registry";

export type ControlPlaneAuthProvider = string;
export type OrgCreationPolicy = "open" | "direct_signup_only";

export interface LoginField {
  name: string;
  type: "email" | "password" | "text";
  placeholder: string;
  autocomplete?: string;
  required?: boolean;
  modes?: string[];
  hiddenWhenInvite?: boolean;
}

export interface LoginMode {
  id: string;
  label: string;
}

export interface LoginConfig {
  provider: ControlPlaneAuthProvider;
  modes?: LoginMode[];
  defaultMode?: string;
  fields: LoginField[];
  submitLabel: string;
  submitLabelsByMode?: Record<string, string>;
}

export interface ControlPlaneAuthProviderAdapter {
  id: ControlPlaneAuthProvider;
  expectedDataplaneAuthProfile: string;
  settingsPath?: string;
  settingsLabel?: string;
  logoutEnabled: boolean;
  validateConfig(): void;
  isAuthenticated(request: NextRequest): Promise<boolean>;
  getDataplaneHeaders(
    request: NextRequest | Request,
    extra?: Record<string, string>
  ): Record<string, string>;
  getLoginConfig(): LoginConfig;
  login(request: NextRequest): Promise<NextResponse>;
  logout(response: NextResponse, request: NextRequest): Promise<void> | void;
}

let providersRegistered = false;

export function getControlPlaneAuthProvider(): ControlPlaneAuthProvider {
  const configured = process.env.HINDSIGHT_CP_AUTH_PROVIDER?.trim();
  if (configured) {
    ensureControlPlaneAuthProvidersRegistered();
    if (hasRegisteredControlPlaneAuthProvider(configured)) return configured;
    throw new Error(`Unsupported HINDSIGHT_CP_AUTH_PROVIDER: ${configured}`);
  }
  return process.env.HINDSIGHT_CP_ACCESS_KEY ? "access_key" : "disabled";
}

export function getAuthProviderAdapter(): ControlPlaneAuthProviderAdapter {
  ensureControlPlaneAuthProvidersRegistered();
  const provider = getControlPlaneAuthProvider();
  const adapter = getRegisteredControlPlaneAuthProvider(provider);
  if (!adapter) throw new Error(`Control-plane auth provider is not registered: ${provider}`);
  return adapter;
}

export function getExpectedDataplaneAuthProfile(): string {
  return getAuthProviderAdapter().expectedDataplaneAuthProfile;
}

export function getOrgCreationPolicy(): OrgCreationPolicy {
  const configured = process.env.HINDSIGHT_AUTH_ORG_CREATION_POLICY?.trim();
  if (configured === "direct_signup_only") return configured;
  return "open";
}

export function assertValidControlPlaneAuthConfig(): void {
  getAuthProviderAdapter().validateConfig();
}

export async function isControlPlaneRequestAuthenticated(request: NextRequest): Promise<boolean> {
  return getAuthProviderAdapter().isAuthenticated(request);
}

export function getProviderDataplaneHeaders(
  request: NextRequest | Request,
  extra?: Record<string, string>
): Record<string, string> {
  return getAuthProviderAdapter().getDataplaneHeaders(request, extra);
}

function ensureControlPlaneAuthProvidersRegistered(): void {
  if (providersRegistered) return;
  registerBuiltinAuthProviders();
  registerBundledAuthProfiles();
  providersRegistered = true;
}
