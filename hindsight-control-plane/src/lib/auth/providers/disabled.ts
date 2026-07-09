import { NextResponse, type NextRequest } from "next/server";

import type { ControlPlaneAuthProviderAdapter } from "@/lib/auth/provider";

export const disabledAuthProvider: ControlPlaneAuthProviderAdapter = {
  id: "disabled",
  expectedDataplaneAuthProfile: "disabled",
  logoutEnabled: false,

  validateConfig() {},

  async isAuthenticated() {
    return true;
  },

  getDataplaneHeaders(_request, extra) {
    const headers: Record<string, string> = { ...extra };
    const apiKey = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    return headers;
  },

  getLoginConfig() {
    return {
      provider: "disabled",
      fields: [],
      submitLabel: "Continue",
    };
  },

  async login(_request: NextRequest) {
    return NextResponse.json({ success: true });
  },

  logout() {},
};
