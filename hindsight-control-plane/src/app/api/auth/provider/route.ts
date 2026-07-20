import { NextResponse } from "next/server";

import { getAuthProviderAdapter } from "@/lib/auth/provider";

export async function GET() {
  const provider = getAuthProviderAdapter();
  return NextResponse.json({
    provider: provider.id,
    login: provider.getLoginConfig(),
    settings_path: provider.settingsPath || null,
    settings_label: provider.settingsLabel || null,
    logout_enabled: provider.logoutEnabled,
  });
}
