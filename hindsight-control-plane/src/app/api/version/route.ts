import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { getAuthProviderAdapter, getExpectedDataplaneAuthProfile } from "@/lib/auth/provider";

export async function GET(request: Request) {
  try {
    const response = await sdk.getVersion({
      client: createDataplaneClientForRequest(request),
    });

    if (response.error) {
      console.error("API error getting version:", response.error);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to get version",
          errorKey: "api.errors.version.fetch",
        }),
        { status: 500 }
      );
    }

    const data = response.data as Record<string, unknown>;
    const features = (data.features ?? {}) as Record<string, boolean | string | null>;
    const authProvider = getAuthProviderAdapter();
    const expectedProfile = getExpectedDataplaneAuthProfile();
    features.access_key_auth = !!process.env.HINDSIGHT_CP_ACCESS_KEY;
    features.auth_provider = authProvider.id;
    features.auth_settings_path = authProvider.settingsPath || null;
    features.auth_settings_label = authProvider.settingsLabel || null;
    features.auth_logout_enabled = authProvider.logoutEnabled;
    features.profile_match =
      expectedProfile === "disabled" || features.auth_profile === expectedProfile;
    data.features = features;

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting version:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to get version",
        errorKey: "api.errors.version.fetch",
      }),
      { status: 500 }
    );
  }
}
