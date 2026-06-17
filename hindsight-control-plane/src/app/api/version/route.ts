import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { getControlPlaneAuthProvider } from "@/lib/auth/provider";

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
    const features = (data.features ?? {}) as Record<string, unknown>;
    const authProvider = getControlPlaneAuthProvider();
    const dataplaneProfile = features.authz_profile;
    features.access_key_auth = !!process.env.HINDSIGHT_CP_ACCESS_KEY;
    features.auth_provider = authProvider;
    features.profile_match =
      authProvider !== "supabase_org" ||
      (dataplaneProfile === "supabase_org" && features.supabase_org_ready === true);
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
