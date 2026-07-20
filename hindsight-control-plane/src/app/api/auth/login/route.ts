import { NextRequest } from "next/server";

import { getAuthProviderAdapter } from "@/lib/auth/provider";

export async function POST(request: NextRequest) {
  return getAuthProviderAdapter().login(request);
}
