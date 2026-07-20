import { NextResponse } from "next/server";

import { getOrgCreationPolicy } from "@/lib/auth/provider";
import {
  createOrganization,
  getAuthenticatedUser,
  jsonError,
  listOrganizationsForUser,
} from "@/lib/supabase-org/store";

export async function GET(request: Request) {
  try {
    const user = await getAuthenticatedUser(request);
    return NextResponse.json(
      { organizations: await listOrganizationsForUser(user.id) },
      { status: 200 }
    );
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to list organizations", 401);
  }
}

export async function POST(request: Request) {
  try {
    if (getOrgCreationPolicy() === "direct_signup_only") {
      return jsonError("Organization creation is only allowed during direct signup", 403);
    }
    const body = (await request.json()) as { name?: string };
    if (!body.name) return jsonError("name is required", 400);
    const user = await getAuthenticatedUser(request);
    return NextResponse.json(
      { organization: await createOrganization(user, body.name) },
      { status: 201 }
    );
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to create organization", 400);
  }
}
