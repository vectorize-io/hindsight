/**
 * GET  /api/memory          — list all memories for the user
 * POST /api/memory          — retain a new memory
 * POST /api/memory/profile  — store structured user profile
 * GET  /api/memory/profile  — reflect a profile summary
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listAllMemories,
  retainMemory,
  getCreatorProfileSummary,
  storeUserProfile,
  storeCreativePreferences,
  storeWorkflowPreferences,
} from "@/lib/hindsight";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const userId = req.headers.get("x-user-id") ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";
  const { searchParams } = new URL(req.url);
  const mode = searchParams.get("mode");

  try {
    if (mode === "profile") {
      const profile = await getCreatorProfileSummary(userId);
      return NextResponse.json({ success: true, data: profile });
    }

    const memories = await listAllMemories(userId);
    return NextResponse.json({ success: true, data: { memories } });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to fetch memories";
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const userId = req.headers.get("x-user-id") ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";

  try {
    const body = await req.json() as {
      type?: "memory" | "profile" | "creative" | "workflow";
      content?: string;
      category?: "profile" | "creative" | "workflow" | "project";
      profile?: Record<string, unknown>;
      creative?: Record<string, unknown>;
      workflow?: Record<string, unknown>;
    };

    const { type = "memory" } = body;

    if (type === "profile" && body.profile) {
      await storeUserProfile(body.profile as Parameters<typeof storeUserProfile>[0], userId);
      return NextResponse.json({ success: true, message: "Profile stored in memory" });
    }

    if (type === "creative" && body.creative) {
      await storeCreativePreferences(body.creative as Parameters<typeof storeCreativePreferences>[0], userId);
      return NextResponse.json({ success: true, message: "Creative preferences stored" });
    }

    if (type === "workflow" && body.workflow) {
      await storeWorkflowPreferences(body.workflow as Parameters<typeof storeWorkflowPreferences>[0], userId);
      return NextResponse.json({ success: true, message: "Workflow preferences stored" });
    }

    // Generic memory retain
    if (body.content) {
      await retainMemory(body.content, body.category ?? "profile", userId);
      return NextResponse.json({ success: true, message: "Memory stored" });
    }

    return NextResponse.json({ success: false, error: "No content to store" }, { status: 400 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to store memory";
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
