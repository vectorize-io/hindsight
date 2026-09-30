/**
 * POST /api/demo
 * Seeds the demo mode — creates a pre-analyzed demo project and stores
 * demo creator preferences in Hindsight.
 * Called once when the user clicks "Try Demo".
 */

import { NextRequest, NextResponse } from "next/server";
import { createProject, getProjectById } from "@/lib/db";
import { storeUserProfile, storeCreativePreferences, storeWorkflowPreferences, ensureBankExists } from "@/lib/hindsight";
import { DEMO_PROJECT } from "@/data/demo";
import { DEMO_USER_ID } from "@/lib/utils";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(_req: NextRequest) {
  try {
    // ── Ensure demo bank exists ──────────────────────────────────────────────
    if (process.env.HINDSIGHT_API_KEY) {
      try { await ensureBankExists(DEMO_USER_ID); } catch { /* ignore */ }
    }

    // ── Seed demo project ────────────────────────────────────────────────────
    const existing = getProjectById(DEMO_PROJECT.id);
    if (!existing) {
      createProject({ ...DEMO_PROJECT, userId: DEMO_USER_ID });
    }

    // ── Seed demo creator preferences into Hindsight ─────────────────────────
    if (process.env.HINDSIGHT_API_KEY) {
      try {
        await storeUserProfile(
          {
            skillLevel: "Beginner",
            preferredFilmingDevice: "iPhone",
            preferredEditingSoftware: "CapCut",
            preferredPlatforms: ["Instagram", "TikTok"],
            preferredReelDuration: "20-30 seconds",
          },
          DEMO_USER_ID
        );

        await storeCreativePreferences(
          {
            pacing: "Fast-paced",
            editingStyle: "Jump cuts with music sync",
            colorStyle: "Warm and vibrant",
            contentCategories: ["Food", "Lifestyle"],
            hookStyle: "Question hook",
          },
          DEMO_USER_ID
        );

        await storeWorkflowPreferences(
          {
            editingSoftware: "CapCut",
            filmingSetup: "Phone on tripod with natural window light",
            appsOwned: ["CapCut", "Canva", "Instagram"],
          },
          DEMO_USER_ID
        );
      } catch {
        /* Hindsight not configured — demo still works without memory */
      }
    }

    return NextResponse.json({
      success: true,
      data: {
        projectId: DEMO_PROJECT.id,
        userId: DEMO_USER_ID,
        message: "Demo seeded successfully",
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Demo setup failed";
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
