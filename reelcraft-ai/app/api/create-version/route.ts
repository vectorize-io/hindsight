/**
 * POST /api/create-version
 * Generate a completely original Reel production plan inspired by a reference.
 * Pipeline:
 *   1. Load reference project guide
 *   2. Recall creator preferences from Hindsight
 *   3. Generate custom Reel plan (original content, adapted structure)
 *   4. Return the plan
 */

import { NextRequest, NextResponse } from "next/server";
import { generateCustomVersion } from "@/lib/agent";
import { buildCreateMyVersionPrompt } from "@/lib/agent/prompts";
import { getCreatorProfileSummary, storeProjectInsight } from "@/lib/hindsight";
import { getProjectById } from "@/lib/db";
import type { CreateMyVersionRequest } from "@/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const userId = req.headers.get("x-user-id") ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";

  if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === "your_gemini_api_key_here") {
    return NextResponse.json(
      { success: false, error: "GEMINI_API_KEY is not configured. Get a free key at https://aistudio.google.com/apikey" },
      { status: 503 }
    );
  }

  try {
    const body = await req.json() as CreateMyVersionRequest;

    const {
      referenceProjectId,
      topic,
      promotingOrShowing,
      equipment,
      phoneOrCamera,
      editingSoftware,
      skillLevel,
      platform,
      desiredDuration,
      style,
      differences,
    } = body;

    // Validate required fields
    if (!topic || !skillLevel || !platform) {
      return NextResponse.json(
        { success: false, error: "topic, skillLevel, and platform are required" },
        { status: 400 }
      );
    }

    // Load reference project context
    let referenceGuideContext = "No reference project loaded.";
    if (referenceProjectId) {
      const project = getProjectById(referenceProjectId);
      if (project?.guide) {
        const g = project.guide;
        referenceGuideContext = [
          `Reference Reel: "${project.title}"`,
          `Content Category: ${g.section1_overview.contentCategory}`,
          `Overall Style: ${g.section1_overview.overallStyle}`,
          `Skill Level Required: ${g.section2_skillLevel.level}`,
          `Key Techniques: ${g.section6_techniques.slice(0, 5).map((t) => t.name).join(", ")}`,
          `Storytelling Structure: ${g.section8_script.map((s) => s.label).join(" → ")}`,
          `Creative Strategy Hook: ${project.creativeStrategy?.hook ?? "N/A"}`,
          `Pacing: ${project.creativeStrategy?.pacing ?? "N/A"}`,
          `Transitions used: ${g.section17_transitions.slice(0, 3).map((t) => t.type).join(", ")}`,
          `Color Grade Style: ${g.section15_colorGrading.style}`,
        ].join("\n");
      }
    }

    // Recall creator profile
    let creatorProfileSummary = "";
    try {
      const profile = await getCreatorProfileSummary(userId);
      creatorProfileSummary = profile.summary;
    } catch { /* continue without */ }

    // Build prompt and generate
    const prompt = buildCreateMyVersionPrompt({
      referenceGuideContext,
      topic,
      promotingOrShowing: promotingOrShowing ?? topic,
      equipment: equipment ?? "smartphone",
      phoneOrCamera: phoneOrCamera ?? "smartphone",
      editingSoftware: editingSoftware ?? "CapCut",
      skillLevel,
      platform,
      desiredDuration: desiredDuration ?? "30 seconds",
      style: style ?? "engaging",
      differences: differences ?? "Make it feel fresh and original",
      creatorProfileSummary,
    });

    const customGuide = await generateCustomVersion(prompt);

    // Store the project insight
    if (process.env.HINDSIGHT_API_KEY) {
      try {
        await storeProjectInsight(
          `Creator created a custom "${topic}" Reel for ${platform}, skill level: ${skillLevel}, editing in: ${editingSoftware || "CapCut"}, duration: ${desiredDuration || "30s"}`,
          userId
        );
      } catch { /* non-critical */ }
    }

    return NextResponse.json({
      success: true,
      data: {
        customGuide,
        createdAt: new Date().toISOString(),
        referenceProjectId,
        memoryUsed: !!creatorProfileSummary,
        creatorProfileSummary,
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Failed to create version";
    console.error("[/api/create-version]", message);
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
