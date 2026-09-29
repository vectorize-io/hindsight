/**
 * POST /api/chat
 * Persistent AI chatbot with Hindsight memory.
 * Pipeline:
 *   1. Recall relevant memories from Hindsight
 *   2. Build system prompt with creator profile
 *   3. Generate AI response
 *   4. Extract & store any new preferences mentioned
 *   5. Return response + memories used
 */

import { NextRequest, NextResponse } from "next/server";
import { chatWithAgent, extractPreferencesFromMessage } from "@/lib/agent";
import { CHAT_SYSTEM_PROMPT } from "@/lib/agent/prompts";
import { recallMemories, retainBatch, getCreatorProfileSummary } from "@/lib/hindsight";
import { getProjectById } from "@/lib/db";

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
    const body = await req.json() as {
      messages: Array<{ role: "user" | "assistant"; content: string }>;
      projectId?: string;
    };

    const { messages, projectId } = body;

    if (!messages?.length) {
      return NextResponse.json({ success: false, error: "messages array is required" }, { status: 400 });
    }

    const latestUserMessage = messages.filter((m) => m.role === "user").pop()?.content ?? "";

    // ── Recall relevant memories ─────────────────────────────────────────────
    let memoriesUsed: string[] = [];
    let creatorSummary = "";
    try {
      const recalled = await recallMemories(latestUserMessage, userId, 6);
      memoriesUsed = recalled.map((m) => m.text);

      const profile = await getCreatorProfileSummary(userId);
      creatorSummary = profile.summary;
    } catch {
      /* Hindsight unavailable — continue without memory */
    }

    // ── Build enriched system prompt ─────────────────────────────────────────
    let systemPrompt = CHAT_SYSTEM_PROMPT;

    if (creatorSummary) {
      systemPrompt += `\n\n## CREATOR PROFILE (from Hindsight memory)\n${creatorSummary}`;
    }

    if (memoriesUsed.length > 0) {
      systemPrompt += `\n\n## RELEVANT MEMORIES\n${memoriesUsed.map((m) => `• ${m}`).join("\n")}`;
    }

    // ── Load project context if provided ─────────────────────────────────────
    let projectContext: string | undefined;
    if (projectId) {
      const project = getProjectById(projectId);
      if (project?.guide) {
        const g = project.guide;
        projectContext = [
          `Currently discussing project: "${project.title}"`,
          `Content category: ${g.section1_overview.contentCategory}`,
          `Skill level: ${g.section2_skillLevel.level}`,
          `Style: ${g.section1_overview.overallStyle}`,
          `Target audience: ${g.section1_overview.targetAudience}`,
        ].join("\n");
      }
    }

    // ── Generate response ─────────────────────────────────────────────────────
    const response = await chatWithAgent(messages, systemPrompt, projectContext);

    // ── Extract and store new preferences ────────────────────────────────────
    if (process.env.HINDSIGHT_API_KEY && latestUserMessage.length > 10) {
      try {
        const newPrefs = await extractPreferencesFromMessage(latestUserMessage);
        if (newPrefs.length > 0) {
          await retainBatch(newPrefs, userId);
          newPrefs.forEach((p) => memoriesUsed.push(`[Stored] ${p.content}`));
        }
      } catch {
        /* non-critical */
      }
    }

    return NextResponse.json({
      success: true,
      data: {
        message: response,
        memoriesUsed,
        hasMemory: memoriesUsed.length > 0,
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Chat failed";
    console.error("[/api/chat]", message);
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
