/**
 * POST /api/analyze
 * Accepts a multipart/form-data video upload OR a JSON body with a videoUrl.
 * Pipeline:
 *   1. Validate the file / URL
 *   2. Extract metadata + frames (file upload only)
 *   3. Recall creator profile from Hindsight
 *   4. Run Gemini vision analysis → 23-section guide
 *   5. Store project in DB
 *   6. Store project insight in Hindsight
 *   7. Return the full project + memory context
 */

import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import path from "path";
import fs from "fs";
import type { Project } from "@/types";
import { validateUploadedFile, validateVideoUrl, extractVideoMetadata, extractFrames } from "@/lib/video";
import { analyzeVideoFrames, analyzeMetadataOnly } from "@/lib/agent";
import { getCreatorProfileSummary, storeProjectInsight, ensureBankExists } from "@/lib/hindsight";
import { createProject, updateProject, ensureUploadsDir, getTmpDir, cleanupTmpFiles } from "@/lib/db";
import { generateId } from "@/lib/utils";

export const runtime = "nodejs";
export const maxDuration = 120; // seconds

// Disable Next.js default body parser — we handle multipart ourselves
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const userId = req.headers.get("x-user-id") ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";
  const contentType = req.headers.get("content-type") ?? "";
  const isDemo = req.headers.get("x-demo-mode") === "true";

  // ── Validate API keys ────────────────────────────────────────────────────────
  if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === "your_gemini_api_key_here") {
    return NextResponse.json(
      { success: false, error: "GEMINI_API_KEY is not configured. Get a free key at https://aistudio.google.com/apikey and add it to .env.local." },
      { status: 503 }
    );
  }

  const projectId = generateId();
  let filePath: string | undefined;
  let tmpFrameDir: string | undefined;

  try {
    // ── Ensure Hindsight bank exists ─────────────────────────────────────────
    try { await ensureBankExists(userId); } catch { /* continue without memory if Hindsight unavailable */ }

    // ── Recall creator profile from Hindsight ────────────────────────────────
    let creatorProfile = { summary: "", memories: [], hasProfile: false } as Awaited<ReturnType<typeof getCreatorProfileSummary>>;
    try {
      creatorProfile = await getCreatorProfileSummary(userId);
    } catch {
      // Hindsight unavailable — proceed without memory
    }

    // ── Parse request ────────────────────────────────────────────────────────
    if (contentType.includes("multipart/form-data")) {
      // ── File Upload path ─────────────────────────────────────────────────
      const formData = await req.formData();
      const file = formData.get("video") as File | null;

      if (!file) {
        return NextResponse.json({ success: false, error: "No video file provided" }, { status: 400 });
      }

      const validation = validateUploadedFile(file.name, file.type, file.size);
      if (!validation.valid) {
        return NextResponse.json({ success: false, error: validation.reason }, { status: 400 });
      }

      // Save to disk
      const uploadsDir = ensureUploadsDir();
      const ext = path.extname(file.name).toLowerCase() || ".mp4";
      const savedFilename = `${projectId}${ext}`;
      filePath = path.join(uploadsDir, savedFilename);

      const buffer = Buffer.from(await file.arrayBuffer());
      fs.writeFileSync(filePath, buffer);

      // Create pending project
      const pendingProject: Project = {
        id: projectId,
        title: sanitizeFilename(file.name),
        videoPath: `/uploads/${savedFilename}`,
        sourceType: "upload",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: "analyzing",
        userId,
        isDemo,
      };
      createProject(pendingProject);

      // Extract metadata
      const metadata = await extractVideoMetadata(filePath);

      // Extract frames
      tmpFrameDir = path.join(getTmpDir(), `frames-${projectId}`);
      const frames = await extractFrames(filePath, metadata, tmpFrameDir, 10);

      // Run analysis
      const analysisResult = await analyzeVideoFrames(
        metadata,
        frames,
        creatorProfile.summary,
        projectId
      );

      // Build complete guide
      const guide = {
        ...analysisResult.guide,
        projectId,
        analyzedAt: new Date().toISOString(),
      };

      // Update project
      const completedProject = updateProject(projectId, {
        title: analysisResult.videoTitle,
        status: "complete",
        metadata,
        scenes: analysisResult.scenes,
        creativeStrategy: analysisResult.creativeStrategy,
        guide,
      });

      // Store insight in Hindsight
      if (process.env.HINDSIGHT_API_KEY) {
        try {
          await storeProjectInsight(
            `Creator analyzed a Reel: "${analysisResult.videoTitle}" — category: ${analysisResult.creativeStrategy.contentCategory}, style: ${analysisResult.guide.section1_overview.overallStyle}, required skill: ${analysisResult.guide.section2_skillLevel.level}`,
            userId
          );
        } catch { /* non-critical */ }
      }

      // Cleanup temp frames
      if (tmpFrameDir) cleanupTmpFiles(tmpFrameDir);

      return NextResponse.json({
        success: true,
        data: {
          project: completedProject,
          memoryUsed: creatorProfile.hasProfile,
          recalledPreferences: creatorProfile.memories.map((m) => m.text),
        },
      });

    } else {
      // ── URL / JSON path ──────────────────────────────────────────────────
      const body = await req.json().catch(() => ({}));
      const { videoUrl, additionalContext } = body as { videoUrl?: string; additionalContext?: string };

      if (!videoUrl) {
        return NextResponse.json({ success: false, error: "Provide a video file or a videoUrl" }, { status: 400 });
      }

      const urlValidation = validateVideoUrl(videoUrl);
      if (!urlValidation.valid) {
        return NextResponse.json({ success: false, error: urlValidation.reason }, { status: 400 });
      }

      // URL analysis — metadata-only (we can't download arbitrary URLs securely)
      const fakeMetadata = {
        duration: 30,
        width: 1080,
        height: 1920,
        aspectRatio: "9:16",
        frameRate: 30,
        orientation: "portrait" as const,
        hasAudio: true,
        fileSize: 0,
        format: "mp4",
      };

      const pendingProject: Project = {
        id: projectId,
        title: `Reel from ${urlValidation.domain}`,
        videoUrl,
        sourceType: "url",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: "analyzing",
        userId,
        isDemo,
      };
      createProject(pendingProject);

      const context = [
        `Video URL: ${videoUrl}`,
        `Platform: ${urlValidation.domain}`,
        additionalContext ?? "",
      ].filter(Boolean).join("\n");

      const analysisResult = await analyzeMetadataOnly(
        fakeMetadata,
        creatorProfile.summary,
        projectId,
        context
      );

      const guide = {
        ...analysisResult.guide,
        projectId,
        analyzedAt: new Date().toISOString(),
      };

      const completedProject = updateProject(projectId, {
        title: analysisResult.videoTitle,
        status: "complete",
        metadata: fakeMetadata,
        scenes: analysisResult.scenes,
        creativeStrategy: analysisResult.creativeStrategy,
        guide,
      });

      if (process.env.HINDSIGHT_API_KEY) {
        try {
          await storeProjectInsight(
            `Creator analyzed a Reel from ${urlValidation.domain}: "${analysisResult.videoTitle}"`,
            userId
          );
        } catch { /* non-critical */ }
      }

      return NextResponse.json({
        success: true,
        data: {
          project: completedProject,
          memoryUsed: creatorProfile.hasProfile,
          recalledPreferences: creatorProfile.memories.map((m) => m.text),
        },
      });
    }
  } catch (error: unknown) {
    // Clean up on error
    if (tmpFrameDir) cleanupTmpFiles(tmpFrameDir);
    if (filePath && fs.existsSync(filePath)) {
      try { fs.unlinkSync(filePath); } catch { /* ignore */ }
    }

    // Mark project as error
    try { updateProject(projectId, { status: "error", error: String(error) }); } catch { /* ignore */ }

    const message = error instanceof Error ? error.message : "Analysis failed";
    console.error("[/api/analyze]", message);

    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}

function sanitizeFilename(name: string): string {
  return name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9\s-_]/g, "").trim() || "Uploaded Reel";
}
