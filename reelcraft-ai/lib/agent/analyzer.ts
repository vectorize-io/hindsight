/**
 * Video Analysis Agent — Google Gemini Edition
 * Uses @google/genai SDK with gemini-3.8-flash (free tier, vision-capable).
 */

import { GoogleGenAI, type Part } from "@google/genai";
import type {
  VideoMetadata,
  RecreationGuide,
  SceneAnalysis,
  CreativeStrategy,
} from "@/types";
import { buildAnalysisPrompt } from "./prompts";
import type { ExtractedFrames } from "@/lib/video/metadata";

// ─── Client ───────────────────────────────────────────────────────────────────

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "your_gemini_api_key_here") {
    throw new Error(
      "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey and add it to .env.local"
    );
  }
  return new GoogleGenAI({ apiKey });
}

const MODEL = () => process.env.GEMINI_MODEL ?? "gemini-3.8-flash";

// ─── Retry helper ─────────────────────────────────────────────────────────────
// Retries up to 4 times with exponential backoff on 503 / rate-limit errors.
async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 4,
  baseDelayMs = 2000
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isRetryable =
        msg.includes("503") ||
        msg.includes("UNAVAILABLE") ||
        msg.includes("high demand") ||
        msg.includes("429") ||
        msg.includes("RESOURCE_EXHAUSTED");

      if (!isRetryable || attempt === maxAttempts) throw err;

      const delay = baseDelayMs * Math.pow(2, attempt - 1); // 2s, 4s, 8s
      console.warn(
        `Gemini attempt ${attempt} failed (${msg.slice(0, 80)}). Retrying in ${delay / 1000}s…`
      );
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastError;
}

// ─── Main Analysis Function ───────────────────────────────────────────────────

export interface AnalysisResult {
  guide: Omit<RecreationGuide, "projectId" | "analyzedAt">;
  scenes: SceneAnalysis[];
  creativeStrategy: CreativeStrategy;
  videoTitle: string;
}

export async function analyzeVideoFrames(
  metadata: VideoMetadata,
  frames: ExtractedFrames,
  creatorProfileSummary: string,
  projectId: string
): Promise<AnalysisResult> {
  const ai = getGeminiClient();

  const systemPrompt = buildAnalysisPrompt(
    metadata,
    frames.count,
    creatorProfileSummary
  );

  // Build content parts: text prompt + image frames
  const parts: Part[] = [
    { text: systemPrompt },
  ];

  // Add frame images as inline base64 parts (max 10)
  const frameTimestamps = frames.timestamps.slice(0, 10);
  for (let i = 0; i < Math.min(frames.frames.length, 10); i++) {
    const dataUri = frames.frames[i];
    // dataUri = "data:image/jpeg;base64,<b64>"
    const base64 = dataUri.replace(/^data:image\/\w+;base64,/, "");
    parts.push({
      inlineData: {
        mimeType: "image/jpeg",
        data: base64,
      },
    });
    parts.push({
      text: `Frame ${i + 1} at ${frameTimestamps[i]?.toFixed(1) ?? "?"}s`,
    });
  }

  parts.push({
    text: "Analyze the above frames and produce the complete JSON guide as specified. Return ONLY valid JSON, no markdown fences.",
  });

  let rawResponse = "";
  try {
    const response = await withRetry(() =>
      ai.models.generateContent({
        model: MODEL(),
        contents: [{ role: "user", parts }],
        config: { temperature: 0.4, maxOutputTokens: 8192 },
      })
    );
    rawResponse = response.text ?? "";
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : "Unknown Gemini error";
    throw new Error(`Gemini API error: ${msg}`);
  }

  return parseAnalysisResponse(rawResponse, projectId, metadata);
}

/**
 * Metadata-only analysis (no frames — URL mode fallback).
 */
export async function analyzeMetadataOnly(
  metadata: VideoMetadata,
  creatorProfileSummary: string,
  projectId: string,
  additionalContext = ""
): Promise<AnalysisResult> {
  const ai = getGeminiClient();

  const prompt = buildAnalysisPrompt(metadata, 0, creatorProfileSummary);

  const fullPrompt = [
    prompt,
    "\nNote: No visual frames were available. Base your analysis on the video metadata and any additional context provided.",
    additionalContext ? `\nADDITIONAL CONTEXT:\n${additionalContext}` : "",
    "\nReturn ONLY valid JSON, no markdown fences.",
  ]
    .filter(Boolean)
    .join("\n");

  const response = await withRetry(() =>
    ai.models.generateContent({
      model: MODEL(),
      contents: [{ role: "user", parts: [{ text: fullPrompt }] }],
      config: { temperature: 0.4, maxOutputTokens: 8192 },
    })
  );

  return parseAnalysisResponse(response.text ?? "", projectId, metadata);
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export async function chatWithAgent(
  messages: Array<{ role: "user" | "assistant"; content: string }>,
  systemPrompt: string,
  projectContext?: string
): Promise<string> {
  const ai = getGeminiClient();

  // Gemini uses "model" for assistant role
  const history = messages.slice(0, -1).map((m) => ({
    role: m.role === "assistant" ? ("model" as const) : ("user" as const),
    parts: [{ text: m.content }],
  }));

  const lastMessage = messages[messages.length - 1];
  const userText = [
    systemPrompt,
    projectContext ? `\n\nCURRENT PROJECT CONTEXT:\n${projectContext}` : "",
    // Prepend context only on first turn; for subsequent turns it's in history
    messages.length === 1 ? "" : "",
    `\n\nUser: ${lastMessage?.content ?? ""}`,
  ]
    .filter(Boolean)
    .join("");

  const response = await withRetry(() =>
    ai.models.generateContent({
      model: MODEL(),
      contents: [
        ...history,
        { role: "user", parts: [{ text: userText }] },
      ],
      config: { temperature: 0.7, maxOutputTokens: 2048 },
    })
  );

  return response.text ?? "I couldn't generate a response. Please try again.";
}

// ─── Memory Extraction ────────────────────────────────────────────────────────

export async function extractPreferencesFromMessage(
  userMessage: string
): Promise<
  Array<{ content: string; category: "profile" | "creative" | "workflow" | "project" }>
> {
  // Quick heuristic — skip API call for messages without preference signals
  const lower = userMessage.toLowerCase();
  const hasSignals = [
    "i use", "i edit", "i shoot", "my phone", "my camera", "i have",
    "i prefer", "i like", "i work", "i am a", "i'm a",
    "beginner", "intermediate", "advanced",
    "premiere", "capcut", "davinci", "iphone", "samsung", "android",
    "fast", "slow", "cinematic", "food", "travel", "fitness",
    "instagram", "tiktok", "youtube", "seconds",
  ].some((s) => lower.includes(s));

  if (!hasSignals) return [];

  try {
    const ai = getGeminiClient();

    const prompt = `Extract creator preferences from this message. Return a JSON array of preferences to store long-term.
Each item: { "content": "...", "category": "profile|creative|workflow|project" }
Return [] if nothing meaningful.

MESSAGE: "${userMessage}"

Return ONLY a valid JSON array, nothing else.`;

    const response = await withRetry(() =>
      ai.models.generateContent({
        model: "gemini-3.8-flash", // always use flash for cheap extraction
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        config: { temperature: 0, maxOutputTokens: 512 },
      })
    );

    const text = response.text ?? "[]";
    const cleaned = text.replace(/```json\n?|\n?```/g, "").trim();
    return JSON.parse(cleaned);
  } catch {
    return [];
  }
}

// ─── Create My Version ────────────────────────────────────────────────────────

export async function generateCustomVersion(prompt: string): Promise<string> {
  const ai = getGeminiClient();

  const response = await withRetry(() =>
    ai.models.generateContent({
      model: MODEL(),
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `You are ReelCraft AI — a professional content creation coach. Generate a completely original, specific, actionable Reel production plan.\n\n${prompt}`,
            },
          ],
        },
      ],
      config: { temperature: 0.7, maxOutputTokens: 4096 },
    })
  );

  return response.text ?? "Unable to generate custom version. Please try again.";
}

// ─── JSON Parser ──────────────────────────────────────────────────────────────

function parseAnalysisResponse(
  raw: string,
  projectId: string,
  metadata: VideoMetadata
): AnalysisResult {
  // Strip markdown fences if Gemini added them
  let cleaned = raw
    .replace(/^```json\s*/m, "")
    .replace(/^```\s*/m, "")
    .replace(/```\s*$/m, "")
    .trim();

  // Extract the outermost JSON object
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start !== -1 && end !== -1) {
    cleaned = cleaned.slice(start, end + 1);
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    console.error("Failed to parse Gemini JSON response. Length:", raw.length);
    return buildFallbackResult(projectId, metadata);
  }

  const videoTitle = (parsed.videoTitle as string) ?? "Analyzed Reel";

  const guide: Omit<RecreationGuide, "projectId" | "analyzedAt"> = {
    videoTitle,
    section1_overview: (parsed.section1_overview as RecreationGuide["section1_overview"]) ?? {
      about: "Analysis completed.",
      whatMakesItEffective: "Unable to determine",
      targetAudience: "General audience",
      contentCategory: "Unknown",
      approximateDuration: `${Math.round(metadata.duration)} seconds`,
      overallStyle: "Modern",
      mainCreativeIdea: "Unable to determine",
    },
    section2_skillLevel: (parsed.section2_skillLevel as RecreationGuide["section2_skillLevel"]) ?? {
      level: "Beginner",
      explanation: "",
      mustKnow: [],
      optional: [],
    },
    section3_skillsToLearn:  (parsed.section3_skillsToLearn  as RecreationGuide["section3_skillsToLearn"])  ?? [],
    section4_equipment:      (parsed.section4_equipment      as RecreationGuide["section4_equipment"])      ?? [],
    section5_apps:           (parsed.section5_apps           as RecreationGuide["section5_apps"])           ?? [],
    section6_techniques:     (parsed.section6_techniques     as RecreationGuide["section6_techniques"])     ?? [],
    section7_preProduction:  (parsed.section7_preProduction  as RecreationGuide["section7_preProduction"])  ?? { steps: [], checklist: [] },
    section8_script:         (parsed.section8_script         as RecreationGuide["section8_script"])         ?? [],
    section9_shotList:       (parsed.section9_shotList       as RecreationGuide["section9_shotList"])       ?? [],
    section10_filmingInstructions:  (parsed.section10_filmingInstructions  as string) ?? "",
    section11_lightingSetup:        (parsed.section11_lightingSetup        as string) ?? "",
    section12_audioMusic:           (parsed.section12_audioMusic           as string) ?? "",
    section13_editingTimeline:      (parsed.section13_editingTimeline      as string) ?? "",
    section14_appSpecificGuide:     (parsed.section14_appSpecificGuide     as string) ?? "",
    section15_colorGrading: (parsed.section15_colorGrading as RecreationGuide["section15_colorGrading"]) ?? {
      style: "natural",
      exposure: "", contrast: "", highlights: "", shadows: "",
      saturation: "", temperature: "", tint: "", skinTones: "",
      beginnerSettings: "",
    },
    section16_textGraphics:         (parsed.section16_textGraphics         as string) ?? "",
    section17_transitions:   (parsed.section17_transitions   as RecreationGuide["section17_transitions"])   ?? [],
    section18_exportSettings:(parsed.section18_exportSettings as RecreationGuide["section18_exportSettings"]) ?? [
      { platform: "Instagram Reels", resolution: "1080×1920", aspectRatio: "9:16", frameRate: "30fps", codec: "H.264", bitrate: "10-20 Mbps", audioFormat: "AAC 320kbps" },
    ],
    section19_publishing: (parsed.section19_publishing as RecreationGuide["section19_publishing"]) ?? {
      captionStructure: "", cta: "", hashtags: [], title: "", coverIdea: "", postingChecklist: [],
    },
    section20_beginnerVersion:      (parsed.section20_beginnerVersion      as string) ?? "",
    section21_professionalVersion:  (parsed.section21_professionalVersion  as string) ?? "",
    section22_timeBudget:           (parsed.section22_timeBudget           as string) ?? "",
    section23_finalChecklist: (parsed.section23_finalChecklist as string[]) ?? [],
  };

  const scenes: SceneAnalysis[]       = (parsed.scenes           as SceneAnalysis[])     ?? [];
  const creativeStrategy: CreativeStrategy = (parsed.creativeStrategy as CreativeStrategy) ?? {
    hook: "", storytellingStructure: "", emotionalTone: "", targetAudience: "",
    contentCategory: "", pacing: "", visualIdentity: "", cta: "",
    retentionTechniques: [], patternInterrupts: [], useOfCuriosity: "", useOfText: "",
  };

  return { guide, scenes, creativeStrategy, videoTitle };
}

function buildFallbackResult(projectId: string, metadata: VideoMetadata): AnalysisResult {
  return {
    videoTitle: "Reel Analysis",
    guide: {
      videoTitle: "Reel Analysis",
      section1_overview: {
        about: "Analysis could not be completed. Please try again.",
        whatMakesItEffective: "", targetAudience: "General audience",
        contentCategory: "Unknown",
        approximateDuration: `${Math.round(metadata.duration)} seconds`,
        overallStyle: "Unknown", mainCreativeIdea: "",
      },
      section2_skillLevel: { level: "Beginner", explanation: "", mustKnow: [], optional: [] },
      section3_skillsToLearn: [], section4_equipment: [], section5_apps: [],
      section6_techniques: [], section7_preProduction: { steps: [], checklist: [] },
      section8_script: [], section9_shotList: [],
      section10_filmingInstructions: "Analysis incomplete — please re-analyze.",
      section11_lightingSetup: "", section12_audioMusic: "",
      section13_editingTimeline: "", section14_appSpecificGuide: "",
      section15_colorGrading: {
        style: "natural", exposure: "", contrast: "", highlights: "",
        shadows: "", saturation: "", temperature: "", tint: "",
        skinTones: "", beginnerSettings: "",
      },
      section16_textGraphics: "", section17_transitions: [],
      section18_exportSettings: [], section19_publishing: {
        captionStructure: "", cta: "", hashtags: [], title: "", coverIdea: "", postingChecklist: [],
      },
      section20_beginnerVersion: "", section21_professionalVersion: "",
      section22_timeBudget: "", section23_finalChecklist: [],
    },
    scenes: [],
    creativeStrategy: {
      hook: "", storytellingStructure: "", emotionalTone: "", targetAudience: "",
      contentCategory: "", pacing: "", visualIdentity: "", cta: "",
      retentionTechniques: [], patternInterrupts: [], useOfCuriosity: "", useOfText: "",
    },
  };
}
