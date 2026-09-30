/**
 * ReelCraft AI — Hindsight Memory Service
 *
 * Stores and retrieves meaningful creator preferences, not raw chat logs.
 * Follows a "store what matters" strategy:
 *   - User profile (skill level, device, software)
 *   - Creative preferences (style, pacing, transitions)
 *   - Workflow (apps, setup)
 *   - Past project insights
 */

import { getBankId, getHindsightClient } from "./client";
import type { MemoryEntry, UserProfile, CreativePreferences, WorkflowPreferences } from "@/types";

// ─── Bank Initialization ─────────────────────────────────────────────────────

export async function ensureBankExists(userId?: string): Promise<string> {
  const client = getHindsightClient();
  const bankId = getBankId(userId);

  try {
    await client.getBankProfile(bankId);
  } catch {
    // Bank doesn't exist yet — create it
    await client.createBank(bankId, {
      name: "ReelCraft AI Creator Memory",
      background:
        "This memory bank stores the creator's skill level, equipment preferences, editing software, creative style preferences, and workflow habits. It is used to personalize Reel recreation guides and content creation advice.",
      disposition: {
        skepticism: 2,   // trust the user's stated preferences
        literalism: 3,   // interpret factual statements literally
        empathy: 5,      // be supportive of the creator's journey
      },
    });
  }

  return bankId;
}

// ─── Retain (Store) ───────────────────────────────────────────────────────────

/**
 * Store a meaningful preference or fact about the user.
 * Only call this with semantically important information — not every message.
 */
export async function retainMemory(
  content: string,
  category: "profile" | "creative" | "workflow" | "project",
  userId?: string
): Promise<void> {
  const client = getHindsightClient();
  const bankId = await ensureBankExists(userId);

  await client.retain(bankId, content, {
    context: `Category: ${category}. ReelCraft AI creator preference.`,
    timestamp: new Date(),
    metadata: { category, app: "reelcraft-ai" },
  });
}

/**
 * Store multiple preferences at once (e.g., after a profile setup conversation).
 */
export async function retainBatch(
  items: Array<{ content: string; category: "profile" | "creative" | "workflow" | "project" }>,
  userId?: string
): Promise<void> {
  const client = getHindsightClient();
  const bankId = await ensureBankExists(userId);

  await client.retainBatch(
    bankId,
    items.map((item) => ({
      content: `[${item.category.toUpperCase()}] ${item.content}`,
    }))
  );
}

// ─── Recall (Search) ──────────────────────────────────────────────────────────

/**
 * Retrieve memories relevant to a query.
 * Returns an array of memory text strings.
 */
export async function recallMemories(
  query: string,
  userId?: string,
  maxResults = 8
): Promise<MemoryEntry[]> {
  const client = getHindsightClient();
  const bankId = await ensureBankExists(userId);

  const result = await client.recall(bankId, query, {
    budget: "mid",
  });

  return result.results.slice(0, maxResults).map((r, idx) => ({
    id: `mem-${Date.now()}-${idx}`,
    text: r.text,
    type: r.type ?? "observation",
    learnedAt: new Date().toISOString(),
    category: inferCategory(r.text),
    whyItMatters: inferWhyItMatters(r.text),
  }));
}

/**
 * Get all stored memories for display on the Creative Memory page.
 */
export async function listAllMemories(userId?: string): Promise<MemoryEntry[]> {
  const client = getHindsightClient();
  const bankId = await ensureBankExists(userId);

  const result = await client.listMemories(bankId, { limit: 100, offset: 0 });

  // The SDK returns MemoryUnitListItem objects; extract the text content safely
  const items = result.items as unknown as Array<string | { text?: string; content?: string }>;
  return items.map((item, idx) => {
    const text = typeof item === "string"
      ? item
      : item.text ?? item.content ?? String(item);
    return {
      id: `mem-${idx}`,
      text,
      type: "observation",
      learnedAt: new Date().toISOString(),
      category: inferCategory(text),
      whyItMatters: inferWhyItMatters(text),
    };
  });
}

// ─── Reflect (AI-powered reasoning over memories) ────────────────────────────

/**
 * Get an AI-synthesized answer based on stored memories.
 * Used to build personalized context before generating a guide.
 */
export async function reflectOnMemory(
  query: string,
  context?: string,
  userId?: string
): Promise<string> {
  const client = getHindsightClient();
  const bankId = await ensureBankExists(userId);

  const response = await client.reflect(bankId, query, {
    context: context ?? "Generating a personalized Reel recreation guide.",
    budget: "mid",
  });

  return response.text;
}

/**
 * Build a comprehensive creator profile summary from memory.
 * This is injected into the system prompt when generating guides.
 */
export async function getCreatorProfileSummary(userId?: string): Promise<{
  summary: string;
  memories: MemoryEntry[];
  hasProfile: boolean;
}> {
  try {
    const memories = await recallMemories(
      "creator skill level, equipment, editing software, preferred style, filming device, platform preferences",
      userId,
      15
    );

    if (memories.length === 0) {
      return { summary: "", memories: [], hasProfile: false };
    }

    const summary = await reflectOnMemory(
      "Summarize this creator's skill level, equipment, editing software, creative preferences, and workflow in 3-4 concise sentences. Focus on what will help personalize a Reel recreation guide.",
      undefined,
      userId
    );

    return { summary, memories, hasProfile: true };
  } catch {
    return { summary: "", memories: [], hasProfile: false };
  }
}

// ─── Smart Memory Extraction ─────────────────────────────────────────────────

/**
 * Analyze a user message and extract any meaningful preferences to store.
 * Call this after every user message that might contain preferences.
 */
export async function extractAndStorePreferences(
  userMessage: string,
  userId?: string
): Promise<string[]> {
  // We use simple heuristics to avoid storing every message.
  // The LLM-based extraction happens in the agent layer.
  const stored: string[] = [];

  const profileKeywords = [
    "beginner", "intermediate", "advanced", "expert",
    "i shoot", "i film", "i use", "my phone", "my camera",
    "premiere pro", "capcut", "davinci", "final cut", "resolve",
    "iphone", "samsung", "pixel", "android",
    "gopro", "mirrorless", "dslr",
    "microphone", "ring light", "softbox", "gimbal", "tripod",
  ];

  const creativeKeywords = [
    "fast-paced", "slow", "cinematic", "minimal", "colorful",
    "warm", "cool", "moody", "vibrant", "aesthetic",
    "food", "travel", "lifestyle", "fitness", "tech", "comedy",
    "voiceover", "no voice", "subtitles", "captions",
    "20 second", "30 second", "60 second", "15 second",
    "instagram", "tiktok", "youtube", "reels", "shorts",
  ];

  const lower = userMessage.toLowerCase();
  const hasProfileInfo = profileKeywords.some((k) => lower.includes(k));
  const hasCreativeInfo = creativeKeywords.some((k) => lower.includes(k));

  if (hasProfileInfo || hasCreativeInfo) {
    stored.push(userMessage);
  }

  return stored;
}

// ─── Profile Structured Storage ──────────────────────────────────────────────

export async function storeUserProfile(
  profile: Partial<UserProfile>,
  userId?: string
): Promise<void> {
  const items: Array<{ content: string; category: "profile" }> = [];

  if (profile.skillLevel)
    items.push({ content: `Creator skill level: ${profile.skillLevel}`, category: "profile" });
  if (profile.preferredEditingSoftware)
    items.push({ content: `Preferred editing software: ${profile.preferredEditingSoftware}`, category: "profile" });
  if (profile.preferredFilmingDevice)
    items.push({ content: `Preferred filming device: ${profile.preferredFilmingDevice}`, category: "profile" });
  if (profile.cameraEquipment)
    items.push({ content: `Camera equipment: ${profile.cameraEquipment}`, category: "profile" });
  if (profile.microphone)
    items.push({ content: `Microphone: ${profile.microphone}`, category: "profile" });
  if (profile.lightingEquipment)
    items.push({ content: `Lighting equipment: ${profile.lightingEquipment}`, category: "profile" });
  if (profile.preferredPlatforms?.length)
    items.push({ content: `Preferred platforms: ${profile.preferredPlatforms.join(", ")}`, category: "profile" });
  if (profile.preferredReelDuration)
    items.push({ content: `Preferred Reel duration: ${profile.preferredReelDuration}`, category: "profile" });
  if (profile.preferredAspectRatio)
    items.push({ content: `Preferred aspect ratio: ${profile.preferredAspectRatio}`, category: "profile" });

  if (items.length > 0) await retainBatch(items, userId);
}

export async function storeCreativePreferences(
  prefs: Partial<CreativePreferences>,
  userId?: string
): Promise<void> {
  const items: Array<{ content: string; category: "creative" }> = [];

  if (prefs.editingStyle)
    items.push({ content: `Preferred editing style: ${prefs.editingStyle}`, category: "creative" });
  if (prefs.pacing)
    items.push({ content: `Preferred editing pacing: ${prefs.pacing}`, category: "creative" });
  if (prefs.transitions)
    items.push({ content: `Preferred transition style: ${prefs.transitions}`, category: "creative" });
  if (prefs.colorStyle)
    items.push({ content: `Preferred color grade style: ${prefs.colorStyle}`, category: "creative" });
  if (prefs.musicStyle)
    items.push({ content: `Preferred music style: ${prefs.musicStyle}`, category: "creative" });
  if (prefs.contentCategories?.length)
    items.push({ content: `Content categories creator makes: ${prefs.contentCategories.join(", ")}`, category: "creative" });
  if (prefs.hookStyle)
    items.push({ content: `Preferred hook style: ${prefs.hookStyle}`, category: "creative" });

  if (items.length > 0) await retainBatch(items, userId);
}

export async function storeWorkflowPreferences(
  workflow: Partial<WorkflowPreferences>,
  userId?: string
): Promise<void> {
  const items: Array<{ content: string; category: "workflow" }> = [];

  if (workflow.editingSoftware)
    items.push({ content: `Editing software used: ${workflow.editingSoftware}`, category: "workflow" });
  if (workflow.filmingSetup)
    items.push({ content: `Filming setup: ${workflow.filmingSetup}`, category: "workflow" });
  if (workflow.appsOwned?.length)
    items.push({ content: `Apps the creator owns: ${workflow.appsOwned.join(", ")}`, category: "workflow" });
  if (workflow.availableEquipment?.length)
    items.push({ content: `Available equipment: ${workflow.availableEquipment.join(", ")}`, category: "workflow" });
  if (workflow.preferredWorkflow)
    items.push({ content: `Preferred workflow: ${workflow.preferredWorkflow}`, category: "workflow" });

  if (items.length > 0) await retainBatch(items, userId);
}

export async function storeProjectInsight(
  insight: string,
  userId?: string
): Promise<void> {
  await retainMemory(insight, "project", userId);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function inferCategory(text: string): "profile" | "creative" | "workflow" | "project" {
  const lower = text.toLowerCase();
  if (
    lower.includes("skill") ||
    lower.includes("device") ||
    lower.includes("camera") ||
    lower.includes("microphone") ||
    lower.includes("platform") ||
    lower.includes("aspect ratio") ||
    lower.includes("filming device")
  )
    return "profile";
  if (
    lower.includes("style") ||
    lower.includes("pacing") ||
    lower.includes("color") ||
    lower.includes("music") ||
    lower.includes("transition") ||
    lower.includes("hook") ||
    lower.includes("content categor")
  )
    return "creative";
  if (
    lower.includes("software") ||
    lower.includes("app") ||
    lower.includes("workflow") ||
    lower.includes("editing") ||
    lower.includes("equipment")
  )
    return "workflow";
  return "project";
}

function inferWhyItMatters(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes("skill")) return "Adapts guide complexity to your experience level";
  if (lower.includes("premiere") || lower.includes("capcut") || lower.includes("davinci") || lower.includes("final cut"))
    return "Tailors editing steps to your specific software";
  if (lower.includes("phone") || lower.includes("iphone") || lower.includes("android"))
    return "Ensures filming instructions match your device";
  if (lower.includes("pacing") || lower.includes("fast") || lower.includes("slow"))
    return "Matches editing rhythm to your preferred style";
  if (lower.includes("platform") || lower.includes("instagram") || lower.includes("tiktok"))
    return "Optimizes export settings for your target platform";
  if (lower.includes("duration") || lower.includes("second"))
    return "Adjusts guide timing to your preferred Reel length";
  if (lower.includes("equipment") || lower.includes("gimbal") || lower.includes("tripod"))
    return "Recommends techniques based on what you own";
  if (lower.includes("style") || lower.includes("aesthetic"))
    return "Shapes creative recommendations toward your aesthetic";
  return "Helps personalize future Reel recreation guides";
}
