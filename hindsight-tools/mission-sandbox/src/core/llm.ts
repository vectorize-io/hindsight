/**
 * LLM layer for mission refinement, backed by Google Gemini (@google/genai).
 *
 * The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) unless passed explicitly. Mission
 * refinement is the only LLM call the tool makes — there is no labeling or scoring.
 */

import { GoogleGenAI, Type } from "@google/genai";
import type { GenerateContentParameters, GenerateContentResponse } from "@google/genai";

import type { MissionKind } from "./types.js";

export interface CoverageResult {
  /** Indices (into the golden list) that are semantically reproduced by the candidate set. */
  coveredIndices: number[];
  missing: string[];
}

/** A golden memory to score, optionally carrying the Phase-1 edit that produced it. */
export interface GoldenForCoverage {
  text: string;
  curatedFrom?: string | null;
  curateReason?: string | null;
}

export const DEFAULT_MODEL = "gemini-2.5-flash";

const KIND_BLURB: Record<MissionKind, string> = {
  retain:
    `A "retain mission" steers what facts and entities get extracted from documents during ` +
    `ingestion. It is injected alongside the system's built-in extraction rules.`,
  observe:
    `An "observation mission" controls how raw facts get consolidated into observations — ` +
    `synthesized summaries derived from multiple facts.`,
};

function systemPrompt(kind: MissionKind): string {
  return `You are an expert at writing ${kind} missions for a memory system.

${KIND_BLURB[kind]}

You are given the current mission and the user's feedback (and optionally concrete failing
examples) gathered from an external evaluation of the memory. Rewrite the mission so it addresses
the feedback while preserving what already works.

Rules:
- Output a single improved mission: concise and actionable (a few sentences to a short paragraph).
- Fold the feedback in directly; do not just append it.
- If there is no current mission, write one from scratch that satisfies the feedback.
- Respond with ONLY the mission text — no preamble, explanation, or wrapper.`;
}

/** The Hindsight deployment's configured LLM, but only when it's a Gemini-family provider. */
function hindsightGeminiConfig(): { key?: string; model?: string } {
  const provider = (process.env.HINDSIGHT_API_LLM_PROVIDER ?? "").toLowerCase();
  // An empty provider is treated as Gemini-compatible; OpenAI/etc. keys won't work here.
  if (provider && !["gemini", "google", "vertexai"].includes(provider)) return {};
  return { key: process.env.HINDSIGHT_API_LLM_API_KEY, model: process.env.HINDSIGHT_API_LLM_MODEL };
}

function resolveApiKey(explicit?: string): string {
  const key =
    explicit ||
    process.env.GEMINI_API_KEY ||
    process.env.GOOGLE_API_KEY ||
    hindsightGeminiConfig().key;
  if (!key) {
    throw new Error(
      "No Gemini API key found. Set GEMINI_API_KEY/GOOGLE_API_KEY, or configure a Gemini " +
        "HINDSIGHT_API_LLM_* in your .env."
    );
  }
  return key;
}

function resolveModel(explicit?: string): string {
  return explicit?.trim() || hindsightGeminiConfig().model || DEFAULT_MODEL;
}

export class MissionLlm {
  private readonly ai: GoogleGenAI;
  readonly model: string;

  constructor(opts: { apiKey?: string; model?: string } = {}) {
    this.ai = new GoogleGenAI({ apiKey: resolveApiKey(opts.apiKey) });
    this.model = resolveModel(opts.model);
  }

  /**
   * generateContent with bounded exponential backoff on TRANSIENT Gemini errors (429/500/503,
   * UNAVAILABLE, overloaded). A single 503 used to abort a whole `retain check` run (no retry on the
   * per-doc coverage judge), so the loop never reached its summary. Permanent 4xx (bad key, etc.)
   * still fail fast.
   */
  private async generate(req: GenerateContentParameters): Promise<GenerateContentResponse> {
    const MAX_ATTEMPTS = 5;
    for (let attempt = 1; ; attempt++) {
      try {
        return await this.ai.models.generateContent(req);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        const transient =
          /"code":\s*(429|500|503)|UNAVAILABLE|INTERNAL|RESOURCE_EXHAUSTED|overloaded/i.test(msg);
        if (!transient || attempt >= MAX_ATTEMPTS) throw e;
        // Exponential backoff: 1s, 2s, 4s, 8s (capped) — attempt-based, no RNG needed.
        await new Promise((r) => setTimeout(r, Math.min(1000 * 2 ** (attempt - 1), 8000)));
      }
    }
  }

  /**
   * Refine (or create) a mission from the current mission + the user's feedback and optional
   * failing examples. This is the tool's single LLM operation.
   */
  async refineMission(
    kind: MissionKind,
    currentMission: string | null,
    feedback: string,
    examples: string[] = []
  ): Promise<string> {
    const examplesSection = examples.length
      ? `\n\n## Failing examples\n${examples.map((e) => `- ${e}`).join("\n")}`
      : "";
    const response = await this.generate({
      model: this.model,
      contents:
        `## Current mission\n${currentMission ?? "(no mission set — using system defaults)"}\n\n` +
        `## Feedback\n${feedback}${examplesSection}\n\nWrite the improved mission.`,
      config: { systemInstruction: systemPrompt(kind), temperature: 0.3 },
    });
    const text = response.text;
    if (!text) throw new Error("Empty response from Gemini while refining mission");
    return text.trim();
  }

  /**
   * Phase 2 objective: which golden memories are semantically reproduced by the candidate set?
   * One structured call — returns the covered golden indices and the missing golden texts.
   */
  async coverage(golden: GoldenForCoverage[], candidate: string[]): Promise<CoverageResult> {
    if (golden.length === 0) return { coveredIndices: [], missing: [] };
    const goldenList = golden
      .map((g, i) => {
        let line = `[${i}] ${g.text}`;
        if (g.curatedFrom) {
          line +=
            `\n    (CURATED — edited from "${g.curatedFrom}"` +
            (g.curateReason ? `; reason: ${g.curateReason}` : "") +
            `. Covered ONLY if the candidate reproduces THIS specific change, not just the original fact.)`;
        }
        return line;
      })
      .join("\n");
    const candList = candidate.length ? candidate.map((c) => `- ${c}`).join("\n") : "(none)";
    const response = await this.generate({
      model: this.model,
      contents:
        `GOLDEN memories (the target):\n${goldenList}\n\n` +
        `CANDIDATE memories (what a mission just extracted):\n${candList}\n\n` +
        `Return the indices of GOLDEN memories whose information is present in the CANDIDATE set ` +
        `(allow paraphrase / different wording), and the texts of those that are missing.`,
      config: {
        systemInstruction:
          "You compare two sets of extracted memories. A golden memory is 'covered' if a candidate " +
          "memory conveys the same fact, even if worded differently or split/merged. For a golden " +
          "memory marked CURATED, it is covered only if the candidate reproduces the curated change " +
          "(the specific improvement noted), not merely the pre-edit fact.",
        temperature: 0,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            coveredIndices: { type: Type.ARRAY, items: { type: Type.INTEGER } },
            missing: { type: Type.ARRAY, items: { type: Type.STRING } },
          },
          required: ["coveredIndices", "missing"],
          propertyOrdering: ["coveredIndices", "missing"],
        },
      },
    });
    const txt = response.text;
    if (!txt) throw new Error("Empty response from Gemini while computing coverage");
    const parsed = JSON.parse(txt) as { coveredIndices?: number[]; missing?: string[] };
    return { coveredIndices: parsed.coveredIndices ?? [], missing: parsed.missing ?? [] };
  }
}
