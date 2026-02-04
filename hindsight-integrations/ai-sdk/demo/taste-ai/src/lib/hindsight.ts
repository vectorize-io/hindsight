// @ts-nocheck - TypeScript has issues resolving tool types from local package
import { createHindsightTools, type HindsightClient } from '@vectorize-io/hindsight-ai-sdk';
import { groq } from '@ai-sdk/groq';

const HINDSIGHT_URL = process.env.HINDSIGHT_URL || 'http://localhost:8888';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';

// Simple HTTP client implementation
class SimpleHindsightClient implements HindsightClient {
  constructor(private baseUrl: string) {}

  async retain(
    bankId: string,
    content: string,
    options?: {
      timestamp?: Date | string;
      context?: string;
      metadata?: Record<string, string>;
      documentId?: string;
      async?: boolean;
    }
  ) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/memories`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        items: [{
          content,
          timestamp: options?.timestamp,
          context: options?.context,
          metadata: options?.metadata,
          document_id: options?.documentId,
        }],
        async: options?.async,
      }),
    });

    if (!response.ok) {
      throw new Error(`Retain failed: ${response.statusText}`);
    }

    return response.json();
  }

  async recall(
    bankId: string,
    query: string,
    options?: {
      types?: string[];
      maxTokens?: number;
      budget?: 'low' | 'mid' | 'high';
      trace?: boolean;
      queryTimestamp?: string;
      includeEntities?: boolean;
      maxEntityTokens?: number;
      includeChunks?: boolean;
      maxChunkTokens?: number;
    }
  ) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/memories/recall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        types: options?.types,
        max_tokens: options?.maxTokens,
        budget: options?.budget,
        trace: options?.trace,
        query_timestamp: options?.queryTimestamp,
        include_entities: options?.includeEntities,
        max_entity_tokens: options?.maxEntityTokens,
        include_chunks: options?.includeChunks,
        max_chunk_tokens: options?.maxChunkTokens,
      }),
    });

    if (!response.ok) {
      throw new Error(`Recall failed: ${response.statusText}`);
    }

    return response.json();
  }

  async reflect(
    bankId: string,
    query: string,
    options?: {
      context?: string;
      budget?: 'low' | 'mid' | 'high';
    }
  ) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/reflect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        context: options?.context,
        budget: options?.budget,
      }),
    });

    if (!response.ok) {
      throw new Error(`Reflect failed: ${response.statusText}`);
    }

    return response.json();
  }

  async createMentalModel(
    bankId: string,
    options?: {
      name?: string;
      sourceQuery?: string;
      tags?: string[];
      maxTokens?: number;
      trigger?: { refresh_after_consolidation?: boolean };
    }
  ) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/mental-models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: options?.name,
        source_query: options?.sourceQuery,
        tags: options?.tags,
        max_tokens: options?.maxTokens,
        trigger: options?.trigger,
      }),
    });

    if (!response.ok) {
      throw new Error(`Create mental model failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getMentalModel(bankId: string, mentalModelId: string) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/mental-models/${mentalModelId}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`Get mental model failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Helper method for document operations
  async getDocument(bankId: string, documentId: string) {
    const response = await fetch(`${this.baseUrl}/v1/default/banks/${bankId}/documents/${documentId}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`Get document failed: ${response.statusText}`);
    }

    return response.json();
  }
}

const hindsightClient = new SimpleHindsightClient(HINDSIGHT_URL);

export const llmModel = groq(GROQ_MODEL);

// AI SDK v6 tools for agent-based interactions
export const hindsightTools = createHindsightTools({
  client: hindsightClient,
});

console.log(`[TasteAI] Connected to Hindsight at ${HINDSIGHT_URL}`);
console.log(`[TasteAI] Using LLM model: ${GROQ_MODEL}`);

// Document storage helpers
export const BANK_ID = 'taste-ai'; // Shared bank for all taste-ai users

// Helper to normalize username for document/tag IDs
export function normalizeUsername(username?: string): string {
  if (!username) return 'guest';
  return username.toLowerCase().replace(/[^a-z0-9]/g, '-');
}

// Get document ID for a specific user
export function getDocumentId(username?: string): string {
  return `app-state-${normalizeUsername(username)}`;
}

export interface StoredMeal {
  id: string;
  name: string;
  emoji: string;
  description?: string;
  type: string;
  date: string;
  timestamp: string;
  healthScore?: number;
  timeMinutes?: number;
  ingredients?: string[];
  instructions?: string;
  tags?: string[];
  action: 'ate' | 'cooked';
}

export interface HealthAssessment {
  score: number;
  trend: 'up' | 'down' | 'stable';
  insight: string;
  assessedAt: string;
  mealsCountAtAssessment: number;
}

export interface UserPreferences {
  nickname?: string;
  language?: string;
  cuisines?: string[];
  dietary?: string[];
  goals?: string[];
  dislikes?: string[];
}

export interface AppDocument {
  username: string; // The user's nickname
  meals: StoredMeal[];
  preferences: UserPreferences;
  healthMentalModelId?: string;
  updatedAt: string;
}

export async function getAppDocument(username: string): Promise<AppDocument> {
  const docId = getDocumentId(username);

  try {
    // @ts-ignore - TS can't resolve tool types from local package
    const response = await hindsightTools.getDocument.execute({
      bankId: BANK_ID,
      documentId: docId,
    });

    if (response?.originalText) {
      const doc = JSON.parse(response.originalText) as AppDocument;

      // Fix nested structure from old data (unwrap if needed)
      if (doc.preferences?.preferences) {
        doc.preferences = doc.preferences.preferences as UserPreferences;
      }

      return doc;
    }
  } catch (e) {
    console.log(`[TasteAI] No app document yet for ${username}, returning empty`);
  }

  return {
    username,
    meals: [],
    preferences: { nickname: username },
    updatedAt: new Date().toISOString()
  };
}

export async function saveAppDocument(doc: AppDocument): Promise<void> {
  const docId = getDocumentId(doc.username);
  const jsonContent = JSON.stringify(doc);

  // Use AI SDK tools for all Hindsight operations
  await hindsightTools.retain.execute({
    bankId: BANK_ID,
    content: jsonContent,
    documentId: docId,
    tags: [`user:${doc.username}`],
  });

  console.log(`[TasteAI] Saved app document for ${doc.username} (${doc.meals.length} meals)`);
}

// Backwards compat alias
export const getMealsDocument = getAppDocument;
export const saveMealsDocument = saveAppDocument;

export async function addMealToDocument(username: string, meal: Omit<StoredMeal, 'id' | 'timestamp'>): Promise<StoredMeal> {
  const doc = await getAppDocument(username);

  const newMeal: StoredMeal = {
    ...meal,
    id: `meal-${Date.now()}`,
    timestamp: new Date().toISOString(),
  };

  doc.meals.unshift(newMeal);
  doc.meals = doc.meals.slice(0, 50); // Keep last 50
  doc.updatedAt = new Date().toISOString();

  await saveAppDocument(doc);

  return newMeal;
}

export async function updatePreferences(username: string, prefs: Partial<UserPreferences>): Promise<void> {
  const doc = await getAppDocument(username);

  // Fix nested structure from old data (unwrap if needed)
  const currentPrefs = doc.preferences?.preferences || doc.preferences || {};
  doc.preferences = { ...currentPrefs, ...prefs };
  doc.updatedAt = new Date().toISOString();

  await saveAppDocument(doc);
  console.log(`[TasteAI] Updated preferences for ${username}:`, prefs);
}
