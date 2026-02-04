// @ts-nocheck - TypeScript has issues resolving tool types from local package
import { generateText } from 'ai';
import { hindsightTools, llmModel, getAppDocument, saveAppDocument, addMealToDocument, BANK_ID } from '@/lib/hindsight';

// Ensure the health mental model exists and return its ID
async function ensureHealthMentalModel(username: string): Promise<string> {
  const doc = await getAppDocument(username);

  // If we already have a mental model ID, return it
  if (doc.healthMentalModelId) {
    console.log('[TasteAI] Using existing health mental model:', doc.healthMentalModelId);
    return doc.healthMentalModelId;
  }

  // Create a new mental model
  try {
    // @ts-ignore - TS can't resolve mental model tools from local package
    const result = (await hindsightTools.createMentalModel.execute({
      bankId: BANK_ID,
      name: `${username}'s Health Assessment Model`,
      sourceQuery: 'meals, eating habits, food choices, dietary patterns',
      maxTokens: 1000,
      tags: [username],
      autoRefresh: true, // Auto-refresh after new consolidations
    })) as { mentalModelId: string; createdAt: string };

    // Store the mental model ID in the app document
    doc.healthMentalModelId = result.mentalModelId;
    await saveAppDocument(doc);

    console.log('[TasteAI] Created health mental model:', result.mentalModelId, 'for', username);
    return result.mentalModelId;
  } catch (e) {
    console.error('[TasteAI] Failed to create health mental model:', e);
    throw e;
  }
}

export async function POST(req: Request) {
  const { username, food, action, mealType } = await req.json();

  if (!username) {
    return Response.json({ error: 'Username required' }, { status: 400 });
  }

  try {
    const today = new Date();
    const mealDate = today.toISOString().split('T')[0];

    // Handle all actions - store meals in document only
    switch (action) {
      case 'ate_today':
      case 'ate_yesterday': {
        // Just a note - we don't store "already ate" in document
        return Response.json({ success: true });
      }

      case 'never': {
        // Store dislike in preferences
        const doc = await getAppDocument(username);
        const currentDislikes = doc.preferences.dislikes || [];
        if (!currentDislikes.includes(food.name)) {
          doc.preferences.dislikes = [...currentDislikes, food.name];
          await saveAppDocument(doc);
        }
        return Response.json({ success: true });
      }

      case 'cook': {
        // Store meal in document
        const storedMeal = await addMealToDocument(username, {
          name: food.name,
          emoji: food.emoji || '🍽️',
          description: food.description,
          type: mealType,
          date: mealDate,
          healthScore: food.healthScore,
          timeMinutes: food.timeMinutes,
          ingredients: food.ingredients,
          instructions: food.instructions,
          tags: food.tags,
          action: 'cooked',
        });

        console.log(`[TasteAI] Stored meal in document for ${username}: "${storedMeal.name}"`);

        // Ensure mental model exists (will auto-refresh after consolidation)
        await ensureHealthMentalModel(username);

        return Response.json({ success: true, meal: storedMeal });
      }

      default:
        return Response.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Log meal error:', error);
    return Response.json({ error: 'Failed to log meal' }, { status: 500 });
  }
}
