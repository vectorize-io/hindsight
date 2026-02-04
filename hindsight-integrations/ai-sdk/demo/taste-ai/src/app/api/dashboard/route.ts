// @ts-nocheck - TypeScript has issues resolving tool types from local package
import { getAppDocument, hindsightTools, BANK_ID } from '@/lib/hindsight';

export async function GET(req: Request) {
  try {
    // Get username from query params
    const { searchParams } = new URL(req.url);
    const username = searchParams.get('username');

    if (!username) {
      return Response.json({ error: 'Username required' }, { status: 400 });
    }

    // Get app document for this user
    const appDoc = await getAppDocument(username);

    // Get meals from document (last 10)
    const meals = appDoc.meals.slice(0, 10);

    // Query mental model fresh for health insights
    let health: any = null;
    if (appDoc.healthMentalModelId && meals.length > 0) {
      try {
        // @ts-ignore - TS can't resolve mental model tools from local package
        const mentalModelResult = await hindsightTools.queryMentalModel.execute({
          bankId: BANK_ID,
          mentalModelId: appDoc.healthMentalModelId,
        });

        if (mentalModelResult?.content) {
          health = {
            score: 0, // Not used anymore
            trend: 'stable', // Not used anymore
            insight: mentalModelResult.content, // Fresh mental model insights
          };
        }
      } catch (e) {
        console.log('[Dashboard] Failed to query mental model:', e);
      }
    }

    // Include preferences for display and onboarding check
    const preferences = appDoc.preferences || {};

    console.log(`[Dashboard] Loaded ${meals.length} meals for ${username}, health: ${health ? 'fresh from mental model' : 'none'}`);

    return Response.json({ health, meals, preferences });
  } catch (error) {
    console.error('Dashboard error:', error);
    return Response.json({ health: null, meals: [], preferences: {} });
  }
}
