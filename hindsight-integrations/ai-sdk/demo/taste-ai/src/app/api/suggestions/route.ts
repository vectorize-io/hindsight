// @ts-nocheck - TypeScript has issues resolving tool types from local package
import { generateText } from 'ai';
import { hindsightTools, llmModel, getMealsDocument, BANK_ID } from '@/lib/hindsight';

interface Recipe {
  id: string;
  name: string;
  emoji: string;
  description: string;
  healthScore: number;
  timeMinutes: number;
  ingredients: string[];
  instructions: string;
  tags: string[];
}

const COOKING_TIME_CONSTRAINTS: Record<string, string> = {
  none: 'No cooking required - ready-to-eat meals only (cereals, yogurt, sandwiches, salads, fruits, pre-cooked items)',
  quick: 'Under 5 minutes cooking time - very quick preparation (toast, instant oatmeal, smoothies)',
  medium: '5-15 minutes cooking time - quick meals (eggs, simple stir-fry, pasta)',
  long: '15-30 minutes cooking time - normal cooking (grilled dishes, rice bowls, baked items)',
  extended: '30+ minutes cooking time - full recipes allowed (slow-cooked, elaborate dishes)',
};

// Protein groups - if user had any fish, avoid all fish
const PROTEIN_GROUPS: Record<string, string[]> = {
  fish: ['fish', 'salmon', 'tuna', 'cod', 'tilapia', 'trout', 'halibut', 'mackerel', 'sardines', 'anchovies'],
  seafood: ['shrimp', 'prawns', 'crab', 'lobster', 'scallops', 'mussels', 'clams', 'oysters', 'calamari', 'squid'],
  poultry: ['chicken', 'turkey', 'duck'],
  red_meat: ['beef', 'steak', 'lamb', 'pork', 'bacon', 'ham'],
  processed_meat: ['sausage', 'hot dog', 'salami', 'pepperoni'],
  plant: ['tofu', 'tempeh', 'seitan'],
  eggs: ['eggs', 'egg', 'omelette', 'frittata'],
};

// Extract recent protein GROUPS from meal names/ingredients
function extractRecentProteins(meals: { name: string; ingredients?: string[] }[]): string[] {
  const foundGroups = new Set<string>();

  for (const meal of meals.slice(0, 5)) { // Only check last 5 meals
    const text = `${meal.name} ${(meal.ingredients || []).join(' ')}`.toLowerCase();

    for (const [group, keywords] of Object.entries(PROTEIN_GROUPS)) {
      for (const keyword of keywords) {
        if (text.includes(keyword)) {
          foundGroups.add(group);
          break;
        }
      }
    }
  }
  return [...foundGroups];
}

// Get specific proteins to avoid based on groups
function getProteinsToAvoid(groups: string[]): string[] {
  const avoid: string[] = [];
  for (const group of groups) {
    if (PROTEIN_GROUPS[group]) {
      avoid.push(...PROTEIN_GROUPS[group]);
    }
  }
  return avoid;
}

export async function POST(req: Request) {
  const { username, mealType, cookingTime } = await req.json();

  if (!username) {
    return Response.json({ error: 'Username required' }, { status: 400 });
  }

  try {
    // Get meals from document to identify recent vs past favorites
    const today = new Date().toISOString().split('T')[0];
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
    let todayYesterdayMeals: string[] = [];
    let pastFavorites: string[] = [];
    let recentProteins: string[] = [];

    try {
      const mealsDoc = await getMealsDocument(username);
      for (const meal of mealsDoc.meals) {
        if (meal.date === today || meal.date === yesterday) {
          todayYesterdayMeals.push(meal.name);
        } else {
          pastFavorites.push(meal.name);
        }
      }
      pastFavorites = [...new Set(pastFavorites)].slice(0, 5);
      recentProteins = extractRecentProteins(mealsDoc.meals);
    } catch (e) {
      console.log('No meals document yet');
    }

    const timeConstraint = cookingTime ? COOKING_TIME_CONSTRAINTS[cookingTime] : '';

    // Use AI SDK v6 with Hindsight tools for personalized context gathering
    // The agent uses recall and reflect tools to understand user preferences
    const contextResult = await generateText({
      model: llmModel,
      tools: {
        // Hindsight SDK tools for memory operations
        recall: hindsightTools.recall,
        reflect: hindsightTools.reflect,
      },
      toolChoice: 'auto',
      prompt: `You are gathering context for personalized ${mealType} recipe suggestions.

Use the recall tool to search for the user's food preferences, dislikes, and recent protein consumption.
Bank ID: "${BANK_ID}"

Then use the reflect tool to analyze:
1. What proteins has the user been eating recently? We need to rotate proteins for variety.
2. What are their dietary preferences and restrictions?
3. What foods do they dislike?
Bank ID: "${BANK_ID}"

After gathering context, summarize:
- User preferences and dislikes
- Recent proteins they've had (so we can suggest DIFFERENT proteins for variety)`,
    });

    // Extract user context from the agent's response
    const userContext = contextResult.text || '';

    // Build insights for the UI - all data comes from Hindsight
    const insights: string[] = [];
    if (userContext) {
      insights.push(`🧠 From memory: ${userContext.slice(0, 120)}${userContext.length > 120 ? '...' : ''}`);
    }
    if (recentProteins.length > 0) {
      const groupLabels: Record<string, string> = {
        fish: 'fish', seafood: 'seafood', poultry: 'poultry',
        red_meat: 'red meat', processed_meat: 'processed meat', plant: 'plant protein', eggs: 'eggs'
      };
      const readableGroups = recentProteins.map(g => groupLabels[g] || g).join(', ');
      insights.push(`🔄 From meal history: avoiding ${readableGroups} (rotating proteins for variety)`);
    }
    if (todayYesterdayMeals.length > 0) {
      insights.push(`⏰ From meal history: avoiding ${todayYesterdayMeals.slice(0, 2).join(', ')}`);
    }
    if (pastFavorites.length > 0) {
      insights.push(`❤️ From meal history: including favorites like ${pastFavorites.slice(0, 2).join(', ')}`);
    }

    // Log what Hindsight returned
    console.log('[TasteAI] === Hindsight Context Gathering ===');
    console.log('[TasteAI] Steps:', contextResult.steps.length);
    console.log('[TasteAI] Tool calls:', JSON.stringify(contextResult.steps.flatMap(s => s.toolCalls), null, 2));
    console.log('[TasteAI] Tool results:', JSON.stringify(contextResult.steps.flatMap(s => s.toolResults), null, 2));
    console.log('[TasteAI] Final context summary:', userContext);
    console.log('[TasteAI] Insights:', insights);
    console.log('[TasteAI] ================================');

    // Generate recipes with the gathered context
    const proteinsToAvoid = getProteinsToAvoid(recentProteins);
    const proteinRotation = recentProteins.length > 0
      ? `PROTEIN ROTATION (STRICT): User recently had ${recentProteins.join(', ')} proteins. DO NOT suggest any of these: ${proteinsToAvoid.join(', ')}. Use completely different protein sources.`
      : '';

    const recipeResult = await generateText({
      model: llmModel,
      prompt: `You are a recipe generator. Generate exactly 10 ${mealType} recipes as a JSON array.

TIME CONSTRAINT (IMPORTANT): ${timeConstraint || 'Any cooking time is fine'}
${proteinRotation}
${userContext ? `User context: ${userContext.slice(0, 500)}` : ''}
${pastFavorites.length > 0 ? `User's past favorites (consider including some): ${pastFavorites.join(', ')}` : ''}
${todayYesterdayMeals.length > 0 ? `Avoid these (eaten today/yesterday): ${todayYesterdayMeals.join(', ')}` : ''}

Output ONLY valid JSON array, no markdown or explanations. Format:
[{"id":"1","name":"Recipe","emoji":"🍳","description":"Short desc","healthScore":7,"timeMinutes":20,"ingredients":["a","b","c"],"instructions":"Cook it.","tags":["quick"]}]`,
    });

    // Parse recipes
    let recipes: Recipe[] = [];
    const text = recipeResult.text.trim();
    const startIdx = text.indexOf('[');
    const endIdx = text.lastIndexOf(']');

    if (startIdx !== -1 && endIdx > startIdx) {
      let jsonStr = text.slice(startIdx, endIdx + 1);
      jsonStr = jsonStr
        .replace(/,\s*]/g, ']')
        .replace(/,\s*}/g, '}');
      recipes = JSON.parse(jsonStr);
    }

    // Validate and filter
    recipes = recipes.filter(r => r.name && r.emoji && r.ingredients && r.instructions);
    recipes = recipes.filter(recipe => {
      const nameLower = recipe.name.toLowerCase();
      return !todayYesterdayMeals.some(meal => nameLower.includes(meal.toLowerCase()));
    });

    if (recipes.length === 0) {
      return Response.json({ error: 'Failed to generate recipes. Please try again.' }, { status: 500 });
    }

    console.log(`[TasteAI] Generated ${recipes.length} personalized recipes using AI SDK v6 + Hindsight`);
    return Response.json({ suggestions: recipes.slice(0, 10), insights });
  } catch (error) {
    console.error('Suggestions error:', error);
    return Response.json({ error: 'Failed to generate recipes. Please try again.' }, { status: 500 });
  }
}
