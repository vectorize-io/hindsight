import { updatePreferences } from '@/lib/hindsight';

export async function POST(req: Request) {
  const body = await req.json();
  const username = body.username;
  const prefs = body.preferences || body;

  if (!username) {
    return Response.json({ error: 'Username required' }, { status: 400 });
  }

  try {
    // Save to document for app state only
    // Preferences are configuration, not temporal events, so we don't retain as memories
    await updatePreferences(username, prefs);

    console.log(`[TasteAI] Updated preferences for ${username}:`, prefs);

    return Response.json({ success: true });
  } catch (error) {
    console.error('Update preferences error:', error);
    return Response.json({ error: 'Failed to update preferences' }, { status: 500 });
  }
}
