// Handler for auto-retaining messages to Hindsight
const handler = async (event) => {
  console.log(`[Hindsight Hook] Received event: ${event.type}`);

  // Only process agent_end events (after each agent turn)
  if (event.type !== 'agent_end') {
    return;
  }

  console.log('[Hindsight Hook] Processing retention after agent turn...');

  try {
    // Get client from global (set by main plugin)
    const clientGlobal = global.__hindsightClient;
    if (!clientGlobal) {
      console.warn('[Hindsight] Client global not found, skipping retain');
      return;
    }

    const client = clientGlobal.getClient();
    if (!client) {
      console.warn('[Hindsight] Client not initialized, skipping retain');
      return;
    }

    // Extract session information
    const { sessionId, sessionKey } = event.context || {};
    if (!sessionId) {
      return;
    }

    // Get messages from the event context
    const sessionEntry = event.context?.sessionEntry;
    if (!sessionEntry || !sessionEntry.messages || sessionEntry.messages.length === 0) {
      return;
    }

    // Format messages into a transcript
    const transcript = sessionEntry.messages
      .map((msg) => {
        const role = msg.role || 'unknown';
        const content = msg.content || '';
        return `${role}: ${content}`;
      })
      .join('\n\n');

    if (!transcript.trim()) {
      return;
    }

    // Retain to Hindsight with session_id as document_id
    await client.retain({
      content: transcript,
      document_id: sessionId,
      metadata: {
        session_key: sessionKey,
        retained_at: new Date().toISOString(),
        message_count: sessionEntry.messages.length,
      },
    });

    console.log(`[Hindsight] Retained ${sessionEntry.messages.length} messages for session ${sessionId}`);
  } catch (error) {
    console.error('[Hindsight] Error retaining messages:', error);
  }
};

export default handler;
