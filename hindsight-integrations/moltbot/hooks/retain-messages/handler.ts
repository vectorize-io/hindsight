// Handler for auto-retaining messages to Hindsight

import type { HookHandler } from 'moltbot/plugin-sdk';

const handler: HookHandler = async (event) => {
  // Only process tool_result_persist and command:new events
  if (
    event.type !== 'tool_result_persist' &&
    !(event.type === 'command' && event.action === 'new')
  ) {
    return;
  }

  try {
    // Get client from global (set by main plugin)
    const clientGlobal = (global as any).__hindsightClient;
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
    // The messages are in event.context.sessionEntry or similar
    const sessionEntry = event.context?.sessionEntry;
    if (!sessionEntry || !sessionEntry.messages || sessionEntry.messages.length === 0) {
      return;
    }

    // Format messages into a transcript
    const transcript = sessionEntry.messages
      .map((msg: any) => {
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
