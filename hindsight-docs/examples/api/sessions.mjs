#!/usr/bin/env node
/**
 * Sessions API examples for Hindsight (Node.js): live sessions.
 * Run: node examples/api/sessions.mjs
 */
import { HindsightClient, sdk, createClient, createConfig } from '@vectorize-io/hindsight-client';

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Setup (not shown in docs)
// =============================================================================
const client = new HindsightClient({ baseUrl: HINDSIGHT_URL });
const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_URL }));
const sessionId = `chat-${Math.random().toString(16).slice(2, 10)}`;

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:sessions-append]
// Send the messages since the last append, in the OpenAI chat format your agent likely
// keeps already, each with your own id: sending one again (a retry after a lost response)
// stores nothing twice. The first append creates the session.
const { data: appended } = await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: {
        session: { live: true, retain: 'turns' },
        messages: [
            { id: '1', role: 'user', content: 'Half cents round the wrong way.' },
            {
                id: '2',
                role: 'assistant',
                tool_calls: [{ id: 't1', type: 'function', function: { name: 'edit_file', arguments: '{"path": "pay/rounding.py"}' } }],
            },
            { id: '3', role: 'tool', tool_call_id: 't1', content: 'updated' },
        ],
    },
});
console.log(appended.notes, appended.checking);
// [/docs:sessions-append]

// [docs:sessions-notes]
// Notes come back with every append. `ack` is the seq of the last note you delivered:
// the response carries the notes after it, so a note is handed over until you confirm it.
let delivered = 0;
const { data: polled } = await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: { ack: delivered },
});
for (const note of polled.notes) {
    console.log(`#${note.seq} ${note.text} (source: ${note.source?.document_id ?? 'memory'})`);
    delivered = note.seq;
}
// [/docs:sessions-notes]

// [docs:sessions-turn-end]
// The end of a turn saves the conversation into the session's document
// (`session:<id>` unless you named one), where memories are extracted from it.
await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: {
        messages: [{ id: '4', role: 'assistant', content: 'Fixed: half cents round down.' }],
        turn_end: true,
        ack: delivered,
    },
});
// [/docs:sessions-turn-end]

// [docs:sessions-follow]
// Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
let after = 0;
const { data: log } = await sdk.listSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    query: { after },
});
for (const entry of log.items) {
    console.log(entry.seq, entry.source, entry.kind);
}
after = log.items.at(-1)?.seq ?? after; // poll again from the last entry you saw
// [/docs:sessions-follow]

// [docs:sessions-list]
const { data: sessions } = await sdk.listSessions({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { limit: 20 },
});
for (const s of sessions.items) {
    console.log(s.session_id, s.last_event_at, s.note_count);
}
// [/docs:sessions-list]


// [docs:sessions-delete]
// Deletes the session and its log. The document it saved stays: that is memory now.
await sdk.deleteSession({ client: apiClient, path: { bank_id: 'my-bank', session_id: sessionId } });
// [/docs:sessions-delete]

// =============================================================================
// Cleanup (not shown in docs)
// =============================================================================
await client.deleteBank('my-bank');
