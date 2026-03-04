import { describe, it, expect } from 'vitest';
import {
  stripMemoryTags,
  extractRecallQuery,
  formatMemories,
  prepareRetentionTranscript,
  sliceLastTurnsByUserBoundary,
  composeRecallQuery,
  truncateRecallQuery,
} from './index.js';
import type { PluginConfig, MemoryResult } from './types.js';

// ---------------------------------------------------------------------------
// stripMemoryTags
// ---------------------------------------------------------------------------

describe('stripMemoryTags', () => {
  it('strips simple hindsight_memories tags', () => {
    const input =
      'User: Hello\n<hindsight_memories>\nRelevant memories here...\n</hindsight_memories>\nAssistant: How can I help?';
    expect(stripMemoryTags(input)).toBe('User: Hello\n\nAssistant: How can I help?');
  });

  it('strips relevant_memories tags', () => {
    const input = 'Before\n<relevant_memories>\nSome data\n</relevant_memories>\nAfter';
    expect(stripMemoryTags(input)).toBe('Before\n\nAfter');
  });

  it('strips multiple hindsight_memories blocks', () => {
    const input =
      'Start\n<hindsight_memories>\nBlock 1\n</hindsight_memories>\nMiddle\n<hindsight_memories>\nBlock 2\n</hindsight_memories>\nEnd';
    expect(stripMemoryTags(input)).toBe('Start\n\nMiddle\n\nEnd');
  });

  it('handles multiline memory blocks with JSON', () => {
    const input =
      'User: What is the weather?\n<hindsight_memories>\n[\n  {"memory": "User likes sunny weather"}\n]\n</hindsight_memories>\nAssistant: Let me check';
    const result = stripMemoryTags(input);
    expect(result).toBe('User: What is the weather?\n\nAssistant: Let me check');
  });

  it('preserves content without memory tags', () => {
    const input = 'User: Hello\nAssistant: Hi there!';
    expect(stripMemoryTags(input)).toBe(input);
  });

  it('strips both tag types in same content', () => {
    const input =
      'A\n<hindsight_memories>\nH mem\n</hindsight_memories>\nB\n<relevant_memories>\nR mem\n</relevant_memories>\nC';
    expect(stripMemoryTags(input)).toBe('A\n\nB\n\nC');
  });

  it('strips tags from a real-world agent conversation with injected memories', () => {
    const input =
      '[role: system]\n<hindsight_memories>\nRelevant memories:\n[{"text": "User prefers dark mode"}]\nUser message: How do I enable dark mode?\n</hindsight_memories>\n[system:end]\n\n[role: user]\nHow do I enable dark mode?\n[user:end]\n\n[role: assistant]\nLet me help you enable dark mode.\n[assistant:end]';

    const result = stripMemoryTags(input);

    expect(result).not.toContain('<hindsight_memories>');
    expect(result).not.toContain('</hindsight_memories>');
    expect(result).not.toContain('User prefers dark mode');
    expect(result).toContain('[role: user]');
    expect(result).toContain('How do I enable dark mode?');
    expect(result).toContain('[role: assistant]');
  });
});

// ---------------------------------------------------------------------------
// extractRecallQuery
// ---------------------------------------------------------------------------

describe('extractRecallQuery', () => {
  it('returns rawMessage when it is long enough', () => {
    expect(extractRecallQuery('What is my favorite food?', undefined)).toBe(
      'What is my favorite food?',
    );
  });

  it('returns null when rawMessage is too short and prompt is absent', () => {
    expect(extractRecallQuery('Hi', undefined)).toBeNull();
    expect(extractRecallQuery('', '')).toBeNull();
    expect(extractRecallQuery(undefined, undefined)).toBeNull();
  });

  it('returns null when both rawMessage and prompt are too short', () => {
    expect(extractRecallQuery('Hey', 'Hey')).toBeNull();
  });

  it('falls back to prompt when rawMessage is absent', () => {
    const result = extractRecallQuery(undefined, 'What programming language do I prefer?');
    expect(result).toBe('What programming language do I prefer?');
  });

  it('returns null when rawMessage is absent and prompt is bare metadata', () => {
    const metadataPrompt = 'Conversation info (untrusted metadata):\n```json\n{"message_id": "abc123"}\n```';
    expect(extractRecallQuery(undefined, metadataPrompt)).toBeNull();
  });

  it('falls back to prompt when rawMessage is metadata but prompt has real content', () => {
    const result = extractRecallQuery(
      'Conversation info (untrusted metadata):',
      'System: You are c0der.\n\nhow many cats do i have?',
    );
    expect(result).toBe('how many cats do i have?');
  });

  it('strips leading System: lines from prompt', () => {
    const prompt = 'System: You are an agent.\nSystem: Use tools wisely.\n\nWhat is my name?';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain('System:');
    expect(result).toContain('What is my name?');
  });

  it('strips [Channel] envelope header and returns inner message', () => {
    const prompt = '[Telegram Chat]\nWhat is my favorite hobby?';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).toBe('What is my favorite hobby?');
  });

  it('strips [from: SenderName] footer from group chat prompts', () => {
    const prompt = '[Slack Channel #general]\nWhat should I eat for lunch?\n[from: Alice]';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain('[from: Alice]');
    expect(result).toContain('What should I eat for lunch?');
  });

  it('handles full envelope with System lines, channel header, and from footer', () => {
    const prompt =
      'System: You are a helpful agent.\n\n[Discord Server]\nRemind me what I said about Python?\n[from: Bob]';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain('System:');
    expect(result).not.toContain('[Discord');
    expect(result).not.toContain('[from: Bob]');
    expect(result).toContain('Remind me what I said about Python?');
  });

  it('strips session abort hint from prompt', () => {
    const prompt =
      'Note: The previous agent run was aborted by the user\n\n[Telegram]\nWhat is my cat\'s name?';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).not.toContain('Note: The previous agent run was aborted');
    expect(result).toContain("What is my cat's name?");
  });

  it('returns null when prompt reduces to < 5 chars after stripping', () => {
    // Envelope with almost-empty inner message
    const prompt = '[Telegram Chat]\nHi';
    const result = extractRecallQuery(undefined, prompt);
    expect(result).toBeNull();
  });

  it('prefers rawMessage over prompt even when prompt is longer', () => {
    const rawMessage = 'What do I like to eat?';
    const prompt = '[Telegram]\nWhat do I like to eat?\n[from: Alice]';
    const result = extractRecallQuery(rawMessage, prompt);
    // Should return the clean rawMessage verbatim
    expect(result).toBe(rawMessage);
    expect(result).not.toContain('[from: Alice]');
  });

  it('trims whitespace from result', () => {
    const result = extractRecallQuery('   What is my job?   ', undefined);
    expect(result).toBe('What is my job?');
  });

  it('rejects OpenClaw untrusted metadata messages as rawMessage', () => {
    const result = extractRecallQuery('Conversation info (untrusted metadata):', undefined);
    expect(result).toBeNull();
  });

  it('rejects untrusted metadata even when prompt is also metadata', () => {
    const result = extractRecallQuery(
      'Conversation info (untrusted metadata):',
      'Conversation info (untrusted metadata): some details',
    );
    expect(result).toBeNull();
  });

  it('falls back to prompt when rawMessage is metadata', () => {
    const result = extractRecallQuery(
      'Conversation info (untrusted metadata):',
      'How many cats do I have?',
    );
    expect(result).toBe('How many cats do I have?');
  });
});


// ---------------------------------------------------------------------------
// formatMemories
// ---------------------------------------------------------------------------

describe('formatMemories', () => {
  const makeMemoryResult = (overrides: Partial<MemoryResult>): MemoryResult => ({
    id: 'mem-1',
    text: 'default text',
    type: 'world',
    entities: [],
    context: '',
    occurred_start: null,
    occurred_end: null,
    mentioned_at: null,
    document_id: null,
    metadata: null,
    chunk_id: null,
    tags: [],
    ...overrides,
  });

  it('formats memories as a bulleted list', () => {
    const memories: MemoryResult[] = [
      makeMemoryResult({ id: '1', text: 'User prefers dark mode', type: 'world', mentioned_at: '2023-01-01T12:00:00Z' }),
      makeMemoryResult({ id: '2', text: 'User is learning Rust', type: 'experience', mentioned_at: null }),
    ];
    const output = formatMemories(memories);
    expect(output).toBe('- User prefers dark mode [world] (2023-01-01T12:00:00Z)\n\n- User is learning Rust [experience]');
  });

  it('returns empty string for empty memories', () => {
    expect(formatMemories([])).toBe('');
  });
});

// ---------------------------------------------------------------------------
// prepareRetentionTranscript
// ---------------------------------------------------------------------------

describe('prepareRetentionTranscript', () => {
  const baseConfig: PluginConfig = {
    dynamicBankId: true,
    retainRoles: ['user', 'assistant'],
  };

  it('returns null if no user message found (turn boundary)', () => {
    const messages = [
      { role: 'assistant', content: 'Hello' },
      { role: 'system', content: 'Context' }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).toBeNull();
  });

  it('retains from last user message onwards', () => {
    const messages = [
      { role: 'user', content: 'Old user' },
      { role: 'assistant', content: 'Old assistant' },
      { role: 'user', content: 'New user' },
      { role: 'assistant', content: 'New assistant' }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).toContain('New user');
    expect(result?.transcript).toContain('New assistant');
    expect(result?.transcript).not.toContain('Old user');
  });

  it('filters out excluded roles', () => {
    const config: PluginConfig = { ...baseConfig, retainRoles: ['user'] };
    const messages = [
      { role: 'user', content: 'User msg' },
      { role: 'assistant', content: 'Assistant msg' }
    ];
    const result = prepareRetentionTranscript(messages, config);
    expect(result).not.toBeNull();
    expect(result?.transcript).toContain('User msg');
    expect(result?.transcript).not.toContain('Assistant msg');
  });

  it('handles array content', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello array' }] }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result?.transcript).toContain('Hello array');
  });

  it('strips memory tags from retained content (feedback loop prevention)', () => {
    const messages = [
      { role: 'user', content: 'What is dark mode?' },
      { role: 'assistant', content: '<hindsight_memories>\nUser prefers dark mode\n</hindsight_memories>\nHere is how to enable dark mode.' }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).not.toContain('<hindsight_memories>');
    expect(result?.transcript).not.toContain('User prefers dark mode');
    expect(result?.transcript).toContain('Here is how to enable dark mode.');
  });

  it('strips memory tags from user message when prependContext is prepended to it', () => {
    // Simulates the host prepending prependContext to the user message content
    const userContent = `<hindsight_memories>\nRelevant memories:\n- User prefers dark mode [world]\n\nUser message: What is dark mode?\n</hindsight_memories>\nWhat is dark mode?`;
    const messages = [
      { role: 'user', content: userContent },
      { role: 'assistant', content: 'Dark mode is a display setting.' }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    expect(result?.transcript).not.toContain('<hindsight_memories>');
    expect(result?.transcript).not.toContain('User prefers dark mode');
    expect(result?.transcript).toContain('What is dark mode?');
    expect(result?.transcript).toContain('Dark mode is a display setting.');
  });

  it('reports accurate messageCount excluding empty messages', () => {
    const messages = [
      { role: 'user', content: 'Real message' },
      { role: 'assistant', content: '<hindsight_memories>\nonly tags\n</hindsight_memories>' },
      { role: 'assistant', content: 'Actual response' }
    ];
    const result = prepareRetentionTranscript(messages, baseConfig);
    expect(result).not.toBeNull();
    // The middle message becomes empty after tag stripping, so messageCount should be 2
    expect(result?.messageCount).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// sliceLastTurnsByUserBoundary
// ---------------------------------------------------------------------------

describe('sliceLastTurnsByUserBoundary', () => {
  it('returns the whole message list when requested turns exceed available user turns', () => {
    const messages = [
      { role: 'system', content: 'System preface' },
      { role: 'user', content: 'Turn 1 user' },
      { role: 'assistant', content: 'Turn 1 assistant' },
      { role: 'user', content: 'Turn 2 user' },
      { role: 'assistant', content: 'Turn 2 assistant' },
    ];

    const result = sliceLastTurnsByUserBoundary(messages, 3);
    expect(result).toEqual(messages);
  });

  it('slices by real user-turn boundaries with system/tool messages present', () => {
    const messages = [
      { role: 'system', content: 'System preface' },
      { role: 'user', content: 'Turn 1 user' },
      { role: 'assistant', content: 'Turn 1 assistant' },
      { role: 'tool', content: 'Tool output in turn 1' },
      { role: 'user', content: 'Turn 2 user' },
      { role: 'assistant', content: 'Turn 2 assistant' },
      { role: 'system', content: 'System note in turn 2' },
      { role: 'user', content: 'Turn 3 user' },
      { role: 'assistant', content: 'Turn 3 assistant' },
    ];

    const result = sliceLastTurnsByUserBoundary(messages, 2);
    expect(result).toEqual(messages.slice(4));
  });

  it('returns empty list for invalid turn counts', () => {
    const messages = [{ role: 'user', content: 'Hello' }];
    expect(sliceLastTurnsByUserBoundary(messages, 0)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// composeRecallQuery + truncateRecallQuery
// ---------------------------------------------------------------------------

describe('composeRecallQuery', () => {
  it('returns latest query unchanged when recallContextTurns is 1', () => {
    const query = composeRecallQuery('What is my preference?', [{ role: 'user', content: 'Old message' }], 1);
    expect(query).toBe('What is my preference?');
  });

  it('includes prior user/assistant context when recallContextTurns > 1', () => {
    const messages = [
      { role: 'user', content: 'I like dark mode.' },
      { role: 'assistant', content: 'Got it, dark mode noted.' },
      { role: 'user', content: 'What theme do I prefer?' },
    ];

    const query = composeRecallQuery('What theme do I prefer?', messages, 2);
    expect(query).toContain('What theme do I prefer?');
    expect(query).toContain('user: I like dark mode.');
    expect(query).toContain('assistant: Got it, dark mode noted.');
    // latest message should appear after prior context
    expect(query.indexOf('Prior context:')).toBeLessThan(query.indexOf('What theme do I prefer?'));
  });

  it('respects recallRoles when building prior context', () => {
    const messages = [
      { role: 'system', content: 'System context' },
      { role: 'assistant', content: 'Assistant context' },
      { role: 'user', content: 'What theme do I prefer?' },
    ];

    const query = composeRecallQuery('What theme do I prefer?', messages, 2, ['user']);
    expect(query).toBe('What theme do I prefer?');
  });

  it('falls back to latest query when context has no usable text', () => {
    const messages = [{ role: 'tool', content: 'binary blob' }];
    const query = composeRecallQuery('Summarize my preference', messages, 3);
    expect(query).toBe('Summarize my preference');
  });
});

describe('truncateRecallQuery', () => {
  it('keeps query unchanged when under max', () => {
    const query = 'short query';
    expect(truncateRecallQuery(query, query, 100)).toBe(query);
  });

  it('falls back to latest query when non-context query is over max', () => {
    const latest = 'What foods do I like?';
    const long = `${latest} ${'x'.repeat(300)}`;
    expect(truncateRecallQuery(long, latest, 20)).toBe(latest.slice(0, 20));
  });

  it('trims prior context first and preserves latest section', () => {
    const latest = 'What foods do I like?';
    const composed = [
      'Prior context:',
      'user: I like sushi.',
      'assistant: You like sushi and ramen.',
      'user: Also pizza.',
      latest,
    ].join('\n\n');

    const truncated = truncateRecallQuery(composed, latest, 180);
    expect(truncated).toContain(latest);
    expect(truncated.length).toBeLessThanOrEqual(180);
  });
});
