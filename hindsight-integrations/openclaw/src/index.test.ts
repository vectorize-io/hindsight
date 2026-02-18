import { describe, it, expect } from 'vitest';
import { stripMemoryTags, extractRecallQuery } from './index.js';

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
});
