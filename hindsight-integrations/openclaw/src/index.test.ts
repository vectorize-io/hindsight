import { describe, it, expect } from 'vitest';

/**
 * Unit tests for the memory feedback loop fix.
 * Verifies that <hindsight_memories> and <relevant_memories> tags
 * are stripped from content before RETAIN to prevent duplicates.
 */
describe('Memory Tag Stripping', () => {
  /**
   * Simulates the tag stripping logic from agent_end hook
   */
  function stripMemoryTags(content: string): string {
    // Strip plugin-injected memory tags to prevent feedback loop
    content = content.replace(/<hindsight_memories>[\s\S]*?<\/hindsight_memories>/g, '');
    content = content.replace(/<relevant_memories>[\s\S]*?<\/relevant_memories>/g, '');
    return content;
  }

  it('should strip simple hindsight_memories tags', () => {
    const input = 'User: Hello\n<hindsight_memories>\nRelevant memories here...\n</hindsight_memories>\nAssistant: How can I help?';
    const expected = 'User: Hello\n\nAssistant: How can I help?';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should strip relevant_memories tags', () => {
    const input = 'Before\n<relevant_memories>\nSome data\n</relevant_memories>\nAfter';
    const expected = 'Before\n\nAfter';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should strip multiple hindsight_memories blocks', () => {
    const input = 'Start\n<hindsight_memories>\nBlock 1\n</hindsight_memories>\nMiddle\n<hindsight_memories>\nBlock 2\n</hindsight_memories>\nEnd';
    const expected = 'Start\n\nMiddle\n\nEnd';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should handle multiline memory blocks with JSON', () => {
    const input = 'User: What is the weather?\n<hindsight_memories>\nRelevant memories:\n{\n  "memory": "User likes sunny weather"\n}\n</hindsight_memories>\nAssistant: Let me check';
    const expected = 'User: What is the weather?\n\nAssistant: Let me check';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should preserve content without memory tags', () => {
    const input = 'User: Hello\nAssistant: Hi there!';
    const expected = 'User: Hello\nAssistant: Hi there!';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should handle nested-like content without actual nesting', () => {
    const input = '<hindsight_memories>Outer start\n</hindsight_memories>\nSafe content\n<hindsight_memories>\nOuter end</hindsight_memories>';
    const expected = '\nSafe content\n';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should strip both tag types in same content', () => {
    const input = 'A\n<hindsight_memories>\nH mem\n</hindsight_memories>\nB\n<relevant_memories>\nR mem\n</relevant_memories>\nC';
    const expected = 'A\n\nB\n\nC';
    const result = stripMemoryTags(input);
    expect(result).toBe(expected);
  });

  it('should handle real-world agent conversation with injected memories', () => {
    const input = '[role: system]\n<hindsight_memories>\nRelevant memories from past conversations (score 1=highest, prioritize recent when conflicting):\n[\n  {\n    "content": "User prefers dark mode",\n    "relevance_score": 0.95\n  }\n]\n\nUser message: How do I enable dark mode?\n</hindsight_memories>\n[system:end]\n\n[role: user]\nHow do I enable dark mode?\n[user:end]\n\n[role: assistant]\nBased on your previous preference, let me help you enable dark mode.\n[assistant:end]';

    const result = stripMemoryTags(input);

    // Should not contain the memory tags
    expect(result).not.toContain('<hindsight_memories>');
    expect(result).not.toContain('</hindsight_memories>');
    expect(result).not.toContain('Relevant memories from past conversations');

    // Should still contain the actual conversation
    expect(result).toContain('[role: user]');
    expect(result).toContain('How do I enable dark mode?');
    expect(result).toContain('[role: assistant]');
  });
});
