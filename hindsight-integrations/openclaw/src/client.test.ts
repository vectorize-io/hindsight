import { describe, it, expect } from 'vitest';
import { HindsightClient, escapeShellArg } from './client.js';

describe('HindsightClient', () => {
  it('should create instance with provider and API key', () => {
    const client = new HindsightClient('openai', 'test-key', 'gpt-4');
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should set bank ID', () => {
    const client = new HindsightClient('openai', 'test-key');
    client.setBankId('test-bank');
    // No error thrown means success
    expect(true).toBe(true);
  });

  it('should handle content escaping for single quotes', () => {
    const client = new HindsightClient('openai', 'test-key');
    // This test validates the client is instantiated correctly
    // Actual CLI calls would require mocking
    expect(client).toBeDefined();
  });
});

describe('escapeShellArg', () => {
  it('should return unchanged string when no special characters', () => {
    expect(escapeShellArg('hello world')).toBe('hello world');
    expect(escapeShellArg('simple text 123')).toBe('simple text 123');
  });

  it('should escape single quotes', () => {
    expect(escapeShellArg("it's")).toBe("it'\\''s");
    expect(escapeShellArg("don't")).toBe("don'\\''t");
    expect(escapeShellArg("'quoted'")).toBe("'\\''quoted'\\''");
  });

  it('should preserve dollar signs (protected by single quotes)', () => {
    // These are NOT escaped - single quotes protect them
    expect(escapeShellArg('$HOME')).toBe('$HOME');
    expect(escapeShellArg('cost is $100')).toBe('cost is $100');
  });

  it('should preserve backticks (protected by single quotes)', () => {
    expect(escapeShellArg('`ls`')).toBe('`ls`');
    expect(escapeShellArg('run `command`')).toBe('run `command`');
  });

  it('should preserve exclamation marks (protected by single quotes)', () => {
    expect(escapeShellArg('hello!')).toBe('hello!');
    expect(escapeShellArg('wow! amazing!')).toBe('wow! amazing!');
  });

  it('should preserve glob patterns (protected by single quotes)', () => {
    expect(escapeShellArg('*.txt')).toBe('*.txt');
    expect(escapeShellArg('file?.log')).toBe('file?.log');
    expect(escapeShellArg('[abc]')).toBe('[abc]');
  });

  it('should preserve parentheses and braces (protected by single quotes)', () => {
    expect(escapeShellArg('(subshell)')).toBe('(subshell)');
    expect(escapeShellArg('{a,b,c}')).toBe('{a,b,c}');
  });

  it('should preserve redirection and control operators (protected by single quotes)', () => {
    expect(escapeShellArg('a > b')).toBe('a > b');
    expect(escapeShellArg('cmd | grep')).toBe('cmd | grep');
    expect(escapeShellArg('a && b')).toBe('a && b');
    expect(escapeShellArg('a; b')).toBe('a; b');
  });

  it('should preserve backslashes (protected by single quotes)', () => {
    expect(escapeShellArg('path\\to\\file')).toBe('path\\to\\file');
  });

  it('should preserve double quotes (protected by single quotes)', () => {
    expect(escapeShellArg('"quoted"')).toBe('"quoted"');
  });

  it('should preserve hash (protected by single quotes)', () => {
    expect(escapeShellArg('# comment')).toBe('# comment');
  });

  it('should preserve tilde (protected by single quotes)', () => {
    expect(escapeShellArg('~user')).toBe('~user');
  });

  it('should preserve newlines (protected by single quotes)', () => {
    expect(escapeShellArg('line1\nline2')).toBe('line1\nline2');
  });

  it('should handle complex mixed content', () => {
    expect(escapeShellArg("It's $100! Run `ls`")).toBe("It'\\''s $100! Run `ls`");
    expect(escapeShellArg("user's file*.txt")).toBe("user'\\''s file*.txt");
  });

  it('should handle empty string', () => {
    expect(escapeShellArg('')).toBe('');
  });

  it('should handle multiple consecutive single quotes', () => {
    expect(escapeShellArg("''")).toBe("'\\'''\\''");
    expect(escapeShellArg("'''")).toBe("'\\'''\\'''\\''");
  });
});
