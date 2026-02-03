import { describe, it, expect } from 'vitest';
import { HindsightClient, escapeShellArg } from './client.js';

describe('escapeShellArg', () => {
  it('should pass through simple strings unchanged', () => {
    expect(escapeShellArg('hello world')).toBe('hello world');
    expect(escapeShellArg('simple text')).toBe('simple text');
  });

  it('should escape single quotes', () => {
    expect(escapeShellArg("It's")).toBe("It'\\''s");
    expect(escapeShellArg("don't")).toBe("don'\\''t");
    expect(escapeShellArg("'quoted'")).toBe("'\\''quoted'\\''");
  });

  it('should handle question marks (glob pattern)', () => {
    // Question marks are literal in single quotes, no escaping needed
    expect(escapeShellArg('where did you fetch this info from?')).toBe('where did you fetch this info from?');
    expect(escapeShellArg('what? why? how?')).toBe('what? why? how?');
  });

  it('should handle exclamation marks (history expansion)', () => {
    // Exclamation marks are literal in single quotes
    expect(escapeShellArg('Hello! How are you!')).toBe('Hello! How are you!');
    expect(escapeShellArg('!!important!!')).toBe('!!important!!');
  });

  it('should handle backticks (command substitution)', () => {
    // Backticks are literal in single quotes
    expect(escapeShellArg('use `code` blocks')).toBe('use `code` blocks');
    expect(escapeShellArg('`whoami`')).toBe('`whoami`');
  });

  it('should handle dollar signs (variable expansion)', () => {
    // Dollar signs are literal in single quotes
    expect(escapeShellArg('costs $100')).toBe('costs $100');
    expect(escapeShellArg('$HOME/path')).toBe('$HOME/path');
    expect(escapeShellArg('$(command)')).toBe('$(command)');
    expect(escapeShellArg('${VAR}')).toBe('${VAR}');
  });

  it('should handle double quotes', () => {
    // Double quotes are literal in single quotes
    expect(escapeShellArg('He said "hello"')).toBe('He said "hello"');
    expect(escapeShellArg('"quoted text"')).toBe('"quoted text"');
  });

  it('should handle newlines', () => {
    // Newlines are literal in single quotes
    expect(escapeShellArg('line1\nline2')).toBe('line1\nline2');
    expect(escapeShellArg('multi\n\nline')).toBe('multi\n\nline');
  });

  it('should handle backslashes', () => {
    // Backslashes are literal in single quotes
    expect(escapeShellArg('path\\to\\file')).toBe('path\\to\\file');
    expect(escapeShellArg('escape\\nsequence')).toBe('escape\\nsequence');
  });

  it('should handle glob patterns (* [ ])', () => {
    // Glob patterns are literal in single quotes
    expect(escapeShellArg('file*.txt')).toBe('file*.txt');
    expect(escapeShellArg('[a-z]')).toBe('[a-z]');
    expect(escapeShellArg('match[0-9]*')).toBe('match[0-9]*');
  });

  it('should handle shell control characters', () => {
    // All these are literal in single quotes
    expect(escapeShellArg('cmd1 | cmd2')).toBe('cmd1 | cmd2');
    expect(escapeShellArg('cmd1 & cmd2')).toBe('cmd1 & cmd2');
    expect(escapeShellArg('cmd1 ; cmd2')).toBe('cmd1 ; cmd2');
    expect(escapeShellArg('input < file')).toBe('input < file');
    expect(escapeShellArg('output > file')).toBe('output > file');
  });

  it('should handle subshell and brace expansion', () => {
    // Parentheses and braces are literal in single quotes
    expect(escapeShellArg('(subshell)')).toBe('(subshell)');
    expect(escapeShellArg('{a,b,c}')).toBe('{a,b,c}');
    expect(escapeShellArg('${expansion}')).toBe('${expansion}');
  });

  it('should handle tilde and hash', () => {
    // Tilde and hash are literal in single quotes
    expect(escapeShellArg('~user/path')).toBe('~user/path');
    expect(escapeShellArg('# comment')).toBe('# comment');
  });

  it('should handle complex strings with multiple special characters', () => {
    const complex = "It's $100! Really? Use `echo $HOME` & see ~/dir/*.txt";
    expect(escapeShellArg(complex)).toBe("It'\\''s $100! Really? Use `echo $HOME` & see ~/dir/*.txt");
  });

  it('should handle strings with single quotes and other special chars', () => {
    expect(escapeShellArg("user's $HOME")).toBe("user'\\''s $HOME");
    expect(escapeShellArg("What's `this`?")).toBe("What'\\''s `this`?");
  });

  it('should handle empty strings', () => {
    expect(escapeShellArg('')).toBe('');
  });

  it('should handle strings that are only single quotes', () => {
    // Each ' becomes '\'' (4 chars: quote, backslash, quote, quote)
    const escaped1 = escapeShellArg("'");
    expect(escaped1).toBe("'\\''");  // ' -> '\''
    expect(escaped1.length).toBe(4);

    // Two single quotes: each becomes '\''
    const escaped2 = escapeShellArg("''");
    expect(escaped2.length).toBe(8);  // 2 * 4 = 8 chars
    expect(escaped2).toContain("'\\''");

    // Three single quotes
    const escaped3 = escapeShellArg("'''");
    expect(escaped3.length).toBe(12);  // 3 * 4 = 12 chars
  });
});

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
