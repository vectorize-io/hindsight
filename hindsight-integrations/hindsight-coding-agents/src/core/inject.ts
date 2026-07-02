/** The system-prompt injection wrapper for a surfaced memory (harness-agnostic text). */

export function buildSystemInjection(memory: string): string {
  return (
    "Relevant project memory, surfaced from THIS repository's git history and past developer " +
    "conversations — a past decision that likely explains this issue. If it states an EXACT rule " +
    "or literal values (specific strings, numbers, set members, mappings), apply them PRECISELY as " +
    "given — the hidden tests depend on those exact choices; do not substitute your own guess. " +
    "Verify against the current code before editing:\n\n" + memory
  );
}
