/**
 * `{placeholder}` substitution, shared by the two things that template: the dynamic bank id
 * (core/bank.ts) and the retain stamp — `retainTags` / `retainMetadata` (core/retain-stamp.ts).
 *
 * Each call site supplies its OWN resolver map rather than drawing on one global set, because the
 * valid placeholders genuinely differ: a bank id cannot reference `{bankId}` without chasing its own
 * tail, and a retain stamp has a session and a bank that bank derivation has not computed yet. An
 * unknown placeholder resolves to "unknown" and says so on stderr, naming the ones that would have
 * worked — a silent empty string is how you end up with a tag like `project:` and no idea why.
 */

const PLACEHOLDER = /\{([a-zA-Z_]+)\}/g;

export type Resolvers = Record<string, () => string>;

/** Substitute every `{name}` in `template` using `resolvers`. `what` names the setting in the
 *  error message, so a typo points at the config key that carries it. */
export function applyTemplate(template: string, resolvers: Resolvers, what: string): string {
  return template.replace(PLACEHOLDER, (_, name: string) => {
    const resolve = resolvers[name];
    if (!resolve) {
      console.error(
        `hindsight: unknown ${what} placeholder "{${name}}" — valid: ` +
          Object.keys(resolvers)
            .sort()
            .map((k) => `{${k}}`)
            .join(", ")
      );
      return "unknown";
    }
    return resolve();
  });
}
