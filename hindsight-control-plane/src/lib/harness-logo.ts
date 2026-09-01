/**
 * Which coding agent wrote a document, resolved to a logo.
 *
 * The convention is set by the writers, not by this UI. `hindsight-coding-agents`
 * stamps the harness on every document it retains, twice:
 *
 *   document_metadata.harness = "claude-code"      (the authoritative field)
 *   tags                      = ["harness:claude-code", ...]   (so it's filterable)
 *
 * Metadata wins over the tag: the tag is a projection of it and can be edited by
 * hand in the UI.
 *
 * The ids below are exactly the ones that integration emits — see its
 * `src/harness/hook-lifecycle.ts` (one HookSpec per hook-driven agent) and
 * `src/harness/registry.ts` (the persistent-plugin agents, whose id is the
 * `createPluginEntry(...)` argument of their entrypoint). Do not add entries for
 * agents that cannot appear yet: an id nothing writes is a logo nothing renders.
 * When that integration gains a harness, add it here in the same change — drop
 * its icon in `public/img/harness/` (copied from
 * `hindsight-docs/static/img/icons/`, or taken from the agent's own brand assets
 * when the docs site carries none; the docs site and the control plane are
 * separate packages and cannot share a static dir) and add one entry.
 *
 * An unknown harness is not an error: it renders no logo, and the value still
 * shows as an ordinary metadata chip.
 */

export interface HarnessLogoEntry {
  /** Canonical harness id, e.g. "claude-code". */
  id: string;
  /** Display name, a proper noun — deliberately not translated. */
  label: string;
  /** Path under `public/`. Callers must run it through `withBasePath`. */
  src: string;
  /**
   * Marks a mark that is dark-on-transparent, so it disappears against the dark
   * theme's background. Only ever set for monochrome logos — inverting a
   * multi-colour mark (Gemini, Claude Code) would misrepresent the brand.
   */
  invertOnDark?: boolean;
}

/** Exported for the test that checks every entry's asset is actually shipped. */
export const HARNESS_LOGO_REGISTRY: Record<string, HarnessLogoEntry> = {
  "antigravity-cli": {
    id: "antigravity-cli",
    label: "Antigravity CLI",
    src: "/img/harness/antigravity-cli.png",
  },
  "claude-code": { id: "claude-code", label: "Claude Code", src: "/img/harness/claude-code.png" },
  "cline-cli": {
    id: "cline-cli",
    label: "Cline CLI",
    src: "/img/harness/cline-cli.svg",
    invertOnDark: true,
  },
  codex: { id: "codex", label: "Codex", src: "/img/harness/codex.svg", invertOnDark: true },
  dcode: { id: "dcode", label: "DeepAgents Dcode", src: "/img/harness/dcode.svg" },
  "copilot-cli": {
    id: "copilot-cli",
    label: "GitHub Copilot CLI",
    src: "/img/harness/copilot-cli.svg",
    invertOnDark: true,
  },
  "cursor-cli": {
    id: "cursor-cli",
    label: "Cursor CLI",
    src: "/img/harness/cursor-cli.svg",
    invertOnDark: true,
  },
  "devin-cli": {
    id: "devin-cli",
    label: "Devin CLI",
    src: "/img/harness/devin-cli.svg",
    invertOnDark: true,
  },
  // The DeepSeek Harness mark, taken from that project's own favicon (MIT). Its
  // upstream copy carries a prefers-color-scheme rule; that was dropped so the
  // control plane's own theme decides, like every other monochrome mark here.
  dsh: { id: "dsh", label: "DeepSeek Harness", src: "/img/harness/dsh.svg", invertOnDark: true },
  // Retired: the Gemini CLI harness was replaced by `antigravity-cli`, so nothing
  // emits this id any more. The entry stays because documents retained while it
  // did are still in people's banks, and they should keep their logo.
  gemini: { id: "gemini", label: "Gemini", src: "/img/harness/gemini.svg" },
  // Not inverted: unlike the other monochrome marks this one is a filled black
  // tile with a white glyph, so it stays legible on dark — inverting it would
  // burn a white square into the row.
  "grok-build": { id: "grok-build", label: "Grok Build", src: "/img/harness/grok-build.svg" },
  kilo: { id: "kilo", label: "Kilo CLI", src: "/img/harness/kilo.svg" },
  opencode: { id: "opencode", label: "OpenCode", src: "/img/harness/opencode.png" },
  // opencode v2 ships as a separate `opencode2` binary alongside v1 and reports
  // its own harness id, but it is the same product — so it deliberately shares
  // v1's brand mark rather than inventing a second one.
  opencode2: { id: "opencode2", label: "OpenCode 2", src: "/img/harness/opencode.png" },
  "qwen-code": { id: "qwen-code", label: "Qwen Code", src: "/img/harness/qwen-code.svg" },
  "prime-agent": {
    id: "prime-agent",
    label: "Prime Agent",
    src: "/img/harness/prime-agent.svg",
    invertOnDark: true,
  },
};

const HARNESS_TAG_PREFIX = "harness:";

/**
 * The logo for a harness value, or null when it is unknown or empty.
 *
 * Case and separators are normalised because the harness can also come from the
 * user's own `harness` config key, not only from a HookSpec constant — so
 * `Claude_Code` and `claude code` both land on "claude-code".
 */
export function resolveHarnessLogo(value: string | null | undefined): HarnessLogoEntry | null {
  if (!value) return null;
  const key = value
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-");
  return HARNESS_LOGO_REGISTRY[key] ?? null;
}

/**
 * The harness a document was written by: the `harness` metadata field, falling
 * back to a `harness:<id>` tag. Returns the raw value — resolving it to a logo
 * is a separate step so an unknown harness can still be reported as text.
 */
export function documentHarness(
  metadata: Record<string, unknown> | null | undefined,
  tags: readonly string[] | null | undefined
): string | null {
  const fromMetadata = metadata?.harness;
  if (typeof fromMetadata === "string" && fromMetadata.trim()) {
    return fromMetadata.trim();
  }
  for (const tag of tags ?? []) {
    if (tag.toLowerCase().startsWith(HARNESS_TAG_PREFIX)) {
      const value = tag.slice(HARNESS_TAG_PREFIX.length).trim();
      if (value) return value;
    }
  }
  return null;
}
