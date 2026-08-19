/**
 * `retainTags` / `retainMetadata` — user-defined provenance on the memories this plugin writes.
 *
 * Every conversation retain already carries `source:chat` and `harness:<id>`, which say WHAT wrote
 * the memory but nothing about WHERE it came from. That is fine while each repo has its own bank,
 * since the bank itself is the answer. It stops being fine on a deliberately SHARED bank — one bank
 * holding cross-project knowledge so facts recall everywhere — where a retained fact then carries no
 * record of the repository it came out of (#3269).
 *
 * So both settings take `{placeholder}` templates, resolved per retain against the same vocabulary
 * the dynamic bank id uses, plus the things only a retain knows (its bank and session):
 *
 *   {gitProject} worktree-aware repo name      {project}  working-directory basename
 *   {harness}    the agent that wrote it       {bankId}   the bank being written to
 *   {sessionId}  the agent session             {timestamp} ISO-8601, resolved at retain time
 *   {channel}    $HINDSIGHT_CHANNEL_ID         {user}     $HINDSIGHT_USER_ID
 *
 * The built-in tags and metadata always win over configured ones. `harness` in particular is load-
 * bearing: the control plane resolves a document's agent logo from `metadata.harness` and the
 * `harness:<id>` tag, so letting a template overwrite them would break attribution for everyone
 * looking at the documents list.
 */
import { applyTemplate, type Resolvers } from "./template";
import { log } from "./log";
import { projectNameOf } from "./bank";
import { basename } from "node:path";

/** Tag namespaces the plugin owns — see the filter in buildRetainStamp. */
const RESERVED_TAG_PREFIX = /^(source|harness):/;

export interface RetainStampContext {
  /** Working directory the retain is being written from — the repo, for {project}/{gitProject}. */
  directory: string;
  /** Where the session started, when known. {gitProject} must name the same project the bank id
   *  does, so it falls back to this exactly as bank resolution does (see core/bank.ts). */
  sessionRoot?: string;
  harness: string;
  bankId: string;
  /** Agent session when the document comes from one; absent for non-session documents. */
  sessionId?: string;
}

export interface RetainStamp {
  tags: string[];
  metadata: Record<string, string>;
}

export interface RetainStampConfig {
  retainTags?: string[];
  retainMetadata?: Record<string, string>;
}

function resolversFor(ctx: RetainStampContext): Resolvers {
  return {
    // Worktree-aware like the bank id, so every linked worktree of a repo stamps the SAME name —
    // otherwise a shared bank ends up with `project:app` and `project:app-wt2` for one repository.
    gitProject: () => projectNameOf(ctx.directory, ctx.sessionRoot),
    project: () => (ctx.directory ? basename(ctx.directory) : "unknown"),
    harness: () => ctx.harness,
    bankId: () => ctx.bankId,
    sessionId: () => ctx.sessionId ?? "unknown",
    timestamp: () => new Date().toISOString(),
    channel: () => process.env.HINDSIGHT_CHANNEL_ID || "default",
    user: () => process.env.HINDSIGHT_USER_ID || "anonymous",
  };
}

/**
 * Resolve the configured tags and metadata for one retain. Returns empty collections when neither
 * setting is configured, which is the default — this adds nothing to a retain unless asked.
 */
export function buildRetainStamp(cfg: RetainStampConfig, ctx: RetainStampContext): RetainStamp {
  const hasTags = Boolean(cfg.retainTags?.length);
  const hasMetadata = Boolean(cfg.retainMetadata && Object.keys(cfg.retainMetadata).length);
  if (!hasTags && !hasMetadata) return { tags: [], metadata: {} };

  // Built lazily and memoized: {gitProject} shells out to git, and a config using it in both a tag
  // and a metadata value should not pay for that twice per retain.
  const base = resolversFor(ctx);
  const cache = new Map<string, string>();
  const resolvers: Resolvers = Object.fromEntries(
    Object.entries(base).map(([name, resolve]) => [
      name,
      () => {
        const hit = cache.get(name);
        if (hit !== undefined) return hit;
        const value = resolve();
        cache.set(name, value);
        return value;
      },
    ])
  );

  const tags = (cfg.retainTags ?? [])
    .map((t) => applyTemplate(t, resolvers, "retainTags").trim())
    .filter(Boolean)
    // `source:` and `harness:` are the plugin's own namespaces: the documents list filters on them
    // and resolves each document's agent logo from `harness:<id>`. A configured `harness:something`
    // would sit alongside the real one and make the document match a filter for an agent that never
    // wrote it, so those are dropped with a warning rather than silently honoured.
    .filter((t) => {
      if (!RESERVED_TAG_PREFIX.test(t)) return true;
      log.warn("retain-stamp", "ignoring reserved retainTags entry", { tag: t });
      return false;
    });
  const metadata: Record<string, string> = {};
  for (const [key, value] of Object.entries(cfg.retainMetadata ?? {})) {
    metadata[key] = applyTemplate(value, resolvers, "retainMetadata");
  }
  return { tags, metadata };
}
