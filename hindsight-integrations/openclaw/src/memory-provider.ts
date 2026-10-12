/**
 * OpenClaw native memory provider (#4916).
 *
 * When Hindsight holds OpenClaw's memory slot, OpenClaw's own memory surfaces
 * (Memory page, `doctor.memory.status`, gateway `memory.status` / `memory.search`)
 * read through the slot owner's `providerRuntime`. Without one they all reported
 * "memory plugin unavailable" while recall and retain worked fine.
 *
 * Deliberately read-only and narrow: search, get and health. No automatic-recall
 * candidates (Active Memory would duplicate the auto-recall hook) and no
 * pre-compaction flush (every turn is already retained).
 *
 * Access rule: an agent-only bank holds every conversation of that agent, so
 * only callers allowed to see all of it may read it: an operator, or the
 * agent owner's own private session. A bank keyed by user or channel can't be
 * picked from an agent id alone, so search is refused there instead of guessed.
 * Health answers every caller and reveals no content.
 */

// Structural copies of OpenClaw's provider types (src/plugins/memory-provider-types.ts,
// 2026.10.x). The plugin compiles without OpenClaw's SDK, like the other host types.
type MemoryAudience =
  | { kind: "owner-private"; agentId: string }
  | { kind: "conversation"; agentId: string; sessionKey: string; sessionId: string };

type MemoryCallerAuthority =
  | { kind: "operator"; scopes: readonly string[]; connId?: string }
  | { kind: "session"; sessionKey: string; sandboxed: boolean; audience?: MemoryAudience }
  | { kind: "host"; operation: string };

interface MemoryReference {
  providerId: string;
  id: string;
}

interface MemorySearchHit {
  reference: MemoryReference;
  excerpt: string;
  score?: number;
  source?: string;
}

interface MemoryProviderHandle {
  capabilities: {
    sources: readonly ("memory" | "sessions")[];
    pagination: boolean;
    candidates: readonly ("trigger" | "project")[];
    projectFilter: boolean;
  };
  search(request: {
    query: string;
    maxResults?: number;
    minScore?: number;
  }): Promise<{ hits: MemorySearchHit[] }>;
  get(request: {
    reference: MemoryReference;
  }): Promise<{ status: "ok"; reference: MemoryReference; text: string } | { status: "not_found" }>;
  health(): Promise<{ status: "ready" | "degraded" | "unavailable"; message?: string }>;
  close(): Promise<void>;
}

export interface MemoryProviderRuntime {
  open(params: {
    agentId: string;
    context: { authority: MemoryCallerAuthority; assertCurrent(): void; signal?: AbortSignal };
  }): Promise<{ provider: MemoryProviderHandle | null; error?: string }>;
}

/** What the provider needs from the plugin; index.ts supplies the live implementations. */
export interface MemoryProviderDeps {
  providerId: string;
  /** The bank for this agent when it depends on the agent alone, else undefined. */
  agentBank(agentId: string): string | undefined;
  recall(
    bankId: string,
    query: string,
    signal: AbortSignal | undefined
  ): Promise<Array<{ id: string; text: string; score?: number | null }>>;
  getMemory(
    bankId: string,
    memoryId: string,
    signal: AbortSignal | undefined
  ): Promise<string | null>;
  healthy(): Promise<boolean>;
}

const DEFAULT_MAX_RESULTS = 10;

function mayRead(authority: MemoryCallerAuthority, agentId: string): boolean {
  if (authority.kind === "operator") return true;
  return (
    authority.kind === "session" &&
    authority.audience?.kind === "owner-private" &&
    authority.audience.agentId === agentId
  );
}

export function createMemoryProviderRuntime(deps: MemoryProviderDeps): MemoryProviderRuntime {
  return {
    async open({ agentId, context }) {
      const bankId = deps.agentBank(agentId);
      const readable = mayRead(context.authority, agentId);

      // Denial and failure must not become empty success, so both throw.
      const readableBank = (): string => {
        if (!readable) throw new Error("Hindsight memory is not readable by this caller");
        if (!bankId) {
          throw new Error(
            "Hindsight banks here are keyed by user or channel; search them from a conversation"
          );
        }
        return bankId;
      };

      const provider: MemoryProviderHandle = {
        capabilities: {
          sources: ["memory"],
          pagination: false,
          candidates: [],
          projectFilter: false,
        },

        async search({ query, maxResults, minScore }) {
          const bank = readableBank();
          context.assertCurrent();
          const results = await deps.recall(bank, query, context.signal);
          context.assertCurrent();
          const hits = results
            // An unscored result can't be shown to clear a floor, so a floor drops it.
            .filter((r) => minScore === undefined || (r.score != null && r.score >= minScore))
            .slice(0, maxResults ?? DEFAULT_MAX_RESULTS)
            .map((r) => ({
              reference: { providerId: deps.providerId, id: r.id },
              excerpt: r.text,
              ...(r.score != null ? { score: r.score } : {}),
              source: "memory",
            }));
          return { hits };
        },

        async get({ reference }) {
          const bank = readableBank();
          if (reference.providerId !== deps.providerId) return { status: "not_found" };
          context.assertCurrent();
          const text = await deps.getMemory(bank, reference.id, context.signal);
          context.assertCurrent();
          return text === null
            ? { status: "not_found" }
            : { status: "ok", reference: { providerId: deps.providerId, id: reference.id }, text };
        },

        async health() {
          return (await deps.healthy())
            ? { status: "ready" }
            : { status: "unavailable", message: "Hindsight API is not reachable" };
        },

        async close() {},
      };
      return { provider };
    },
  };
}
