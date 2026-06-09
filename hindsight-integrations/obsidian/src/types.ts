/**
 * Typed request/response shapes for the Hindsight HTTP API surface used by this
 * plugin. Only the fields the plugin actually reads are modeled — the server
 * returns more.
 */

export type Budget = "low" | "mid" | "high";

export type TagMatch = "any" | "all" | "any_strict" | "all_strict";

/** A leaf tag filter. `all_strict` = AND match that also excludes untagged memories. */
export interface TagLeaf {
  tags: string[];
  match?: TagMatch;
}

/** Compound AND of tag groups (server key is `and`). */
export interface TagAnd {
  and: TagGroup[];
}

/** Subset of the server's TagGroup union — leaves and AND are all the chat needs. */
export type TagGroup = TagLeaf | TagAnd;

export interface RetainOptions {
  tags?: string[];
  metadata?: Record<string, string>;
  /** ISO 8601 timestamp, or "unset" for timeless content. */
  timestamp?: string;
  context?: string;
  updateMode?: "replace" | "append";
}

/** A fact cited by reflect via `based_on.memories`. */
export interface ReflectFact {
  id: string;
  text: string;
  document_id?: string;
}

/** A mental model cited by reflect via `based_on.mental_models`. */
export interface ReflectMentalModel {
  id: string;
  name?: string;
  text?: string;
}

export interface ReflectBasedOn {
  memories?: ReflectFact[];
  mental_models?: ReflectMentalModel[];
}

export interface ReflectToolCall {
  tool: string;
  input?: Record<string, unknown>;
  /** Tool result (included when include.tool_calls.output is true — the default). */
  output?: Record<string, unknown>;
  duration_ms?: number;
  iteration?: number;
}

export interface ReflectTrace {
  tool_calls?: ReflectToolCall[];
}

export interface ReflectResponse {
  text: string;
  based_on?: ReflectBasedOn;
  trace?: ReflectTrace;
}

export interface ReflectOptions {
  budget?: Budget;
  /** When true, ask the server to return `based_on` citations + `trace`. */
  includeCitations?: boolean;
  tags?: string[];
  /** Compound tag filter (mutually exclusive with `tags`). */
  tagGroups?: TagGroup[];
}
