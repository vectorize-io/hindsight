// Per-bank recall settings shown in the bank config's Recall section.
//
// Each is tri-state: null inherits the server default, true/false overrides it for
// this bank. The four pipeline stages ship enabled (explicit false switches one off,
// trading recall breadth for latency; semantic always runs as the baseline arm). The
// result filters ship disabled: prefer_observations, and the decision-model
// reranker's prune and dedup, which only act when a decision model is set up for
// recall. A recall request that sets one of these still overrides the bank value.

/** Each field with its message-key prefix (`${key}Label`, `${key}Description`). */
export const RECALL_FIELDS = [
  ["enable_text_search", "recallTextSearch"],
  ["enable_temporal_retrieval", "recallTemporalRetrieval"],
  ["enable_graph_retrieval", "recallGraphRetrieval"],
  ["enable_reranking", "recallReranking"],
  ["recall_prefer_observations", "recallPreferObservations"],
  ["recall_decision_model_prune", "recallDecisionModelPrune"],
  ["recall_decision_model_dedup", "recallDecisionModelDedup"],
] as const;

export type RecallField = (typeof RECALL_FIELDS)[number][0];

export type RecallEdits = Record<RecallField, boolean | null>;

/** The bank's own recall overrides; a field it does not override is null (inherit). */
export function recallSlice(overrides: Record<string, unknown>): RecallEdits {
  const edits = {} as RecallEdits;
  for (const [field] of RECALL_FIELDS) {
    const value = overrides[field];
    edits[field] = typeof value === "boolean" ? value : null;
  }
  return edits;
}
