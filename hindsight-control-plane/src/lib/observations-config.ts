export type ObservationsEdits = {
  enable_observations: boolean | null;
  consolidation_llm_batch_size: number | null;
  consolidation_source_facts_max_tokens: number | null;
  consolidation_source_facts_max_tokens_per_observation: number | null;
  observations_mission: string | null;
  max_observations_per_scope: number | null;
  consolidation_strategies: ConsolidationStrategy[] | null;
};

/** The settings a consolidation strategy can override. The bank-level values of
 *  these same four fields are what the UI calls the "Default" strategy: they apply
 *  to every scope no strategy claims, and to any setting a strategy leaves empty. */
export type ConsolidationSettings = Pick<
  ObservationsEdits,
  | "observations_mission"
  | "max_observations_per_scope"
  | "consolidation_source_facts_max_tokens"
  | "consolidation_source_facts_max_tokens_per_observation"
>;

/** One `consolidation_strategies` entry: the rules it claims scopes with, and
 *  their settings. A strategy claims a consolidation pass when any rule matches
 *  it (see `ScopePattern`). Every setting is optional. */
export type ConsolidationStrategy = {
  /** Alternatives: the strategy claims a scope when any pattern matches it. */
  scopes: ScopePattern[];
} & Partial<ConsolidationSettings>;

/** One alternative: tags that must all be on the scope, and whether others may
 *  be too — "all" (default, omitted when stored) or "exact". Per pattern, so one
 *  strategy can mix them. */
export type ScopePattern = { tags: string[]; tags_match?: StrategyTagsMatch };

export type StrategyTagsMatch = "all" | "exact";

type ObservationsConfig = Partial<ObservationsEdits> & Record<string, unknown>;
type ObservationsOverridesSnapshot = Pick<Partial<ObservationsEdits>, "enable_observations"> &
  Record<string, unknown>;
type ObservationsOverridesResponse = ObservationsOverridesSnapshot | null | undefined;
const OBSERVATIONS_KEYS = [
  "enable_observations",
  "consolidation_llm_batch_size",
  "consolidation_source_facts_max_tokens",
  "consolidation_source_facts_max_tokens_per_observation",
  "observations_mission",
  "max_observations_per_scope",
  "consolidation_strategies",
] as const satisfies readonly (keyof ObservationsEdits)[];

function resolvedObservationsSlice(resolvedConfig: ObservationsConfig): ObservationsEdits {
  return {
    enable_observations: resolvedConfig.enable_observations ?? null,
    consolidation_llm_batch_size: resolvedConfig.consolidation_llm_batch_size ?? null,
    consolidation_source_facts_max_tokens:
      resolvedConfig.consolidation_source_facts_max_tokens ?? null,
    consolidation_source_facts_max_tokens_per_observation:
      resolvedConfig.consolidation_source_facts_max_tokens_per_observation ?? null,
    observations_mission: resolvedConfig.observations_mission ?? null,
    max_observations_per_scope: resolvedConfig.max_observations_per_scope ?? null,
    consolidation_strategies: resolvedConfig.consolidation_strategies ?? null,
  };
}

export function observationsSlice(
  resolvedConfig: ObservationsConfig,
  overrides: ObservationsOverridesResponse
): ObservationsEdits {
  return {
    ...resolvedObservationsSlice(resolvedConfig),
    // The resolved value cannot distinguish inheritance from an explicit bank
    // override. Keep only this field override-aware so the other controls retain
    // their existing resolved-value behavior.
    enable_observations: overrides?.enable_observations ?? null,
  };
}

export function mergeResolvedObservations(
  currentConfig: Record<string, unknown>,
  submittedEdits: ObservationsEdits,
  resolvedConfig: ObservationsConfig
): Record<string, unknown> {
  // A section save must not move another editor's baseline if the response also
  // reflects a concurrent or canonicalized value outside Observations. Config
  // may omit permission-filtered fields, so accepted submitted values become
  // their baseline unless the response supplies a canonical value.
  const next = { ...currentConfig };
  for (const key of OBSERVATIONS_KEYS) {
    if (Object.prototype.hasOwnProperty.call(resolvedConfig, key)) {
      next[key] = resolvedConfig[key] ?? null;
    } else if (key === "enable_observations" && submittedEdits[key] === null) {
      // After clearing an override, the old resolved value represented that
      // override. Drop it when permissions hide the new parent value.
      delete next[key];
    } else {
      next[key] = submittedEdits[key];
    }
  }
  return next;
}

export function mergeObservationsOverrides(
  currentOverrides: Record<string, unknown>,
  responseOverrides: ObservationsOverridesResponse
): Record<string, unknown> {
  // PATCH returns a complete bank-override snapshot. An absent key therefore
  // means the null tombstone was applied and the bank now inherits its parent.
  const next = { ...currentOverrides };
  const value = responseOverrides?.enable_observations;
  if (value === null || value === undefined) delete next.enable_observations;
  else next.enable_observations = value;
  return next;
}

export function reconcileObservationsEdits(
  currentEdits: ObservationsEdits,
  submittedEdits: ObservationsEdits,
  resolvedConfig: ObservationsConfig,
  responseOverrides: ObservationsOverridesResponse
): ObservationsEdits {
  const responseEdits = observationsSlice(
    { ...submittedEdits, ...resolvedConfig },
    responseOverrides
  );
  const reconcileField = <K extends keyof ObservationsEdits>(key: K): ObservationsEdits[K] =>
    Object.is(currentEdits[key], submittedEdits[key]) ? responseEdits[key] : currentEdits[key];

  // Inputs remain editable during a save. Preserve only fields changed after
  // submission, while accepting canonical response values for untouched fields.
  return {
    enable_observations: reconcileField("enable_observations"),
    consolidation_llm_batch_size: reconcileField("consolidation_llm_batch_size"),
    consolidation_source_facts_max_tokens: reconcileField("consolidation_source_facts_max_tokens"),
    consolidation_source_facts_max_tokens_per_observation: reconcileField(
      "consolidation_source_facts_max_tokens_per_observation"
    ),
    observations_mission: reconcileField("observations_mission"),
    max_observations_per_scope: reconcileField("max_observations_per_scope"),
    consolidation_strategies: reconcileField("consolidation_strategies"),
  };
}

/** A strategy as stored: empty settings are left out instead of written as null,
 *  so "inherit from Default" has one spelling on the wire (absent). A blank
 *  mission counts as empty — the server ignores one anyway. */
export function compactStrategy(strategy: ConsolidationStrategy): ConsolidationStrategy {
  const out: ConsolidationStrategy = {
    scopes: strategy.scopes.map((pattern) =>
      pattern.tags_match === "exact"
        ? { tags: pattern.tags, tags_match: "exact" }
        : { tags: pattern.tags }
    ),
  };
  if (strategy.observations_mission?.trim())
    out.observations_mission = strategy.observations_mission;
  for (const key of [
    "max_observations_per_scope",
    "consolidation_source_facts_max_tokens",
    "consolidation_source_facts_max_tokens_per_observation",
  ] as const) {
    const value = strategy[key];
    if (typeof value === "number") out[key] = value;
  }
  return out;
}

/** Whether a strategy changes anything. The server drops one that does not, so
 *  the editor warns about it instead of letting it look configured. */
export function strategyOverridesSomething(strategy: ConsolidationStrategy): boolean {
  const { scopes: _scopes, ...settings } = compactStrategy(strategy);
  return Object.keys(settings).length > 0;
}

/** A strategy's tab label: its scopes are its name. Tags inside one scope must
 *  all be present ("org:* and shared"); separate scopes are alternatives
 *  ("team:* or org:* and shared"). The joining words are passed in so the label
 *  is translated. Empty scopes — a box still being filled in — are skipped. */
export function scopesLabel(scopes: ScopePattern[], and = "and", or = "or"): string {
  return scopes
    .filter((pattern) => pattern.tags.length > 0)
    .map((pattern) => pattern.tags.join(` ${and} `))
    .join(` ${or} `);
}

/** Autocomplete for the scope editor: the given tags (the editor passes the
 *  bank's tag search results), plus a `key:*` wildcard for each `key:value` tag —
 *  the pattern people almost always want ("every company", "every team"). */
export function suggestedTags(tagGroups: string[][]): string[] {
  const out = new Set<string>();
  for (const tags of tagGroups) {
    for (const tag of tags) {
      out.add(tag);
      const colon = tag.indexOf(":");
      if (colon > 0) out.add(`${tag.slice(0, colon)}:*`);
    }
  }
  return [...out].sort();
}
