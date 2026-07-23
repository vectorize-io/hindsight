export type ObservationsEdits = {
  enable_observations: boolean | null;
  consolidation_llm_batch_size: number | null;
  consolidation_source_facts_max_tokens: number | null;
  consolidation_source_facts_max_tokens_per_observation: number | null;
  observations_mission: string | null;
  max_observations_per_scope: number | null;
};

type ObservationsConfig = Partial<ObservationsEdits> & Record<string, unknown>;
type ObservationsOverridesSnapshot = Pick<Partial<ObservationsEdits>, "enable_observations"> &
  Record<string, unknown>;
const OBSERVATIONS_KEYS = [
  "enable_observations",
  "consolidation_llm_batch_size",
  "consolidation_source_facts_max_tokens",
  "consolidation_source_facts_max_tokens_per_observation",
  "observations_mission",
  "max_observations_per_scope",
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
  };
}

export function observationsSlice(
  resolvedConfig: ObservationsConfig,
  overrides: ObservationsOverridesSnapshot
): ObservationsEdits {
  return {
    ...resolvedObservationsSlice(resolvedConfig),
    // The resolved value cannot distinguish inheritance from an explicit bank
    // override. Keep only this field override-aware so the other controls retain
    // their existing resolved-value behavior.
    enable_observations: overrides.enable_observations ?? null,
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
  responseOverrides: ObservationsOverridesSnapshot
): Record<string, unknown> {
  // PATCH returns a complete bank-override snapshot. An absent key therefore
  // means the null tombstone was applied and the bank now inherits its parent.
  const next = { ...currentOverrides };
  const value = responseOverrides.enable_observations;
  if (value === null || value === undefined) delete next.enable_observations;
  else next.enable_observations = value;
  return next;
}

export function reconcileObservationsEdits(
  currentEdits: ObservationsEdits,
  submittedEdits: ObservationsEdits,
  resolvedConfig: ObservationsConfig,
  responseOverrides: ObservationsOverridesSnapshot
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
  };
}
