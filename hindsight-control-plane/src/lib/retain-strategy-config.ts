export type RetainStrategyValues<TLabels = unknown> = {
  retain_extraction_mode: string | null;
  retain_chunk_size: number | null;
  retain_structured_chunk_size: number | null;
  retain_mission: string | null;
  retain_custom_instructions: string | null;
  entities_allow_free_form: boolean | null;
  entity_labels: TLabels | null;
};

export type RetainStrategy<TLabels = unknown> = {
  id: number;
  name: string;
  values: RetainStrategyValues<TLabels>;
};

export function deserializeRetainStrategies<TLabels>(
  dict: Record<string, Record<string, any>> | null,
  parseEntityLabels: (raw: unknown) => TLabels | null
): RetainStrategy<TLabels>[] {
  if (!dict) return [];
  return Object.entries(dict).map(([name, overrides], i) => {
    return {
      id: i,
      name,
      values: {
        retain_extraction_mode: overrides.retain_extraction_mode ?? null,
        retain_chunk_size: overrides.retain_chunk_size ?? null,
        retain_structured_chunk_size: overrides.retain_structured_chunk_size ?? null,
        retain_mission: overrides.retain_mission ?? null,
        retain_custom_instructions: overrides.retain_custom_instructions ?? null,
        entities_allow_free_form: overrides.entities_allow_free_form ?? null,
        entity_labels: parseEntityLabels(overrides.entity_labels),
      },
    };
  });
}

export function serializeRetainStrategies<TLabels>(
  local: RetainStrategy<TLabels>[]
): Record<string, Record<string, any>> | null {
  const dict: Record<string, Record<string, any>> = {};
  for (const s of local) {
    if (!s.name.trim()) continue;
    const overrides: Record<string, any> = {};
    if (s.values.retain_extraction_mode !== null)
      overrides.retain_extraction_mode = s.values.retain_extraction_mode;
    if (s.values.retain_chunk_size !== null)
      overrides.retain_chunk_size = s.values.retain_chunk_size;
    if (s.values.retain_structured_chunk_size !== null) {
      overrides.retain_structured_chunk_size = s.values.retain_structured_chunk_size;
    }
    if (s.values.retain_mission) overrides.retain_mission = s.values.retain_mission;
    if (s.values.retain_custom_instructions)
      overrides.retain_custom_instructions = s.values.retain_custom_instructions;
    if (s.values.entities_allow_free_form !== null)
      overrides.entities_allow_free_form = s.values.entities_allow_free_form;
    if (s.values.entity_labels !== null) overrides.entity_labels = s.values.entity_labels;
    dict[s.name.trim()] = overrides;
  }
  return Object.keys(dict).length > 0 ? dict : null;
}
