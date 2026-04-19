import type { RecallResult, EntityState, MemoryPromptOptions } from "./types.js";

const DEFAULT_PREAMBLE =
  "You have access to the following memories about this user from previous interactions:";

/**
 * Formats recalled memories and entity observations into a system prompt string.
 *
 * Returns an empty string when there are no memories or entities to include,
 * so callers can safely concatenate or conditionally append.
 */
export function formatMemoriesAsSystemPrompt(
  memories: RecallResult[],
  entities: Record<string, EntityState> | null | undefined,
  options?: MemoryPromptOptions
): string {
  const {
    preamble = DEFAULT_PREAMBLE,
    maxMemories,
    includeTypes,
    includeEntities = true,
  } = options ?? {};

  let filtered = memories;

  if (includeTypes && includeTypes.length > 0) {
    filtered = filtered.filter((m) => m.type != null && includeTypes.includes(m.type as never));
  }

  if (maxMemories != null && maxMemories > 0) {
    filtered = filtered.slice(0, maxMemories);
  }

  const hasMemories = filtered.length > 0;
  const entityEntries = entities ? Object.values(entities) : [];
  const hasEntities = includeEntities && entityEntries.length > 0;

  if (!hasMemories && !hasEntities) {
    return "";
  }

  const parts: string[] = [preamble, ""];

  if (hasMemories) {
    parts.push("<memories>");
    for (const memory of filtered) {
      const typeSuffix = memory.type ? ` [${memory.type}]` : "";
      parts.push(`- ${memory.text}${typeSuffix}`);
    }
    parts.push("</memories>");
  }

  if (hasEntities) {
    if (hasMemories) parts.push("");
    parts.push("<entity_observations>");
    for (const entity of entityEntries) {
      parts.push(`## ${entity.canonical_name}`);
      for (const obs of entity.observations) {
        parts.push(`- ${obs.text}`);
      }
    }
    parts.push("</entity_observations>");
  }

  return parts.join("\n");
}
