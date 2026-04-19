/**
 * Content processing utilities.
 *
 * Port of the Claude Code plugin's content.py:
 *   - Memory tag stripping (anti-feedback-loop)
 *   - Recall query composition and truncation
 *   - Memory formatting for context injection
 *   - Retention transcript formatting
 */

/** Strip <hindsight_memories> and <relevant_memories> blocks to prevent retain feedback loops. */
export function stripMemoryTags(content: string): string {
  content = content.replace(/<hindsight_memories>[\s\S]*?<\/hindsight_memories>/g, "");
  content = content.replace(/<relevant_memories>[\s\S]*?<\/relevant_memories>/g, "");
  return content;
}

export interface RecallResult {
  text: string;
  type?: string | null;
  mentioned_at?: string | null;
}

/** Format recall results into human-readable text for context injection. */
export function formatMemories(results: RecallResult[]): string {
  if (!results.length) return "";
  return results
    .map((r) => {
      const typeStr = r.type ? ` [${r.type}]` : "";
      const dateStr = r.mentioned_at ? ` (${r.mentioned_at})` : "";
      return `- ${r.text}${typeStr}${dateStr}`;
    })
    .join("\n\n");
}

/** Format current UTC time for recall context. */
export function formatCurrentTime(): string {
  const now = new Date();
  const y = now.getUTCFullYear();
  const m = String(now.getUTCMonth() + 1).padStart(2, "0");
  const d = String(now.getUTCDate()).padStart(2, "0");
  const h = String(now.getUTCHours()).padStart(2, "0");
  const min = String(now.getUTCMinutes()).padStart(2, "0");
  return `${y}-${m}-${d} ${h}:${min}`;
}

export interface Message {
  role: string;
  content: string;
}

/**
 * Compose a multi-turn recall query from conversation history.
 *
 * When recallContextTurns > 1, includes prior context above the latest query.
 */
export function composeRecallQuery(
  latestQuery: string,
  messages: Message[],
  recallContextTurns: number
): string {
  const latest = latestQuery.trim();
  if (recallContextTurns <= 1 || !messages.length) return latest;

  const contextual = sliceLastTurnsByUserBoundary(messages, recallContextTurns);
  const contextLines: string[] = [];

  for (const msg of contextual) {
    const content = stripMemoryTags(msg.content).trim();
    if (!content) continue;
    if (msg.role === "user" && content === latest) continue;
    contextLines.push(`${msg.role}: ${content}`);
  }

  if (!contextLines.length) return latest;

  return ["Prior context:", contextLines.join("\n"), latest].join("\n\n");
}

/**
 * Truncate a composed recall query to maxChars.
 * Preserves the latest user message, drops oldest context lines first.
 */
export function truncateRecallQuery(query: string, latestQuery: string, maxChars: number): string {
  if (maxChars <= 0 || query.length <= maxChars) return query;

  const latest = latestQuery.trim();
  const latestOnly = latest.length > maxChars ? latest.slice(0, maxChars) : latest;

  if (!query.includes("Prior context:")) return latestOnly;

  const contextMarker = "Prior context:\n\n";
  const markerIndex = query.indexOf(contextMarker);
  if (markerIndex === -1) return latestOnly;

  const suffix = "\n\n" + latest;
  const suffixIndex = query.lastIndexOf(suffix);
  if (suffixIndex === -1) return latestOnly;
  if (suffix.length >= maxChars) return latestOnly;

  const contextBody = query.slice(markerIndex + contextMarker.length, suffixIndex);
  const contextLines = contextBody.split("\n").filter(Boolean);

  const kept: string[] = [];
  for (let i = contextLines.length - 1; i >= 0; i--) {
    kept.unshift(contextLines[i]);
    const candidate = `${contextMarker}${kept.join("\n")}${suffix}`;
    if (candidate.length > maxChars) {
      kept.shift();
      break;
    }
  }

  if (kept.length) return `${contextMarker}${kept.join("\n")}${suffix}`;
  return latestOnly;
}

/** Slice messages to the last N turns, where a turn starts at a user message. */
export function sliceLastTurnsByUserBoundary(messages: Message[], turns: number): Message[] {
  if (!messages.length || turns <= 0) return [];

  let userTurnsSeen = 0;
  let startIndex = -1;

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      userTurnsSeen++;
      if (userTurnsSeen >= turns) {
        startIndex = i;
        break;
      }
    }
  }

  return startIndex === -1 ? [...messages] : messages.slice(startIndex);
}

/**
 * Format messages into a retention transcript.
 *
 * Uses [role: ...]...[role:end] markers for structured retention.
 */
export function prepareRetentionTranscript(
  messages: Message[],
  retainFullWindow: boolean = false
): { transcript: string | null; messageCount: number } {
  if (!messages.length) return { transcript: null, messageCount: 0 };

  let targetMessages: Message[];
  if (retainFullWindow) {
    targetMessages = messages;
  } else {
    // Default: retain only the last turn
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx === -1) return { transcript: null, messageCount: 0 };
    targetMessages = messages.slice(lastUserIdx);
  }

  const parts: string[] = [];
  for (const msg of targetMessages) {
    const content = stripMemoryTags(msg.content).trim();
    if (!content) continue;
    parts.push(`[role: ${msg.role}]\n${content}\n[${msg.role}:end]`);
  }

  if (!parts.length) return { transcript: null, messageCount: 0 };

  const transcript = parts.join("\n\n");
  if (transcript.trim().length < 10) return { transcript: null, messageCount: 0 };

  return { transcript, messageCount: parts.length };
}
