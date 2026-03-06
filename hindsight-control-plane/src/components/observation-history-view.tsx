"use client";

export interface HistoryEntry {
  previous_text: string;
  previous_tags: string[];
  previous_occurred_start: string | null;
  previous_occurred_end: string | null;
  previous_mentioned_at: string | null;
  changed_at: string;
  new_source_memory_ids: string[];
}

interface CurrentState {
  text: string;
  tags: string[];
  occurred_start: string | null;
  occurred_end: string | null;
  mentioned_at: string | null;
}

function diffWords(a: string, b: string): { type: "same" | "removed" | "added"; text: string }[] {
  const aWords = a.split(/(\s+)/);
  const bWords = b.split(/(\s+)/);
  const m = aWords.length;
  const n = bWords.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        aWords[i - 1] === bWords[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  let i = m,
    j = n;
  const ops: { type: "same" | "removed" | "added"; text: string }[] = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && aWords[i - 1] === bWords[j - 1]) {
      ops.push({ type: "same", text: aWords[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ type: "added", text: bWords[j - 1] });
      j--;
    } else {
      ops.push({ type: "removed", text: aWords[i - 1] });
      i--;
    }
  }
  return ops.reverse();
}

function TextDiff({ before, after }: { before: string; after: string }) {
  const parts = diffWords(before, after);
  const hasChanges = parts.some((p) => p.type !== "same");
  if (!hasChanges) return <span className="text-sm text-muted-foreground italic">unchanged</span>;
  return (
    <span className="text-sm leading-relaxed">
      {parts.map((part, idx) =>
        part.type === "same" ? (
          <span key={idx}>{part.text}</span>
        ) : part.type === "removed" ? (
          <span
            key={idx}
            className="bg-red-500/15 text-red-700 dark:text-red-400 line-through rounded-sm px-0.5"
          >
            {part.text}
          </span>
        ) : (
          <span
            key={idx}
            className="bg-green-500/15 text-green-700 dark:text-green-400 rounded-sm px-0.5"
          >
            {part.text}
          </span>
        )
      )}
    </span>
  );
}

function TagsDiff({ before, after }: { before: string[]; after: string[] }) {
  const removed = before.filter((t) => !after.includes(t));
  const added = after.filter((t) => !before.includes(t));
  const kept = before.filter((t) => after.includes(t));
  if (removed.length === 0 && added.length === 0)
    return <span className="text-sm text-muted-foreground italic">unchanged</span>;
  return (
    <div className="flex gap-1 flex-wrap">
      {kept.map((t, idx) => (
        <span
          key={idx}
          className="text-[10px] px-1.5 py-0.5 rounded-md bg-amber-500/10 text-amber-700 border border-amber-500/20 font-mono"
        >
          #{t}
        </span>
      ))}
      {removed.map((t, idx) => (
        <span
          key={idx}
          className="text-[10px] px-1.5 py-0.5 rounded-md bg-red-500/15 text-red-700 dark:text-red-400 border border-red-500/20 font-mono line-through"
        >
          #{t}
        </span>
      ))}
      {added.map((t, idx) => (
        <span
          key={idx}
          className="text-[10px] px-1.5 py-0.5 rounded-md bg-green-500/15 text-green-700 dark:text-green-400 border border-green-500/20 font-mono"
        >
          +#{t}
        </span>
      ))}
    </div>
  );
}

function DateDiff({
  label,
  before,
  after,
}: {
  label: string;
  before: string | null;
  after: string | null;
}) {
  if (!before && !after) return null;
  const changed = before !== after;
  return (
    <div>
      <span className="text-xs text-muted-foreground">{label}: </span>
      {changed ? (
        <>
          <span className="text-xs bg-red-500/15 text-red-700 dark:text-red-400 line-through rounded-sm px-0.5">
            {before ? new Date(before).toLocaleString() : "—"}
          </span>
          {" → "}
          <span className="text-xs bg-green-500/15 text-green-700 dark:text-green-400 rounded-sm px-0.5">
            {after ? new Date(after).toLocaleString() : "—"}
          </span>
        </>
      ) : (
        <span className="text-xs">{after ? new Date(after).toLocaleString() : "—"}</span>
      )}
    </div>
  );
}

export function ObservationHistoryView({
  history,
  current,
}: {
  history: HistoryEntry[];
  current: CurrentState;
}) {
  const entries = [...history].reverse();

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        {history.length} change{history.length !== 1 ? "s" : ""} recorded. Most recent first.
      </p>
      {entries.map((entry, idx) => {
        const isLatest = idx === 0;
        const afterText = isLatest ? current.text : entries[idx - 1].previous_text;
        const afterTags = isLatest ? current.tags : entries[idx - 1].previous_tags;
        const afterOccurredStart = isLatest
          ? current.occurred_start
          : entries[idx - 1].previous_occurred_start;
        const afterOccurredEnd = isLatest
          ? current.occurred_end
          : entries[idx - 1].previous_occurred_end;
        const afterMentionedAt = isLatest
          ? current.mentioned_at
          : entries[idx - 1].previous_mentioned_at;

        return (
          <div key={idx} className="border border-border rounded-lg p-3 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-muted-foreground uppercase">
                Change #{history.length - idx}
              </span>
              <span className="text-xs text-muted-foreground">
                {new Date(entry.changed_at).toLocaleString()}
              </span>
            </div>

            <div>
              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">Text</div>
              <TextDiff before={entry.previous_text} after={afterText} />
            </div>

            <div>
              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">Tags</div>
              <TagsDiff before={entry.previous_tags} after={afterTags} />
            </div>

            <div className="space-y-1">
              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">Dates</div>
              <DateDiff
                label="Occurred start"
                before={entry.previous_occurred_start}
                after={afterOccurredStart}
              />
              <DateDiff
                label="Occurred end"
                before={entry.previous_occurred_end}
                after={afterOccurredEnd}
              />
              <DateDiff
                label="Mentioned at"
                before={entry.previous_mentioned_at}
                after={afterMentionedAt}
              />
            </div>

            {entry.new_source_memory_ids && entry.new_source_memory_ids.length > 0 && (
              <div>
                <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                  New Sources ({entry.new_source_memory_ids.length})
                </div>
                <div className="space-y-1">
                  {entry.new_source_memory_ids.map((id) => (
                    <code
                      key={id}
                      className="block text-xs font-mono text-muted-foreground truncate"
                    >
                      {id}
                    </code>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
