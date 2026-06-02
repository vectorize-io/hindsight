"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface HistoryEntry {
  previous_text: string;
  previous_tags: string[];
  previous_occurred_start: string | null;
  previous_occurred_end: string | null;
  previous_mentioned_at: string | null;
  changed_at: string;
  new_source_memory_ids: string[];
  source_facts?: {
    id: string;
    text: string | null;
    type: string | null;
    context: string | null;
    is_new: boolean;
  }[];
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
  const t = useTranslations("observationHistory");
  const parts = diffWords(before, after);
  const hasChanges = parts.some((p) => p.type !== "same");
  if (!hasChanges)
    return <span className="text-sm text-muted-foreground italic">{t("unchanged")}</span>;
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
  const tr = useTranslations("observationHistory");
  const removed = before.filter((t) => !after.includes(t));
  const added = after.filter((t) => !before.includes(t));
  const kept = before.filter((t) => after.includes(t));
  if (removed.length === 0 && added.length === 0)
    return <span className="text-sm text-muted-foreground italic">{tr("unchanged")}</span>;
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

function SourceFactItem({ fact }: { fact: NonNullable<HistoryEntry["source_facts"]>[number] }) {
  const t = useTranslations("observationHistory");
  const typeColors =
    fact.type === "experience"
      ? "bg-green-500/10 text-green-700 dark:text-green-400"
      : "bg-blue-500/10 text-blue-700 dark:text-blue-400";

  return (
    <div
      className={`p-2 rounded border space-y-1 ${
        fact.is_new ? "border-green-500/40 bg-green-500/5" : "border-border/50 bg-muted/30"
      }`}
    >
      <div className="flex items-center gap-1.5">
        {fact.type && (
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded font-medium flex-shrink-0 ${typeColors}`}
          >
            {fact.type}
          </span>
        )}
        {fact.is_new && (
          <span className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-green-500/15 text-green-700 dark:text-green-400 border border-green-500/30">
            {t("new")}
          </span>
        )}
        {fact.context && (
          <span className="text-[10px] text-muted-foreground italic truncate">{fact.context}</span>
        )}
      </div>
      {fact.text ? (
        <p className="text-xs text-foreground leading-relaxed">{fact.text}</p>
      ) : (
        <p className="text-xs text-muted-foreground italic">{t("memoryNoLongerAvailable")}</p>
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
  const t = useTranslations("observationHistory");
  // index 0 = most recent change
  const entries = [...history].reverse();
  const [idx, setIdx] = useState(0);

  const entry = entries[idx];
  const isLatest = idx === 0;
  const afterText = isLatest ? current.text : entries[idx - 1].previous_text;
  const afterTags = isLatest ? current.tags : entries[idx - 1].previous_tags;
  const afterOccurredStart = isLatest
    ? current.occurred_start
    : entries[idx - 1].previous_occurred_start;
  const afterOccurredEnd = isLatest ? current.occurred_end : entries[idx - 1].previous_occurred_end;
  const afterMentionedAt = isLatest ? current.mentioned_at : entries[idx - 1].previous_mentioned_at;

  return (
    <div className="space-y-3">
      {/* Navigation header */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {t("changeOf", { current: history.length - idx, total: history.length })} &middot;{" "}
          {new Date(entry.changed_at).toLocaleString()}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={idx === entries.length - 1}
            onClick={() => setIdx(idx + 1)}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={idx === 0}
            onClick={() => setIdx(idx - 1)}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Change card */}
      <div className="border border-border rounded-lg p-3 space-y-3">
        <div>
          <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
            {t("sectionText")}
          </div>
          <TextDiff before={entry.previous_text} after={afterText} />
        </div>

        <div>
          <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
            {t("sectionTags")}
          </div>
          <TagsDiff before={entry.previous_tags} after={afterTags} />
        </div>

        <div className="space-y-1">
          <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
            {t("sectionDates")}
          </div>
          <DateDiff
            label={t("occurredStart")}
            before={entry.previous_occurred_start}
            after={afterOccurredStart}
          />
          <DateDiff
            label={t("occurredEnd")}
            before={entry.previous_occurred_end}
            after={afterOccurredEnd}
          />
          <DateDiff
            label={t("mentionedAt")}
            before={entry.previous_mentioned_at}
            after={afterMentionedAt}
          />
        </div>

        {entry.source_facts && entry.source_facts.length > 0 && (
          <div>
            <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
              {t("sourceFacts", { count: entry.source_facts.length })}
            </div>
            <div className="space-y-1.5">
              {entry.source_facts.map((fact) => (
                <SourceFactItem key={fact.id} fact={fact} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
