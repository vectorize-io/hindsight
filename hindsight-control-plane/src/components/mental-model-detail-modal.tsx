"use client";

import { useState, useEffect, useMemo } from "react";
import { client, MentalModel } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Loader2,
  Zap,
  FileText,
  History as HistoryIcon,
  Settings,
  ChevronLeft,
  ChevronRight,
  MoreVertical,
  Pencil,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { CompactMarkdown } from "./compact-markdown";
import { MemoryDetailModal } from "./memory-detail-modal";
import { DirectiveDetailModal } from "./directive-detail-modal";
import { formatAbsoluteDateTime as formatDateTime, formatRelativeTime } from "@/lib/relative-time";

type BasedOnFact = {
  id: string;
  text: string;
  type?: string;
  context?: string | null;
};

type ReflectResponseSnapshot = {
  text?: string;
  based_on?: Record<string, BasedOnFact[]>;
  mental_models?: unknown[];
} | null;

type HistoryEntry = {
  previous_content: string | null;
  previous_reflect_response?: ReflectResponseSnapshot;
  changed_at: string;
};

function getFactTypeDisplay(factType: string) {
  if (factType === "directives") {
    return { label: "directive", color: "bg-purple-500/10 text-purple-600 dark:text-purple-400" };
  }
  if (factType === "mental-models") {
    return {
      label: "mental model",
      color: "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400",
    };
  }
  if (factType === "world") {
    return { label: "world", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" };
  }
  if (factType === "experience") {
    return { label: "experience", color: "bg-green-500/10 text-green-600 dark:text-green-400" };
  }
  if (factType === "observation") {
    return {
      label: "observation",
      color: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
    };
  }
  return { label: factType, color: "bg-slate-500/10 text-slate-600 dark:text-slate-400" };
}

const FACT_TYPE_ORDER = ["observation", "experience", "world", "directives", "mental-models"];

function BasedOnList({
  based_on,
  onViewMemory,
  onViewDirective,
}: {
  based_on: Record<string, BasedOnFact[]> | undefined;
  onViewMemory?: (id: string) => void;
  onViewDirective?: (id: string) => void;
}) {
  const groups = useMemo(() => {
    if (!based_on) return [] as Array<{ factType: string; facts: BasedOnFact[] }>;
    const all = Object.entries(based_on)
      .map(([factType, facts]) => ({ factType, facts: facts ?? [] }))
      .filter((g) => g.facts.length > 0);
    all.sort((a, b) => {
      const ai = FACT_TYPE_ORDER.indexOf(a.factType);
      const bi = FACT_TYPE_ORDER.indexOf(b.factType);
      return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    });
    return all;
  }, [based_on]);

  if (groups.length === 0) {
    return <p className="text-sm text-muted-foreground italic">No based_on data.</p>;
  }

  return (
    <div className="space-y-4">
      {groups.map((group) => {
        const display = getFactTypeDisplay(group.factType);
        return (
          <div key={group.factType} className="rounded-lg border border-border/60 overflow-hidden">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/40 border-b border-border/60">
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${display.color}`}
              >
                {display.label}
              </span>
              <span className="text-xs text-muted-foreground">{group.facts.length}</span>
            </div>
            <ul className="divide-y divide-border/50">
              {group.facts.map((fact, i) => {
                const canView =
                  (group.factType === "directives" && !!onViewDirective) ||
                  (group.factType !== "directives" &&
                    group.factType !== "mental-models" &&
                    !!onViewMemory);
                return (
                  <li
                    key={fact.id || i}
                    className="group flex items-start gap-3 px-3 py-2 hover:bg-muted/30"
                  >
                    <p className="text-sm text-foreground leading-relaxed flex-1 min-w-0">
                      {fact.text}
                      {fact.context && (
                        <span className="block text-xs text-muted-foreground italic mt-0.5">
                          {fact.context}
                        </span>
                      )}
                    </p>
                    {canView && (
                      <button
                        className="text-xs text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0 mt-0.5"
                        onClick={() => {
                          if (group.factType === "directives") onViewDirective?.(fact.id);
                          else onViewMemory?.(fact.id);
                        }}
                      >
                        View →
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

type LineKind = "same" | "removed" | "added";
type LineDiff = { type: LineKind; text: string };
type TokenSpan = { type: LineKind; text: string };
type AnnotatedLine = { type: LineKind; spans: TokenSpan[] };

/** Tokenise into words, whitespace, and single punctuation marks so the
 *  per-character diff aligns on natural boundaries instead of mid-word. */
function tokenize(s: string): string[] {
  return s.match(/\s+|[A-Za-z0-9_]+|[^\s\w]/g) || [];
}

/** Normalize a token for equality: collapse runs of whitespace so that
 *  reflowing whitespace doesn't show as a diff. */
function normToken(t: string): string {
  return /^\s+$/.test(t) ? " " : t;
}

/** Word-level LCS over two single lines, returning inline spans for
 *  before/after. Pure-whitespace differences are folded into "same" so
 *  the diff highlights actual content changes only. */
function diffTokensInline(a: string, b: string): { left: TokenSpan[]; right: TokenSpan[] } {
  const aTok = tokenize(a);
  const bTok = tokenize(b);
  const m = aTok.length;
  const n = bTok.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        normToken(aTok[i - 1]) === normToken(bTok[j - 1])
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const left: TokenSpan[] = [];
  const right: TokenSpan[] = [];
  let i = m;
  let j = n;
  const leftStack: TokenSpan[] = [];
  const rightStack: TokenSpan[] = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && normToken(aTok[i - 1]) === normToken(bTok[j - 1])) {
      leftStack.push({ type: "same", text: aTok[i - 1] });
      rightStack.push({ type: "same", text: bTok[j - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      // Whitespace-only insertions render as "same" so reflow noise is hidden.
      const t = bTok[j - 1];
      rightStack.push({ type: /^\s+$/.test(t) ? "same" : "added", text: t });
      j--;
    } else {
      const t = aTok[i - 1];
      leftStack.push({ type: /^\s+$/.test(t) ? "same" : "removed", text: t });
      i--;
    }
  }
  while (leftStack.length) left.push(leftStack.pop()!);
  while (rightStack.length) right.push(rightStack.pop()!);
  return { left, right };
}

function diffLines(a: string, b: string): { left: AnnotatedLine[]; right: AnnotatedLine[] } {
  const aLines = a.split("\n");
  const bLines = b.split("\n");
  const m = aLines.length;
  const n = bLines.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        aLines[i - 1] === bLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const ops: LineDiff[] = [];
  let i = m;
  let j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && aLines[i - 1] === bLines[j - 1]) {
      ops.push({ type: "same", text: aLines[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ type: "added", text: bLines[j - 1] });
      j--;
    } else {
      ops.push({ type: "removed", text: aLines[i - 1] });
      i--;
    }
  }
  ops.reverse();

  const left: AnnotatedLine[] = [];
  const right: AnnotatedLine[] = [];
  let k = 0;
  while (k < ops.length) {
    const op = ops[k];
    if (op.type === "same") {
      left.push({ type: "same", spans: [{ type: "same", text: op.text }] });
      right.push({ type: "same", spans: [{ type: "same", text: op.text }] });
      k++;
      continue;
    }
    // Collect a contiguous run of removed/added lines and pair them up so
    // we can emit per-token inline diffs for the overlap.
    const removed: string[] = [];
    const added: string[] = [];
    while (k < ops.length && ops[k].type !== "same") {
      if (ops[k].type === "removed") removed.push(ops[k].text);
      else added.push(ops[k].text);
      k++;
    }
    const maxLen = Math.max(removed.length, added.length);
    for (let r = 0; r < maxLen; r++) {
      const r0 = r < removed.length ? removed[r] : null;
      const a0 = r < added.length ? added[r] : null;
      if (r0 !== null && a0 !== null) {
        const inline = diffTokensInline(r0, a0);
        // Demote whole-line classification to "same" if every span is
        // already "same" (purely whitespace-only difference).
        const leftKind: LineKind = inline.left.some((s) => s.type !== "same") ? "removed" : "same";
        const rightKind: LineKind = inline.right.some((s) => s.type !== "same") ? "added" : "same";
        left.push({ type: leftKind, spans: inline.left });
        right.push({ type: rightKind, spans: inline.right });
      } else if (r0 !== null) {
        left.push({ type: "removed", spans: [{ type: "removed", text: r0 }] });
        right.push({ type: "same", spans: [{ type: "same", text: "" }] });
      } else if (a0 !== null) {
        left.push({ type: "same", spans: [{ type: "same", text: "" }] });
        right.push({ type: "added", spans: [{ type: "added", text: a0 }] });
      }
    }
  }
  return { left, right };
}

function renderSpans(spans: TokenSpan[], side: "left" | "right") {
  return spans.map((s, i) => {
    if (s.type === "same") return <span key={i}>{s.text}</span>;
    const cls =
      side === "left"
        ? "bg-red-500/25 text-red-800 dark:text-red-300 rounded-sm px-0.5"
        : "bg-green-500/25 text-green-800 dark:text-green-300 rounded-sm px-0.5";
    return (
      <span key={i} className={cls}>
        {s.text}
      </span>
    );
  });
}

function SideBySideDiff({ before, after }: { before: string; after: string }) {
  const { left, right } = diffLines(before, after);
  const hasChanges = left.some((l) => l.type !== "same") || right.some((r) => r.type !== "same");
  if (!hasChanges) return <span className="text-sm text-muted-foreground italic">unchanged</span>;

  return (
    <div className="grid grid-cols-2 divide-x divide-border border border-border rounded-md overflow-hidden text-xs font-mono">
      <div>
        <div className="px-3 py-1.5 bg-muted text-muted-foreground font-sans font-semibold text-xs uppercase tracking-wide border-b border-border">
          Before
        </div>
        {left.map((line, idx) => (
          <div
            key={idx}
            className={`px-3 py-0.5 whitespace-pre-wrap leading-5 min-h-[1.25rem] ${
              line.type === "removed" ? "bg-red-500/5" : ""
            }`}
          >
            {renderSpans(line.spans, "left")}
          </div>
        ))}
      </div>
      <div>
        <div className="px-3 py-1.5 bg-muted text-muted-foreground font-sans font-semibold text-xs uppercase tracking-wide border-b border-border">
          After
        </div>
        {right.map((line, idx) => (
          <div
            key={idx}
            className={`px-3 py-0.5 whitespace-pre-wrap leading-5 min-h-[1.25rem] ${
              line.type === "added" ? "bg-green-500/5" : ""
            }`}
          >
            {renderSpans(line.spans, "right")}
          </div>
        ))}
      </div>
    </div>
  );
}

function BasedOnDiff({
  before,
  after,
  onViewMemory,
  onViewDirective,
}: {
  before: Record<string, BasedOnFact[]> | undefined;
  after: Record<string, BasedOnFact[]> | undefined;
  onViewMemory: (id: string) => void;
  onViewDirective: (id: string) => void;
}) {
  const diff = useMemo(() => {
    const types = new Set<string>([...Object.keys(before ?? {}), ...Object.keys(after ?? {})]);
    const groups: Array<{
      factType: string;
      added: BasedOnFact[];
      removed: BasedOnFact[];
      kept: BasedOnFact[];
    }> = [];
    for (const factType of types) {
      const b = (before?.[factType] ?? []).filter(Boolean);
      const a = (after?.[factType] ?? []).filter(Boolean);
      const bIds = new Set(b.map((f) => f.id));
      const aIds = new Set(a.map((f) => f.id));
      const added = a.filter((f) => !bIds.has(f.id));
      const removed = b.filter((f) => !aIds.has(f.id));
      const kept = a.filter((f) => bIds.has(f.id));
      if (added.length === 0 && removed.length === 0 && kept.length === 0) continue;
      groups.push({ factType, added, removed, kept });
    }
    groups.sort((x, y) => {
      const xi = FACT_TYPE_ORDER.indexOf(x.factType);
      const yi = FACT_TYPE_ORDER.indexOf(y.factType);
      return (xi === -1 ? 999 : xi) - (yi === -1 ? 999 : yi);
    });
    return groups;
  }, [before, after]);

  if (diff.length === 0) {
    return <p className="text-sm text-muted-foreground italic">No based_on data.</p>;
  }

  const renderFact = (fact: BasedOnFact, factType: string, mode: "added" | "removed" | "kept") => {
    const canView =
      (factType === "directives" && !!onViewDirective) ||
      (factType !== "directives" && factType !== "mental-models" && !!onViewMemory);
    const rowCls =
      mode === "added"
        ? "bg-green-500/10 border-l-2 border-green-500"
        : mode === "removed"
          ? "bg-red-500/10 border-l-2 border-red-500 text-muted-foreground line-through decoration-red-500/50"
          : "";
    const marker = mode === "added" ? "+" : mode === "removed" ? "−" : " ";
    const markerCls =
      mode === "added"
        ? "text-green-600 dark:text-green-400"
        : mode === "removed"
          ? "text-red-600 dark:text-red-400"
          : "text-muted-foreground/40";
    return (
      <li key={`${mode}-${fact.id}`} className={`group flex items-start gap-2 px-3 py-2 ${rowCls}`}>
        <span className={`font-mono text-sm shrink-0 ${markerCls}`}>{marker}</span>
        <p className="text-sm leading-relaxed flex-1 min-w-0">
          {fact.text}
          {fact.context && (
            <span className="block text-xs text-muted-foreground italic mt-0.5 no-underline">
              {fact.context}
            </span>
          )}
        </p>
        {canView && mode !== "removed" && (
          <button
            className="text-xs text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0 mt-0.5"
            onClick={() => {
              if (factType === "directives") onViewDirective(fact.id);
              else onViewMemory(fact.id);
            }}
          >
            View →
          </button>
        )}
      </li>
    );
  };

  return (
    <div className="space-y-4">
      {diff.map((group) => {
        const display = getFactTypeDisplay(group.factType);
        return (
          <div key={group.factType} className="rounded-lg border border-border/60 overflow-hidden">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/40 border-b border-border/60">
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${display.color}`}
              >
                {display.label}
              </span>
              {group.added.length > 0 && (
                <span className="text-xs text-green-600 dark:text-green-400">
                  +{group.added.length}
                </span>
              )}
              {group.removed.length > 0 && (
                <span className="text-xs text-red-600 dark:text-red-400">
                  −{group.removed.length}
                </span>
              )}
              {group.kept.length > 0 && (
                <span className="text-xs text-muted-foreground">{group.kept.length} kept</span>
              )}
            </div>
            <ul className="divide-y divide-border/40">
              {group.added.map((f) => renderFact(f, group.factType, "added"))}
              {group.removed.map((f) => renderFact(f, group.factType, "removed"))}
              {group.kept.map((f) => renderFact(f, group.factType, "kept"))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

function MentalModelHistoryView({
  history,
  currentContent,
  currentBasedOn,
  onViewMemory,
  onViewDirective,
}: {
  history: HistoryEntry[];
  currentContent: string;
  currentBasedOn: Record<string, BasedOnFact[]> | undefined;
  onViewMemory: (id: string) => void;
  onViewDirective: (id: string) => void;
}) {
  const [idx, setIdx] = useState(0);
  const entry = history[idx];
  const afterContent = idx === 0 ? currentContent : (history[idx - 1].previous_content ?? "");
  const beforeBasedOn = entry.previous_reflect_response?.based_on;
  const afterBasedOn =
    idx === 0 ? currentBasedOn : history[idx - 1].previous_reflect_response?.based_on;
  const snapshot = entry.previous_reflect_response ?? null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">v{history.length - idx}</span> →{" "}
          <span className="font-semibold text-foreground">
            v{history.length - idx + 1}
            {idx === 0 ? " (current)" : ""}
          </span>{" "}
          &middot; changed {new Date(entry.changed_at).toLocaleString()}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={idx === history.length - 1}
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

      <div>
        <SectionLabel>Content diff</SectionLabel>
        {entry.previous_content !== null ? (
          <SideBySideDiff before={entry.previous_content} after={afterContent} />
        ) : (
          <div className="border border-border rounded-lg p-3">
            <span className="text-sm text-muted-foreground italic">
              Previous content not available
            </span>
          </div>
        )}
      </div>

      <div>
        <SectionLabel>Based on diff</SectionLabel>
        {snapshot?.based_on ? (
          <BasedOnDiff
            before={beforeBasedOn}
            after={afterBasedOn}
            onViewMemory={onViewMemory}
            onViewDirective={onViewDirective}
          />
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Not captured for this version (recorded before reflect snapshots were tracked).
          </p>
        )}
      </div>
    </div>
  );
}

interface MentalModelDetailModalProps {
  mentalModelId: string | null;
  onClose: () => void;
  onEdit?: (m: MentalModel) => void;
  onDelete?: (m: MentalModel) => void;
  onRefreshed?: (m: MentalModel) => void;
  initialTab?: "content" | "configuration" | "history";
}

export function MentalModelDetailModal({
  mentalModelId,
  onClose,
  onEdit,
  onDelete,
  onRefreshed,
  initialTab = "content",
}: MentalModelDetailModalProps) {
  const { currentBank } = useBank();
  const [mentalModel, setMentalModel] = useState<MentalModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"content" | "configuration" | "history">(initialTab);
  const [refreshing, setRefreshing] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [viewMemoryId, setViewMemoryId] = useState<string | null>(null);
  const [viewDirectiveId, setViewDirectiveId] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    if (!mentalModelId || !currentBank) return;

    const load = async () => {
      setLoading(true);
      setError(null);
      setMentalModel(null);
      setHistory(null);
      setActiveTab(initialTab);

      try {
        const data = await client.getMentalModel(currentBank, mentalModelId);
        setMentalModel(data);
      } catch (err) {
        console.error("Error loading mental model:", err);
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [mentalModelId, currentBank, initialTab]);

  useEffect(() => {
    if (activeTab !== "history" || !mentalModel || !currentBank || history !== null) return;

    const loadHistory = async () => {
      setLoadingHistory(true);
      try {
        const data = await client.getMentalModelHistory(currentBank, mentalModel.id);
        setHistory(data);
      } catch (err) {
        console.error("Error loading mental model history:", err);
        setHistory([]);
      } finally {
        setLoadingHistory(false);
      }
    };

    loadHistory();
  }, [activeTab, mentalModel, currentBank, history]);

  const handleReload = async () => {
    if (!currentBank || !mentalModel) return;
    setReloading(true);
    try {
      const updated = await client.getMentalModel(currentBank, mentalModel.id);
      setMentalModel(updated);
      setHistory(null);
    } catch (err) {
      console.error("Error reloading mental model:", err);
    } finally {
      setReloading(false);
    }
  };

  const handleRefresh = async () => {
    if (!currentBank || !mentalModel) return;
    setRefreshing(true);
    const originalRefreshedAt = mentalModel.last_refreshed_at;

    try {
      await client.refreshMentalModel(currentBank, mentalModel.id);

      const pollInterval = 1000;
      const maxAttempts = 120;
      let attempts = 0;

      const poll = async (): Promise<void> => {
        attempts++;
        try {
          const updated = await client.getMentalModel(currentBank, mentalModel.id);
          if (updated.last_refreshed_at !== originalRefreshedAt) {
            setMentalModel(updated);
            setHistory(null);
            onRefreshed?.(updated);
            setRefreshing(false);
            return;
          }
          if (attempts >= maxAttempts) {
            setRefreshing(false);
            toast.error("Refresh timeout", {
              description:
                "Refresh is taking longer than expected. Check the operations list for status.",
            });
            return;
          }
          setTimeout(poll, pollInterval);
        } catch (err) {
          console.error("Error polling mental model:", err);
          setRefreshing(false);
        }
      };
      setTimeout(poll, pollInterval);
    } catch {
      setRefreshing(false);
    }
  };

  const isOpen = mentalModelId !== null;

  const basedOn = mentalModel?.reflect_response?.based_on as
    | Record<string, BasedOnFact[]>
    | undefined;
  const basedOnCount = basedOn
    ? Object.values(basedOn).reduce((acc, facts) => acc + (facts?.length ?? 0), 0)
    : 0;

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
          <DialogHeader className="pr-10">
            <DialogTitle className="flex items-center gap-2">
              <span className="truncate">{mentalModel?.name ?? "Mental Model"}</span>
              {mentalModel && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 shrink-0"
                  onClick={handleReload}
                  disabled={reloading}
                  title="Reload data"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${reloading ? "animate-spin" : ""}`} />
                </Button>
              )}
              {mentalModel?.trigger?.refresh_after_consolidation && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-600 dark:text-green-400 text-xs font-medium">
                  <Zap className="w-3 h-3" />
                  Auto refresh
                </span>
              )}
            </DialogTitle>
          </DialogHeader>

          {loading ? (
            <div className="flex items-center justify-center flex-1">
              <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center flex-1">
              <div className="text-center text-destructive">
                <div className="text-sm">Error: {error}</div>
              </div>
            </div>
          ) : mentalModel ? (
            <Tabs
              value={activeTab}
              onValueChange={(v) => setActiveTab(v as "content" | "configuration" | "history")}
              className="flex-1 flex flex-col overflow-hidden"
            >
              <div className="flex items-center justify-between gap-2">
                <TabsList className="grid grid-cols-3 w-full max-w-md">
                  <TabsTrigger value="content" className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    Content
                  </TabsTrigger>
                  <TabsTrigger value="configuration" className="flex items-center gap-1.5">
                    <Settings className="w-3.5 h-3.5" />
                    Configuration
                  </TabsTrigger>
                  <TabsTrigger value="history" className="flex items-center gap-1.5">
                    <HistoryIcon className="w-3.5 h-3.5" />
                    History
                  </TabsTrigger>
                </TabsList>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 shrink-0"
                      disabled={refreshing}
                      aria-label="Actions"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {onEdit && (
                      <DropdownMenuItem onClick={() => onEdit(mentalModel)}>
                        <Pencil className="h-4 w-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuItem onClick={handleRefresh} disabled={refreshing}>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh Manually
                    </DropdownMenuItem>
                    {onDelete && (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => onDelete(mentalModel)}
                          className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex-1 overflow-y-auto mt-4">
                <TabsContent value="content" className="mt-0 space-y-6">
                  <div className="rounded-lg border border-border bg-muted/30 overflow-hidden">
                    <div className="flex items-center justify-between gap-2 px-4 py-2 border-b border-border bg-muted/50 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        <span className="font-semibold uppercase tracking-wide">
                          Stored content
                        </span>
                        <span className="text-muted-foreground/70">
                          &middot; {mentalModel.content.length.toLocaleString()} chars
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {mentalModel.is_stale === true ? (
                          <span
                            className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-amber-500/15 text-amber-700 dark:text-amber-400"
                            title="New memories in this mental model's scope have been ingested since it was last refreshed"
                          >
                            Stale
                          </span>
                        ) : mentalModel.is_stale === false ? (
                          <span
                            className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-green-500/15 text-green-700 dark:text-green-400"
                            title="No new memories in this mental model's scope since it was last refreshed"
                          >
                            In sync
                          </span>
                        ) : null}
                        <span
                          title={formatDateTime(mentalModel.last_refreshed_at)}
                          className="flex items-center gap-1"
                        >
                          <RefreshCw className="w-3 h-3" />
                          Last refreshed {formatRelativeTime(mentalModel.last_refreshed_at)}
                        </span>
                      </div>
                    </div>
                    <CompactMarkdown className="p-4">{mentalModel.content}</CompactMarkdown>
                  </div>
                  <div>
                    <SectionLabel>
                      Based On{basedOnCount > 0 ? ` (${basedOnCount})` : ""}
                    </SectionLabel>
                    {mentalModel.reflect_response ? (
                      <BasedOnList
                        based_on={basedOn}
                        onViewMemory={(id) => setViewMemoryId(id)}
                        onViewDirective={(id) => setViewDirectiveId(id)}
                      />
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No source data available. Click &quot;Refresh&quot; to regenerate with
                        source tracking.
                      </p>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="configuration" className="mt-0">
                  <ConfigurationTab mentalModel={mentalModel} />
                </TabsContent>

                <TabsContent value="history" className="mt-0">
                  {loadingHistory ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                    </div>
                  ) : history && history.length > 0 ? (
                    <MentalModelHistoryView
                      history={history}
                      currentContent={mentalModel.content}
                      currentBasedOn={basedOn}
                      onViewMemory={(id) => setViewMemoryId(id)}
                      onViewDirective={(id) => setViewDirectiveId(id)}
                    />
                  ) : (
                    <p className="text-sm text-muted-foreground italic">No history recorded yet.</p>
                  )}
                </TabsContent>
              </div>
            </Tabs>
          ) : null}
        </DialogContent>
      </Dialog>

      {viewMemoryId && (
        <MemoryDetailModal memoryId={viewMemoryId} onClose={() => setViewMemoryId(null)} />
      )}
      {viewDirectiveId && (
        <DirectiveDetailModal
          directiveId={viewDirectiveId}
          onClose={() => setViewDirectiveId(null)}
        />
      )}
    </>
  );
}

function Metadata({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  );
}

function InfoCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/20 overflow-hidden">
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-border bg-muted/40 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
      </div>
      <div className="p-4 space-y-4">{children}</div>
    </div>
  );
}

function Pill({ label, color }: { label: string; color?: string }) {
  return (
    <span
      className={`px-2 py-0.5 rounded text-xs ${
        color ?? "bg-muted text-foreground border border-border/60"
      }`}
    >
      {label}
    </span>
  );
}

function ConfigurationTab({ mentalModel }: { mentalModel: MentalModel }) {
  const t = mentalModel.trigger ?? { refresh_after_consolidation: false };
  const factTypes = t.fact_types ?? [];
  const tagGroups = t.tag_groups ?? [];
  const excludeIds = t.exclude_mental_model_ids ?? [];
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <InfoCard title="Identity" icon={<FileText className="w-3.5 h-3.5" />}>
        <Metadata
          label="ID"
          value={
            <code className="text-xs font-mono text-muted-foreground break-all">
              {mentalModel.id}
            </code>
          }
        />
        <Metadata label="Name" value={mentalModel.name} />
        {mentalModel.source_query && (
          <Metadata label="Source Query" value={mentalModel.source_query} />
        )}
        <Metadata
          label="Tags"
          value={
            mentalModel.tags?.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {mentalModel.tags.map((tag) => (
                  <Pill
                    key={tag}
                    label={tag}
                    color="bg-amber-500/10 text-amber-600 dark:text-amber-400"
                  />
                ))}
              </div>
            ) : (
              <span className="text-muted-foreground italic text-sm">none</span>
            )
          }
        />
      </InfoCard>

      <InfoCard title="Timing" icon={<RefreshCw className="w-3.5 h-3.5" />}>
        <Metadata label="Created" value={formatDateTime(mentalModel.created_at)} />
        <Metadata
          label="Last Refreshed"
          value={
            <span title={formatDateTime(mentalModel.last_refreshed_at)}>
              {formatRelativeTime(mentalModel.last_refreshed_at)}
            </span>
          }
        />
        <Metadata label="Max Tokens" value={mentalModel.max_tokens.toLocaleString()} />
      </InfoCard>

      <InfoCard title="Refresh Trigger" icon={<Zap className="w-3.5 h-3.5" />}>
        <Metadata
          label="Refresh mode"
          value={
            t.mode === "delta" ? (
              <Pill label="Delta" color="bg-purple-500/10 text-purple-600 dark:text-purple-400" />
            ) : (
              <Pill label="Full" />
            )
          }
        />
        <Metadata
          label="Auto-refresh after consolidation"
          value={
            t.refresh_after_consolidation ? (
              <Pill label="Enabled" color="bg-green-500/10 text-green-600 dark:text-green-400" />
            ) : (
              <Pill label="Disabled" />
            )
          }
        />
        <Metadata
          label="Fact types"
          value={
            factTypes.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {factTypes.map((ft) => {
                  const d = getFactTypeDisplay(ft);
                  return <Pill key={ft} label={d.label} color={d.color} />;
                })}
              </div>
            ) : (
              <span className="text-muted-foreground italic text-sm">all</span>
            )
          }
        />
        <Metadata label="Exclude mental models" value={t.exclude_mental_models ? "Yes" : "No"} />
        {excludeIds.length > 0 && (
          <Metadata
            label="Excluded IDs"
            value={
              <div className="flex flex-wrap gap-1.5">
                {excludeIds.map((id) => (
                  <code
                    key={id}
                    className="text-xs font-mono px-1.5 py-0.5 rounded bg-muted border border-border/60"
                  >
                    {id}
                  </code>
                ))}
              </div>
            }
          />
        )}
      </InfoCard>

      <InfoCard title="Recall Parameters" icon={<Settings className="w-3.5 h-3.5" />}>
        <Metadata
          label="Include chunks"
          value={t.include_chunks == null ? "default" : t.include_chunks ? "Yes" : "No"}
        />
        <Metadata
          label="Recall max tokens"
          value={t.recall_max_tokens != null ? t.recall_max_tokens.toLocaleString() : "default"}
        />
        <Metadata
          label="Recall chunks max tokens"
          value={
            t.recall_chunks_max_tokens != null
              ? t.recall_chunks_max_tokens.toLocaleString()
              : "default"
          }
        />
        <Metadata
          label="Tags match"
          value={t.tags_match ? <Pill label={t.tags_match} /> : "default"}
        />
      </InfoCard>

      {tagGroups.length > 0 && (
        <div className="md:col-span-2">
          <InfoCard title="Tag Groups" icon={<Settings className="w-3.5 h-3.5" />}>
            <pre className="text-xs font-mono bg-muted/40 rounded p-3 overflow-x-auto border border-border/60">
              {JSON.stringify(tagGroups, null, 2)}
            </pre>
          </InfoCard>
        </div>
      )}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
      {children}
    </div>
  );
}
