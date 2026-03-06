"use client";

import { useState, useEffect } from "react";
import { client, MentalModel } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { Loader2, Zap, FileText, History, ChevronLeft, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MentalModelDetailContentProps {
  mentalModel: MentalModel;
}

const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr);
  return `${date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })} at ${date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })}`;
};

/**
 * Shared content component for displaying mental model details.
 * Matches the layout of MentalModelDetailPanel for consistency.
 */
export function MentalModelDetailContent({ mentalModel }: MentalModelDetailContentProps) {
  return (
    <div className="space-y-6">
      {/* Header: Name, ID, Source Query */}
      <div className="pb-5 border-b border-border">
        <div className="flex items-center gap-2">
          <h3 className="text-xl font-bold text-foreground">{mentalModel.name}</h3>
          {mentalModel.trigger?.refresh_after_consolidation && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs font-medium">
              <Zap className="w-3 h-3" />
              Auto refresh
            </span>
          )}
        </div>
        <code className="text-xs font-mono text-muted-foreground/70">{mentalModel.id}</code>
        {mentalModel.source_query && (
          <p className="text-sm text-muted-foreground mt-1">{mentalModel.source_query}</p>
        )}
      </div>

      {/* Created / Last Refreshed */}
      <div className="flex gap-8">
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Created
          </div>
          <div className="text-sm text-foreground">{formatDateTime(mentalModel.created_at)}</div>
        </div>
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Last Refreshed
          </div>
          <div className="text-sm text-foreground">
            {formatDateTime(mentalModel.last_refreshed_at)}
          </div>
        </div>
      </div>

      {/* Content */}
      <div>
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
          Content
        </div>
        <div className="prose prose-base dark:prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{mentalModel.content}</ReactMarkdown>
        </div>
      </div>

      {/* Tags */}
      {mentalModel.tags && mentalModel.tags.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Tags
          </div>
          <div className="flex flex-wrap gap-1.5">
            {mentalModel.tags.map((tag: string, idx: number) => (
              <span
                key={idx}
                className="px-2 py-0.5 bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded text-xs"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

type HistoryEntry = { previous_content: string | null; changed_at: string };

type LineDiff = { type: "same" | "removed" | "added"; text: string };

function diffLines(a: string, b: string): { left: LineDiff[]; right: LineDiff[] } {
  const aLines = a.split("\n");
  const bLines = b.split("\n");
  const m = aLines.length;
  const n = bLines.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] =
        aLines[i - 1] === bLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);

  const ops: LineDiff[] = [];
  let i = m,
    j = n;
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

  // Pair removed/added lines side-by-side; same lines appear on both sides
  const left: LineDiff[] = [];
  const right: LineDiff[] = [];
  let k = 0;
  while (k < ops.length) {
    const op = ops[k];
    if (op.type === "same") {
      left.push(op);
      right.push(op);
      k++;
    } else {
      // collect a block of removed/added and align them
      const removed: string[] = [];
      const added: string[] = [];
      while (k < ops.length && ops[k].type !== "same") {
        if (ops[k].type === "removed") removed.push(ops[k].text);
        else added.push(ops[k].text);
        k++;
      }
      const maxLen = Math.max(removed.length, added.length);
      for (let r = 0; r < maxLen; r++) {
        left.push(
          r < removed.length ? { type: "removed", text: removed[r] } : { type: "same", text: "" }
        );
        right.push(
          r < added.length ? { type: "added", text: added[r] } : { type: "same", text: "" }
        );
      }
    }
  }
  return { left, right };
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
              line.type === "removed"
                ? "bg-red-500/10 text-red-700 dark:text-red-400"
                : "text-foreground"
            }`}
          >
            {line.text}
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
              line.type === "added"
                ? "bg-green-500/10 text-green-700 dark:text-green-400"
                : "text-foreground"
            }`}
          >
            {line.text}
          </div>
        ))}
      </div>
    </div>
  );
}

function MentalModelHistoryView({
  history,
  currentContent,
}: {
  history: HistoryEntry[];
  currentContent: string;
}) {
  const [idx, setIdx] = useState(0);
  const entry = history[idx];
  const afterContent = idx === 0 ? currentContent : (history[idx - 1].previous_content ?? "");

  return (
    <div className="space-y-3">
      {/* Navigation header */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          Change <span className="font-semibold text-foreground">{history.length - idx}</span> of{" "}
          {history.length} &middot; {new Date(entry.changed_at).toLocaleString()}
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

      {/* Change card */}
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
  );
}

interface MentalModelDetailModalProps {
  mentalModelId: string | null;
  onClose: () => void;
  initialTab?: string;
}

/**
 * Modal wrapper for MentalModelDetailContent.
 * Fetches the mental model by ID and displays it in a dialog.
 */
export function MentalModelDetailModal({
  mentalModelId,
  onClose,
  initialTab,
}: MentalModelDetailModalProps) {
  const { currentBank } = useBank();
  const [mentalModel, setMentalModel] = useState<MentalModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(initialTab ?? "model");

  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    if (!mentalModelId || !currentBank) return;

    const loadMentalModel = async () => {
      setLoading(true);
      setError(null);
      setMentalModel(null);
      setHistory(null);
      setActiveTab(initialTab ?? "model");

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

    loadMentalModel();
  }, [mentalModelId, currentBank]);

  // Load history lazily when history tab is selected
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

  const isOpen = mentalModelId !== null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col p-6">
        <VisuallyHidden>
          <DialogTitle>Mental Model Details</DialogTitle>
        </VisuallyHidden>
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center text-destructive">
              <div className="text-sm">Error: {error}</div>
            </div>
          </div>
        ) : mentalModel ? (
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col overflow-hidden"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="model" className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" />
                Mental Model
              </TabsTrigger>
              <TabsTrigger value="history" className="flex items-center gap-1.5">
                <History className="w-3.5 h-3.5" />
                History
                {history && history.length > 0 ? ` (${history.length})` : ""}
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-y-auto mt-4">
              <TabsContent value="model" className="mt-0">
                <MentalModelDetailContent mentalModel={mentalModel} />
              </TabsContent>

              <TabsContent value="history" className="mt-0">
                {loadingHistory ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                  </div>
                ) : history && history.length > 0 ? (
                  <MentalModelHistoryView history={history} currentContent={mentalModel.content} />
                ) : (
                  <p className="text-sm text-muted-foreground italic">No history recorded yet.</p>
                )}
              </TabsContent>
            </div>
          </Tabs>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
