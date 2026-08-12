"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useTranslations, useLocale } from "next-intl";
import { toast } from "sonner";
import { client, LLMRequestEntry } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { DataView } from "./data-view";
import { TraceDialog } from "./llm-requests-view";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  X,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pencil,
  Check,
  RefreshCw,
  MoreVertical,
  FileText,
  Settings,
  Layers,
  ChevronDown,
  Network,
  Eye,
  Activity,
  Download,
  Upload,
  Lock,
  RotateCcw,
  Search,
} from "lucide-react";
import { TagFilterInput } from "./tag-filter-input";
import { FacetLegend, MetadataChip, TagChip } from "@/components/ui/facet-chip";
import { Spinner } from "@/components/ui/spinner";
import { HarnessLogo } from "@/components/ui/harness-logo";
import { documentHarness, resolveHarnessLogo } from "@/lib/harness-logo";

const ITEMS_PER_PAGE = 50;

// Show in-flight/failed uploads that don't have a real document row yet. The
// source of truth is the server's file_convert_retain operations, not any
// client-side store — so the status survives reloads, tabs and devices.
const PENDING_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const PENDING_POLL_INTERVAL_MS = 4000;
// Idle cadence for auto-refreshing the whole table (new/updated docs, counts,
// "Updating" badges) so it stays live without a manual reload. Gentler than the
// active poll above — a quiet table shouldn't hammer the API.
const DOCUMENTS_AUTO_REFRESH_MS = 8000;
// A file_convert_retain operation flips to "completed" a couple of seconds
// before the document becomes visible in listDocuments. Keep showing the
// pending row for recently-completed operations so it stays on screen until
// the real document row takes over (dedup by document_id) — no flicker.
const PENDING_BRIDGE_MS = 30000;
const DOCUMENTS_REFRESH_EVENT = "hindsight:documents-refresh";
// Debounce between a filter edit (search text, tags) and the list request.
const FILTER_DEBOUNCE_MS = 250;

type PendingUpload = {
  operationId: string;
  id: string;
  filename: string | null;
  status: "processing" | "failed";
  error?: string | null;
  createdAt: string;
};

function formatRelativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

// Live "Refreshed N seconds ago" label next to the documents count. It owns a
// 1s ticker so only this label re-renders each second (not the whole table), and
// uses Intl.RelativeTimeFormat with the active locale — no per-unit translation
// keys needed. The list's own formatRelativeTime floors anything under a minute
// to "just now", which is useless for an ~8s auto-refresh, so we format seconds
// here directly.
function LastRefreshedLabel({ at }: { at: number }) {
  const t = useTranslations("documentsView");
  const locale = useLocale();
  const [, tick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, []);

  const sec = Math.max(0, Math.round((Date.now() - at) / 1000));
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const rel =
    sec < 60
      ? rtf.format(-sec, "second")
      : sec < 3600
        ? rtf.format(-Math.round(sec / 60), "minute")
        : rtf.format(-Math.round(sec / 3600), "hour");

  return (
    <span className="text-xs text-muted-foreground/70">· {t("lastRefreshed", { time: rel })}</span>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Detail-dialog rendering of document metadata. Unlike the list row this shows
// every entry with its full value and no truncation — the dialog is where you
// come to read the whole thing, so a "+N" here would just be a dead end.
function MetadataBadges({ metadata }: { metadata: Record<string, any> }) {
  const entries = Object.entries(metadata);
  if (entries.length === 0) return <span>-</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([k, v]) => (
        <MetadataChip key={k} entryKey={k} value={String(v)} />
      ))}
    </div>
  );
}

// Tags and metadata share one list column: both are short chips describing the
// same document, and two half-empty columns wasted the width that the size and
// memory-unit numbers need.
//
// The column header carries a legend — one specimen chip per kind, labelled —
// so the two treatments are defined once, up front, instead of being asserted
// per row. That is what lets the rows stay bare chips: nothing is spent
// restating the category on every one of them.
//
// Colours and shapes come from ui/facet-chip, the single place tags, entities
// and metadata are styled app-wide. Do not hand-roll chip classes here.
//
// The cap is on the TOTAL chip count, not per kind, because the column is
// narrow enough that each extra chip costs a whole wrapped line. Tags fill the
// slots first (they're the actionable half) and metadata takes what's left;
// the rest collapses into a "+N" tooltip. The full, untruncated set lives in
// the document dialog — this column is for scanning, not reading.
const ROW_CHIP_LIMIT = 3;

// Bounds a single chip so one long metadata value can't claim the whole row.
const ROW_CHIP_WIDTH = "max-w-[180px]";

function TagsAndMetadataCell({
  tags,
  metadata,
  selectedTags,
  onToggleTag,
  harnessShownAsLogo = false,
}: {
  tags: string[];
  metadata: Record<string, any> | null | undefined;
  selectedTags: string[];
  onToggleTag: (tag: string) => void;
  /**
   * The row already shows the harness as a logo, so `harness=…` would be the
   * same fact twice — and chip slots here are scarce. The `harness:<id>` TAG
   * stays: unlike the metadata chip it filters the list when clicked.
   */
  harnessShownAsLogo?: boolean;
}) {
  const t = useTranslations("documentsView");
  const metadataEntries = Object.entries(metadata ?? {}).filter(
    ([k]) => !(harnessShownAsLogo && k === "harness")
  );
  const shownTags = tags.slice(0, ROW_CHIP_LIMIT);
  const shownMetadata = metadataEntries.slice(0, ROW_CHIP_LIMIT - shownTags.length);
  const overflow = [
    ...tags.slice(shownTags.length).map((tag) => `#${tag}`),
    ...metadataEntries.slice(shownMetadata.length).map(([k, v]) => `${k}=${String(v)}`),
  ];

  if (shownTags.length === 0 && shownMetadata.length === 0) {
    return <span className="text-muted-foreground">-</span>;
  }

  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
      {shownTags.map((tag) => (
        <TagChip
          key={`tag-${tag}`}
          tag={tag}
          truncate
          className={ROW_CHIP_WIDTH}
          active={selectedTags.includes(tag)}
          title={`${t("labelTags")}: ${tag} — ${t("filterByThisTag")}`}
          onClick={() => onToggleTag(tag)}
        />
      ))}
      {shownMetadata.map(([k, v]) => (
        <MetadataChip
          key={`meta-${k}`}
          entryKey={k}
          value={String(v)}
          truncate
          className={ROW_CHIP_WIDTH}
          title={`${t("labelMetadata")}: ${k}=${String(v)}`}
        />
      ))}
      {overflow.length > 0 && (
        <span
          className="text-xs px-1.5 py-0.5 text-muted-foreground"
          title={`${t("seeAllInDocument")}\n\n${overflow.join("\n")}`}
        >
          +{overflow.length}
        </span>
      )}
    </div>
  );
}

/* ── Shared helper components (match mental-model-detail-modal pattern) ── */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground mb-1">
      {children}
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

interface RetainRun {
  traceId: string;
  entry: LLMRequestEntry;
  calls: number;
  tokens: number;
  status: string;
  start: string | null;
}

// Lists the retain traces that processed this document (one per retain/
// re-retain run) and opens the trace dialog. Renders nothing when tracing was
// off at retain time (no rows) — so it's invisible unless there's data.
function DocumentRetainTraces({ bankId, documentId }: { bankId: string; documentId: string }) {
  const t = useTranslations("documentsView");
  const [runs, setRuns] = useState<RetainRun[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [dialogEntry, setDialogEntry] = useState<LLMRequestEntry | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await client.listLLMRequests(bankId, {
          document_id: documentId,
          group: true,
          limit: 50,
        });
        if (cancelled) return;
        const byTrace = new Map<string, LLMRequestEntry[]>();
        for (const it of data.items || []) {
          const key = it.trace_id || it.id;
          if (!byTrace.has(key)) byTrace.set(key, []);
          byTrace.get(key)!.push(it);
        }
        const list: RetainRun[] = [...byTrace.entries()].map(([traceId, rows]) => ({
          traceId,
          entry: rows[0],
          calls: rows.length,
          tokens: rows.reduce((s, r) => s + (r.total_tokens ?? 0), 0),
          status: rows.some((r) => r.status === "error") ? "error" : "success",
          start:
            rows
              .map((r) => r.started_at)
              .filter(Boolean)
              .sort()[0] ?? null,
        }));
        list.sort((a, b) => (b.start || "").localeCompare(a.start || ""));
        setRuns(list);
      } catch {
        // Tracing may be disabled or the endpoint unavailable — stay hidden.
      } finally {
        if (!cancelled) setLoaded(true);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [bankId, documentId]);

  if (!loaded || runs.length === 0) return null;

  return (
    <InfoCard title={t("retainTracesTitle")} icon={<Activity className="w-3.5 h-3.5" />}>
      <div className="space-y-1">
        {runs.map((run) => (
          <button
            key={run.traceId}
            type="button"
            onClick={() => setDialogEntry(run.entry)}
            className="w-full flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-sm text-left hover:bg-muted/50"
          >
            <span className="inline-flex items-center gap-2 min-w-0">
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 ${run.status === "error" ? "bg-red-500" : "bg-green-500"}`}
              />
              <span className="font-mono text-xs">{run.entry.operation || "retain"}</span>
              <span className="text-muted-foreground text-xs truncate">
                {run.start ? new Date(run.start).toLocaleString() : ""}
              </span>
            </span>
            <span className="text-muted-foreground text-xs font-mono shrink-0">
              {t("retainTracesSummary", { calls: run.calls, tokens: run.tokens.toLocaleString() })}
            </span>
          </button>
        ))}
      </div>
      <TraceDialog
        bankId={bankId}
        entry={dialogEntry}
        open={!!dialogEntry}
        onOpenChange={(o) => !o && setDialogEntry(null)}
      />
    </InfoCard>
  );
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  );
}

// Renders the observation_scopes spec a document was retained with: a mode
// keyword ("per_tag" / "combined" / "all_combinations") shown as a mono badge,
// or explicit tag-set lists shown as scope chips. Surfacing this lets you see
// which scoping was requested (e.g. all_combinations on 2 tags → 3 scopes),
// which otherwise only becomes visible once async consolidation finishes.
function ObservationScopesValue({ spec }: { spec: string | string[][] }) {
  if (typeof spec === "string") {
    return <span className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded w-fit">{spec}</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {spec.map((scope, j) => (
        <span
          key={j}
          className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 font-medium"
        >
          {scope.length === 0 ? "—" : scope.map((tag) => `#${tag}`).join(" ")}
        </span>
      ))}
    </div>
  );
}

const COMPOSITION_COLORS = {
  world: "#8b5cf6",
  experience: "#ec4899",
  observation: "#6366f1",
};

function MemoryComposition({
  nodesByFactType,
}: {
  nodesByFactType: { world: number; experience: number; observation: number } | undefined;
}) {
  const t = useTranslations("dataView");
  const counts = nodesByFactType ?? { world: 0, experience: 0, observation: 0 };
  const total = counts.world + counts.experience + counts.observation;
  const items = [
    { name: "World", value: counts.world, color: COMPOSITION_COLORS.world },
    { name: "Experience", value: counts.experience, color: COMPOSITION_COLORS.experience },
    { name: "Observations", value: counts.observation, color: COMPOSITION_COLORS.observation },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
          {t("memoryComposition")}
        </h4>
        <span className="text-xs text-muted-foreground tabular-nums">{total.toLocaleString()}</span>
      </div>
      {total === 0 ? (
        <div className="text-xs text-muted-foreground py-2">{t("noMemoriesYet")}</div>
      ) : (
        <>
          <div className="h-1.5 flex w-full rounded-full overflow-hidden bg-muted">
            {items
              .filter((d) => d.value > 0)
              .map((d) => (
                <div
                  key={d.name}
                  className="h-full"
                  style={{ width: `${(d.value / total) * 100}%`, backgroundColor: d.color }}
                  title={`${d.name}: ${d.value.toLocaleString()}`}
                />
              ))}
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm">
            {items.map((d) => (
              <div key={d.name} className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-[2px]" style={{ backgroundColor: d.color }} />
                  <span className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium">
                    {d.name}
                  </span>
                </div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-base font-semibold tabular-nums">
                    {d.value.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {total > 0 ? `${((d.value / total) * 100).toFixed(0)}%` : "0%"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

type ChunkFactType = "world" | "experience" | "observation";

function ChunkMemoriesHeader({
  chunkFactType,
  setChunkFactType,
}: {
  chunkFactType: ChunkFactType;
  setChunkFactType: (ft: ChunkFactType) => void;
}) {
  return (
    <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border bg-muted/30">
      {(["world", "experience", "observation"] as const).map((ft) => (
        <button
          key={ft}
          onClick={() => setChunkFactType(ft)}
          className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
            chunkFactType === ft
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {ft === "observation" ? "Obs" : ft.charAt(0).toUpperCase() + ft.slice(1)}
        </button>
      ))}
    </div>
  );
}

// Document-level audit: facts extracted from this document that were later
// invalidated (moved to the curation archive, so they no longer appear in the
// chunk memory views). Each can be restored in place.
function InvalidatedFactsSection({ bankId, documentId }: { bankId: string; documentId: string }) {
  const t = useTranslations("documentsView");
  const tCuration = useTranslations("memoryDetailPanel");
  const [rows, setRows] = useState<any[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!bankId || !documentId) return;
    try {
      const resp: any = await client.listMemories(bankId, {
        state: "invalidated",
        documentId,
        limit: 200,
      });
      setRows(resp?.items ?? []);
    } catch {
      setRows([]);
    } finally {
      setLoaded(true);
    }
  }, [bankId, documentId]);

  useEffect(() => {
    load();
  }, [load]);

  const restore = async (id: string) => {
    setRestoringId(id);
    try {
      await client.updateMemory(id, bankId, { state: "valid" });
      await load();
    } finally {
      setRestoringId(null);
    }
  };

  if (!loaded || rows.length === 0) return null;

  return (
    <InfoCard title={`${t("invalidatedFactsTitle")} (${rows.length})`}>
      <div className="space-y-2">
        {rows.map((row) => (
          <div key={row.id} className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                {row.fact_type && (
                  <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-medium capitalize">
                    {row.fact_type}
                  </span>
                )}
                <div className="text-sm text-foreground line-clamp-2">{row.text}</div>
              </div>
              {row.entities && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {row.entities
                    .split(", ")
                    .filter(Boolean)
                    .slice(0, 6)
                    .map((e: string, i: number) => (
                      <span
                        key={i}
                        className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground"
                      >
                        {e}
                      </span>
                    ))}
                </div>
              )}
              {row.occurred_start && (
                <div className="text-xs text-muted-foreground mt-0.5">
                  {new Date(row.occurred_start).toLocaleDateString()}
                  {row.occurred_end && row.occurred_end !== row.occurred_start && (
                    <> → {new Date(row.occurred_end).toLocaleDateString()}</>
                  )}
                </div>
              )}
              {(row.invalidation_reason || row.invalidated_at) && (
                <div className="text-xs text-muted-foreground mt-0.5">
                  {row.invalidation_reason && (
                    <>
                      {tCuration("curationReasonLabel")}: {row.invalidation_reason}
                    </>
                  )}
                  {row.invalidation_reason && row.invalidated_at && " · "}
                  {row.invalidated_at && new Date(row.invalidated_at).toLocaleString()}
                </div>
              )}
            </div>
            <Button
              variant="secondary"
              size="sm"
              disabled={restoringId === row.id}
              onClick={() => restore(row.id)}
              className="shrink-0 h-7 px-2 text-xs gap-1"
            >
              <RotateCcw className="w-3 h-3" />
              {tCuration("curationRevert")}
            </Button>
          </div>
        ))}
      </div>
    </InfoCard>
  );
}

function ChunkRow({ chunk }: { chunk: any }) {
  const [expanded, setExpanded] = useState(false);
  const [memoriesExpanded, setMemoriesExpanded] = useState(false);
  const [chunkFactType, setChunkFactType] = useState<ChunkFactType>("world");
  const previewLength = 150;
  const text = chunk.chunk_text ?? "";
  const preview = text.length > previewLength ? text.slice(0, previewLength) + "..." : text;

  return (
    <div>
      <button
        className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-muted/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform ${expanded ? "" : "-rotate-90"}`}
        />
        <span className="text-xs font-mono text-muted-foreground shrink-0">
          #{chunk.chunk_index}
        </span>
        <span className="text-[11px] text-muted-foreground/60 shrink-0">
          {text.length.toLocaleString()} chars
        </span>
        {!expanded && <span className="text-xs text-foreground/50 truncate">{preview}</span>}
      </button>
      {expanded &&
        (memoriesExpanded ? (
          /* Full-width memories view with controls (text hidden) */
          <div style={{ height: "500px" }} className="flex flex-col border-t border-border">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/30 border-b border-border">
              <ChunkMemoriesHeader
                chunkFactType={chunkFactType}
                setChunkFactType={setChunkFactType}
              />
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setMemoriesExpanded(false)}
                className="h-6 px-2 text-xs gap-1"
              >
                <Eye className="w-3 h-3" />
                Compact
              </Button>
            </div>
            <div className="flex-1 min-h-0">
              <DataView
                key={`${chunk.chunk_id}-${chunkFactType}-full`}
                factType={chunkFactType}
                chunkId={chunk.chunk_id}
              />
            </div>
          </div>
        ) : (
          /* Split view: left text, right compact memories */
          <div className="grid grid-cols-2 divide-x divide-border" style={{ height: "350px" }}>
            <div className="overflow-y-auto">
              <pre className="px-4 py-3 text-[11px] leading-5 text-foreground/80 whitespace-pre-wrap font-mono">
                {text}
              </pre>
            </div>
            <div className="flex flex-col overflow-hidden">
              <ChunkMemoriesHeader
                chunkFactType={chunkFactType}
                setChunkFactType={setChunkFactType}
              />
              <div className="flex-1 min-h-0">
                <DataView
                  key={`${chunk.chunk_id}-${chunkFactType}-compact`}
                  factType={chunkFactType}
                  chunkId={chunk.chunk_id}
                  compact
                  onExpandToggle={() => setMemoriesExpanded(true)}
                />
              </div>
            </div>
          </div>
        ))}
    </div>
  );
}

export function DocumentsView() {
  const t = useTranslations("documentsView");
  const tCommon = useTranslations("common");
  const tBank = useTranslations("bank");
  const tApiError = useTranslations("api.errors.documents");
  const tOperations = useTranslations("bankOperations");
  const { currentBank } = useBank();
  const { features } = useFeatures();
  const [documents, setDocuments] = useState<any[]>([]);
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  // Document IDs targeted by a pending/processing retain op → badged as "updating".
  const [inFlightDocIds, setInFlightDocIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  // Whether the first document fetch has completed. The mount fetch is debounced
  // (see the load effect), so without this the empty state ("No documents found")
  // flashes for the initial paint + debounce window before `loading` ever flips.
  // Gate the empty state on this so we show the loader until we actually know.
  const [loaded, setLoaded] = useState(false);
  // Wall-clock time of the last list refresh, shown next to the count so the
  // auto-refresh is visible ("Refreshed 14:41:32").
  const [lastRefreshedAt, setLastRefreshedAt] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  // The UI exposes the two useful modes; both map to their *_strict variant so
  // that filtering by a tag never surfaces untagged documents.
  const [tagsMatch, setTagsMatch] = useState<"any" | "all">("any");
  const [total, setTotal] = useState(0);

  // Document transfer (export/import) state
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportIncludeObservations, setExportIncludeObservations] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importOnConflict, setImportOnConflict] = useState<"skip" | "replace" | "new-id">("skip");

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);
  const offset = (currentPage - 1) * ITEMS_PER_PAGE;

  // Document view panel state
  const [selectedDocument, setSelectedDocument] = useState<any>(null);
  const [loadingDocument, setLoadingDocument] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [deletingUploadOperationId, setDeletingUploadOperationId] = useState<string | null>(null);

  // Tag editing state
  const [editingTags, setEditingTags] = useState(false);
  const [tagInput, setTagInput] = useState("");
  const [savingTags, setSavingTags] = useState(false);

  // Content editing state
  const [editingContent, setEditingContent] = useState(false);
  const [contentInput, setContentInput] = useState("");
  const [savingContent, setSavingContent] = useState(false);

  // Chunks state
  const [chunks, setChunks] = useState<any[]>([]);
  const [chunksTotal, setChunksTotal] = useState(0);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunksLoaded, setChunksLoaded] = useState(false);

  // Reprocess state
  const [reprocessing, setReprocessing] = useState(false);
  const [reprocessResult, setReprocessResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  // Delete confirmation dialog state
  const [documentToDelete, setDocumentToDelete] = useState<{
    id: string;
    memoryCount?: number;
  } | null>(null);
  const [deleteResult, setDeleteResult] = useState<{ success: boolean; message: string } | null>(
    null
  );

  const loadDocuments = useCallback(
    async (page: number = 1) => {
      if (!currentBank) return;

      setLoading(true);
      try {
        const pageOffset = (page - 1) * ITEMS_PER_PAGE;
        const data: any = await client.listDocuments({
          bank_id: currentBank,
          q: searchQuery,
          tags: selectedTags,
          tags_match: tagsMatch === "all" ? "all_strict" : "any_strict",
          limit: ITEMS_PER_PAGE,
          offset: pageOffset,
        });
        setDocuments(data.items || []);
        setTotal(data.total || 0);
      } catch (error) {
        // Error toast is shown automatically by the API client interceptor
      } finally {
        setLoading(false);
        setLoaded(true);
        setLastRefreshedAt(Date.now());
      }
    },
    [currentBank, searchQuery, selectedTags, tagsMatch]
  );

  // Pull in-flight/failed file uploads straight from the server's
  // file_convert_retain operations. Completed operations drop off here and
  // reappear as real document rows once listDocuments returns them.
  const loadPendingUploads = useCallback(async () => {
    if (!currentBank) {
      setPendingUploads([]);
      return;
    }
    try {
      const data = await client.listOperations(currentBank, {
        type: "file_convert_retain",
        limit: 100,
      });
      const now = Date.now();
      const cutoff = now - PENDING_MAX_AGE_MS;
      const uploads: PendingUpload[] = (data.operations || [])
        .filter((op) => {
          const created = Date.parse(op.created_at);
          if (!Number.isFinite(created) || created < cutoff) return false;
          if (op.status === "pending" || op.status === "processing" || op.status === "failed") {
            return true;
          }
          // Bridge the window between completion and the document appearing in
          // the list. After PENDING_BRIDGE_MS we stop bridging so older
          // completed uploads (whose row simply sits on another page) don't
          // resurface as phantom "Processing" rows.
          if (op.status === "completed") {
            const settled = Date.parse(op.updated_at || op.created_at);
            return Number.isFinite(settled) && now - settled < PENDING_BRIDGE_MS;
          }
          return false;
        })
        .map((op) => ({
          operationId: op.id,
          id: op.document_id || op.id,
          filename: op.filename ?? null,
          status: op.status === "failed" ? "failed" : "processing",
          error: op.error_message,
          createdAt: op.created_at,
        }));
      setPendingUploads(uploads);
    } catch {
      // Keep whatever we had; the Operations tab remains the detailed source of truth.
    }
  }, [currentBank]);

  // Cross-check in-flight retain operations against the visible documents: a
  // pending/processing op that targets an existing document_id means that
  // document is being rewritten. Fetched broadly (not just file uploads) so text
  // re-retains and reprocesses are caught too — the API surfaces the target
  // document_id for single-document retains (batch/file), which is the case here.
  const loadUpdatingOps = useCallback(async () => {
    if (!currentBank) {
      setInFlightDocIds(new Set());
      return;
    }
    try {
      const data = await client.listOperations(currentBank, { limit: 100 });
      const ids = new Set<string>();
      for (const op of data.operations || []) {
        if ((op.status === "pending" || op.status === "processing") && op.document_id) {
          ids.add(op.document_id);
        }
      }
      setInFlightDocIds(ids);
    } catch {
      // Keep the previous set; the Operations tab is the detailed source of truth.
    }
  }, [currentBank]);

  // Only badge documents that are actually visible in the table.
  const updatingDocIds = useMemo(() => {
    if (inFlightDocIds.size === 0) return new Set<string>();
    const realIds = new Set(documents.map((doc) => doc.id));
    return new Set([...inFlightDocIds].filter((id) => realIds.has(id)));
  }, [inFlightDocIds, documents]);
  const hasUpdatingDocs = updatingDocIds.size > 0;

  // Pending rows: in-flight/failed uploads that aren't yet in the real list.
  // A tag filter hides them entirely — their tags only exist on the document
  // row the conversion hasn't produced yet, so we can't honestly match them.
  const pendingRows = useMemo<PendingUpload[]>(() => {
    if (selectedTags.length > 0) return [];
    const realIds = new Set(documents.map((doc) => doc.id));
    const q = searchQuery.trim().toLowerCase();
    return pendingUploads
      .filter((upload) => !realIds.has(upload.id))
      .filter((upload) => {
        if (!q) return true;
        return (
          upload.id.toLowerCase().includes(q) ||
          (upload.filename?.toLowerCase().includes(q) ?? false)
        );
      });
  }, [documents, pendingUploads, searchQuery, selectedTags]);

  const hasActiveFilters = searchQuery.trim().length > 0 || selectedTags.length > 0;

  const clearFilters = () => {
    setSearchQuery("");
    setSelectedTags([]);
  };

  // Clicking a tag chip in the table toggles it in the filter.
  const toggleTagFilter = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  // Keep polling while any *visible* pending row is still processing (i.e. its
  // real document hasn't shown up yet). Basing this on pendingRows rather than
  // pendingUploads stops the poll as soon as the document takes over.
  const hasActiveUploads = useMemo(
    () => pendingRows.some((upload) => upload.status === "processing"),
    [pendingRows]
  );
  const displayTotal = total + pendingRows.length;

  // The detail response nests the document's metadata under `retain_params`,
  // where the list response returns it flat as `document_metadata` — same map,
  // two shapes, so the harness is resolved once per surface rather than inline.
  const selectedHarness = documentHarness(
    selectedDocument?.retain_params?.metadata,
    selectedDocument?.tags
  );
  const selectedHarnessLogo = resolveHarnessLogo(selectedHarness);

  // Handle page change
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
    loadDocuments(newPage);
  };

  const viewDocumentText = async (documentId: string) => {
    if (!currentBank) return;

    setLoadingDocument(true);
    setSelectedDocument({ id: documentId }); // Set placeholder to show loading
    setEditingTags(false);
    setTagInput("");
    setEditingContent(false);
    setContentInput("");
    setChunks([]);
    setChunksTotal(0);
    setChunksLoaded(false);

    try {
      const doc: any = await client.getDocument(documentId, currentBank);
      setSelectedDocument(doc);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
      setSelectedDocument(null);
    } finally {
      setLoadingDocument(false);
    }
  };

  const loadChunks = async (documentId: string) => {
    if (!currentBank) return;

    setLoadingChunks(true);
    try {
      const data: any = await client.listDocumentChunks({
        document_id: documentId,
        bank_id: currentBank,
        limit: 100,
      });
      setChunks(data.items || []);
      setChunksTotal(data.total || 0);
      setChunksLoaded(true);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoadingChunks(false);
    }
  };

  const reprocessDocument = async () => {
    if (!currentBank || !selectedDocument) return;

    setReprocessing(true);
    try {
      const result = await client.reprocessDocument(selectedDocument.id, currentBank);
      setReprocessResult({
        success: true,
        message: `Reprocessing started (operation: ${result.operation_id})`,
      });
      // Surface the "Updating" badge immediately instead of waiting for the next
      // poll tick; the poll then keeps it live and clears it when the op finishes.
      loadUpdatingOps();
    } catch (error) {
      setReprocessResult({
        success: false,
        message: "Error reprocessing document: " + (error as Error).message,
      });
    } finally {
      setReprocessing(false);
    }
  };

  const confirmDeleteDocument = async () => {
    if (!currentBank || !documentToDelete) return;

    const documentId = documentToDelete.id;
    setDeletingDocumentId(documentId);
    setDocumentToDelete(null);

    try {
      const result = await client.deleteDocument(documentId, currentBank);
      setDeleteResult({
        success: true,
        message: t("toastDeletedDocumentAndUnits", { count: result.memory_units_deleted }),
      });

      // Close panel if this document was selected
      if (selectedDocument?.id === documentId) {
        setSelectedDocument(null);
      }

      // Reload documents list at current page
      loadDocuments(currentPage);
    } catch (error) {
      console.error("Error deleting document:", error);
      setDeleteResult({
        success: false,
        message: t("toastErrorDeletingDocument") + (error as Error).message,
      });
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const requestDeleteDocument = (documentId: string, memoryCount?: number) => {
    setDocumentToDelete({ id: documentId, memoryCount });
  };

  const deleteFailedUpload = async (operationId: string) => {
    if (!currentBank) return;

    setDeletingUploadOperationId(operationId);
    try {
      // Failed uploads have no document row to delete. Remove their terminal
      // operation record instead, using the same API as the Operations view.
      await client.deleteOperation(currentBank, operationId);
      await loadPendingUploads();
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setDeletingUploadOperationId(null);
    }
  };

  const startEditTags = () => {
    setTagInput((selectedDocument?.tags ?? []).join(", "));
    setEditingTags(true);
  };

  const cancelEditTags = () => {
    setEditingTags(false);
    setTagInput("");
  };

  const startEditContent = () => {
    setContentInput(selectedDocument?.original_text ?? "");
    setEditingContent(true);
  };

  const cancelEditContent = () => {
    setEditingContent(false);
    setContentInput("");
  };

  const saveDocumentContent = async () => {
    if (!currentBank || !selectedDocument) return;

    const newContent = contentInput;
    if (!newContent.trim()) return;

    const retainParams = selectedDocument.retain_params ?? {};
    const item: Parameters<typeof client.retain>[0]["items"][number] = {
      content: newContent,
      document_id: selectedDocument.id,
    };
    if (retainParams.context) item.context = retainParams.context;
    if (retainParams.event_date) item.timestamp = retainParams.event_date;
    if (retainParams.metadata && Object.keys(retainParams.metadata).length > 0) {
      item.metadata = retainParams.metadata;
    }
    if (selectedDocument.tags && selectedDocument.tags.length > 0) {
      item.tags = selectedDocument.tags;
    }

    setSavingContent(true);
    try {
      await client.retain({
        bank_id: currentBank,
        items: [item],
        async: false,
      });
      // Refresh the document and the list
      const doc: any = await client.getDocument(selectedDocument.id, currentBank);
      setSelectedDocument(doc);
      setEditingContent(false);
      setContentInput("");
      loadDocuments(currentPage);
    } catch (error) {
      console.error("Error updating document content:", error);
    } finally {
      setSavingContent(false);
    }
  };

  const saveDocumentTags = async () => {
    if (!currentBank || !selectedDocument) return;

    const newTags = tagInput
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    setSavingTags(true);
    try {
      await client.updateDocument(selectedDocument.id, currentBank, newTags);
      setSelectedDocument({ ...selectedDocument, tags: newTags });
      // Update tags in the documents list too
      setDocuments((prev) =>
        prev.map((d) => (d.id === selectedDocument.id ? { ...d, tags: newTags } : d))
      );
      setEditingTags(false);
      setTagInput("");
    } catch (error) {
      console.error("Error updating document tags:", error);
    } finally {
      setSavingTags(false);
    }
  };

  // Load page 1 on mount, on bank change and whenever a filter changes.
  // `loadDocuments` re-identifies exactly on those inputs, so this single
  // debounced effect covers all of them — typing in the search box no longer
  // fires one request per keystroke *plus* a second undebounced one.
  useEffect(() => {
    if (!currentBank) return;
    const timeoutId = setTimeout(() => {
      setCurrentPage(1);
      loadDocuments(1);
    }, FILTER_DEBOUNCE_MS);
    return () => clearTimeout(timeoutId);
  }, [currentBank, loadDocuments]);

  useEffect(() => {
    if (currentBank) {
      loadPendingUploads();
      loadUpdatingOps();
    } else {
      setPendingUploads([]);
      setInFlightDocIds(new Set());
    }
  }, [currentBank, loadPendingUploads, loadUpdatingOps]);

  // Auto-refresh the whole table on a timer so new/updated documents, counts,
  // pending uploads, and "Updating" badges all appear without a manual reload.
  // Refresh faster while something is actively in flight (uploads converting or a
  // document being rewritten), and on the gentler idle cadence otherwise.
  useEffect(() => {
    if (!currentBank) return;

    const period =
      hasActiveUploads || hasUpdatingDocs ? PENDING_POLL_INTERVAL_MS : DOCUMENTS_AUTO_REFRESH_MS;
    const interval = window.setInterval(() => {
      loadUpdatingOps();
      loadPendingUploads();
      loadDocuments(currentPage);
    }, period);

    return () => window.clearInterval(interval);
  }, [
    currentBank,
    hasActiveUploads,
    hasUpdatingDocs,
    currentPage,
    loadDocuments,
    loadPendingUploads,
    loadUpdatingOps,
  ]);

  // Refresh immediately after an upload is submitted elsewhere (bank selector).
  useEffect(() => {
    if (!currentBank) return;

    const onRefresh = () => {
      loadPendingUploads();
      loadUpdatingOps();
      loadDocuments(currentPage);
    };
    window.addEventListener(DOCUMENTS_REFRESH_EVENT, onRefresh);
    return () => window.removeEventListener(DOCUMENTS_REFRESH_EVENT, onRefresh);
  }, [currentBank, currentPage, loadDocuments, loadPendingUploads, loadUpdatingOps]);

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const exportDocuments = async (documentIds?: string[], includeObservations = false) => {
    if (!currentBank || exporting) return;
    setExporting(true);
    try {
      const blob = await client.exportDocuments(currentBank, documentIds, includeObservations);
      const suffix = documentIds && documentIds.length === 1 ? `-${documentIds[0]}` : "-documents";
      triggerDownload(blob, `${currentBank}${suffix}.zip`);
      toast.success(t("exportSuccess"));
      setExportDialogOpen(false);
    } catch (error) {
      // Binary transfer requests bypass the API client's shared error interceptor.
      toast.error(error instanceof Error ? error.message : tApiError("export"));
    } finally {
      setExporting(false);
    }
  };

  const runImport = async (file: File) => {
    if (!file || !currentBank) return;
    setImporting(true);
    try {
      // Import is an async operation: submit, then poll until it completes.
      const { operation_id } = await client.importDocuments(currentBank, file, importOnConflict);
      const deadline = Date.now() + 5 * 60 * 1000; // give large imports up to 5 min
      let meta: Record<string, any> | null = null;
      while (Date.now() < deadline) {
        const op = await client.getOperationStatus(currentBank, operation_id);
        if (op.status === "completed") {
          meta = op.result_metadata ?? {};
          break;
        }
        if (op.status === "failed") {
          toast.error(op.error_message || t("importFailed"));
          return;
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (meta === null) {
        toast.error(t("importTimeout"));
        return;
      }
      toast.success(
        t("importSuccess", {
          imported: meta.documents_imported ?? 0,
          facts: meta.facts_imported ?? 0,
          skipped: meta.documents_skipped ?? 0,
        })
      );
      loadDocuments(currentPage);
      setImportDialogOpen(false);
      setImportFile(null);
    } catch (error) {
      // Multipart transfer requests bypass the API client's shared error interceptor.
      toast.error(error instanceof Error ? error.message : tApiError("import"));
    } finally {
      setImporting(false);
    }
  };

  const canExport = features?.document_export_api ?? false;
  const canImport = features?.document_import_api ?? false;

  return (
    <div>
      {/* Page header: the bank-page title for the Documents tab, with the
          Export/Import Actions menu on the same row (right-aligned). */}
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-foreground">{tBank("documents")}</h1>
          <p className="text-muted-foreground">{tBank("documentsDescription")}</p>
        </div>
        {(canExport || canImport) && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="h-8 shrink-0"
                disabled={!currentBank || exporting || importing}
              >
                {exporting ? t("exporting") : importing ? t("importing") : t("actionsButton")}
                <ChevronDown className="w-4 h-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {canExport && (
                <DropdownMenuItem
                  onClick={() => {
                    setExportIncludeObservations(false);
                    setExportDialogOpen(true);
                  }}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {t("exportButton")}
                </DropdownMenuItem>
              )}
              {canImport && (
                <DropdownMenuItem onClick={() => setImportDialogOpen(true)}>
                  <Upload className="h-4 w-4 mr-2" />
                  {t("importButton")}
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Export dialog: explains the action and offers the observations choice. */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("exportDialogTitle")}</DialogTitle>
            <DialogDescription>{t("exportDialogDescription")}</DialogDescription>
          </DialogHeader>
          <div className="flex items-start gap-2 py-2">
            <Checkbox
              id="export-include-observations"
              checked={exportIncludeObservations}
              onCheckedChange={(v) => setExportIncludeObservations(v === true)}
            />
            <div className="grid gap-1 leading-none">
              <Label htmlFor="export-include-observations">
                {t("exportIncludeObservationsLabel")}
              </Label>
              <p className="text-xs text-muted-foreground">{t("exportIncludeObservationsHint")}</p>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setExportDialogOpen(false)}
              disabled={exporting}
            >
              {tCommon("cancel")}
            </Button>
            <Button
              size="sm"
              onClick={() => exportDocuments(undefined, exportIncludeObservations)}
              disabled={exporting}
            >
              <Download className="h-4 w-4 mr-2" />
              {exporting ? t("exporting") : t("exportButton")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import dialog: explains the action and only accepts .zip archives. */}
      <Dialog
        open={importDialogOpen}
        onOpenChange={(open) => {
          setImportDialogOpen(open);
          if (!open) setImportFile(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("importDialogTitle")}</DialogTitle>
            <DialogDescription>{t("importDialogDescription")}</DialogDescription>
          </DialogHeader>
          <div className="py-2 space-y-4">
            <div className="grid gap-1.5">
              <input
                type="file"
                accept=".zip,application/zip"
                disabled={importing}
                onChange={(e) => setImportFile(e.target.files?.[0] ?? null)}
                className="block w-full text-sm text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-muted/80"
              />
              {/* Spelled out because "Import from zip" reads like a bulk upload
                  of ordinary files, which this is not — that's Add Document. */}
              <p className="text-xs text-muted-foreground">{t("importFileHint")}</p>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="import-on-conflict">{t("importConflictLabel")}</Label>
              <Select
                value={importOnConflict}
                onValueChange={(v) => setImportOnConflict(v as "skip" | "replace" | "new-id")}
                disabled={importing}
              >
                <SelectTrigger id="import-on-conflict" className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="skip">{t("importConflictSkip")}</SelectItem>
                  <SelectItem value="replace">{t("importConflictReplace")}</SelectItem>
                  <SelectItem value="new-id">{t("importConflictNewId")}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setImportDialogOpen(false);
                setImportFile(null);
              }}
              disabled={importing}
            >
              {tCommon("cancel")}
            </Button>
            <Button
              size="sm"
              onClick={() => importFile && runImport(importFile)}
              disabled={!importFile || importing}
            >
              <Upload className="h-4 w-4 mr-2" />
              {importing ? t("importing") : t("importButton")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {/* Filter toolbar — rendered even when nothing matches, so a filter that
          empties the list can still be undone. */}
      {/* items-start: the tag filter is a two-row block (controls + applied
          chips), so the other controls align to its first row. */}
      <div className="mb-3 flex flex-wrap items-start gap-3">
        <div className="relative w-72">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t("searchPlaceholder")}
            className="pl-8 pr-8 h-9"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              aria-label={t("clearSearch")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <TagFilterInput
          value={selectedTags}
          onChange={setSelectedTags}
          bankId={currentBank}
          matchMode={tagsMatch}
          onMatchModeChange={setTagsMatch}
          className="flex-1 min-w-[260px]"
        />
        {hasActiveFilters && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearFilters}
            className="h-9 gap-1 text-xs shrink-0"
          >
            <X className="h-3.5 w-3.5" />
            {t("clearFilters")}
          </Button>
        )}
      </div>

      {/* Hide the count during the very first load so it doesn't read
          "0 total documents" above the loading spinner. */}
      {(loaded || documents.length > 0 || pendingRows.length > 0) && (
        <div className="mb-4 flex items-baseline gap-2 text-sm text-muted-foreground">
          <span>
            {hasActiveFilters
              ? t("matchingDocuments", { total: displayTotal })
              : t("totalDocuments", { total: displayTotal })}
          </span>
          {lastRefreshedAt !== null && <LastRefreshedLabel at={lastRefreshedAt} />}
        </div>
      )}

      {/* Documents List Section */}
      {/* Show the loader until the first fetch resolves (`!loaded`), not just while
          `loading`. Two reasons the empty state would otherwise flash first:
          (1) the mount fetch is debounced, and (2) on a hard refresh currentBank
          is null until bank-context resolves it from the URL in an effect (after
          the first paint / before hydration + theme). `!loaded` covers both — and
          it can't get stuck: on any /banks/[id] route currentBank always resolves,
          the fetch runs, and its `finally` flips `loaded` (even if the bank is
          invalid and the fetch errors). */}
      {(loading || !loaded) && documents.length === 0 && pendingRows.length === 0 ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <Spinner size="xl" variant="jump" className="mx-auto mb-2" />
            <div className="text-sm text-muted-foreground">{t("loadingDocuments")}</div>
          </div>
        </div>
      ) : documents.length > 0 || pendingRows.length > 0 ? (
        <>
          {/* Documents Table */}
          <div className="w-full">
            <div className="overflow-x-auto pb-5">
              {/* table-fixed so the column widths below are honoured. With the
                  default auto layout the long unbreakable mono document IDs
                  sized the first column themselves, starving the chip column
                  and stopping `truncate` from ever kicking in. */}
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[38%]">{t("colDocument")}</TableHead>
                    <TableHead>
                      <FacetLegend
                        items={[
                          { kind: "tag", label: t("labelTags") },
                          { kind: "metadata", label: t("labelMetadata") },
                        ]}
                      />
                    </TableHead>
                    <TableHead className="w-[110px] text-right whitespace-nowrap">
                      {t("colSize")}
                    </TableHead>
                    <TableHead className="w-[130px] text-right whitespace-nowrap">
                      {t("colMemoryUnits")}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pendingRows.map((upload) => (
                    <TableRow key={`pending-${upload.id}`} className="bg-muted/30">
                      <TableCell className="text-card-foreground">
                        <div className="min-w-0">
                          <div className="font-mono text-sm truncate" title={upload.id}>
                            {upload.id}
                          </div>
                          <div className="mt-0.5 text-xs text-muted-foreground truncate">
                            {upload.filename ? `${upload.filename} · ` : ""}
                            <span title={new Date(upload.createdAt).toLocaleString()}>
                              {formatRelativeTime(upload.createdAt)}
                            </span>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-card-foreground">
                        {upload.status === "failed" ? (
                          <div className="flex items-center gap-2">
                            <span
                              className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20"
                              title={upload.error || t("pendingUploadFailed")}
                            >
                              <X className="w-3 h-3" />
                              {t("pendingUploadFailedStatus")}
                            </span>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                              onClick={() => deleteFailedUpload(upload.operationId)}
                              disabled={deletingUploadOperationId === upload.operationId}
                            >
                              {deletingUploadOperationId === upload.operationId ? (
                                <Spinner size="xs" />
                              ) : (
                                <Trash2 className="w-3 h-3 mr-1" />
                              )}
                              {deletingUploadOperationId === upload.operationId
                                ? ""
                                : tOperations("action.delete")}
                            </Button>
                          </div>
                        ) : (
                          <span
                            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20"
                            title={t("pendingUploadMetadata")}
                          >
                            <RefreshCw className="w-3 h-3 animate-spin" />
                            {t("pendingUploadProcessingStatus")}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-card-foreground text-right">-</TableCell>
                      <TableCell className="text-card-foreground text-right">-</TableCell>
                    </TableRow>
                  ))}
                  {documents.length > 0 ? (
                    documents.map((doc) => {
                      const harness = documentHarness(doc.document_metadata, doc.tags);
                      const harnessLogo = resolveHarnessLogo(harness);
                      return (
                        <TableRow
                          key={doc.id}
                          className={`cursor-pointer hover:bg-muted/50 ${selectedDocument?.id === doc.id ? "bg-primary/10" : ""}`}
                          onClick={() => viewDocumentText(doc.id)}
                        >
                          {/* Identity block: the ID reads first, with when it was
                              last touched underneath. Created-at was dropped —
                              for almost every document it repeated updated-at. */}
                          <TableCell className="text-card-foreground">
                            <div className="min-w-0">
                              <div className="font-mono text-sm truncate" title={doc.id}>
                                {doc.id}
                              </div>
                              {/* The harness logo trails the timestamp rather
                                  than leading the ID: as a leading mark it only
                                  exists on some rows, so every ID shifted
                                  horizontally depending on whether its document
                                  had one. Here it appends to a line that is
                                  already ragged, and nothing moves. */}
                              <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                                {doc.updated_at ? (
                                  <span title={new Date(doc.updated_at).toLocaleString()}>
                                    {t("colUpdated")} {formatRelativeTime(doc.updated_at)}
                                  </span>
                                ) : (
                                  "N/A"
                                )}
                                <HarnessLogo
                                  harness={harness}
                                  size={14}
                                  titlePrefix={tCommon("harness")}
                                />
                                {updatingDocIds.has(doc.id) && (
                                  <span
                                    className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground"
                                    title={t("documentUpdating")}
                                  >
                                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500/70 animate-pulse" />
                                    {t("documentUpdating")}
                                  </span>
                                )}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="text-card-foreground">
                            <TagsAndMetadataCell
                              tags={doc.tags ?? []}
                              metadata={doc.document_metadata}
                              selectedTags={selectedTags}
                              onToggleTag={toggleTagFilter}
                              harnessShownAsLogo={!!harnessLogo}
                            />
                          </TableCell>
                          <TableCell className="text-card-foreground text-right tabular-nums whitespace-nowrap font-medium">
                            {formatBytes(doc.text_length || 0)}
                          </TableCell>
                          <TableCell className="text-right tabular-nums font-semibold text-foreground">
                            {doc.memory_unit_count}
                          </TableCell>
                        </TableRow>
                      );
                    })
                  ) : pendingRows.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center">
                        {t("clickLoadDocumentsToView")}
                      </TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-3 pt-3 border-t">
                <div className="text-xs text-muted-foreground">
                  {offset + 1}-{Math.min(offset + ITEMS_PER_PAGE, total)} of {total}
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(1)}
                    disabled={currentPage === 1 || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronsLeft className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1 || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronLeft className="h-3 w-3" />
                  </Button>
                  <span className="text-xs px-2">
                    {currentPage} / {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronRight className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(totalPages)}
                    disabled={currentPage === totalPages || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronsRight className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <FileText className="w-10 h-10 mx-auto mb-3 text-muted-foreground/50" />
            <div className="text-sm text-muted-foreground">
              {hasActiveFilters ? t("noDocumentsMatchSearch") : t("noDocumentsFound")}
            </div>
            {hasActiveFilters && (
              <Button variant="outline" size="sm" onClick={clearFilters} className="mt-3 gap-1">
                <X className="h-3.5 w-3.5" />
                {t("clearFilters")}
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Document Detail Dialog */}
      <Dialog open={!!selectedDocument} onOpenChange={(open) => !open && setSelectedDocument(null)}>
        <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
          <DialogHeader className="pr-10">
            <DialogTitle className="flex items-center gap-2">
              <HarnessLogo harness={selectedHarness} size={18} titlePrefix={tCommon("harness")} />
              <span className="truncate font-mono text-sm">
                {selectedDocument?.id ?? "Document"}
              </span>
            </DialogTitle>
          </DialogHeader>

          {loadingDocument ? (
            <div className="flex items-center justify-center flex-1">
              <div className="text-center">
                <Spinner size="xl" variant="jump" className="mx-auto mb-2" />
                <div className="text-sm text-muted-foreground">{t("loadingDocument")}</div>
              </div>
            </div>
          ) : selectedDocument ? (
            <Tabs defaultValue="general" className="flex-1 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between gap-2">
                <TabsList className="grid grid-cols-3 w-full max-w-md">
                  <TabsTrigger value="general" className="flex items-center gap-1.5">
                    <Settings className="w-3.5 h-3.5" />
                    General
                  </TabsTrigger>
                  <TabsTrigger value="memories" className="flex items-center gap-1.5">
                    <Network className="w-3.5 h-3.5" />
                    Memories
                  </TabsTrigger>
                  <TabsTrigger
                    value="chunks"
                    className="flex items-center gap-1.5"
                    onClick={() => {
                      if (!chunksLoaded && selectedDocument?.id) {
                        loadChunks(selectedDocument.id);
                      }
                    }}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    Chunks{chunksLoaded ? ` (${chunksTotal})` : ""}
                  </TabsTrigger>
                </TabsList>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 shrink-0"
                      disabled={reprocessing}
                      aria-label={tCommon("actions")}
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={reprocessDocument} disabled={reprocessing}>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Reprocess
                    </DropdownMenuItem>
                    {canExport && (
                      <DropdownMenuItem
                        onClick={() => exportDocuments([selectedDocument.id])}
                        disabled={exporting}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        {t("exportButton")}
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onClick={() =>
                        requestDeleteDocument(
                          selectedDocument.id,
                          selectedDocument.memory_unit_count
                        )
                      }
                      className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {t("deleteButton")}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex-1 overflow-y-auto mt-4">
                {/* Memories Tab — constellation/graph + invalidated facts */}
                <TabsContent value="memories" className="mt-0">
                  <div className="space-y-4">
                    <Tabs defaultValue="world" className="flex flex-col">
                      <TabsList className="w-fit">
                        <TabsTrigger value="world">World</TabsTrigger>
                        <TabsTrigger value="experience">Experience</TabsTrigger>
                        <TabsTrigger value="observation">Observations</TabsTrigger>
                      </TabsList>
                      <div className="mt-2">
                        <TabsContent value="world" className="mt-0">
                          <DataView factType="world" documentId={selectedDocument.id} compact />
                        </TabsContent>
                        <TabsContent value="experience" className="mt-0">
                          <DataView
                            factType="experience"
                            documentId={selectedDocument.id}
                            compact
                          />
                        </TabsContent>
                        <TabsContent value="observation" className="mt-0">
                          <DataView
                            factType="observation"
                            documentId={selectedDocument.id}
                            compact
                          />
                        </TabsContent>
                      </div>
                    </Tabs>
                    {currentBank && selectedDocument?.id && (
                      <InvalidatedFactsSection
                        bankId={currentBank}
                        documentId={selectedDocument.id}
                      />
                    )}
                  </div>
                </TabsContent>

                {/* General Tab */}
                <TabsContent value="general" className="mt-0">
                  <div className="space-y-4">
                    {/* Info cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InfoCard title="Document" icon={<FileText className="w-3.5 h-3.5" />}>
                        {selectedDocument.created_at && (
                          <MetadataRow
                            label={t("labelCreated")}
                            value={new Date(selectedDocument.created_at).toLocaleString()}
                          />
                        )}
                        {selectedDocument.updated_at && (
                          <MetadataRow
                            label={t("labelUpdated")}
                            value={new Date(selectedDocument.updated_at).toLocaleString()}
                          />
                        )}
                        {selectedDocument.original_text && (
                          <MetadataRow
                            label={t("colSize")}
                            value={formatBytes(new Blob([selectedDocument.original_text]).size)}
                          />
                        )}
                        <MetadataRow
                          label={t("labelTags")}
                          value={
                            editingTags ? (
                              <div className="flex items-center gap-2">
                                <Input
                                  value={tagInput}
                                  onChange={(e) => setTagInput(e.target.value)}
                                  placeholder={t("tagsInputPlaceholder")}
                                  className="text-sm h-7 w-64"
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") saveDocumentTags();
                                    if (e.key === "Escape") cancelEditTags();
                                  }}
                                  autoFocus
                                />
                                <Button
                                  size="sm"
                                  onClick={saveDocumentTags}
                                  disabled={savingTags}
                                  className="h-7 w-7 p-0"
                                >
                                  {savingTags ? (
                                    <Spinner size="xs" />
                                  ) : (
                                    <Check className="h-3 w-3" />
                                  )}
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={cancelEditTags}
                                  disabled={savingTags}
                                  className="h-7 w-7 p-0"
                                >
                                  <X className="h-3 w-3" />
                                </Button>
                              </div>
                            ) : (
                              <div className="flex items-center gap-2">
                                {selectedDocument.tags && selectedDocument.tags.length > 0 ? (
                                  <div className="flex flex-wrap gap-1.5">
                                    {selectedDocument.tags.map((tag: string, i: number) => (
                                      <TagChip key={i} tag={tag} />
                                    ))}
                                  </div>
                                ) : (
                                  <span className="text-sm text-muted-foreground italic">none</span>
                                )}
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={startEditTags}
                                  className="h-6 w-6 p-0"
                                >
                                  <Pencil className="h-3 w-3" />
                                </Button>
                              </div>
                            )
                          }
                        />
                        {/* An unknown harness still gets its name spelled out
                            here — only the logo is registry-gated. */}
                        {selectedHarness && (
                          <MetadataRow
                            label={tCommon("harness")}
                            value={
                              <span className="inline-flex items-center gap-1.5">
                                <HarnessLogo harness={selectedHarness} />
                                <span className="text-sm">
                                  {selectedHarnessLogo?.label ?? selectedHarness}
                                </span>
                              </span>
                            }
                          />
                        )}
                        {selectedDocument.retain_params?.context && (
                          <MetadataRow
                            label="Context"
                            value={selectedDocument.retain_params.context}
                          />
                        )}
                        {selectedDocument.retain_params?.event_date && (
                          <MetadataRow
                            label={t("labelEventDate")}
                            value={new Date(
                              selectedDocument.retain_params.event_date
                            ).toLocaleString()}
                          />
                        )}
                        {selectedDocument.retain_params?.metadata &&
                          Object.keys(selectedDocument.retain_params.metadata).length > 0 && (
                            <MetadataRow
                              label={t("labelMetadata")}
                              value={
                                <MetadataBadges
                                  metadata={selectedDocument.retain_params.metadata}
                                />
                              }
                            />
                          )}
                        {selectedDocument.observation_scopes && (
                          <MetadataRow
                            label={t("labelObservationScopes")}
                            value={
                              <ObservationScopesValue spec={selectedDocument.observation_scopes} />
                            }
                          />
                        )}
                      </InfoCard>

                      <InfoCard
                        title={t("memoryCompositionTitle")}
                        icon={<Network className="w-3.5 h-3.5" />}
                      >
                        <MemoryComposition nodesByFactType={selectedDocument.nodes_by_fact_type} />
                      </InfoCard>

                      {currentBank && selectedDocument?.id && (
                        <DocumentRetainTraces
                          bankId={currentBank}
                          documentId={selectedDocument.id}
                        />
                      )}
                    </div>

                    {/* Content — stored document text (last) */}
                    {!features?.store_document_text && !selectedDocument.original_text ? (
                      <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-700 dark:text-amber-400">
                        <Lock className="h-4 w-4 mt-0.5 shrink-0" />
                        <span>{t("textNotStoredWarning")}</span>
                      </div>
                    ) : selectedDocument.original_text !== undefined ? (
                      editingContent ? (
                        <div className="space-y-2">
                          <div className="flex items-center justify-end mb-2">
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={saveDocumentContent}
                                disabled={savingContent || !contentInput.trim()}
                                className="h-7 px-3 gap-1 text-xs"
                              >
                                {savingContent ? (
                                  <Spinner size="xs" />
                                ) : (
                                  <Check className="h-3 w-3" />
                                )}
                                {t("saveButton")}
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={cancelEditContent}
                                disabled={savingContent}
                                className="h-7 px-3 gap-1 text-xs"
                              >
                                <X className="h-3 w-3" />
                                {t("cancelButton")}
                              </Button>
                            </div>
                          </div>
                          <textarea
                            value={contentInput}
                            onChange={(e) => setContentInput(e.target.value)}
                            className="w-full min-h-[400px] max-h-[600px] p-4 bg-muted/50 rounded-lg border border-border text-sm font-mono leading-relaxed text-card-foreground resize-y"
                            autoFocus
                          />
                          <p className="text-xs text-muted-foreground">{t("saveHint")}</p>
                        </div>
                      ) : (
                        <div className="rounded-lg border border-border bg-muted/30 overflow-hidden">
                          <div className="flex items-center justify-between gap-2 px-4 py-2 border-b border-border bg-muted/50 text-xs text-muted-foreground">
                            <div className="flex items-center gap-1.5">
                              <FileText className="w-3.5 h-3.5" />
                              <span className="font-semibold uppercase tracking-wide">Content</span>
                              <span className="text-muted-foreground/70">
                                &middot;{" "}
                                {selectedDocument.original_text?.length?.toLocaleString() ?? 0}{" "}
                                chars
                              </span>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={startEditContent}
                              className="h-6 px-2 gap-1 text-xs"
                            >
                              <Pencil className="h-3 w-3" />
                              {t("editButton")}
                            </Button>
                          </div>
                          <pre className="p-4 text-[11px] leading-5 text-foreground/80 whitespace-pre-wrap font-mono">
                            {selectedDocument.original_text}
                          </pre>
                        </div>
                      )
                    ) : null}
                  </div>
                </TabsContent>

                {/* Chunks Tab */}
                <TabsContent value="chunks" className="mt-0">
                  {!features?.store_document_text && (
                    <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-3 mb-3 text-xs text-amber-700 dark:text-amber-400">
                      <Lock className="h-4 w-4 mt-0.5 shrink-0" />
                      <span>{t("textNotStoredWarning")}</span>
                    </div>
                  )}
                  {loadingChunks ? (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <Spinner size="xl" variant="jump" className="mx-auto mb-2" />
                        <div className="text-sm text-muted-foreground">{t("loadingChunks")}</div>
                      </div>
                    </div>
                  ) : chunks.length > 0 ? (
                    <div className="rounded-lg border border-border overflow-hidden divide-y divide-border">
                      {chunks.map((chunk) => (
                        <ChunkRow key={chunk.chunk_id} chunk={chunk} />
                      ))}
                    </div>
                  ) : chunksLoaded ? (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <FileText className="w-10 h-10 mx-auto mb-3 text-muted-foreground/50" />
                        <div className="text-sm text-muted-foreground">{t("noChunksFound")}</div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">
                          {t("clickChunksTabToLoad")}
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </div>
            </Tabs>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={!!documentToDelete}
        onOpenChange={(open) => !open && setDocumentToDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteDialogTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteDialogDescription")}{" "}
              <span className="font-mono font-semibold">&quot;{documentToDelete?.id}&quot;</span>?
              <br />
              <br />
              {t("deleteDialogWillDelete")}{" "}
              {documentToDelete?.memoryCount !== undefined ? (
                <span className="font-semibold">
                  {t("deleteDialogMemoryUnits", { count: documentToDelete.memoryCount })}
                </span>
              ) : (
                t("deleteDialogAllMemoryUnits")
              )}{" "}
              {t("deleteDialogExtracted")}
              <br />
              <br />
              <span className="text-destructive font-semibold">{t("deleteDialogCannotUndo")}</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("cancelButton")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteDocument}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("deleteButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Result Dialog */}
      <AlertDialog open={!!deleteResult} onOpenChange={(open) => !open && setDeleteResult(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteResult?.success ? t("deleteResultSuccessTitle") : t("deleteResultErrorTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>{deleteResult?.message}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setDeleteResult(null)}>
              {t("okButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Reprocess Result Dialog */}
      <AlertDialog
        open={!!reprocessResult}
        onOpenChange={(open) => !open && setReprocessResult(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {reprocessResult?.success ? "Reprocessing Started" : t("deleteResultErrorTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>{reprocessResult?.message}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setReprocessResult(null)}>
              {t("okButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
