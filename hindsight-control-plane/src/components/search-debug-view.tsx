"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { client } from "@/lib/api";
import { resolveTemporalWindow } from "@/lib/temporal-window";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { FactType, FactTypeFilter } from "@/components/fact-type-filter";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Search,
  Clock,
  Zap,
  ChevronRight,
  ChevronDown,
  Database,
  FileText,
  Users,
  ArrowDown,
  Tag,
  Calendar,
  CalendarRange,
} from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";
import { MemoryDetailModal } from "./memory-detail-modal";
import { AttachmentStrip, RetainedAttachment } from "@/components/ui/inline-attachment-text";
import { EntityChip, TagChip } from "@/components/ui/facet-chip";

type Budget = "low" | "mid" | "high";
type TagsMatch = "any" | "all" | "any_strict" | "all_strict" | "exact";
type ViewMode = "results" | "trace" | "json";

// Significant digits, never a fixed number of decimals. This used to print the
// raw value for a reason: fusion scores cluster around 0.001, where toFixed(3)
// collapses 0.001125 and 0.001004 to the same "0.001" and makes the reranker's
// behaviour impossible to read. Four significant digits keeps those apart while
// still trimming 0.8765526740786869 to 0.8766, and the exact value stays one
// hover away wherever there is room for a tooltip. `null`/`undefined` → em dash.
const fmtScore = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return "—";
  if (!Number.isFinite(v)) return String(v);
  // `Number(...)` drops the trailing zeros toPrecision pads on (0.8 → "0.8000").
  return String(Number(v.toPrecision(4)));
};

// Timestamps are rendered in UTC, matching what the temporal-window hint on this
// same page tells you times are read as — and a fact the extractor dated to a
// day arrives as midnight UTC, which in any other zone would shift it onto the
// wrong date entirely.
//
// The time is dropped when it *is* that midnight: printing "00:00" beside a
// date-only occurrence invents a precision the memory does not have.
const UTC_DATE: Intl.DateTimeFormatOptions = {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
};
const UTC_TIME: Intl.DateTimeFormatOptions = {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "UTC",
};

const fmtWhen = (v: string | null | undefined): string => {
  if (!v) return "";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  const date = d.toLocaleDateString(undefined, UTC_DATE);
  const isDateOnly = d.getUTCHours() === 0 && d.getUTCMinutes() === 0 && d.getUTCSeconds() === 0;
  return isDateOnly ? date : `${date} ${d.toLocaleTimeString(undefined, UTC_TIME)}`;
};

// A day-granularity occurrence is stored as the whole day — 00:00:00 to
// 23:59:59.999 — so rendering it as a range prints "3 Mar 2024 → 3 Mar 2024
// 23:59", which reads as a precision that was never claimed. Collapse it back to
// the single date it means.
const fmtWhenRange = (start: string, end: string | null | undefined): string => {
  if (!end || end === start) return fmtWhen(start);
  const from = new Date(start);
  const to = new Date(end);
  if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime())) return fmtWhen(start);
  const sameDay = from.toISOString().slice(0, 10) === to.toISOString().slice(0, 10);
  const wholeDay =
    from.getUTCHours() === 0 &&
    from.getUTCMinutes() === 0 &&
    from.getUTCSeconds() === 0 &&
    to.getUTCHours() === 23 &&
    to.getUTCMinutes() === 59;
  if (sameDay && wholeDay) return fmtWhen(start);
  return `${fmtWhen(start)} → ${fmtWhen(end)}`;
};

/** The unrounded value, for a `title` beside a rounded one. */
const exactScore = (v: number | null | undefined): string | undefined =>
  v === null || v === undefined ? undefined : String(v);

export function SearchDebugView() {
  const t = useTranslations("searchDebug");
  const { currentBank } = useBank();

  // Query state
  const [query, setQuery] = useState("");
  const [factTypes, setFactTypes] = useState<FactType[]>(["world"]);
  const [budget, setBudget] = useState<Budget>("mid");
  const [maxTokens, setMaxTokens] = useState(4096);
  const [queryDate, setQueryDate] = useState("");
  const [includeChunks, setIncludeChunks] = useState(false);
  // On by default: the entity names shown as chips on each result come back only
  // when entities are included — the flag gates the names, not just the entity
  // observations block — and "what is this fact about" is the first thing worth
  // seeing in a results list. Untick it to recall without the extra lookup.
  const [includeEntities, setIncludeEntities] = useState(true);
  const [windowStart, setWindowStart] = useState("");
  const [windowEnd, setWindowEnd] = useState("");
  const [tags, setTags] = useState("");
  const [tagsMatch, setTagsMatch] = useState<TagsMatch>("any");

  // Results state
  const [results, setResults] = useState<any[] | null>(null);
  const [entities, setEntities] = useState<any[] | null>(null);
  // Keyed by chunk id (`chunk_id -> ChunkData`), which is the shape the API
  // returns — it was typed as an array, which no consumer could index. Only the
  // JSON view reads it: results take their attachments from the fact's own edge,
  // never from the chunk (see attachmentsForResult).
  const [chunks, setChunks] = useState<Record<string, any> | null>(null);
  const [observations, setObservations] = useState<any[] | null>(null);
  const [trace, setTrace] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("results");
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());

  const toggleStep = (step: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(step)) {
        next.delete(step);
      } else {
        next.add(step);
      }
      return next;
    });
  };

  const toggleExpandResults = (key: string) => {
    setExpandedResults((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const INITIAL_RESULTS_COUNT = 5;

  // Trace rows carry the memory id under either key; the dialog fetches the rest
  // itself, so there is nothing to look up in `results` any more.
  const selectMemoryFromTrace = (traceResult: any) => {
    setSelectedMemoryId(traceResult.id || traceResult.node_id || null);
  };

  const temporalWindow = resolveTemporalWindow(windowStart, windowEnd);

  // Only the edge the extractor recorded for this fact. There is no falling back
  // to the chunk's attachments: a chunk lists everything its text references, so
  // a fact drawn from the prose beside a screenshot would be shown that
  // screenshot as its evidence. A fact the extractor attributed to nothing shows
  // nothing, which is the honest answer.
  const attachmentsForResult = (result: any): RetainedAttachment[] =>
    (result?.attachments as RetainedAttachment[] | undefined) ?? [];

  const runSearch = async () => {
    // Guard here, not just on the button: Enter in the query box calls this
    // directly, and searching anyway would silently drop the window the user
    // set rather than telling them it is unusable.
    if (temporalWindow.reversed) {
      toast.error(t("temporalWindowReversed"));
      return;
    }
    if (!currentBank) {
      toast.error(t("errorSelectBank"));
      return;
    }

    if (!query) {
      return;
    }

    // Must select at least one type
    if (factTypes.length === 0) {
      toast.error(t("errorSelectFactType"));
      return;
    }

    setLoading(true);

    try {
      // Parse tags from comma-separated string
      const parsedTags = tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const requestBody: any = {
        bank_id: currentBank,
        query: query,
        types: factTypes,
        budget: budget,
        max_tokens: maxTokens,
        trace: true,
        include: {
          entities: includeEntities ? { max_tokens: 500 } : null,
          chunks: includeChunks ? { max_tokens: 8192 } : null,
        },
        ...(queryDate && { query_timestamp: queryDate }),
        ...(temporalWindow.value && { temporal_window: temporalWindow.value }),
        ...(parsedTags.length > 0 && { tags: parsedTags, tags_match: tagsMatch }),
      };

      const data: any = await client.recall(requestBody);

      setResults(data.results || []);
      setEntities(data.entities || null);
      setChunks(data.chunks || null);
      setObservations(data.observations || null);
      setTrace(data.trace || null);
      setViewMode("results");
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoading(false);
    }
  };

  if (!currentBank) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-16">
          <Database className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">{t("noBankSelected")}</h3>
          <p className="text-muted-foreground">{t("noBankSelectedDescription")}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t("queryPlaceholder")}
                className="pl-10 h-12 text-lg"
                onKeyDown={(e) => e.key === "Enter" && runSearch()}
              />
            </div>
            <Button
              onClick={runSearch}
              disabled={loading || !query || temporalWindow.reversed}
              className="h-12 px-8"
            >
              {loading ? t("searching") : t("recall")}
            </Button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-6 mt-4 pt-4 border-t">
            <FactTypeFilter value={factTypes} onChange={setFactTypes} label={t("typesLabel")} />

            <div className="h-6 w-px bg-border" />

            {/* Budget */}
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              <Select value={budget} onValueChange={(v) => setBudget(v as Budget)}>
                <SelectTrigger className="w-24 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">{t("budgetLow")}</SelectItem>
                  <SelectItem value="mid">{t("budgetMid")}</SelectItem>
                  <SelectItem value="high">{t("budgetHigh")}</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Max Tokens */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">{t("tokensLabel")}</span>
              <Input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-24 h-8"
              />
            </div>

            {/* Query Date */}
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <Input
                type="datetime-local"
                value={queryDate}
                onChange={(e) => setQueryDate(e.target.value)}
                className="h-8"
                placeholder={t("queryDatePlaceholder")}
              />
            </div>

            <div className="h-6 w-px bg-border" />

            {/* Include options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={includeChunks}
                  onCheckedChange={(c) => setIncludeChunks(c as boolean)}
                />
                <FileText className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">{t("chunks")}</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer" title={t("entitiesHint")}>
                <Checkbox
                  checked={includeEntities}
                  onCheckedChange={(c) => setIncludeEntities(c as boolean)}
                />
                <Users className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">{t("entities")}</span>
              </label>
            </div>
          </div>

          {/* Tags Filter */}
          <div className="flex items-center gap-4 mt-4 pt-4 border-t">
            <Tag className="h-4 w-4 text-muted-foreground" />
            <div className="flex-1 max-w-md">
              <Input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder={t("tagsPlaceholder")}
                className="h-8"
              />
            </div>
            <Select value={tagsMatch} onValueChange={(v) => setTagsMatch(v as TagsMatch)}>
              <SelectTrigger className="w-40 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">{t("tagsMatchAny")}</SelectItem>
                <SelectItem value="all">{t("tagsMatchAll")}</SelectItem>
                <SelectItem value="any_strict">{t("tagsMatchAnyStrict")}</SelectItem>
                <SelectItem value="all_strict">{t("tagsMatchAllStrict")}</SelectItem>
                <SelectItem value="exact">{t("tagsMatchExact")}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Temporal window */}
          <div className="flex flex-wrap items-center gap-4 mt-4 pt-4 border-t">
            <CalendarRange className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">{t("temporalWindowLabel")}</span>
            <Input
              type="datetime-local"
              value={windowStart}
              onChange={(e) => setWindowStart(e.target.value)}
              aria-label={t("temporalWindowStart")}
              className="h-8 w-56"
            />
            <span className="text-sm text-muted-foreground">{t("temporalWindowTo")}</span>
            <Input
              type="datetime-local"
              value={windowEnd}
              onChange={(e) => setWindowEnd(e.target.value)}
              aria-label={t("temporalWindowEnd")}
              className="h-8 w-56"
            />
            {(windowStart || windowEnd) && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8"
                onClick={() => {
                  setWindowStart("");
                  setWindowEnd("");
                }}
              >
                {t("temporalWindowClear")}
              </Button>
            )}
            <p
              className={`w-full text-xs ${temporalWindow.reversed ? "text-destructive" : "text-muted-foreground"}`}
            >
              {temporalWindow.reversed ? t("temporalWindowReversed") : t("temporalWindowHint")}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Spinner size="lg" variant="jump" className="mb-4" />
            <p className="text-muted-foreground">{t("searchingMemories")}</p>
          </CardContent>
        </Card>
      )}

      {!loading && results && (
        <div className="space-y-4">
          {/* Summary Stats */}
          {trace?.summary && (
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">{t("resultsLabel")}</span>
                <span className="font-semibold">{results.length}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">{t("durationLabel")}</span>
                <span className="font-semibold">
                  {trace.summary.total_duration_seconds?.toFixed(2)}s
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">{t("nodesVisitedLabel")}</span>
                <span className="font-semibold">{trace.summary.total_nodes_visited}</span>
              </div>

              <div className="flex-1" />

              {/* View Mode Tabs */}
              <div className="flex gap-1 bg-muted p-1 rounded-lg">
                {(["results", "trace", "json"] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      viewMode === mode
                        ? "bg-background shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {mode === "results"
                      ? t("viewResults")
                      : mode === "trace"
                        ? t("viewTrace")
                        : t("viewJson")}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Results View */}
          {viewMode === "results" && (
            <div className="space-y-4">
              {/* Observations Section */}
              {observations && observations.length > 0 && (
                <Card className="border-orange-500/30 bg-orange-500/5">
                  <CardHeader className="py-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Database className="h-4 w-4 text-orange-500" />
                      <span>Observations</span>
                      <span className="text-xs text-muted-foreground">({observations.length})</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0 space-y-2">
                    {observations.map((obs: any, idx: number) => (
                      <div
                        key={obs.id || idx}
                        className="p-3 bg-background rounded-lg border border-orange-500/20"
                      >
                        <p className="text-sm text-foreground">{obs.text}</p>
                        <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                          <span className="px-2 py-0.5 rounded bg-orange-500/10 text-orange-600">
                            Observation
                          </span>
                          <span>{t("proofCount", { count: obs.proof_count || 1 })}</span>
                          <span>{t("relevance", { value: fmtScore(obs.relevance) })}</span>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Memories Section */}
              <div className="space-y-3">
                {results.length === 0 && (!observations || observations.length === 0) ? (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Search className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">{t("noMemoriesFound")}</p>
                    </CardContent>
                  </Card>
                ) : (
                  results.map((result: any, idx: number) => {
                    const visit = trace?.visits?.find((v: any) => v.node_id === result.id);
                    const score = visit ? visit.weights.final_weight : result.scores?.final;

                    return (
                      <Card
                        key={idx}
                        className="cursor-pointer hover:border-primary/50 transition-colors"
                        onClick={() => setSelectedMemoryId(result.id)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-start gap-4">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                              <span className="text-sm font-semibold text-primary">{idx + 1}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-foreground">{result.text}</p>
                              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-muted-foreground">
                                <span className="px-2 py-0.5 rounded bg-muted capitalize">
                                  {result.type || "world"}
                                </span>
                                {result.context && (
                                  <span className="truncate max-w-xs">{result.context}</span>
                                )}
                                {result.occurred_start && (
                                  <span
                                    className="inline-flex items-center gap-1 whitespace-nowrap"
                                    title={result.occurred_start}
                                  >
                                    <Calendar className="h-3 w-3 shrink-0" />
                                    {t("occurredLabel")}{" "}
                                    {fmtWhenRange(result.occurred_start, result.occurred_end)}
                                  </span>
                                )}
                                {result.mentioned_at && (
                                  <span
                                    className="inline-flex items-center gap-1 whitespace-nowrap"
                                    title={result.mentioned_at}
                                  >
                                    <Clock className="h-3 w-3 shrink-0" />
                                    {t("mentionedLabel")} {fmtWhen(result.mentioned_at)}
                                  </span>
                                )}
                              </div>
                              {/* Boolean, not a bare length: `0 && ...` renders a
                                  literal "0" for a fact with empty arrays. */}
                              {((result.entities?.length ?? 0) > 0 ||
                                (result.tags?.length ?? 0) > 0) && (
                                // Entities and tags, not the score breakdown:
                                // what the fact is *about* is what a reader
                                // scanning results needs. The per-signal scores
                                // (semantic/keyword/reranker and the boosts they
                                // feed) are a retrieval-debugging concern and
                                // live in the Trace tab, which shows them per
                                // stage rather than as a flat row here.
                                <div className="flex flex-wrap gap-1.5 mt-2">
                                  {(result.entities ?? []).map((entity: string) => (
                                    <EntityChip
                                      key={`e-${entity}`}
                                      entity={entity}
                                      size="xs"
                                      truncate
                                      className="max-w-[220px]"
                                    />
                                  ))}
                                  {(result.tags ?? []).map((tag: string) => (
                                    <TagChip
                                      key={`t-${tag}`}
                                      tag={tag}
                                      size="xs"
                                      truncate
                                      className="max-w-[220px]"
                                    />
                                  ))}
                                </div>
                              )}
                              {(() => {
                                const attachments = attachmentsForResult(result);
                                if (attachments.length === 0) return null;
                                return (
                                  // Each attachment is a link to its own bytes;
                                  // without stopping the click here it would also
                                  // bubble to the card and open the memory dialog
                                  // on top of the image the reader just asked for.
                                  <div
                                    className="mt-3"
                                    onClick={(e) => e.stopPropagation()}
                                    role="presentation"
                                  >
                                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                                      {t("factAttachments")}
                                    </div>
                                    <AttachmentStrip
                                      bankId={currentBank}
                                      attachments={attachments}
                                      className="mt-1"
                                    />
                                  </div>
                                );
                              })()}
                            </div>
                            <div className="flex-shrink-0 text-right">
                              <div className="text-sm font-semibold" title={exactScore(score)}>
                                {fmtScore(score)}
                              </div>
                              <div className="text-xs text-muted-foreground">{t("scoreLabel")}</div>
                            </div>
                            <ChevronRight className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })
                )}
              </div>
            </div>
          )}

          {/* Trace View */}
          {viewMode === "trace" && trace && (
            <div className="space-y-4">
              {/* Parallel Retrieval Methods - Grouped by Fact Type */}
              {trace.retrieval_results &&
                trace.retrieval_results.length > 0 &&
                (() => {
                  // Group retrieval results by fact type
                  const factTypeGroups: Record<string, any[]> = {};
                  trace.retrieval_results.forEach((method: any) => {
                    const ft = method.fact_type || "all";
                    if (!factTypeGroups[ft]) factTypeGroups[ft] = [];
                    factTypeGroups[ft].push(method);
                  });
                  const factTypes = Object.keys(factTypeGroups);

                  return (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-2">
                        <div className="flex-1 h-px bg-border" />
                        <span>{t("parallelRetrieval")}</span>
                        <div className="flex-1 h-px bg-border" />
                      </div>

                      {/* Fact type lanes */}
                      <div className="space-y-2">
                        {factTypes.map((factType, ftIdx) => {
                          const methods = factTypeGroups[factType];
                          const laneKey = `lane-${factType}`;
                          const isLaneExpanded = expandedSteps.has(laneKey);
                          const totalResults = methods.reduce(
                            (sum: number, m: any) => sum + (m.results?.length || 0),
                            0
                          );
                          const totalDuration = Math.max(
                            ...methods.map((m: any) => m.duration_seconds || 0)
                          );

                          // Color coding for fact types
                          const ftColors: Record<
                            string,
                            { bg: string; text: string; border: string }
                          > = {
                            world: {
                              bg: "bg-blue-500/10",
                              text: "text-blue-500",
                              border: "border-blue-500/30",
                            },
                            experience: {
                              bg: "bg-green-500/10",
                              text: "text-green-500",
                              border: "border-green-500/30",
                            },
                            opinion: {
                              bg: "bg-purple-500/10",
                              text: "text-purple-500",
                              border: "border-purple-500/30",
                            },
                            all: {
                              bg: "bg-gray-500/10",
                              text: "text-gray-500",
                              border: "border-gray-500/30",
                            },
                          };
                          const colors = ftColors[factType] || ftColors.all;

                          return (
                            <Card
                              key={laneKey}
                              className={`transition-colors ${isLaneExpanded ? "border-primary" : colors.border}`}
                            >
                              <CardContent className="py-3 px-4">
                                {/* Lane Header */}
                                <div
                                  className="flex items-center gap-3 cursor-pointer"
                                  onClick={() => toggleStep(laneKey)}
                                >
                                  <div
                                    className={`w-8 h-8 rounded-lg ${colors.bg} flex items-center justify-center`}
                                  >
                                    <span className={`text-sm font-bold ${colors.text} capitalize`}>
                                      {factType.charAt(0).toUpperCase()}
                                    </span>
                                  </div>
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                      <span className="font-semibold text-foreground capitalize">
                                        {factType}
                                      </span>
                                      <span className="text-xs text-muted-foreground">
                                        {t("methodsCount", { count: methods.length })}
                                      </span>
                                    </div>
                                    {/* Method summary pills */}
                                    <div className="flex gap-1.5 mt-1">
                                      {methods.map((m: any, mIdx: number) => (
                                        <span
                                          key={mIdx}
                                          className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground capitalize"
                                        >
                                          {m.method_name}: {m.results?.length || 0}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-2xl font-bold text-foreground">
                                      {totalResults}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground">
                                      {totalDuration.toFixed(2)}s
                                    </div>
                                  </div>
                                  {isLaneExpanded ? (
                                    <ChevronDown className="h-5 w-5 text-muted-foreground" />
                                  ) : (
                                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                                  )}
                                </div>

                                {/* Expanded: Show methods grid */}
                                {isLaneExpanded && (
                                  <div className="mt-4 pt-4 border-t border-border">
                                    <div
                                      className={`grid gap-3 ${
                                        methods.length === 1
                                          ? "grid-cols-1"
                                          : methods.length === 2
                                            ? "grid-cols-2"
                                            : methods.length === 3
                                              ? "grid-cols-3"
                                              : "grid-cols-4"
                                      }`}
                                    >
                                      {methods.map((method: any, mIdx: number) => {
                                        const methodKey = `${laneKey}-method-${mIdx}`;
                                        const isMethodExpanded = expandedSteps.has(methodKey);
                                        const methodResults = method.results || [];

                                        return (
                                          <div key={methodKey} className="flex flex-col">
                                            <div
                                              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                                                isMethodExpanded
                                                  ? "bg-primary/10 border border-primary"
                                                  : "bg-muted/50 hover:bg-muted"
                                              }`}
                                              onClick={(e) => {
                                                e.stopPropagation();
                                                toggleStep(methodKey);
                                              }}
                                            >
                                              <div className="flex items-center justify-between mb-1">
                                                <div className="flex items-center gap-2">
                                                  <span className="font-medium text-sm text-foreground capitalize">
                                                    {method.method_name}
                                                  </span>
                                                  {/* Show temporal range inline */}
                                                  {method.method_name === "temporal" &&
                                                    method.metadata?.constraint && (
                                                      <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                                                        <Calendar className="h-3 w-3" />
                                                        {method.metadata.constraint.start
                                                          ? new Date(
                                                              method.metadata.constraint.start
                                                            ).toLocaleDateString()
                                                          : "any"}
                                                        {" → "}
                                                        {method.metadata.constraint.end
                                                          ? new Date(
                                                              method.metadata.constraint.end
                                                            ).toLocaleDateString()
                                                          : "any"}
                                                      </span>
                                                    )}
                                                </div>
                                                {isMethodExpanded ? (
                                                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                                                ) : (
                                                  <ChevronRight className="h-3 w-3 text-muted-foreground" />
                                                )}
                                              </div>
                                              <div className="flex items-end justify-between">
                                                <div className="text-2xl font-bold text-foreground">
                                                  {methodResults.length}
                                                </div>
                                                <div className="text-[10px] text-muted-foreground">
                                                  {method.duration_seconds?.toFixed(2)}s
                                                </div>
                                              </div>
                                            </div>

                                            {/* Method Results */}
                                            {isMethodExpanded &&
                                              methodResults.length > 0 &&
                                              (() => {
                                                const resultsKey = `results-${methodKey}`;
                                                const showAll = expandedResults.has(resultsKey);
                                                const displayResults = showAll
                                                  ? methodResults
                                                  : methodResults.slice(0, INITIAL_RESULTS_COUNT);
                                                const hasMore =
                                                  methodResults.length > INITIAL_RESULTS_COUNT;

                                                return (
                                                  <div className="mt-2 space-y-1.5 max-h-[300px] overflow-y-auto">
                                                    {displayResults.map((r: any, rIdx: number) => (
                                                      <div
                                                        key={rIdx}
                                                        className="p-2 bg-background rounded cursor-pointer hover:bg-muted/50 transition-colors border border-border"
                                                        onClick={(e) => {
                                                          e.stopPropagation();
                                                          selectMemoryFromTrace(r);
                                                        }}
                                                      >
                                                        <div className="flex items-start gap-2">
                                                          <span className="text-[10px] font-mono text-muted-foreground mt-0.5">
                                                            {rIdx + 1}
                                                          </span>
                                                          <div className="flex-1 min-w-0">
                                                            <p className="text-xs text-foreground line-clamp-2">
                                                              {r.text}
                                                            </p>
                                                            <div className="flex items-center gap-2 mt-1">
                                                              <span className="text-[10px] text-muted-foreground">
                                                                {fmtScore(r.score ?? r.similarity)}
                                                              </span>
                                                            </div>
                                                          </div>
                                                        </div>
                                                      </div>
                                                    ))}
                                                    {hasMore && (
                                                      <button
                                                        className="w-full text-[10px] text-primary hover:text-primary/80 py-1.5 hover:bg-muted/50 rounded transition-colors"
                                                        onClick={(e) => {
                                                          e.stopPropagation();
                                                          toggleExpandResults(resultsKey);
                                                        }}
                                                      >
                                                        {showAll
                                                          ? t("showLess")
                                                          : t("viewAllResults", {
                                                              count: methodResults.length,
                                                            })}
                                                      </button>
                                                    )}
                                                  </div>
                                                );
                                              })()}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          );
                        })}
                      </div>

                      {/* Parallel indicator - vertical lines showing all run together */}
                      <div className="flex justify-center py-2">
                        <div className="flex items-center gap-2">
                          {factTypes.map((ft, i) => {
                            const ftColors: Record<string, string> = {
                              world: "bg-blue-500",
                              experience: "bg-green-500",
                              opinion: "bg-purple-500",
                              all: "bg-gray-500",
                            };
                            return (
                              <div key={i} className="flex flex-col items-center">
                                <div
                                  className={`w-1 h-4 ${ftColors[ft] || ftColors.all} rounded-full opacity-50`}
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      <div className="flex justify-center">
                        <ArrowDown className="h-5 w-5 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Step 2: RRF Merge */}
              {trace.rrf_merged &&
                (() => {
                  const stepKey = "rrf-merge";
                  const isExpanded = expandedSteps.has(stepKey);

                  return (
                    <div>
                      <Card
                        className={`cursor-pointer transition-colors ${isExpanded ? "border-primary" : "hover:border-primary/50"}`}
                        onClick={() => toggleStep(stepKey)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                              <span className="text-sm font-bold text-purple-500">∪</span>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-foreground">
                                  {t("rrfFusion")}
                                </span>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                                  {t("rrfMerge")}
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground mt-0.5">
                                {t("rrfDescription")}
                              </div>
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                              {trace.rrf_merged.length}
                            </div>
                            {isExpanded ? (
                              <ChevronDown className="h-5 w-5 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-5 w-5 text-muted-foreground" />
                            )}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Expanded Results */}
                      {isExpanded &&
                        trace.rrf_merged.length > 0 &&
                        (() => {
                          const resultsKey = "results-rrf";
                          const showAll = expandedResults.has(resultsKey);
                          const displayResults = showAll
                            ? trace.rrf_merged
                            : trace.rrf_merged.slice(0, INITIAL_RESULTS_COUNT);
                          const hasMore = trace.rrf_merged.length > INITIAL_RESULTS_COUNT;

                          return (
                            <div className="ml-6 mt-2 space-y-2 border-l-2 border-muted pl-4 max-h-[400px] overflow-y-auto">
                              {displayResults.map((r: any, rIdx: number) => (
                                <div
                                  key={rIdx}
                                  className="p-3 bg-muted/30 rounded-lg cursor-pointer hover:bg-muted/50 transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    selectMemoryFromTrace(r);
                                  }}
                                >
                                  <div className="flex items-start gap-3">
                                    <span className="text-xs font-mono text-muted-foreground">
                                      {rIdx + 1}
                                    </span>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm text-foreground line-clamp-2">
                                        {r.text}
                                      </p>
                                      <div className="text-xs text-muted-foreground mt-1">
                                        {t("rrfScore")} {fmtScore(r.rrf_score ?? r.score)}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                              {hasMore && (
                                <button
                                  className="w-full text-xs text-primary hover:text-primary/80 py-2 hover:bg-muted/50 rounded transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandResults(resultsKey);
                                  }}
                                >
                                  {showAll
                                    ? t("showLess")
                                    : t("viewAllResults", { count: trace.rrf_merged.length })}
                                </button>
                              )}
                            </div>
                          );
                        })()}

                      {/* Arrow */}
                      <div className="flex justify-center py-2">
                        <ArrowDown className="h-4 w-4 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Step 3: Combined Scoring */}
              {trace.reranked &&
                (() => {
                  const stepKey = "reranking";
                  const isExpanded = expandedSteps.has(stepKey);

                  return (
                    <div>
                      <Card
                        className={`cursor-pointer transition-colors ${isExpanded ? "border-primary" : "hover:border-primary/50"}`}
                        onClick={() => toggleStep(stepKey)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-amber-500/10 flex items-center justify-center">
                              <span className="text-sm font-bold text-amber-500">⚡</span>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-foreground">
                                  {t("combinedScoring")}
                                </span>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                                  {t("rerank")}
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground mt-0.5">
                                <span className="font-mono text-xs">
                                  reranker_score × recency_boost(±10%) × temporal_boost(±10%) ×
                                  proof_boost(±5%)
                                </span>
                              </div>
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                              {trace.reranked.length}
                            </div>
                            {isExpanded ? (
                              <ChevronDown className="h-5 w-5 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-5 w-5 text-muted-foreground" />
                            )}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Expanded Results */}
                      {isExpanded &&
                        trace.reranked.length > 0 &&
                        (() => {
                          const resultsKey = "results-rerank";
                          const showAll = expandedResults.has(resultsKey);
                          const displayResults = showAll
                            ? trace.reranked
                            : trace.reranked.slice(0, INITIAL_RESULTS_COUNT);
                          const hasMore = trace.reranked.length > INITIAL_RESULTS_COUNT;

                          return (
                            <div className="ml-6 mt-2 space-y-2 border-l-2 border-muted pl-4 max-h-[400px] overflow-y-auto">
                              {displayResults.map((r: any, rIdx: number) => {
                                const sc = r.score_components || {};
                                // The combined score is CE × multiplicative boosts, where each
                                // boost = 1 + alpha·(signal − 0.5) (neutral 1.0 at signal 0.5).
                                // The trace carries the raw 0–1 signals, so derive the actual
                                // multipliers here — otherwise CE × "Rec 1.000" can't reproduce
                                // the displayed total (e.g. 0.999 × recency-boost 1.100 ≈ 1.099).
                                const boost = (signal: number, alpha: number) =>
                                  1 + alpha * (signal - 0.5);
                                const recBoost =
                                  sc.recency !== undefined ? boost(sc.recency, 0.2) : undefined;
                                const tmpBoost =
                                  sc.temporal !== undefined ? boost(sc.temporal, 0.2) : undefined;
                                const proofBoost =
                                  sc.proof_norm !== undefined
                                    ? boost(sc.proof_norm, 0.1)
                                    : undefined;
                                return (
                                  <div
                                    key={rIdx}
                                    className="p-3 bg-muted/30 rounded-lg cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      selectMemoryFromTrace(r);
                                    }}
                                  >
                                    <div className="flex items-start gap-3">
                                      <span className="text-xs font-mono text-muted-foreground">
                                        {rIdx + 1}
                                      </span>
                                      <div className="flex-1 min-w-0">
                                        <p className="text-sm text-foreground line-clamp-2">
                                          {r.text}
                                        </p>
                                        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px] text-muted-foreground font-mono">
                                          <span className="font-semibold text-foreground">
                                            = {fmtScore(r.rerank_score ?? r.score)}
                                          </span>
                                          {sc.cross_encoder_score_normalized !== undefined && (
                                            <span title={t("tooltipCrossEncoder")}>
                                              reranker {fmtScore(sc.cross_encoder_score_normalized)}
                                            </span>
                                          )}
                                          {recBoost !== undefined && (
                                            <span
                                              title={`${t("tooltipRecency")} — signal ${fmtScore(sc.recency)} → ×${fmtScore(recBoost)}`}
                                            >
                                              × rec {fmtScore(recBoost)}
                                            </span>
                                          )}
                                          {tmpBoost !== undefined && sc.temporal !== 0.5 && (
                                            <span
                                              title={`${t("tooltipTemporal")} — signal ${fmtScore(sc.temporal)} → ×${fmtScore(tmpBoost)}`}
                                            >
                                              × tmp {fmtScore(tmpBoost)}
                                            </span>
                                          )}
                                          {proofBoost !== undefined && sc.proof_norm !== 0.5 && (
                                            <span
                                              title={`Proof-count boost (±5%) — signal ${fmtScore(sc.proof_norm)} → ×${fmtScore(proofBoost)}`}
                                            >
                                              × proof {fmtScore(proofBoost)}
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                              {hasMore && (
                                <button
                                  className="w-full text-xs text-primary hover:text-primary/80 py-2 hover:bg-muted/50 rounded transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandResults(resultsKey);
                                  }}
                                >
                                  {showAll
                                    ? t("showLess")
                                    : t("viewAllResults", { count: trace.reranked.length })}
                                </button>
                              )}
                            </div>
                          );
                        })()}

                      {/* Arrow */}
                      <div className="flex justify-center py-2">
                        <ArrowDown className="h-4 w-4 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Final: Results */}
              <Card className="border-primary bg-primary/5">
                <CardContent className="py-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                      <span className="text-sm font-bold text-primary">✓</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-foreground">{t("finalResults")}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                          {t("output")}
                        </span>
                      </div>
                      <div className="text-sm text-muted-foreground mt-0.5">
                        {t("finalResultsDescription")}
                      </div>
                    </div>
                    <div className="text-2xl font-bold text-primary">{results?.length || 0}</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* JSON View */}
          {viewMode === "json" && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">{t("rawResponse")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted p-4 rounded-lg overflow-auto max-h-[600px]">
                  <JsonView
                    src={{
                      results,
                      ...(entities && { entities }),
                      ...(chunks && { chunks }),
                      ...(observations && { observations }),
                      trace,
                    }}
                    collapsed={2}
                    theme="default"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Empty State */}
      {!loading && !results && (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Search className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">{t("readyToRecall")}</h3>
            <p className="text-muted-foreground text-center max-w-md">
              {t("readyToRecallDescription")}
            </p>
          </CardContent>
        </Card>
      )}

      {/* The same dialog every other view opens a memory in. It was a fixed
          right-hand drawer here, which meant one memory rendered two different
          ways depending on where you clicked — and the drawer was fed a recall
          *result*, which carries none of the detail (attachments included) that
          fetching the memory by id returns. */}
      <MemoryDetailModal memoryId={selectedMemoryId} onClose={() => setSelectedMemoryId(null)} />
    </div>
  );
}
