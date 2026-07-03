"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useTranslations } from "next-intl";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Calendar,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Settings2,
  Eye,
  RefreshCw,
  CheckCircle,
  Clock,
  List,
  Search,
  Layers,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MemoryDetailModal } from "./memory-detail-modal";
import { convertHindsightGraphData, GraphNode } from "./graph-data";
import { Constellation } from "./constellation";
import { TagFilterInput } from "./tag-filter-input";
import { ObservationScopeFilter, ObservationScope } from "./observation-scope-filter";
import { ScatterChart, Plus, FileText } from "lucide-react";

type FactType = "world" | "experience" | "observation";
type ViewMode = "table" | "timeline" | "constellation";

// Categorical palette for coloring observation scopes (exact tag sets) when
// "Group by scope" clusters the constellation. Distinct, reasonably separable hues.
const SCOPE_PALETTE = [
  "#0074d9",
  "#e11d48",
  "#16a34a",
  "#f59e0b",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#65a30d",
  "#f97316",
  "#6366f1",
];

// Stable key for a scope = its tag set, order-normalized (matches the backend's
// normalized scope enumeration so colors are consistent regardless of tag order).
function scopeKeyOf(tags: string[] | undefined): string {
  return JSON.stringify([...(tags || [])].sort());
}

interface DataViewProps {
  factType: FactType;
  documentId?: string;
  chunkId?: string;
  compact?: boolean;
  onExpandToggle?: () => void;
}

export function DataView({
  factType,
  documentId,
  chunkId,
  compact = false,
  onExpandToggle,
}: DataViewProps) {
  const t = useTranslations("dataView");
  const tAddDoc = useTranslations("addDocument");
  const { currentBank } = useBank();
  const [viewMode, setViewMode] = useState<ViewMode>("constellation");
  const [compactMode, setCompactMode] = useState(compact);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [tagFilters, setTagFilters] = useState<string[]>([]);
  // Observation scope filtering: the distinct scopes available, and the selected
  // one. `null` = all scopes; `[]` = the global (untagged) scope; otherwise an
  // exact tag set. Mutually exclusive with the free-form tag filter above.
  const [scopes, setScopes] = useState<ObservationScope[]>([]);
  const [selectedScope, setSelectedScope] = useState<string[] | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [modalMemoryId, setModalMemoryId] = useState<string | null>(null);
  // Table view: toggle between live facts (graph-fed) and invalidated facts (archive).
  const [showInvalidated, setShowInvalidated] = useState(false);
  const [invalidatedRows, setInvalidatedRows] = useState<any[]>([]);
  const itemsPerPage = 100;

  // Fetch limit state - how many memories to load from the API
  const [fetchLimit, setFetchLimit] = useState(1000);

  // Which timestamp drives the constellation recency color
  type RecencyBasis = "mentioned_at" | "occurred_start" | "occurred_end";
  const RECENCY_BASIS_LABEL: Record<RecencyBasis, string> = {
    mentioned_at: t("recencyBasisMentioned"),
    occurred_start: t("recencyBasisOccurredStart"),
    occurred_end: t("recencyBasisOccurredEnd"),
  };
  const [recencyBasis, setRecencyBasis] = useState<RecencyBasis>("mentioned_at");
  // Constellation: group observations into per-scope clusters (with colored blobs).
  const [groupByScope, setGroupByScope] = useState(false);

  // Consolidation status for mental models
  const [consolidationStatus, setConsolidationStatus] = useState<{
    pending_consolidation: number;
    last_consolidated_at: string | null;
  } | null>(null);

  // Constellation controls state
  const [visibleLinkTypes, setVisibleLinkTypes] = useState<Set<string>>(
    new Set(["semantic", "temporal", "entity", "causal"])
  );

  const toggleLinkType = (type: string) => {
    setVisibleLinkTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  // `silent` skips the loading spinner — used by the background consolidation
  // poll so the view refreshes in place without flashing.
  const loadData = async (
    limit?: number,
    q?: string,
    tags?: string[],
    tagsMatch?: string,
    silent = false
  ) => {
    if (!currentBank) return;

    if (!silent) setLoading(true);
    try {
      const graphData: any = await client.getGraph({
        bank_id: currentBank,
        type: factType,
        limit: limit ?? fetchLimit,
        q,
        tags,
        tags_match: tagsMatch,
        document_id: documentId,
        chunk_id: chunkId,
      });
      setData(graphData);

      // Fetch consolidation status for observations
      if (factType === "observation") {
        const stats: any = await client.getBankStats(currentBank);
        setConsolidationStatus({
          pending_consolidation: stats.pending_consolidation || 0,
          last_consolidated_at: stats.last_consolidated_at || null,
        });
      }
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      if (!silent) setLoading(false);
    }
  };

  // Invalidated facts live in a separate archive, not the graph — fetch them via list.
  const loadInvalidated = useCallback(async () => {
    if (!currentBank) return;
    try {
      const resp: any = await client.listMemories(currentBank, {
        state: "invalidated",
        type: factType,
        limit: fetchLimit,
      });
      setInvalidatedRows(resp?.items ?? []);
    } catch {
      setInvalidatedRows([]);
    }
  }, [currentBank, factType, fetchLimit]);

  useEffect(() => {
    if (showInvalidated && viewMode === "table") {
      loadInvalidated();
    }
  }, [showInvalidated, viewMode, loadInvalidated]);

  // Table rows: live rows are graph-fed (filtered server-side); invalidated rows
  // come from the archive via list.
  const filteredTableRows = useMemo(() => {
    if (showInvalidated) return invalidatedRows;
    return data?.table_rows ?? [];
  }, [data, showInvalidated, invalidatedRows]);
  const hasActiveMemoryFilters =
    searchQuery.trim().length > 0 || tagFilters.length > 0 || selectedScope !== null;

  // Helper to get normalized link type
  const getLinkTypeCategory = (type: string | undefined): string => {
    if (!type) return "semantic";
    if (type === "semantic" || type === "temporal" || type === "entity") return type;
    if (["causes", "caused_by", "enables", "prevents"].includes(type)) return "causal";
    return "semantic";
  };

  // Convert data for the constellation (graph data is already filtered server-side)
  const graph2DData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const fullData = convertHindsightGraphData(data);

    // Filter links based on visible link types
    const links = fullData.links.filter((link) => {
      const category = getLinkTypeCategory(link.type);
      return visibleLinkTypes.has(category);
    });

    return { nodes: fullData.nodes, links };
  }, [data, visibleLinkTypes]);

  // Handle node click in graph - show in panel
  const handleGraphNodeClick = useCallback((node: GraphNode) => {
    // Open the memory dialog for the clicked node (same dialog the table/timeline use).
    setModalMemoryId(node.id);
  }, []);

  // Memoized color functions to prevent graph re-initialization
  // Uses brand colors: primary blue (#0074d9), teal (#009296), amber for entity, purple for causal
  const nodeColorFn = useCallback((node: GraphNode) => node.color || "#0074d9", []);

  // For observations, size nodes by their proof_count (number of source facts
  // consolidated into this observation) so "stronger" observations stand out.
  const observationSizeLookup = useMemo(() => {
    if (factType !== "observation" || !data?.table_rows) return null;
    const counts = new Map<string, number>();
    let max = 1;
    for (const row of data.table_rows as Array<{ id: string; proof_count?: number | null }>) {
      const c = row.proof_count ?? 1;
      counts.set(row.id, c);
      if (c > max) max = c;
    }
    return { counts, max };
  }, [factType, data]);

  // Recency heat — map each memory's chosen timestamp to 0..1 (oldest → newest).
  // Linear so the position on the gradient bar reflects the actual time fraction
  // between the oldest and newest memory in view.
  const recencyLookup = useMemo(() => {
    if (!data?.table_rows?.length) return null;
    type Row = {
      id: string;
      mentioned_at?: string | null;
      occurred_start?: string | null;
      occurred_end?: string | null;
    };
    const times = new Map<string, number>();
    let minT = Infinity;
    let maxT = -Infinity;
    for (const row of data.table_rows as Row[]) {
      const ts = row[recencyBasis];
      if (!ts) continue;
      const t = Date.parse(ts);
      if (Number.isNaN(t)) continue;
      times.set(row.id, t);
      if (t < minT) minT = t;
      if (t > maxT) maxT = t;
    }
    if (!Number.isFinite(minT) || !Number.isFinite(maxT) || maxT === minT) {
      return null;
    }
    return { times, minT, maxT };
  }, [data, recencyBasis]);

  const recencyHeatFn = useCallback(
    (node: GraphNode) => {
      if (!recencyLookup) return 0.5;
      const t = recencyLookup.times.get(node.id);
      if (t === undefined) return 0;
      return (t - recencyLookup.minT) / (recencyLookup.maxT - recencyLookup.minT);
    },
    [recencyLookup]
  );

  // Assign each distinct observation scope (exact tag set) a stable color from
  // the palette, in order of first appearance, for the "Group by scope" clusters.
  const scopeColorLookup = useMemo(() => {
    if (factType !== "observation" || !data?.table_rows) return null;
    const map = new Map<string, string>();
    let i = 0;
    for (const row of data.table_rows as Array<{ tags?: string[] }>) {
      const key = scopeKeyOf(row.tags);
      if (!map.has(key)) map.set(key, SCOPE_PALETTE[i++ % SCOPE_PALETTE.length]);
    }
    return map;
  }, [factType, data]);

  const scopeClusterKeyFn = useCallback(
    (node: GraphNode) => scopeKeyOf(node.metadata?.tags as string[] | undefined),
    []
  );
  const scopeClusterColorFn = useCallback(
    (key: string) => scopeColorLookup?.get(key) || "#0074d9",
    [scopeColorLookup]
  );
  const scopeClusterLabelFn = useCallback(
    (key: string) => {
      try {
        const tags = JSON.parse(key) as string[];
        return tags.length ? tags.map((tag) => `#${tag}`).join(" ") : t("scopeGlobal");
      } catch {
        return key;
      }
    },
    [t]
  );

  const observationNodeSizeFn = useCallback(
    (node: GraphNode) => {
      if (!observationSizeLookup) return 3;
      const c = observationSizeLookup.counts.get(node.id) ?? 1;
      // proof_count=1 matches the default memory dot size; grows with sqrt so
      // heavily-supported observations stand out without dwarfing the canvas.
      return 3 + Math.min(Math.sqrt(c - 1) * 2, 11);
    },
    [observationSizeLookup]
  );
  const linkColorFn = useCallback((link: any) => {
    if (link.type === "temporal") return "#009296"; // Brand teal
    if (link.type === "entity") return "#f59e0b"; // Amber
    if (
      link.type === "causes" ||
      link.type === "caused_by" ||
      link.type === "enables" ||
      link.type === "prevents"
    ) {
      return "#8b5cf6"; // Purple for causal
    }
    return "#0074d9"; // Brand primary blue for semantic
  }, []);

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [tagFilters, selectedScope]);

  // Resolve the active tag filter into (tags, tags_match) for the graph query.
  // A selected observation scope takes precedence and uses exact set-equality
  // matching (so scope [a] excludes [a, b]); otherwise the free-form tag filter
  // uses the default contains semantics. `null` scope means "no scope filter".
  const resolveTagQuery = useCallback(
    (scopeOverride?: string[] | null): { tags?: string[]; match?: string } => {
      const scope = scopeOverride === undefined ? selectedScope : scopeOverride;
      if (scope !== null) {
        return { tags: scope, match: "exact" };
      }
      return { tags: tagFilters.length > 0 ? tagFilters : undefined };
    },
    [selectedScope, tagFilters]
  );

  // Trigger text search on Enter key
  const executeSearch = () => {
    if (currentBank) {
      setCurrentPage(1);
      const { tags, match } = resolveTagQuery();
      loadData(undefined, searchQuery || undefined, tags, match);
    }
  };

  // Single auto-loader for the graph data. This deliberately replaces what used
  // to be two effects (mount/context + filter change) that BOTH fired on mount,
  // doubling the initial /api/graph request (see issue #2158). When the context
  // (factType/bank/document/chunk) changes we drop the now-meaningless scope
  // filter and feed the cleared value straight into the same reload, so the
  // scope reset never triggers a second fetch.
  const contextKeyRef = useRef<string | null>(null);
  const skipScopeResetReload = useRef(false);
  const lastAutoLoadSig = useRef<string | null>(null);
  useEffect(() => {
    if (!currentBank) return;
    // The previous run already issued the reload with scope=null; this run is
    // only the echo of our own setSelectedScope(null), so skip it.
    if (skipScopeResetReload.current) {
      skipScopeResetReload.current = false;
      return;
    }
    const contextKey = `${factType} ${currentBank} ${documentId ?? ""} ${chunkId ?? ""}`;
    const contextChanged = contextKeyRef.current !== contextKey;
    contextKeyRef.current = contextKey;

    let scope = selectedScope;
    if (contextChanged && selectedScope !== null) {
      scope = null;
      skipScopeResetReload.current = true;
      setSelectedScope(null);
    }
    const { tags, match } = resolveTagQuery(scope);
    // Collapse identical consecutive auto-loads into a single request. This makes
    // the effect idempotent, so React's mount-effect double-invoke (dev
    // StrictMode, and any redundant re-render) can't re-issue the same /api/graph
    // query. Manual reloads (search, load-more, consolidation poll) call loadData
    // directly and intentionally bypass this guard.
    const sig = JSON.stringify([contextKey, tags ?? null, match ?? null]);
    if (sig === lastAutoLoadSig.current) return;
    lastAutoLoadSig.current = sig;
    loadData(undefined, searchQuery || undefined, tags, match);
  }, [factType, currentBank, documentId, chunkId, tagFilters, selectedScope]);

  // Load the available observation scopes for the scope filter dropdown.
  const loadScopes = useCallback(async () => {
    if (!currentBank || factType !== "observation") {
      setScopes([]);
      return;
    }
    try {
      const resp = await client.listObservationScopes(currentBank);
      setScopes(resp.scopes ?? []);
    } catch {
      setScopes([]);
    }
  }, [currentBank, factType]);

  useEffect(() => {
    loadScopes();
  }, [loadScopes]);

  // While consolidation is in progress, poll so the observations + scopes (and
  // the "In Sync" badge) refresh live instead of showing a stale, one-shot read
  // (bank stats are also cached for up to 60s, so a single fetch can lag well
  // behind reality). Silent reloads avoid flashing the spinner. The effect only
  // restarts when consolidation starts/stops, not on every tick.
  const isConsolidating =
    factType === "observation" && (consolidationStatus?.pending_consolidation ?? 0) > 0;
  useEffect(() => {
    if (!isConsolidating || !currentBank) return;
    const id = setInterval(() => {
      const { tags, match } = resolveTagQuery();
      loadData(undefined, searchQuery || undefined, tags, match, true);
      loadScopes();
    }, 4000);
    return () => clearInterval(id);
  }, [isConsolidating, currentBank]);

  return (
    <div>
      {loading && !data ? (
        <div className="text-center py-12">
          <RefreshCw className="w-8 h-8 mx-auto mb-3 text-muted-foreground animate-spin" />
          <p className="text-muted-foreground">{t("loadingMemories")}</p>
        </div>
      ) : data && data.total_units === 0 && !hasActiveMemoryFilters ? (
        <div className="text-center py-20">
          <FileText className="w-10 h-10 mx-auto mb-4 text-muted-foreground/50" />
          <h3 className="text-base font-medium text-foreground mb-1">{t("noMemoriesYet")}</h3>
          {!documentId && !chunkId && (
            <>
              <p className="text-sm text-muted-foreground mb-6">{t("noMemoriesDescription")}</p>
              <Button
                variant="default"
                size="sm"
                className="gap-1.5"
                onClick={() => {
                  const btn = document.querySelector<HTMLButtonElement>("[data-add-document]");
                  btn?.click();
                }}
              >
                <Plus className="w-4 h-4" />
                {tAddDoc("addDocumentButton")}
              </Button>
            </>
          )}
        </div>
      ) : data ? (
        <>
          {/* Always visible filters */}
          {!compactMode && (
            <div className="mb-4 space-y-2">
              <div className="flex items-center gap-2">
                {/* Text search */}
                <div className="relative max-w-xs flex-1">
                  {loading ? (
                    <RefreshCw className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none animate-spin" />
                  ) : (
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                  )}
                  <Input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        executeSearch();
                      }
                    }}
                    placeholder={t("filterByTextPlaceholder")}
                    className="pl-8 h-9"
                  />
                </div>
                {/* Tag input. Setting a tag filter clears any selected scope
                    so the two filters never fight over the same query. */}
                <TagFilterInput
                  value={tagFilters}
                  onChange={(next) => {
                    if (next.length > 0) setSelectedScope(null);
                    setTagFilters(next);
                  }}
                  bankId={currentBank}
                />
                {/* Observation scope filter. Selecting a scope clears the
                    free-form tag filter (mutually exclusive). */}
                {factType === "observation" && scopes.length > 0 && (
                  <ObservationScopeFilter
                    scopes={scopes}
                    value={selectedScope}
                    onChange={(scope) => {
                      if (scope !== null) setTagFilters([]);
                      setSelectedScope(scope);
                    }}
                  />
                )}
              </div>
            </div>
          )}

          {compactMode ? (
            <div className="flex items-center justify-between mb-2 px-1">
              <div className="text-xs text-muted-foreground">{data.total_units} memories</div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (onExpandToggle) {
                    onExpandToggle();
                  } else {
                    setCompactMode(false);
                  }
                }}
                className="h-6 px-2 text-xs gap-1"
              >
                <Settings2 className="w-3 h-3" />
                Expand
              </Button>
            </div>
          ) : (
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                {compact && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      if (onExpandToggle) {
                        onExpandToggle();
                      } else {
                        setCompactMode(true);
                      }
                    }}
                    className="h-7 px-2 text-xs gap-1"
                  >
                    <Eye className="w-3 h-3" />
                    Compact
                  </Button>
                )}
                <div className="text-sm text-muted-foreground">
                  {hasActiveMemoryFilters ? (
                    t("matchingMemories", { count: filteredTableRows.length })
                  ) : data.table_rows?.length < data.total_units ? (
                    <span>
                      {t("showingMemories", {
                        shown: data.table_rows?.length ?? 0,
                        total: data.total_units,
                      })}
                      <button
                        onClick={() => {
                          const newLimit = Math.min(data.total_units, fetchLimit + 1000);
                          setFetchLimit(newLimit);
                          const { tags, match } = resolveTagQuery();
                          loadData(newLimit, searchQuery || undefined, tags, match);
                        }}
                        className="ml-2 text-primary hover:underline"
                      >
                        {t("loadMore")}
                      </button>
                    </span>
                  ) : (
                    t("totalMemories", { count: data.total_units })
                  )}
                </div>

                {/* Consolidation status for observations */}
                {factType === "observation" && consolidationStatus && (
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${
                      consolidationStatus.pending_consolidation === 0
                        ? "bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20"
                        : "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20"
                    }`}
                    title={
                      consolidationStatus.pending_consolidation === 0
                        ? consolidationStatus.last_consolidated_at
                          ? t("allConsolidatedWithDate", {
                              date: new Date(
                                consolidationStatus.last_consolidated_at
                              ).toLocaleString(),
                            })
                          : t("allConsolidated")
                        : t("pendingConsolidation", {
                            count: consolidationStatus.pending_consolidation,
                          })
                    }
                  >
                    {consolidationStatus.pending_consolidation === 0 ? (
                      <>
                        <CheckCircle className="w-3 h-3" />
                        {t("inSync")}
                      </>
                    ) : (
                      <>
                        <Clock className="w-3 h-3" />
                        {t("pendingCount", { count: consolidationStatus.pending_consolidation })}
                        <button
                          onClick={() =>
                            (() => {
                              const { tags, match } = resolveTagQuery();
                              loadData(fetchLimit, searchQuery || undefined, tags, match);
                            })()
                          }
                          disabled={loading}
                          className="ml-0.5 opacity-70 hover:opacity-100 disabled:opacity-40 transition-opacity"
                          title={t("refreshMemories")}
                        >
                          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
                        </button>
                      </>
                    )}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
                <button
                  onClick={() => setViewMode("constellation")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "constellation"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <ScatterChart className="w-4 h-4" />
                  {t("constellation")}
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "table"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <List className="w-4 h-4" />
                  {t("table")}
                </button>
                <button
                  onClick={() => setViewMode("timeline")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "timeline"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Calendar className="w-4 h-4" />
                  {t("timeline")}
                </button>
              </div>
            </div>
          )}

          {(compactMode || viewMode === "constellation") && (
            <div className="space-y-3">
              {/* Constellation controls — moved out of the old side panel to sit
                  inline above the graph, next to the view toggle / filters. */}
              {!compactMode && (
                <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                  {factType === "observation" && (
                    <div className="flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5 text-muted-foreground" />
                      <span className="text-xs font-medium text-muted-foreground">
                        {t("groupByScope")}
                      </span>
                      <Switch checked={groupByScope} onCheckedChange={setGroupByScope} />
                    </div>
                  )}
                  {!(factType === "observation" && groupByScope) && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-muted-foreground">
                        {t("colorBy")}
                      </span>
                      <Select
                        value={recencyBasis}
                        onValueChange={(v) => setRecencyBasis(v as RecencyBasis)}
                      >
                        <SelectTrigger className="h-8 w-44 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mentioned_at">{t("mentioned")}</SelectItem>
                          <SelectItem value="occurred_start">{t("occurredStart")}</SelectItem>
                          <SelectItem value="occurred_end">{t("occurredEnd")}</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-medium text-muted-foreground">
                      {t("linkTypes")}
                    </span>
                    {Object.entries({
                      semantic: "#0074d9",
                      temporal: "#009296",
                      entity: "#f59e0b",
                      causal: "#8b5cf6",
                    }).map(([type, color]) => (
                      <button
                        key={type}
                        type="button"
                        className="flex items-center gap-1.5"
                        onClick={() => toggleLinkType(type)}
                      >
                        <span
                          className="w-3 h-3 rounded-full"
                          style={{
                            backgroundColor: color,
                            opacity: visibleLinkTypes.has(type) ? 1 : 0.2,
                          }}
                        />
                        <span
                          className={`text-xs capitalize ${visibleLinkTypes.has(type) ? "text-foreground" : "text-muted-foreground line-through"}`}
                        >
                          {type}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="border border-border rounded-lg overflow-hidden">
                <Constellation
                  key={compactMode ? "compact" : "full"}
                  data={graph2DData}
                  height={compactMode ? 300 : 700}
                  onNodeClick={handleGraphNodeClick}
                  nodeColorFn={nodeColorFn}
                  linkColorFn={linkColorFn}
                  nodeSizeFn={factType === "observation" ? observationNodeSizeFn : undefined}
                  sizeLegendLabel={factType === "observation" ? t("sourceFactsLabel") : undefined}
                  clusterKeyFn={
                    factType === "observation" && groupByScope ? scopeClusterKeyFn : undefined
                  }
                  clusterColorFn={
                    factType === "observation" && groupByScope ? scopeClusterColorFn : undefined
                  }
                  clusterLabelFn={
                    factType === "observation" && groupByScope ? scopeClusterLabelFn : undefined
                  }
                  // When grouping by scope, color encodes scope (not recency), so
                  // suppress the recency heat to avoid a misleading legend.
                  nodeHeatFn={
                    !(factType === "observation" && groupByScope) && recencyLookup
                      ? recencyHeatFn
                      : undefined
                  }
                  heatLegendLabel={
                    !(factType === "observation" && groupByScope) && recencyLookup
                      ? t("recencyLabel", { basis: RECENCY_BASIS_LABEL[recencyBasis] })
                      : undefined
                  }
                  heatLegendEndpoints={
                    !(factType === "observation" && groupByScope) && recencyLookup
                      ? [
                          new Date(recencyLookup.minT).toISOString().slice(0, 10),
                          new Date(recencyLookup.maxT).toISOString().slice(0, 10),
                        ]
                      : undefined
                  }
                />
              </div>
            </div>
          )}

          {!compactMode && viewMode === "table" && (
            <div>
              {factType !== "observation" && (
                <div className="flex items-center gap-2 mb-3">
                  <div className="flex items-center gap-1 bg-muted rounded-lg p-1">
                    <button
                      onClick={() => {
                        setShowInvalidated(false);
                        setCurrentPage(1);
                      }}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                        !showInvalidated
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("filterActive")}
                    </button>
                    <button
                      onClick={() => {
                        setShowInvalidated(true);
                        setCurrentPage(1);
                      }}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                        showInvalidated
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("filterInvalidated")}
                    </button>
                  </div>
                  {showInvalidated && (
                    <span className="text-xs text-muted-foreground">{t("invalidatedHint")}</span>
                  )}
                </div>
              )}
              <div className="w-full">
                <div className="pb-4">
                  {filteredTableRows.length > 0 ? (
                    (() => {
                      const totalPages = Math.ceil(filteredTableRows.length / itemsPerPage);
                      const startIndex = (currentPage - 1) * itemsPerPage;
                      const endIndex = startIndex + itemsPerPage;
                      const paginatedRows = filteredTableRows.slice(startIndex, endIndex);

                      return (
                        <>
                          <Table className="table-fixed">
                            <TableHeader>
                              <TableRow>
                                <TableHead
                                  className={factType === "observation" ? "w-[35%]" : "w-[38%]"}
                                >
                                  {factType === "observation"
                                    ? t("columnObservation")
                                    : t("columnMemory")}
                                </TableHead>
                                <TableHead className="w-[15%]">{t("columnEntities")}</TableHead>
                                <TableHead className="w-[15%]">{t("columnTags")}</TableHead>
                                {factType === "observation" && (
                                  <TableHead className="w-[10%]">{t("columnSources")}</TableHead>
                                )}
                                <TableHead
                                  className={factType === "observation" ? "w-[12%]" : "w-[16%]"}
                                >
                                  {t("columnOccurred")}
                                </TableHead>
                                <TableHead
                                  className={factType === "observation" ? "w-[13%]" : "w-[16%]"}
                                >
                                  {t("columnMentioned")}
                                </TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {paginatedRows.map((row: any, idx: number) => {
                                const occurredDisplay = row.occurred_start
                                  ? new Date(row.occurred_start).toLocaleDateString("en-US", {
                                      month: "short",
                                      day: "numeric",
                                      year: "numeric",
                                    })
                                  : null;
                                const mentionedDisplay = row.mentioned_at
                                  ? new Date(row.mentioned_at).toLocaleDateString("en-US", {
                                      month: "short",
                                      day: "numeric",
                                      year: "numeric",
                                    })
                                  : null;

                                return (
                                  <TableRow
                                    key={row.id || idx}
                                    onClick={() => setModalMemoryId(row.id)}
                                    className="cursor-pointer hover:bg-muted/50"
                                  >
                                    <TableCell className="py-2">
                                      <div className="line-clamp-2 text-sm leading-snug text-foreground">
                                        {row.text}
                                      </div>
                                      {row.context && factType !== "observation" && (
                                        <div className="text-xs text-muted-foreground mt-0.5 truncate">
                                          {row.context}
                                        </div>
                                      )}
                                    </TableCell>
                                    <TableCell className="py-2">
                                      {row.entities ? (
                                        <div className="flex gap-1 flex-wrap">
                                          {row.entities
                                            .split(", ")
                                            .slice(0, 2)
                                            .map((entity: string, i: number) => (
                                              <span
                                                key={i}
                                                className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-medium"
                                              >
                                                {entity}
                                              </span>
                                            ))}
                                          {row.entities.split(", ").length > 2 && (
                                            <span className="text-[10px] text-muted-foreground">
                                              +{row.entities.split(", ").length - 2}
                                            </span>
                                          )}
                                        </div>
                                      ) : (
                                        <span className="text-xs text-muted-foreground">-</span>
                                      )}
                                    </TableCell>
                                    <TableCell className="py-2">
                                      {row.tags && row.tags.length > 0 ? (
                                        <div className="flex gap-1 flex-wrap">
                                          {(row.tags as string[])
                                            .slice(0, 2)
                                            .map((tag: string, i: number) => (
                                              <span
                                                key={i}
                                                className="text-[10px] px-1.5 py-0.5 rounded-md bg-amber-500/10 text-amber-700 border border-amber-500/20 font-medium font-mono"
                                              >
                                                #{tag}
                                              </span>
                                            ))}
                                          {row.tags.length > 2 && (
                                            <span className="text-[10px] text-muted-foreground">
                                              +{row.tags.length - 2}
                                            </span>
                                          )}
                                        </div>
                                      ) : (
                                        <span className="text-xs text-muted-foreground">-</span>
                                      )}
                                    </TableCell>
                                    {factType === "observation" && (
                                      <TableCell className="text-xs py-2 text-foreground">
                                        {row.proof_count ?? 1}
                                      </TableCell>
                                    )}
                                    <TableCell className="text-xs py-2 text-foreground">
                                      {occurredDisplay || (
                                        <span className="text-muted-foreground">-</span>
                                      )}
                                    </TableCell>
                                    <TableCell className="text-xs py-2 text-foreground">
                                      {mentionedDisplay || (
                                        <span className="text-muted-foreground">-</span>
                                      )}
                                    </TableCell>
                                  </TableRow>
                                );
                              })}
                            </TableBody>
                          </Table>

                          {/* Pagination Controls */}
                          {totalPages > 1 && (
                            <div className="flex items-center justify-between mt-3 pt-3 border-t">
                              <div className="text-xs text-muted-foreground">
                                {startIndex + 1}-{Math.min(endIndex, filteredTableRows.length)} of{" "}
                                {filteredTableRows.length}
                              </div>
                              <div className="flex items-center gap-1">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(1)}
                                  disabled={currentPage === 1}
                                  className="h-7 w-7 p-0"
                                >
                                  <ChevronsLeft className="h-3 w-3" />
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                  disabled={currentPage === 1}
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
                                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                  disabled={currentPage === totalPages}
                                  className="h-7 w-7 p-0"
                                >
                                  <ChevronRight className="h-3 w-3" />
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(totalPages)}
                                  disabled={currentPage === totalPages}
                                  className="h-7 w-7 p-0"
                                >
                                  <ChevronsRight className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      {hasActiveMemoryFilters ? t("noMemoriesMatchFilter") : t("noMemoriesFound")}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {!compactMode && viewMode === "timeline" && (
            <TimelineView
              data={data}
              filteredRows={filteredTableRows}
              bankId={currentBank || undefined}
              onMemoryClick={(id) => setModalMemoryId(id)}
            />
          )}
        </>
      ) : (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <div className="text-sm text-muted-foreground">{t("noDataAvailable")}</div>
          </div>
        </div>
      )}

      {/* Memory Detail Modal */}
      <MemoryDetailModal
        memoryId={modalMemoryId}
        onClose={() => setModalMemoryId(null)}
        onChanged={() => {
          if (showInvalidated) loadInvalidated();
        }}
      />
    </div>
  );
}

// Timeline View Component - Custom compact timeline with zoom and navigation
type Granularity = "year" | "month" | "week" | "day";

function TimelineView({
  data,
  filteredRows,
  bankId,
  onMemoryClick,
}: {
  data: any;
  filteredRows: any[];
  bankId?: string;
  onMemoryClick: (id: string) => void;
}) {
  const t = useTranslations("dataView");
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [currentIndex, setCurrentIndex] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Filter and sort items that have occurred_start dates (using filtered data)
  const { sortedItems, itemsWithoutDates } = useMemo(() => {
    if (!filteredRows || filteredRows.length === 0)
      return { sortedItems: [], itemsWithoutDates: [] };

    const withDates = filteredRows
      .filter((row: any) => row.occurred_start)
      .sort((a: any, b: any) => {
        const dateA = new Date(a.occurred_start).getTime();
        const dateB = new Date(b.occurred_start).getTime();
        return dateA - dateB;
      });

    const withoutDates = filteredRows.filter((row: any) => !row.occurred_start);

    return { sortedItems: withDates, itemsWithoutDates: withoutDates };
  }, [filteredRows]);

  // Group items by granularity
  const timelineGroups = useMemo(() => {
    if (sortedItems.length === 0) return [];

    const getGroupKey = (date: Date): string => {
      const year = date.getFullYear();
      const month = date.getMonth();
      const day = date.getDate();

      switch (granularity) {
        case "year":
          return `${year}`;
        case "month":
          return `${year}-${String(month + 1).padStart(2, "0")}`;
        case "week":
          const startOfWeek = new Date(date);
          startOfWeek.setDate(day - date.getDay());
          return `${startOfWeek.getFullYear()}-W${String(Math.ceil(startOfWeek.getDate() / 7)).padStart(2, "0")}-${String(startOfWeek.getMonth() + 1).padStart(2, "0")}-${String(startOfWeek.getDate()).padStart(2, "0")}`;
        case "day":
          return `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      }
    };

    const getGroupLabel = (key: string, date: Date): string => {
      switch (granularity) {
        case "year":
          return key;
        case "month":
          return date.toLocaleDateString("en-US", { year: "numeric", month: "short" });
        case "week":
          const endOfWeek = new Date(date);
          endOfWeek.setDate(date.getDate() + 6);
          return `${date.toLocaleDateString("en-US", { month: "short", day: "numeric" })} - ${endOfWeek.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}`;
        case "day":
          return date.toLocaleDateString("en-US", {
            weekday: "short",
            month: "short",
            day: "numeric",
            year: "numeric",
          });
      }
    };

    const groups: { [key: string]: { items: any[]; date: Date } } = {};
    sortedItems.forEach((row: any) => {
      const date = new Date(row.occurred_start);
      const key = getGroupKey(date);
      if (!groups[key]) {
        // For week, parse the start date from key
        let groupDate = date;
        if (granularity === "week") {
          const parts = key.split("-");
          groupDate = new Date(parseInt(parts[0]), parseInt(parts[2]) - 1, parseInt(parts[3]));
        }
        groups[key] = { items: [], date: groupDate };
      }
      groups[key].items.push(row);
    });

    return Object.entries(groups)
      .sort(([, a], [, b]) => a.date.getTime() - b.date.getTime())
      .map(([key, { items, date }]) => ({
        key,
        label: getGroupLabel(key, date),
        items,
        date,
      }));
  }, [sortedItems, granularity]);

  // Get date range info
  const dateRange = useMemo(() => {
    if (sortedItems.length === 0) return null;
    const first = new Date(sortedItems[0].occurred_start);
    const last = new Date(sortedItems[sortedItems.length - 1].occurred_start);
    return { first, last };
  }, [sortedItems]);

  // Navigation
  const scrollToGroup = (index: number) => {
    const clampedIndex = Math.max(0, Math.min(index, timelineGroups.length - 1));
    setCurrentIndex(clampedIndex);
    const element = document.getElementById(`timeline-group-${clampedIndex}`);
    element?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const zoomIn = () => {
    const levels: Granularity[] = ["year", "month", "week", "day"];
    const currentIdx = levels.indexOf(granularity);
    if (currentIdx < levels.length - 1) {
      setGranularity(levels[currentIdx + 1]);
    }
  };

  const zoomOut = () => {
    const levels: Granularity[] = ["year", "month", "week", "day"];
    const currentIdx = levels.indexOf(granularity);
    if (currentIdx > 0) {
      setGranularity(levels[currentIdx - 1]);
    }
  };

  if (sortedItems.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Calendar className="w-12 h-12 text-muted-foreground mb-3" />
        <div className="text-base font-medium text-foreground mb-1">{t("noTimelineData")}</div>
        <div className="text-xs text-muted-foreground text-center max-w-md">
          {t("noTimelineDataDescription")}
          {itemsWithoutDates.length > 0 && (
            <span className="block mt-1">
              {t("memoriesWithoutDatesInTable", { count: itemsWithoutDates.length })}
            </span>
          )}
        </div>
      </div>
    );
  }

  const formatDateTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const dateFormatted = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    const timeFormatted = date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    return { date: dateFormatted, time: timeFormatted };
  };

  const granularityLabels: Record<Granularity, string> = {
    year: t("granularityYear"),
    month: t("granularityMonth"),
    week: t("granularityWeek"),
    day: t("granularityDay"),
  };

  return (
    <div className="px-4">
      {/* Timeline */}
      <div>
        {/* Controls */}
        <div className="flex items-center justify-between mb-3 gap-4">
          <div className="text-xs text-muted-foreground">
            {t("timelineMemoriesCount", { count: sortedItems.length })}
            {itemsWithoutDates.length > 0 &&
              ` ${t("timelineWithoutDates", { count: itemsWithoutDates.length })}`}
            {dateRange && (
              <span className="ml-2 text-foreground">
                ({dateRange.first.toLocaleDateString("en-US", { month: "short", year: "numeric" })}{" "}
                → {dateRange.last.toLocaleDateString("en-US", { month: "short", year: "numeric" })})
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            {/* Zoom controls */}
            <div className="flex items-center border border-border rounded mr-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={zoomOut}
                disabled={granularity === "year"}
                className="h-7 w-7 p-0"
                title={t("zoomOut")}
              >
                <ZoomOut className="h-3 w-3" />
              </Button>
              <span className="text-[10px] px-2 min-w-[50px] text-center border-x border-border text-foreground">
                {granularityLabels[granularity]}
              </span>
              <Button
                variant="secondary"
                size="sm"
                onClick={zoomIn}
                disabled={granularity === "day"}
                className="h-7 w-7 p-0"
                title={t("zoomIn")}
              >
                <ZoomIn className="h-3 w-3" />
              </Button>
            </div>

            {/* Navigation controls */}
            <div className="flex items-center border border-border rounded">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(0)}
                disabled={timelineGroups.length <= 1}
                className="h-7 w-7 p-0"
                title={t("first")}
              >
                <ChevronsLeft className="h-3 w-3" />
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(currentIndex - 1)}
                disabled={currentIndex === 0}
                className="h-7 w-7 p-0"
                title={t("previous")}
              >
                <ChevronLeft className="h-3 w-3" />
              </Button>
              <span className="text-[10px] px-2 min-w-[60px] text-center border-x border-border text-foreground">
                {currentIndex + 1} / {timelineGroups.length}
              </span>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(currentIndex + 1)}
                disabled={currentIndex >= timelineGroups.length - 1}
                className="h-7 w-7 p-0"
                title={t("next")}
              >
                <ChevronRight className="h-3 w-3" />
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(timelineGroups.length - 1)}
                disabled={timelineGroups.length <= 1}
                className="h-7 w-7 p-0"
                title={t("last")}
              >
                <ChevronsRight className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </div>

        <div ref={timelineRef} className="relative max-h-[550px] overflow-y-auto pr-2">
          {/* Vertical line */}
          <div className="absolute left-[60px] top-0 bottom-0 w-0.5 bg-border" />

          {timelineGroups.map((group, groupIdx) => (
            <div key={group.key} id={`timeline-group-${groupIdx}`} className="mb-4">
              {/* Group header */}
              <div
                className="flex items-center mb-2 cursor-pointer hover:opacity-80"
                onClick={() => setCurrentIndex(groupIdx)}
              >
                <div className="w-[60px] text-right pr-3">
                  <span className="text-xs font-semibold text-primary">{group.label}</span>
                </div>
                <div className="w-2 h-2 rounded-full bg-primary z-10" />
                <span className="ml-2 text-[10px] text-muted-foreground">
                  {group.items.length}{" "}
                  {group.items.length === 1 ? t("timelineItem") : t("timelineItems")}
                </span>
              </div>

              {/* Items in this month */}
              <div className="space-y-1">
                {group.items.map((item: any, idx: number) => (
                  <div
                    key={item.id || idx}
                    onClick={() => onMemoryClick(item.id)}
                    className={`flex items-start cursor-pointer group ${"hover:opacity-80"}`}
                  >
                    {/* Date & Time */}
                    <div className="w-[60px] text-right pr-3 pt-1 flex-shrink-0">
                      <div className="text-[10px] text-muted-foreground">
                        {formatDateTime(item.occurred_start).date}
                      </div>
                      <div className="text-[9px] text-muted-foreground/70">
                        {formatDateTime(item.occurred_start).time}
                      </div>
                    </div>

                    {/* Connector dot */}
                    <div className="flex-shrink-0 pt-2">
                      <div
                        className={`w-1.5 h-1.5 rounded-full z-10 ${"bg-muted-foreground/50 group-hover:bg-primary"}`}
                      />
                    </div>

                    {/* Card */}
                    <div
                      className={`ml-3 flex-1 p-2 rounded border transition-colors ${"bg-card border-border hover:border-primary/50"}`}
                    >
                      <p className="text-xs text-foreground line-clamp-2 leading-relaxed">
                        {item.text}
                      </p>
                      {item.context && (
                        <p className="text-[10px] text-muted-foreground mt-1 truncate">
                          {item.context}
                        </p>
                      )}
                      {item.entities && (
                        <div className="flex gap-1 mt-1 flex-wrap">
                          {item.entities
                            .split(", ")
                            .slice(0, 3)
                            .map((entity: string, i: number) => (
                              <span
                                key={i}
                                className="text-[9px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-medium"
                              >
                                {entity}
                              </span>
                            ))}
                          {item.entities.split(", ").length > 3 && (
                            <span className="text-[9px] text-muted-foreground">
                              +{item.entities.split(", ").length - 3}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
