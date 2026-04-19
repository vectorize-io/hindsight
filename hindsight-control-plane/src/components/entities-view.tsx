"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  List,
  ScatterChart,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Constellation } from "./constellation";
import { convertHindsightGraphData, GraphNode } from "./graph-2d";

type EntityGraphResponse = Awaited<ReturnType<typeof client.getEntityGraph>>;

interface Entity {
  id: string;
  canonical_name: string;
  mention_count: number;
  first_seen?: string;
  last_seen?: string;
  metadata?: Record<string, any>;
}

type EntityDetail = Entity;

type ViewMode = "relations" | "list";

const ITEMS_PER_PAGE = 50;

export function EntitiesView() {
  const { currentBank } = useBank();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("relations");
  const [graphData, setGraphData] = useState<EntityGraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [total, setTotal] = useState(0);

  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);
  const offset = (currentPage - 1) * ITEMS_PER_PAGE;

  const loadEntities = async (page: number = 1) => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const pageOffset = (page - 1) * ITEMS_PER_PAGE;
      const result = await client.listEntities({
        bank_id: currentBank,
        limit: ITEMS_PER_PAGE,
        offset: pageOffset,
      });
      setEntities(result.items || []);
      setTotal(result.total || 0);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoading(false);
    }
  };

  const loadEntityDetail = async (entityId: string) => {
    if (!currentBank) return;

    setLoadingDetail(true);
    try {
      const result: any = await client.getEntity(entityId, currentBank);
      setSelectedEntity(result);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoadingDetail(false);
    }
  };

  // Handle page change
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
    loadEntities(newPage);
  };

  const loadGraph = useCallback(async () => {
    if (!currentBank) return;
    setGraphLoading(true);
    try {
      const result = await client.getEntityGraph({
        bank_id: currentBank,
        limit: 2000,
        min_count: 1,
      });
      setGraphData(result);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setGraphLoading(false);
    }
  }, [currentBank]);

  useEffect(() => {
    if (currentBank) {
      setCurrentPage(1);
      loadEntities(1);
      setSelectedEntity(null);
      setGraphData(null);
    }
  }, [currentBank]);

  useEffect(() => {
    if (viewMode === "relations" && currentBank && !graphData && !graphLoading) {
      loadGraph();
    }
  }, [viewMode, currentBank, graphData, graphLoading, loadGraph]);

  const constellationData = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] };
    return convertHindsightGraphData(graphData);
  }, [graphData]);

  // Sum co-occurrence counts (edge weights) per entity, then map to a dot
  // radius. Log scaling keeps hubs big without letting them dwarf everything.
  const nodeWeights = useMemo(() => {
    const weights = new Map<string, number>();
    for (const link of constellationData.links) {
      const w = typeof link.weight === "number" && link.weight > 0 ? link.weight : 1;
      weights.set(link.source, (weights.get(link.source) || 0) + w);
      weights.set(link.target, (weights.get(link.target) || 0) + w);
    }
    return weights;
  }, [constellationData]);

  const maxNodeWeight = useMemo(() => {
    let max = 1;
    for (const w of nodeWeights.values()) if (w > max) max = w;
    return max;
  }, [nodeWeights]);

  const nodeSizeFn = useCallback(
    (node: GraphNode) => {
      const w = nodeWeights.get(node.id) || 0;
      // 3px (isolated) → 14px (the hub); sqrt flattens the long tail.
      const t = Math.sqrt(w / maxNodeWeight);
      return 3 + t * 11;
    },
    [nodeWeights, maxNodeWeight]
  );

  // Recency heat per entity — the most recent co-occurrence across any of its
  // edges. Lets color encode "fresh vs stale" while size encodes co-occurrence
  // volume, so the two axes stay orthogonal.
  const recencyLookup = useMemo(() => {
    const edges = graphData?.edges || [];
    if (!edges.length) return null;
    const times = new Map<string, number>();
    let minT = Infinity;
    let maxT = -Infinity;
    for (const e of edges) {
      const iso = e.data.lastCooccurred;
      if (!iso) continue;
      const t = Date.parse(iso);
      if (Number.isNaN(t)) continue;
      for (const id of [e.data.source, e.data.target]) {
        const prev = times.get(id);
        if (prev === undefined || t > prev) times.set(id, t);
      }
      if (t < minT) minT = t;
      if (t > maxT) maxT = t;
    }
    if (!Number.isFinite(minT) || !Number.isFinite(maxT) || maxT === minT) return null;
    return { times, minT, maxT };
  }, [graphData]);

  const nodeHeatFn = useCallback(
    (node: GraphNode) => {
      if (!recencyLookup) return 0.5;
      const t = recencyLookup.times.get(node.id);
      if (t === undefined) return 0;
      return (t - recencyLookup.minT) / (recencyLookup.maxT - recencyLookup.minT);
    },
    [recencyLookup]
  );

  const handleConstellationNodeClick = useCallback(
    (node: GraphNode) => {
      loadEntityDetail(node.id);
    },
    [currentBank]
  );

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "N/A";
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div>
      {/* View mode toggle — same segmented control as memories page */}
      <div className="mb-4 flex items-center justify-end">
        <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
          <button
            onClick={() => setViewMode("relations")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
              viewMode === "relations"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <ScatterChart className="w-4 h-4" />
            Relations
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
              viewMode === "list"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <List className="w-4 h-4" />
            List
          </button>
        </div>
      </div>

      {viewMode === "relations" && (
        <div className="border border-border rounded-lg overflow-hidden">
          {graphLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="text-4xl mb-2">...</div>
                <div className="text-sm text-muted-foreground">Loading entity graph...</div>
              </div>
            </div>
          ) : constellationData.nodes.length > 0 ? (
            <Constellation
              data={constellationData}
              height={700}
              onNodeClick={handleConstellationNodeClick}
              nodeSizeFn={nodeSizeFn}
              nodeHeatFn={recencyLookup ? nodeHeatFn : undefined}
              heatLegendLabel={recencyLookup ? "recency · last co-occurrence" : undefined}
              heatLegendEndpoints={
                recencyLookup
                  ? [
                      new Date(recencyLookup.minT).toISOString().slice(0, 10),
                      new Date(recencyLookup.maxT).toISOString().slice(0, 10),
                    ]
                  : undefined
              }
              sizeLegendLabel="co-occurrences"
              compactLabels
            />
          ) : (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="text-sm text-muted-foreground">No entity co-occurrences yet</div>
                <div className="text-xs text-muted-foreground mt-1">
                  Two entities co-occur when they appear in the same memory.
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Entity List */}
      {viewMode === "list" && (
        <div>
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="text-4xl mb-2">...</div>
                <div className="text-sm text-muted-foreground">Loading entities...</div>
              </div>
            </div>
          ) : entities.length > 0 ? (
            <>
              <div className="mb-4 text-sm text-muted-foreground">
                {total} {total === 1 ? "entity" : "entities"}
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Mentions</TableHead>
                      <TableHead>First Seen</TableHead>
                      <TableHead>Last Seen</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {entities.map((entity) => (
                      <TableRow
                        key={entity.id}
                        onClick={() => loadEntityDetail(entity.id)}
                        className={`cursor-pointer hover:bg-muted/50 ${
                          selectedEntity?.id === entity.id ? "bg-primary/10" : ""
                        }`}
                      >
                        <TableCell className="font-medium text-card-foreground">
                          {entity.canonical_name}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {entity.mention_count}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {formatDate(entity.first_seen)}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {formatDate(entity.last_seen)}
                        </TableCell>
                      </TableRow>
                    ))}
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
            </>
          ) : (
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="text-4xl mb-2">...</div>
                <div className="text-sm text-muted-foreground">No entities found</div>
                <div className="text-xs text-muted-foreground mt-1">
                  Entities are extracted from facts when memories are added.
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Entity Detail Panel - Fixed overlay */}
      {selectedEntity && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
          <div className="p-5">
            {/* Header */}
            <div className="flex justify-between items-center mb-6 pb-4 border-b border-border">
              <div>
                <h3 className="text-xl font-bold text-card-foreground">
                  {selectedEntity.canonical_name}
                </h3>
                <p className="text-sm text-muted-foreground mt-1">Entity details</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedEntity(null)}
                className="h-8 w-8 p-0"
              >
                <span className="text-lg">x</span>
              </Button>
            </div>

            <div className="space-y-5">
              {/* Entity Info */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Mentions
                  </div>
                  <div className="text-lg font-semibold text-card-foreground">
                    {selectedEntity.mention_count}
                  </div>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    First Seen
                  </div>
                  <div className="text-sm font-medium text-card-foreground">
                    {formatDate(selectedEntity.first_seen)}
                  </div>
                </div>
              </div>

              {/* ID */}
              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                  Entity ID
                </div>
                <code className="text-xs font-mono break-all text-muted-foreground">
                  {selectedEntity.id}
                </code>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
