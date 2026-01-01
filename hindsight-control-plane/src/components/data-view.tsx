"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Copy,
  Check,
  Calendar,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Settings2,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { MemoryDetailPanel } from "./memory-detail-panel";
import { Graph2D, convertHindsightGraphData, GraphNode } from "./graph-2d";

type FactType = "world" | "experience" | "opinion";
type ViewMode = "graph" | "table" | "timeline";

interface DataViewProps {
  factType: FactType;
}

export function DataView({ factType }: DataViewProps) {
  const { currentBank } = useBank();
  const [viewMode, setViewMode] = useState<ViewMode>("graph");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedGraphNode, setSelectedGraphNode] = useState<any>(null);
  const [selectedTableMemory, setSelectedTableMemory] = useState<any>(null);
  const itemsPerPage = 100;

  // Graph controls state
  const [showLabels, setShowLabels] = useState(true);
  const [maxNodes, setMaxNodes] = useState<number | undefined>(undefined);
  const [showControlPanel, setShowControlPanel] = useState(true);
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

  // Esc key handler to deselect graph node
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedGraphNode) {
        setSelectedGraphNode(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedGraphNode]);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(text);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const graphData: any = await client.getGraph({
        bank_id: currentBank,
        type: factType,
      });
      setData(graphData);
    } catch (error) {
      console.error("Error loading data:", error);
      alert(`Error loading ${factType} data: ` + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // Filter table rows based on search query (text only)
  const filteredTableRows = useMemo(() => {
    if (!data?.table_rows) return [];
    if (!searchQuery) return data.table_rows;

    const query = searchQuery.toLowerCase();
    return data.table_rows.filter((row: any) => row.text?.toLowerCase().includes(query));
  }, [data, searchQuery]);

  // Get filtered node IDs for graph filtering
  const filteredNodeIds = useMemo(() => {
    return new Set(filteredTableRows.map((row: any) => row.id));
  }, [filteredTableRows]);

  // Helper to get normalized link type
  const getLinkTypeCategory = (type: string | undefined): string => {
    if (!type) return "semantic";
    if (type === "semantic" || type === "temporal" || type === "entity") return type;
    if (["causes", "caused_by", "enables", "prevents"].includes(type)) return "causal";
    return "semantic";
  };

  // Convert data for Graph2D with filtering
  const graph2DData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const fullData = convertHindsightGraphData(data);

    let nodes = fullData.nodes;
    let links = fullData.links;

    // Filter nodes based on search query
    if (searchQuery) {
      const filteredNodes = fullData.nodes.filter((node) => filteredNodeIds.has(node.id));
      const filteredNodeIdSet = new Set(filteredNodes.map((n) => n.id));
      nodes = filteredNodes;
      links = fullData.links.filter(
        (link) => filteredNodeIdSet.has(link.source) && filteredNodeIdSet.has(link.target)
      );
    }

    // Filter links based on visible link types
    links = links.filter((link) => {
      const category = getLinkTypeCategory(link.type);
      return visibleLinkTypes.has(category);
    });

    return { nodes, links };
  }, [data, searchQuery, filteredNodeIds, visibleLinkTypes]);

  // Calculate link stats for display
  const linkStats = useMemo(() => {
    let semantic = 0,
      temporal = 0,
      entity = 0,
      causal = 0,
      total = 0;
    const otherTypes: Record<string, number> = {};
    graph2DData.links.forEach((l) => {
      total++;
      const type = l.type || "unknown";
      if (type === "semantic") semantic++;
      else if (type === "temporal") temporal++;
      else if (type === "entity") entity++;
      else if (
        type === "causes" ||
        type === "caused_by" ||
        type === "enables" ||
        type === "prevents"
      )
        causal++;
      else {
        otherTypes[type] = (otherTypes[type] || 0) + 1;
      }
    });
    return { semantic, temporal, entity, causal, total, otherTypes };
  }, [graph2DData]);

  // Handle node click in graph - show in panel
  const handleGraphNodeClick = useCallback(
    (node: GraphNode) => {
      const nodeData = data?.table_rows?.find((row: any) => row.id === node.id);
      if (nodeData) {
        setSelectedGraphNode(nodeData);
      }
    },
    [data]
  );

  // Memoized color functions to prevent graph re-initialization
  // Uses brand colors: primary blue (#0074d9), teal (#009296), amber for entity, purple for causal
  const nodeColorFn = useCallback((node: GraphNode) => node.color || "#0074d9", []);
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

  // Reset to first page when search query changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery]);

  // Auto-load data when component mounts or factType/currentBank changes
  useEffect(() => {
    if (currentBank) {
      loadData();
    }
  }, [factType, currentBank]);

  // Enforce 50 node limit to prevent UI instability, default to 20 or max whichever is smaller
  useEffect(() => {
    if (data && maxNodes === undefined) {
      if (graph2DData.nodes.length > 50) {
        // Always set maxNodes to 20 when we have >50 nodes (never leave as undefined)
        setMaxNodes(20);
      } else if (graph2DData.nodes.length > 20) {
        setMaxNodes(20);
      }
      // If ≤20 nodes, leave maxNodes undefined to show all
    }
  }, [data, graph2DData.nodes.length, maxNodes]);

  return (
    <div>
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">⏳</div>
            <div className="text-sm text-muted-foreground">Loading memories...</div>
          </div>
        </div>
      ) : data ? (
        <>
          {/* Always visible filter */}
          <div className="mb-4">
            <Input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter memories by text..."
              className="max-w-md"
            />
          </div>

          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-muted-foreground">
              {searchQuery
                ? `${filteredTableRows.length} of ${data.total_units} memories`
                : `${data.total_units} total memories`}
            </div>
            <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
              <button
                onClick={() => setViewMode("graph")}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === "graph"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Graph View
              </button>
              <button
                onClick={() => setViewMode("table")}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === "table"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Table View
              </button>
              <button
                onClick={() => setViewMode("timeline")}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === "timeline"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Timeline View
              </button>
            </div>
          </div>

          {viewMode === "graph" && (
            <div className="flex gap-0">
              {/* Graph */}
              <div className="flex-1 min-w-0">
                <Graph2D
                  data={graph2DData}
                  height={700}
                  showLabels={showLabels}
                  onNodeClick={handleGraphNodeClick}
                  maxNodes={maxNodes}
                  nodeColorFn={nodeColorFn}
                  linkColorFn={linkColorFn}
                />
              </div>

              {/* Right Toggle Button */}
              <button
                onClick={() => setShowControlPanel(!showControlPanel)}
                className="flex-shrink-0 w-5 h-[700px] bg-transparent hover:bg-muted/50 flex items-center justify-center transition-colors"
                title={showControlPanel ? "Hide panel" : "Show panel"}
              >
                {showControlPanel ? (
                  <ChevronRight className="w-3 h-3 text-muted-foreground/60" />
                ) : (
                  <ChevronLeft className="w-3 h-3 text-muted-foreground/60" />
                )}
              </button>

              {/* Right Panel - Legend/Controls OR Memory Details */}
              <div
                className={`${showControlPanel ? "w-80" : "w-0"} transition-all duration-300 overflow-hidden flex-shrink-0`}
              >
                <div className="w-80 h-[700px] bg-card border-l border-border overflow-y-auto">
                  {selectedGraphNode ? (
                    /* Memory Detail View */
                    <MemoryDetailPanel
                      memory={selectedGraphNode}
                      onClose={() => setSelectedGraphNode(null)}
                      inPanel
                    />
                  ) : (
                    /* Legend & Controls View */
                    <div className="p-4 space-y-5">
                      {/* Legend & Stats */}
                      <div>
                        <h3 className="text-sm font-semibold mb-3 text-foreground">Graph</h3>
                        <div className="space-y-2">
                          {/* Nodes */}
                          <div className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              <div
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: "#0074d9" }}
                              />
                              <span className="text-foreground">Nodes</span>
                            </div>
                            <span className="font-mono text-foreground">
                              {Math.min(
                                maxNodes ?? graph2DData.nodes.length,
                                graph2DData.nodes.length
                              )}
                              /{graph2DData.nodes.length}
                            </span>
                          </div>

                          <div className="text-xs font-medium text-muted-foreground mt-2 mb-1">
                            Links ({linkStats.total}){" "}
                            <span className="text-muted-foreground/60">· click to filter</span>
                          </div>
                          <button
                            onClick={() => toggleLinkType("semantic")}
                            className={`w-full flex items-center justify-between text-sm px-2 py-1 rounded transition-all ${
                              visibleLinkTypes.has("semantic")
                                ? "hover:bg-muted"
                                : "opacity-40 hover:opacity-60"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-0.5 bg-[#0074d9]" />
                              <span className="text-foreground">Semantic</span>
                            </div>
                            <span
                              className={`font-mono ${linkStats.semantic === 0 ? "text-destructive" : "text-foreground"}`}
                            >
                              {linkStats.semantic}
                            </span>
                          </button>
                          <button
                            onClick={() => toggleLinkType("temporal")}
                            className={`w-full flex items-center justify-between text-sm px-2 py-1 rounded transition-all ${
                              visibleLinkTypes.has("temporal")
                                ? "hover:bg-muted"
                                : "opacity-40 hover:opacity-60"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-0.5 bg-[#009296]" />
                              <span className="text-foreground">Temporal</span>
                            </div>
                            <span
                              className={`font-mono ${linkStats.temporal === 0 ? "text-destructive" : "text-foreground"}`}
                            >
                              {linkStats.temporal}
                            </span>
                          </button>
                          <button
                            onClick={() => toggleLinkType("entity")}
                            className={`w-full flex items-center justify-between text-sm px-2 py-1 rounded transition-all ${
                              visibleLinkTypes.has("entity")
                                ? "hover:bg-muted"
                                : "opacity-40 hover:opacity-60"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-0.5 bg-[#f59e0b]" />
                              <span className="text-foreground">Entity</span>
                            </div>
                            <span className="font-mono text-foreground">{linkStats.entity}</span>
                          </button>
                          <button
                            onClick={() => toggleLinkType("causal")}
                            className={`w-full flex items-center justify-between text-sm px-2 py-1 rounded transition-all ${
                              visibleLinkTypes.has("causal")
                                ? "hover:bg-muted"
                                : "opacity-40 hover:opacity-60"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-0.5 bg-[#8b5cf6]" />
                              <span className="text-foreground">Causal</span>
                            </div>
                            <span
                              className={`font-mono ${linkStats.causal === 0 ? "text-muted-foreground" : "text-foreground"}`}
                            >
                              {linkStats.causal}
                            </span>
                          </button>
                          {Object.entries(linkStats.otherTypes || {}).map(([type, count]) => (
                            <div key={type} className="flex items-center justify-between text-sm">
                              <span className="text-muted-foreground capitalize ml-6">{type}</span>
                              <span className="font-mono text-muted-foreground">
                                {count as number}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="border-t border-border" />

                      {/* Controls Section */}
                      <div>
                        <h3 className="text-sm font-semibold mb-3 text-foreground">Display</h3>
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="show-labels" className="text-sm text-foreground">
                              Show labels
                            </Label>
                            <Switch
                              id="show-labels"
                              checked={showLabels}
                              onCheckedChange={setShowLabels}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="border-t border-border" />

                      {/* Limits Section */}
                      <div>
                        <h3 className="text-sm font-semibold mb-3 text-foreground">Performance</h3>
                        <div className="space-y-4">
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <Label className="text-sm text-foreground">Max nodes</Label>
                              <span className="text-xs text-muted-foreground">
                                {graph2DData.nodes.length > 50
                                  ? `${maxNodes ?? 50} / ${graph2DData.nodes.length}`
                                  : `${maxNodes ?? "All"} / ${graph2DData.nodes.length}`}
                              </span>
                            </div>
                            <Slider
                              value={[
                                graph2DData.nodes.length > 50
                                  ? maxNodes || 20
                                  : maxNodes || Math.min(graph2DData.nodes.length, 20),
                              ]}
                              min={10}
                              max={Math.min(Math.max(graph2DData.nodes.length, 10), 50)}
                              step={10}
                              onValueChange={([v]) => {
                                const effectiveMax = Math.min(graph2DData.nodes.length, 50);
                                // If we have >50 nodes, never allow "All" (undefined), cap at 50
                                if (graph2DData.nodes.length > 50) {
                                  setMaxNodes(v);
                                } else {
                                  // Original behavior for ≤50 nodes: allow "All" when slider reaches max
                                  setMaxNodes(v >= effectiveMax ? undefined : v);
                                }
                              }}
                              className="w-full"
                            />
                          </div>
                          <p className="text-xs text-muted-foreground">
                            All links between visible nodes are shown.
                            {graph2DData.nodes.length > 50 && (
                              <span className="block text-amber-600 dark:text-amber-400 mt-1">
                                ⚠️ Limited to 50 nodes for performance. Total:{" "}
                                {graph2DData.nodes.length}
                              </span>
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="border-t border-border" />

                      {/* Hint */}
                      <div className="text-xs text-muted-foreground/60 text-center pt-2">
                        Click a node to see details
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {viewMode === "table" && (
            <div>
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
                          <div className="border rounded-lg overflow-hidden">
                            <Table className="table-fixed">
                              <TableHeader>
                                <TableRow className="bg-muted/50">
                                  <TableHead className="w-[45%]">Memory</TableHead>
                                  <TableHead className="w-[20%]">Entities</TableHead>
                                  <TableHead className="w-[15%]">Occurred</TableHead>
                                  <TableHead className="w-[15%]">Mentioned</TableHead>
                                  <TableHead className="w-[5%]"></TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {paginatedRows.map((row: any, idx: number) => {
                                  const occurredDisplay = row.occurred_start
                                    ? new Date(row.occurred_start).toLocaleDateString("en-US", {
                                        month: "short",
                                        day: "numeric",
                                      })
                                    : null;
                                  const mentionedDisplay = row.mentioned_at
                                    ? new Date(row.mentioned_at).toLocaleDateString("en-US", {
                                        month: "short",
                                        day: "numeric",
                                      })
                                    : null;

                                  return (
                                    <TableRow
                                      key={row.id || idx}
                                      onClick={() => setSelectedTableMemory(row)}
                                      className={`cursor-pointer hover:bg-muted/50 ${
                                        selectedTableMemory?.id === row.id ? "bg-primary/10" : ""
                                      }`}
                                    >
                                      <TableCell className="py-2">
                                        <div className="line-clamp-2 text-sm leading-snug text-foreground">
                                          {row.text}
                                        </div>
                                        {row.context && (
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
                                      <TableCell className="py-2">
                                        <Button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            copyToClipboard(row.id);
                                          }}
                                          size="sm"
                                          variant="secondary"
                                          className="h-6 w-6 p-0"
                                          title="Copy ID"
                                        >
                                          {copiedId === row.id ? (
                                            <Check className="h-3 w-3 text-green-600" />
                                          ) : (
                                            <Copy className="h-3 w-3" />
                                          )}
                                        </Button>
                                      </TableCell>
                                    </TableRow>
                                  );
                                })}
                              </TableBody>
                            </Table>
                          </div>

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
                      {data.table_rows?.length > 0
                        ? "No memories match your filter"
                        : "No memories found"}
                    </div>
                  )}
                </div>
              </div>

              {/* Memory Detail Panel for Table View - Fixed on Right */}
              {selectedTableMemory && (
                <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
                  <MemoryDetailPanel
                    memory={selectedTableMemory}
                    onClose={() => setSelectedTableMemory(null)}
                    inPanel
                  />
                </div>
              )}
            </div>
          )}

          {viewMode === "timeline" && <TimelineView data={data} filteredRows={filteredTableRows} />}
        </>
      ) : (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </div>
      )}
    </div>
  );
}

// Timeline View Component - Custom compact timeline with zoom and navigation
type Granularity = "year" | "month" | "week" | "day";

function TimelineView({ data, filteredRows }: { data: any; filteredRows: any[] }) {
  const [selectedItem, setSelectedItem] = useState<any>(null);
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
      .sort(([a], [b]) => a.localeCompare(b))
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
        <div className="text-base font-medium text-foreground mb-1">No Timeline Data</div>
        <div className="text-xs text-muted-foreground text-center max-w-md">
          No memories have occurred_at dates.
          {itemsWithoutDates.length > 0 && (
            <span className="block mt-1">
              {itemsWithoutDates.length} memories without dates in Table View.
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
    year: "Year",
    month: "Month",
    week: "Week",
    day: "Day",
  };

  return (
    <div className="px-4">
      {/* Timeline */}
      <div>
        {/* Controls */}
        <div className="flex items-center justify-between mb-3 gap-4">
          <div className="text-xs text-muted-foreground">
            {sortedItems.length} memories
            {itemsWithoutDates.length > 0 && ` · ${itemsWithoutDates.length} without dates`}
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
                title="Zoom out"
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
                title="Zoom in"
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
                title="First"
              >
                <ChevronsLeft className="h-3 w-3" />
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(currentIndex - 1)}
                disabled={currentIndex === 0}
                className="h-7 w-7 p-0"
                title="Previous"
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
                title="Next"
              >
                <ChevronRight className="h-3 w-3" />
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => scrollToGroup(timelineGroups.length - 1)}
                disabled={timelineGroups.length <= 1}
                className="h-7 w-7 p-0"
                title="Last"
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
                  {group.items.length} {group.items.length === 1 ? "item" : "items"}
                </span>
              </div>

              {/* Items in this month */}
              <div className="space-y-1">
                {group.items.map((item: any, idx: number) => (
                  <div
                    key={item.id || idx}
                    onClick={() => setSelectedItem(item)}
                    className={`flex items-start cursor-pointer group ${
                      selectedItem?.id === item.id ? "opacity-100" : "hover:opacity-80"
                    }`}
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
                        className={`w-1.5 h-1.5 rounded-full z-10 ${
                          selectedItem?.id === item.id
                            ? "bg-primary"
                            : "bg-muted-foreground/50 group-hover:bg-primary"
                        }`}
                      />
                    </div>

                    {/* Card */}
                    <div
                      className={`ml-3 flex-1 p-2 rounded border transition-colors ${
                        selectedItem?.id === item.id
                          ? "bg-primary/10 border-primary"
                          : "bg-card border-border hover:border-primary/50"
                      }`}
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

      {/* Detail Panel - Fixed on Right */}
      {selectedItem && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
          <MemoryDetailPanel memory={selectedItem} onClose={() => setSelectedItem(null)} inPanel />
        </div>
      )}
    </div>
  );
}
