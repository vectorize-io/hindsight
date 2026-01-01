"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Graph2D, convertHindsightGraphData, GraphNode } from "./graph-2d";

interface Entity {
  id: string;
  canonical_name: string;
  mention_count: number;
  first_seen?: string;
  last_seen?: string;
  metadata?: Record<string, any>;
}

interface EntityDetail extends Entity {
  observations: Array<{
    text: string;
    mentioned_at?: string;
  }>;
}

export function EntitiesView() {
  const { currentBank } = useBank();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [entityGraphData, setEntityGraphData] = useState<any>(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [showGraph, setShowGraph] = useState(false);

  const loadEntities = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const result: any = await client.listEntities({
        bank_id: currentBank,
        limit: 100,
      });
      setEntities(result.items || []);
    } catch (error) {
      console.error("Error loading entities:", error);
      alert("Error loading entities: " + (error as Error).message);
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
      console.error("Error loading entity detail:", error);
      alert("Error loading entity detail: " + (error as Error).message);
    } finally {
      setLoadingDetail(false);
    }
  };

  const loadEntityGraph = async (entityName: string) => {
    if (!currentBank) return;

    setLoadingGraph(true);
    try {
      // Get all graph data and filter for the entity
      const graphData: any = await client.getGraph({
        bank_id: currentBank,
      });

      // Filter for memories that mention this entity
      const entityMemories =
        graphData.table_rows?.filter((row: any) =>
          row.entities?.toLowerCase().includes(entityName.toLowerCase())
        ) || [];

      // Get the IDs of memories that mention this entity
      const entityMemoryIds = new Set(entityMemories.map((row: any) => row.id));

      // Filter nodes and edges for the entity context
      const filteredNodes =
        graphData.nodes?.filter((node: any) => entityMemoryIds.has(node.data.id)) || [];

      const filteredNodeIds = new Set(filteredNodes.map((n: any) => n.data.id));
      const filteredEdges =
        graphData.edges?.filter(
          (edge: any) =>
            filteredNodeIds.has(edge.data.source) && filteredNodeIds.has(edge.data.target)
        ) || [];

      setEntityGraphData({
        nodes: filteredNodes,
        edges: filteredEdges,
        table_rows: entityMemories,
      });
    } catch (error) {
      console.error("Error loading entity graph:", error);
      alert("Error loading entity graph: " + (error as Error).message);
    } finally {
      setLoadingGraph(false);
    }
  };

  const regenerateObservations = async () => {
    if (!currentBank || !selectedEntity) return;

    setRegenerating(true);
    try {
      await client.regenerateEntityObservations(selectedEntity.id, currentBank);
      // Reload entity detail to show new observations
      await loadEntityDetail(selectedEntity.id);
    } catch (error) {
      console.error("Error regenerating observations:", error);
      alert("Error regenerating observations: " + (error as Error).message);
    } finally {
      setRegenerating(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadEntities();
      setSelectedEntity(null);
    }
  }, [currentBank]);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "N/A";
    return new Date(dateStr).toLocaleDateString();
  };

  // Convert entity graph data for Graph2D component
  const graph2DData = useMemo(() => {
    if (!entityGraphData) return { nodes: [], links: [] };
    return convertHindsightGraphData(entityGraphData);
  }, [entityGraphData]);

  // Color functions for graph
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

  return (
    <div>
      {/* Entity List */}
      <div>
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="text-4xl mb-2">⏳</div>
              <div className="text-sm text-muted-foreground">Loading entities...</div>
            </div>
          </div>
        ) : entities.length > 0 ? (
          <>
            <div className="mb-4 text-sm text-muted-foreground">{entities.length} entities</div>
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
                      onClick={() => {
                        loadEntityDetail(entity.id);
                        loadEntityGraph(entity.canonical_name);
                        setShowGraph(false); // Start with details view
                      }}
                      className={`cursor-pointer hover:bg-muted/50 ${
                        selectedEntity?.id === entity.id ? "bg-primary/10" : ""
                      }`}
                    >
                      <TableCell className="font-medium text-card-foreground">
                        {entity.canonical_name}
                      </TableCell>
                      <TableCell className="text-card-foreground">{entity.mention_count}</TableCell>
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
          </>
        ) : (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="text-4xl mb-2">👥</div>
              <div className="text-sm text-muted-foreground">No entities found</div>
              <div className="text-xs text-muted-foreground mt-1">
                Entities are extracted from facts when memories are added.
              </div>
            </div>
          </div>
        )}
      </div>

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
                <span className="text-lg">×</span>
              </Button>
            </div>

            {/* View Toggle */}
            <div className="flex justify-center mb-6">
              <div className="bg-muted rounded-lg p-1">
                <Button
                  variant={!showGraph ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setShowGraph(false)}
                  className="text-xs px-4"
                >
                  Details
                </Button>
                <Button
                  variant={showGraph ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setShowGraph(true)}
                  className="text-xs px-4"
                >
                  Graph
                </Button>
              </div>
            </div>

            {showGraph ? (
              /* Graph View */
              <div className="space-y-4">
                {loadingGraph ? (
                  <div className="flex items-center justify-center h-64">
                    <div className="text-center">
                      <div className="text-2xl mb-2">⏳</div>
                      <div className="text-sm text-muted-foreground">Loading graph...</div>
                    </div>
                  </div>
                ) : entityGraphData ? (
                  <div className="h-96 border border-border rounded-lg overflow-hidden">
                    <Graph2D
                      data={graph2DData}
                      nodeColorFn={nodeColorFn}
                      linkColorFn={linkColorFn}
                    />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-64">
                    <div className="text-center">
                      <div className="text-2xl mb-2">📊</div>
                      <div className="text-sm text-muted-foreground">No graph data available</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Details View */
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

                {/* Observations */}
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <div className="text-xs font-bold text-muted-foreground uppercase">
                      Observations
                    </div>
                    <Button
                      onClick={regenerateObservations}
                      disabled={regenerating}
                      variant="outline"
                      size="sm"
                    >
                      {regenerating ? "Regenerating..." : "Regenerate"}
                    </Button>
                  </div>

                  {loadingDetail ? (
                    <div className="text-muted-foreground text-sm">Loading observations...</div>
                  ) : selectedEntity.observations && selectedEntity.observations.length > 0 ? (
                    <ul className="space-y-2">
                      {selectedEntity.observations.map((obs, idx) => (
                        <li key={idx} className="p-3 bg-muted/50 rounded-lg">
                          <div className="text-sm text-card-foreground">{obs.text}</div>
                          {obs.mentioned_at && (
                            <div className="text-xs text-muted-foreground mt-2">
                              {formatDate(obs.mentioned_at)}
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-muted-foreground text-sm p-4 bg-muted/50 rounded-lg">
                      No observations yet. Click &quot;Regenerate&quot; to generate observations
                      from facts.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
