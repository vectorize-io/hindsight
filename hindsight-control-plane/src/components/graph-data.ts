// Shared graph data model + conversion used by the memory visualizations
// (Constellation, entities view). The Cytoscape-based "Graph" view that used to
// live here was removed; only the framework-agnostic types and the API-response
// converter remain, since the constellation and entity views build on them.

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface GraphNode {
  id: string;
  label?: string;
  color?: string;
  size?: number;
  group?: string;
  metadata?: Record<string, any>;
}

export interface GraphLink {
  source: string;
  target: string;
  color?: string;
  width?: number;
  type?: string;
  entity?: string;
  weight?: number;
  metadata?: Record<string, any>;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// ============================================================================
// Utility Functions
// ============================================================================

export function convertHindsightGraphData(hindsightData: {
  nodes?: Array<{ data: { id: string; label?: string; color?: string } }>;
  edges?: Array<{
    data: {
      source: string;
      target: string;
      color?: string;
      lineStyle?: string;
      linkType?: string;
      entityName?: string;
      weight?: number;
      similarity?: number;
    };
  }>;
  table_rows?: Array<{ id: string; text: string; entities?: string; context?: string }>;
}): GraphData {
  const nodes: GraphNode[] = (hindsightData.nodes || []).map((n) => {
    const tableRow = hindsightData.table_rows?.find((r) => r.id === n.data.id);
    // Use memory text as label, truncated to ~40 chars
    let label = n.data.label;
    if (!label && tableRow?.text) {
      label = tableRow.text.length > 40 ? tableRow.text.substring(0, 40) + "..." : tableRow.text;
    }
    if (!label) {
      label = n.data.id.substring(0, 8);
    }
    return {
      id: n.data.id,
      label,
      color: n.data.color,
      metadata: tableRow,
    };
  });

  const links: GraphLink[] = (hindsightData.edges || []).map((e) => ({
    source: e.data.source,
    target: e.data.target,
    color: e.data.color,
    // Use linkType directly from API, fallback to lineStyle check, default to semantic
    type: e.data.linkType || (e.data.lineStyle === "dashed" ? "temporal" : "semantic"),
    entity: e.data.entityName, // API returns entityName
    weight: e.data.weight ?? e.data.similarity,
  }));

  return { nodes, links };
}
