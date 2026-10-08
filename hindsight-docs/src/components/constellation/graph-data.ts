// Graph data model for the Constellation view. Kept byte-identical in shape to
// hindsight-control-plane/src/components/graph-data.ts so the component can be
// re-synced from there; the API-response converter is control-plane only.

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
