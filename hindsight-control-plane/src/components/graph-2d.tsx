'use client';

import { useRef, useEffect, useState, useMemo } from 'react';
import cytoscape, { Core, NodeSingular } from 'cytoscape';

// Hook to detect dark mode
function useIsDarkMode() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDark = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };

    checkDark();

    // Watch for theme changes
    const observer = new MutationObserver(checkDark);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    return () => observer.disconnect();
  }, []);

  return isDark;
}

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

export interface Graph2DProps {
  data: GraphData;
  height?: number;
  showLabels?: boolean;
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  nodeColorFn?: (node: GraphNode) => string;
  nodeSizeFn?: (node: GraphNode) => number;
  linkColorFn?: (link: GraphLink) => string;
  linkWidthFn?: (link: GraphLink) => number;
  maxNodes?: number;
}

// ============================================================================
// Default Values
// ============================================================================

// Brand colors
const BRAND_PRIMARY = '#0074d9';
const BRAND_TEAL = '#009296';
const LINK_SEMANTIC = '#0074d9';  // Primary blue for semantic
const LINK_TEMPORAL = '#009296';  // Teal for temporal
const LINK_ENTITY = '#f59e0b';    // Amber for entity

const DEFAULT_NODE_COLOR = BRAND_PRIMARY;
const DEFAULT_LINK_COLOR = LINK_SEMANTIC;
const DEFAULT_NODE_SIZE = 20;
const DEFAULT_LINK_WIDTH = 1;

// ============================================================================
// Component
// ============================================================================

export function Graph2D({
  data,
  height = 600,
  showLabels = true,
  onNodeClick,
  onNodeHover,
  nodeColorFn,
  nodeSizeFn,
  linkColorFn,
  linkWidthFn,
  maxNodes,
}: Graph2DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [hoveredLink, setHoveredLink] = useState<GraphLink | null>(null);
  const [linkTooltipPos, setLinkTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const isDarkMode = useIsDarkMode();

  // Use refs to store callbacks and data to prevent re-renders from resetting the graph
  const onNodeClickRef = useRef(onNodeClick);
  const onNodeHoverRef = useRef(onNodeHover);
  const fullDataRef = useRef(data);
  const nodeColorFnRef = useRef(nodeColorFn);
  const linkColorFnRef = useRef(linkColorFn);
  onNodeClickRef.current = onNodeClick;
  onNodeHoverRef.current = onNodeHover;
  fullDataRef.current = data;
  nodeColorFnRef.current = nodeColorFn;
  linkColorFnRef.current = linkColorFn;

  // Transform and limit data - only limit nodes, show ALL links between visible nodes
  const graphData = useMemo(() => {
    let nodes = [...data.nodes];

    // Limit nodes if needed
    if (maxNodes && nodes.length > maxNodes) {
      nodes = nodes.slice(0, maxNodes);
    }

    // Show ALL links between visible nodes (no random link limiting)
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = data.links.filter(l => nodeIds.has(l.source) && nodeIds.has(l.target));

    return { nodes, links };
  }, [data, maxNodes]);

  // Convert to Cytoscape format
  const cyElements = useMemo(() => {
    const nodes = graphData.nodes.map(node => ({
      data: {
        id: node.id,
        label: node.label || node.id.substring(0, 8),
        color: nodeColorFn ? nodeColorFn(node) : (node.color || DEFAULT_NODE_COLOR),
        size: nodeSizeFn ? nodeSizeFn(node) : (node.size || DEFAULT_NODE_SIZE),
        originalNode: node,
      },
    }));

    const edges = graphData.links.map((link, idx) => ({
      data: {
        id: `edge-${idx}`,
        source: link.source,
        target: link.target,
        color: linkColorFn ? linkColorFn(link) : (link.color || DEFAULT_LINK_COLOR),
        width: linkWidthFn ? linkWidthFn(link) : (link.width || DEFAULT_LINK_WIDTH),
        type: link.type,
        entity: link.entity,
        weight: link.weight,
        originalLink: link,
      },
    }));

    return [...nodes, ...edges];
  }, [graphData, nodeColorFn, nodeSizeFn, linkColorFn, linkWidthFn]);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    // Handle empty data case
    if (cyElements.length === 0) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);

    // Theme-aware colors
    const textColor = isDarkMode ? '#ffffff' : '#1f2937';
    const textBgColor = isDarkMode ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)';
    const borderColor = isDarkMode ? '#ffffff' : '#374151';

    const cy = cytoscape({
      container: containerRef.current,
      elements: cyElements,
      style: [
        {
          selector: 'node',
          style: {
            'background-fill': 'radial-gradient',
            'background-gradient-stop-colors': ['#0074d9', '#005bb5'],
            'background-gradient-stop-positions': ['0%', '100%'],
            'width': 'data(size)',
            'height': 'data(size)',
            'label': showLabels ? 'data(label)' : '',
            'color': textColor,
            'text-valign': 'bottom',
            'text-halign': 'center',
            'font-size': '8px',
            'font-weight': 500,
            'text-margin-y': 3,
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            'text-background-color': textBgColor,
            'text-background-opacity': 0.9,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
            'border-width': 0,
            'z-index': 0,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#0074d9',
            'border-opacity': 1,
          },
        },
        {
          selector: 'node:active',
          style: {
            'overlay-opacity': 0,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(width)',
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'curve-style': 'bezier',
            'opacity': isDarkMode ? 0.5 : 0.6,
            'z-index': 1,
          },
        },
        {
          selector: 'edge:selected',
          style: {
            'opacity': 1,
            'width': 3,
          },
        },
        // Dimmed state for non-selected elements
        {
          selector: '.dimmed',
          style: {
            'opacity': 0.15,
          },
        },
        // Highlighted state for selected node and neighbors
        {
          selector: 'node.highlighted',
          style: {
            'opacity': 1,
            'border-width': 3,
            'border-color': '#0074d9',
            'border-opacity': 1,
          },
        },
        {
          selector: 'edge.highlighted',
          style: {
            'opacity': 0.9,
            'width': 2,
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: false,
        randomize: true,
        nodeRepulsion: () => 100000,
        idealEdgeLength: () => 300,
        edgeElasticity: () => 20,
        nestingFactor: 0.1,
        gravity: 0.01,
        numIter: 2500,
        coolingFactor: 0.95,
        minTemp: 1.0,
        nodeOverlap: 20,
        nodeDimensionsIncludeLabels: true,
        padding: 50,
      } as any,
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
    });

    cyRef.current = cy;

    // Event handlers
    cy.on('tap', 'node', (evt) => {
      const node = evt.target as NodeSingular;
      const originalNode = node.data('originalNode') as GraphNode;
      if (onNodeClickRef.current && originalNode) {
        onNodeClickRef.current(originalNode);
      }

      // Find ALL connected nodes from full data (not just visible ones)
      const fullData = fullDataRef.current;
      const clickedNodeId = originalNode.id;

      // Find all links connected to this node from full data
      const connectedLinks = fullData.links.filter(
        l => l.source === clickedNodeId || l.target === clickedNodeId
      );

      // Find all connected node IDs
      const connectedNodeIds = new Set<string>();
      connectedLinks.forEach(l => {
        connectedNodeIds.add(l.source);
        connectedNodeIds.add(l.target);
      });

      // Add any missing nodes to the graph
      const existingNodeIds = new Set(cy.nodes().map(n => n.id()));
      const nodesToAdd: any[] = [];
      const edgesToAdd: any[] = [];

      connectedNodeIds.forEach(nodeId => {
        if (!existingNodeIds.has(nodeId)) {
          const nodeData = fullData.nodes.find(n => n.id === nodeId);
          if (nodeData) {
            nodesToAdd.push({
              group: 'nodes',
              data: {
                id: nodeData.id,
                label: nodeData.label || nodeData.id.substring(0, 8),
                color: nodeColorFnRef.current ? nodeColorFnRef.current(nodeData) : (nodeData.color || DEFAULT_NODE_COLOR),
                size: nodeData.size || DEFAULT_NODE_SIZE,
                originalNode: nodeData,
                isTemporary: true, // Mark as temporarily added
              },
            });
          }
        }
      });

      // Add missing edges
      const existingEdgeIds = new Set(cy.edges().map(e => `${e.data('source')}-${e.data('target')}`));
      connectedLinks.forEach((link, idx) => {
        const edgeKey = `${link.source}-${link.target}`;
        const reverseKey = `${link.target}-${link.source}`;
        if (!existingEdgeIds.has(edgeKey) && !existingEdgeIds.has(reverseKey)) {
          edgesToAdd.push({
            group: 'edges',
            data: {
              id: `temp-edge-${idx}-${Date.now()}`,
              source: link.source,
              target: link.target,
              color: linkColorFnRef.current ? linkColorFnRef.current(link) : (link.color || DEFAULT_LINK_COLOR),
              width: link.width || DEFAULT_LINK_WIDTH,
              type: link.type,
              isTemporary: true,
            },
          });
        }
      });

      // Add new elements to graph
      if (nodesToAdd.length > 0 || edgesToAdd.length > 0) {
        cy.add([...nodesToAdd, ...edgesToAdd]);

        // Position new nodes near the clicked node
        const clickedPos = node.position();
        cy.nodes('[?isTemporary]').forEach((n, i) => {
          const angle = (2 * Math.PI * i) / nodesToAdd.length;
          const radius = 150;
          n.position({
            x: clickedPos.x + radius * Math.cos(angle),
            y: clickedPos.y + radius * Math.sin(angle),
          });
        });
      }

      // Get all connected elements (including newly added)
      const neighborhood = node.neighborhood().add(node);

      // Dim all elements first
      cy.elements().addClass('dimmed');

      // Highlight the neighborhood
      neighborhood.removeClass('dimmed');
      neighborhood.addClass('highlighted');

      // Center on the neighborhood without changing positions
      cy.animate({
        fit: { eles: neighborhood, padding: 50 },
      }, { duration: 400 });
    });

    // Click on background to reset
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        // Remove temporary nodes and edges
        cy.elements('[?isTemporary]').remove();

        cy.elements().removeClass('dimmed highlighted');
        cy.animate({
          fit: { eles: cy.elements(), padding: 50 },
        }, { duration: 400 });
      }
    });

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target as NodeSingular;
      const originalNode = node.data('originalNode') as GraphNode;
      setHoveredNode(originalNode);
      if (onNodeHoverRef.current && originalNode) {
        onNodeHoverRef.current(originalNode);
      }
      containerRef.current!.style.cursor = 'pointer';
    });

    cy.on('mouseout', 'node', () => {
      setHoveredNode(null);
      if (onNodeHoverRef.current) {
        onNodeHoverRef.current(null);
      }
      containerRef.current!.style.cursor = 'default';
    });

    // Edge hover handlers
    cy.on('mouseover', 'edge', (evt) => {
      const edge = evt.target;
      const originalLink = edge.data('originalLink') as GraphLink;
      if (originalLink) {
        setHoveredLink(originalLink);
        // Get position for tooltip
        const renderedPos = edge.renderedMidpoint();
        setLinkTooltipPos({ x: renderedPos.x, y: renderedPos.y });
      }
      containerRef.current!.style.cursor = 'pointer';
    });

    cy.on('mouseout', 'edge', () => {
      setHoveredLink(null);
      setLinkTooltipPos(null);
      containerRef.current!.style.cursor = 'default';
    });

    // Run layout
    cy.layout({
      name: 'cose',
      animate: false,
      randomize: true,
      nodeRepulsion: () => 100000,
      idealEdgeLength: () => 300,
      edgeElasticity: () => 20,
      nestingFactor: 0.1,
      gravity: 0.01,
      numIter: 2500,
      coolingFactor: 0.95,
      minTemp: 1.0,
      nodeOverlap: 20,
      nodeDimensionsIncludeLabels: true,
      padding: 50,
    } as any).run();

    // Fit to viewport
    cy.fit(undefined, 50);
    setIsLoading(false);

    return () => {
      cy.destroy();
    };
  }, [cyElements, showLabels, isDarkMode]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 50);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="relative w-full rounded-lg overflow-hidden border border-border" style={{ height }}>
      {/* Loading state */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">Loading graph...</p>
          </div>
        </div>
      )}

      {/* Cytoscape container */}
      <div
        ref={containerRef}
        className="w-full h-full"
        style={{
          background: isDarkMode
            ? 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.08) 1px, transparent 0)'
            : 'radial-gradient(circle at 1px 1px, rgba(0,0,0,0.06) 1px, transparent 0)',
          backgroundSize: '20px 20px',
          backgroundColor: isDarkMode ? '#0f1419' : '#f8fafc',
        }}
      />

      {/* Empty state */}
      {!isLoading && graphData.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-muted-foreground">No memories to display</p>
          </div>
        </div>
      )}

      {/* Link hover tooltip */}
      {hoveredLink && linkTooltipPos && (
        <div
          className="absolute z-30 pointer-events-none"
          style={{
            left: linkTooltipPos.x,
            top: linkTooltipPos.y,
            transform: 'translate(-50%, -100%) translateY(-8px)',
          }}
        >
          <div className={`px-3 py-2 rounded-lg shadow-lg text-sm ${
            isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900 border border-gray-200'
          }`}>
            <div className="font-medium capitalize mb-1">
              {(() => {
                const type = hoveredLink.type || 'semantic';
                if (['causes', 'caused_by', 'enables', 'prevents'].includes(type)) {
                  return `Causal (${type.replace('_', ' ')})`;
                }
                return `${type} link`;
              })()}
            </div>
            {hoveredLink.entity && (
              <div className="text-xs opacity-80">
                Entity: <span className="font-medium">{hoveredLink.entity}</span>
              </div>
            )}
            {hoveredLink.weight !== undefined && (
              <div className="text-xs opacity-80">
                Weight: <span className="font-medium">{hoveredLink.weight.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Controls hint */}
      <div className="absolute bottom-4 right-4 text-xs text-muted-foreground/60 z-20">
        Drag to pan • Scroll to zoom • Click node to focus
      </div>
    </div>
  );
}

// ============================================================================
// Utility Functions
// ============================================================================

export function convertHindsightGraphData(hindsightData: {
  nodes?: Array<{ data: { id: string; label?: string; color?: string } }>;
  edges?: Array<{ data: { source: string; target: string; color?: string; lineStyle?: string; linkType?: string; entityName?: string; weight?: number; similarity?: number } }>;
  table_rows?: Array<{ id: string; text: string; entities?: string; context?: string }>;
}): GraphData {
  const nodes: GraphNode[] = (hindsightData.nodes || []).map(n => {
    const tableRow = hindsightData.table_rows?.find(r => r.id === n.data.id);
    // Use memory text as label, truncated to ~40 chars
    let label = n.data.label;
    if (!label && tableRow?.text) {
      label = tableRow.text.length > 40 ? tableRow.text.substring(0, 40) + '...' : tableRow.text;
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

  const links: GraphLink[] = (hindsightData.edges || []).map(e => ({
    source: e.data.source,
    target: e.data.target,
    color: e.data.color,
    // Use linkType directly from API, fallback to lineStyle check, default to semantic
    type: e.data.linkType || (e.data.lineStyle === 'dashed' ? 'temporal' : 'semantic'),
    entity: e.data.entityName,  // API returns entityName
    weight: e.data.weight ?? e.data.similarity,
  }));

  return { nodes, links };
}
