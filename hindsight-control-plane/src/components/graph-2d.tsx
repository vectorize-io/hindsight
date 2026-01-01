"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import cytoscape from "cytoscape";

import fcose from "cytoscape-fcose";

// Register the fcose extension
cytoscape.use(fcose);

// Hook to detect dark mode
function useIsDarkMode() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDark = () => {
      setIsDark(document.documentElement.classList.contains("dark"));
    };

    checkDark();

    // Watch for theme changes
    const observer = new MutationObserver(checkDark);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

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
const BRAND_PRIMARY = "#0074d9";
const LINK_SEMANTIC = "#0074d9"; // Primary blue for semantic

const DEFAULT_NODE_COLOR = BRAND_PRIMARY;
const DEFAULT_LINK_COLOR = LINK_SEMANTIC;
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
  const [containerDiv, setContainerDiv] = useState<HTMLDivElement | null>(null);
  const cyRef = useRef<any>(null);
  const isInitializingRef = useRef(false);
  const lastDataSignatureRef = useRef<string>("");
  const [_hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [hoveredLink, setHoveredLink] = useState<GraphLink | null>(null);
  const [linkTooltipPos, setLinkTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMounted, setIsMounted] = useState(false);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const isDarkMode = useIsDarkMode();

  // Use refs to store callbacks and data to prevent re-renders from resetting the graph
  const onNodeClickRef = useRef(onNodeClick);
  const onNodeHoverRef = useRef(onNodeHover);
  const fullDataRef = useRef(data);
  const nodeColorFnRef = useRef(nodeColorFn);
  const linkColorFnRef = useRef(linkColorFn);
  const isFocusModeRef = useRef(isFocusMode);
  onNodeClickRef.current = onNodeClick;
  onNodeHoverRef.current = onNodeHover;
  fullDataRef.current = data;
  nodeColorFnRef.current = nodeColorFn;
  linkColorFnRef.current = linkColorFn;
  isFocusModeRef.current = isFocusMode;

  // Transform and limit data - only limit nodes, show ALL links between visible nodes
  const graphData = useMemo(() => {
    let nodes = [...data.nodes];

    // Limit nodes if needed
    if (maxNodes && nodes.length > maxNodes) {
      nodes = nodes.slice(0, maxNodes);
    }

    // Show ALL links between visible nodes (no random link limiting)
    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = data.links.filter((l) => nodeIds.has(l.source) && nodeIds.has(l.target));

    return { nodes, links };
  }, [data, maxNodes]);

  // Track mounting state
  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  // Convert to Cytoscape format
  const cyElements = useMemo(() => {
    // Calculate node importance based on connections
    const nodeConnections = new Map<string, number>();
    graphData.links.forEach((link) => {
      nodeConnections.set(link.source, (nodeConnections.get(link.source) || 0) + 1);
      nodeConnections.set(link.target, (nodeConnections.get(link.target) || 0) + 1);
    });

    const nodes = graphData.nodes.map((node) => {
      const connections = nodeConnections.get(node.id) || 0;
      const dynamicSize = nodeSizeFn
        ? nodeSizeFn(node)
        : Math.max(16, Math.min(40, 16 + connections * 4)); // Smaller, more subtle sizing

      return {
        data: {
          id: node.id,
          label: node.label || node.id.substring(0, 8),
          color: nodeColorFn ? nodeColorFn(node) : node.color || DEFAULT_NODE_COLOR,
          size: node.size || dynamicSize,
          originalNode: node,
          connections: connections,
        },
      };
    });

    const edges = graphData.links.map((link, idx) => ({
      data: {
        id: `edge-${idx}`,
        source: link.source,
        target: link.target,
        color: linkColorFn ? linkColorFn(link) : link.color || DEFAULT_LINK_COLOR,
        width: linkWidthFn ? linkWidthFn(link) : link.width || DEFAULT_LINK_WIDTH,
        type: link.type,
        entity: link.entity,
        weight: link.weight,
        originalLink: link,
      },
    }));

    return [...nodes, ...edges];
  }, [graphData, nodeColorFn, nodeSizeFn, linkColorFn, linkWidthFn]);

  // Create data signature to prevent double initialization
  const dataSignature = useMemo(() => {
    return JSON.stringify({
      nodeCount: graphData.nodes.length,
      linkCount: graphData.links.length,
      nodeIds: graphData.nodes
        .map((n) => n.id)
        .sort()
        .join(","),
      showLabels,
      isDarkMode,
      maxNodes,
    });
  }, [graphData.nodes, graphData.links, showLabels, isDarkMode, maxNodes]);

  // Initialize Cytoscape
  useEffect(() => {
    let isCancelled = false;

    // Small delay to ensure container is mounted
    const timeout = setTimeout(() => {
      if (isCancelled || !isMounted || !containerDiv || isInitializingRef.current) return;

      // Check if data has actually changed to prevent double initialization
      if (lastDataSignatureRef.current === dataSignature) {
        console.log("Data signature unchanged, skipping graph initialization");
        return;
      }

      // Additional validation - check if element has dimensions
      const rect = containerDiv.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        console.warn("Container has no dimensions, skipping cytoscape initialization");
        setIsLoading(false);
        return;
      }

      // Handle empty data case
      if (cyElements.length === 0) {
        setIsLoading(false);
        return;
      }

      // Check if we already have a graph with the same data
      if (cyRef.current && !cyRef.current.destroyed()) {
        const currentNodes = cyRef.current.nodes().length;
        const currentEdges = cyRef.current.edges().length;
        const newNodes = cyElements.filter((el) => !(el.data as any).source).length;
        const newEdges = cyElements.filter((el) => (el.data as any).source).length;

        // If the element counts are the same, just update styles and skip reinitialization
        if (currentNodes === newNodes && currentEdges === newEdges) {
          console.log("Graph already initialized with same data, skipping reinitialization");
          setIsLoading(false);
          return;
        }

        // Clean up existing graph before creating new one
        console.log("Data changed, destroying existing graph");
        cyRef.current.destroy();
        cyRef.current = null;
      }

      setIsLoading(true);
      isInitializingRef.current = true;

      // Theme-aware colors
      const textColor = isDarkMode ? "#ffffff" : "#1f2937";
      const textBgColor = isDarkMode ? "rgba(0,0,0,0.8)" : "rgba(255,255,255,0.9)";

      try {
        console.log("Initializing cytoscape with container:", containerDiv);
        console.log("Elements count:", cyElements.length);
        console.log("Sample elements:", cyElements.slice(0, 2));

        // Try minimal initialization first
        const cy = cytoscape({
          container: containerDiv,
          elements: [],
          // Disable edge selection to prevent gray border on click
          selectionType: "single",
          userZoomingEnabled: true,
          userPanningEnabled: true,
          boxSelectionEnabled: false,
          // Disable automatic layout on initialization
          layout: { name: "preset" },
          style: [
            {
              selector: "node",
              style: {
                "background-color": "data(color)",
                width: "data(size)",
                height: "data(size)",
                label: showLabels ? "data(label)" : "",
                color: textColor,
                "text-valign": "bottom",
                "text-halign": "center",
                "font-size": "8px",
                "font-weight": 500,
                "text-margin-y": 3,
                "text-wrap": "wrap",
                "text-max-width": "80px",
                "text-background-color": textBgColor,
                "text-background-opacity": 0.9,
                "text-background-padding": "2px",
                "text-background-shape": "roundrectangle",
                "border-width": 1,
                "border-color": isDarkMode ? "#ffffff20" : "#00000020",
                "border-opacity": 0.3,
              },
            },
            {
              selector: "node:selected",
              style: {
                "border-width": 3,
                "border-color": "#0074d9",
                "border-opacity": 1,
              },
            },
            {
              selector: "edge",
              style: {
                width: "data(width)",
                "line-color": "data(color)",
                "target-arrow-color": "data(color)",
                "target-arrow-shape": "triangle",
                "target-arrow-size": 6,
                "curve-style": "bezier",
                opacity: isDarkMode ? 0.6 : 0.7,
              },
            },
            // Focus mode styles
            {
              selector: ".dimmed",
              style: {
                opacity: 0.2,
              },
            },
            {
              selector: ".focused",
              style: {
                "border-width": 4,
                "border-color": "#ff6b35",
                "border-opacity": 1,
                "z-index": 999,
              },
            },
            {
              selector: ".connected",
              style: {
                "border-width": 2,
                "border-color": "#0074d9",
                "border-opacity": 0.8,
                opacity: 1,
              },
            },
            {
              selector: "edge.connection",
              style: {
                width: 2,
                opacity: 1,
                "z-index": 100,
              },
            },
            {
              selector: "edge.connection:hover",
              style: {
                width: 3,
                opacity: 1,
                "z-index": 200,
              },
            },
            // Disable edge selection styling
            {
              selector: "edge:selected",
              style: {
                "overlay-opacity": 0,
                "overlay-color": "transparent",
                "overlay-padding": 0,
              },
            },
          ],
        });

        cyRef.current = cy;

        console.log("Cytoscape initialized successfully");

        // Add elements after initialization
        if (cyElements.length > 0) {
          console.log("Adding elements to cytoscape");
          cy.add(cyElements);
          cy.layout({
            name: "fcose",
            quality: "default",
            randomize: false,
            animate: true,
            animationDuration: 1500,
            // Separation settings - increase to spread nodes more
            nodeSeparation: 200,
            idealEdgeLength: () => 250,
            edgeElasticity: () => 0.05,
            nestingFactor: 0.05,
            gravity: 0.05, // Reduced gravity spreads nodes more
            numIter: 2500,
            // Overlap prevention
            nodeOverlap: 30,
            avoidOverlap: true,
            nodeDimensionsIncludeLabels: true,
            // Layout bounds - reduce padding to use more space
            padding: 20,
            boundingBox: undefined,
            // Tiling - increase spacing between disconnected components
            tile: true,
            tilingPaddingVertical: 30,
            tilingPaddingHorizontal: 30,
            // Force more spread
            uniformNodeDimensions: false,
            packComponents: false, // Don't pack components tightly
          }).run();

          // Fit to viewport
          cy.fit();
        }

        // Add basic interactions
        cy.on("tap", "node", (evt: any) => {
          const node = evt.target as cytoscape.NodeSingular;
          const originalNode = node.data("originalNode") as GraphNode;
          if (onNodeClickRef.current && originalNode) {
            onNodeClickRef.current(originalNode);
          }
        });

        cy.on("mouseover", "node", (evt: any) => {
          const node = evt.target as cytoscape.NodeSingular;
          const originalNode = node.data("originalNode") as GraphNode;
          setHoveredNode(originalNode);
          if (onNodeHoverRef.current && originalNode) {
            onNodeHoverRef.current(originalNode);
          }
          if (containerDiv) containerDiv.style.cursor = "pointer";
        });

        cy.on("mouseout", "node", () => {
          setHoveredNode(null);
          if (onNodeHoverRef.current) {
            onNodeHoverRef.current(null);
          }
          if (containerDiv) containerDiv.style.cursor = "default";
        });

        // Edge hover handlers - only work in focus mode and on highlighted edges
        cy.on("mouseover", "edge", (evt: any) => {
          const edge = evt.target;

          // Only allow interaction if we're in focus mode and edge is highlighted
          if (!isFocusModeRef.current || !edge.hasClass("connection")) {
            return;
          }

          const originalLink = edge.data("originalLink") as GraphLink;
          if (originalLink) {
            setHoveredLink(originalLink);
            // Get position for tooltip
            const renderedPos = edge.renderedMidpoint();
            setLinkTooltipPos({ x: renderedPos.x, y: renderedPos.y });
          }
        });

        cy.on("mouseout", "edge", (evt: any) => {
          const edge = evt.target;

          // Only clear hover state if we were actually hovering a highlighted edge
          if (!isFocusModeRef.current || !edge.hasClass("connection")) {
            return;
          }

          setHoveredLink(null);
          setLinkTooltipPos(null);
        });

        // Prevent edge selection to avoid gray border on click
        cy.on("select", "edge", (evt: any) => {
          evt.target.unselect();
        });

        // Double-click to focus on node and its connections
        cy.on("dblclick", "node", (evt: any) => {
          const focusedNode = evt.target as cytoscape.NodeSingular;
          const focusedNodeId = focusedNode.id();

          console.log("Double-clicked node:", focusedNodeId);

          // Enter focus mode
          setIsFocusMode(true);

          // Clear any existing focus classes
          cy.elements().removeClass("dimmed focused connected connection");

          // Get all connected nodes and edges
          const connectedElements = focusedNode.neighborhood();
          const connectedNodes = connectedElements.nodes();
          const connectedEdges = connectedElements.edges();

          // Apply styling classes
          cy.elements().addClass("dimmed"); // Dim everything first
          focusedNode.removeClass("dimmed").addClass("focused"); // Highlight the focused node
          connectedNodes.removeClass("dimmed").addClass("connected"); // Highlight connected nodes
          connectedEdges.removeClass("dimmed").addClass("connection"); // Highlight connecting edges

          // Create a collection of all relevant elements for positioning
          const relevantElements = focusedNode.union(connectedElements);

          // Reorient the graph to focus on this subgraph
          cy.animate(
            {
              fit: {
                eles: relevantElements,
                padding: 100,
              },
              center: {
                eles: focusedNode,
              },
            },
            {
              duration: 800,
              easing: "ease-out-cubic",
            }
          );
        });

        // Click on background to reset focus
        cy.on("tap", (evt: any) => {
          if (evt.target === cy) {
            console.log("Clicked background - resetting focus");

            // Exit focus mode
            setIsFocusMode(false);

            // Remove all focus classes
            cy.elements().removeClass("dimmed focused connected connection");

            // Zoom out to show all elements
            cy.animate(
              {
                fit: {
                  eles: cy.elements(),
                  padding: 50,
                },
              },
              {
                duration: 600,
                easing: "ease-out",
              }
            );
          }
        });

        setIsLoading(false);
        isInitializingRef.current = false;
        lastDataSignatureRef.current = dataSignature;
      } catch (error) {
        console.error("Error initializing cytoscape:", error);
        setIsLoading(false);
        isInitializingRef.current = false;
      }
    }, 100); // 100ms delay

    return () => {
      isCancelled = true;
      clearTimeout(timeout);
      isInitializingRef.current = false;
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [dataSignature, isMounted, containerDiv]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 80);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      className="relative w-full rounded-lg overflow-hidden border border-border"
      style={{ height }}
    >
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
      {isMounted && (
        <div
          ref={setContainerDiv}
          className="w-full h-full"
          style={{
            backgroundImage: isDarkMode
              ? "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.08) 1px, transparent 0)"
              : "radial-gradient(circle at 1px 1px, rgba(0,0,0,0.06) 1px, transparent 0)",
            backgroundSize: "20px 20px",
            backgroundColor: isDarkMode ? "#0f1419" : "#f8fafc",
          }}
        />
      )}

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
            transform: "translate(-50%, -100%) translateY(-8px)",
          }}
        >
          <div
            className={`px-3 py-2 rounded-lg shadow-lg text-sm ${
              isDarkMode
                ? "bg-gray-800 text-white"
                : "bg-white text-gray-900 border border-gray-200"
            }`}
          >
            <div className="font-medium capitalize mb-1">
              {(() => {
                const type = hoveredLink.type || "semantic";
                if (["causes", "caused_by", "enables", "prevents"].includes(type)) {
                  return `Causal (${type.replace("_", " ")})`;
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
        Drag to pan • Scroll to zoom • Double-click node to focus • Click background to reset
      </div>
    </div>
  );
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
