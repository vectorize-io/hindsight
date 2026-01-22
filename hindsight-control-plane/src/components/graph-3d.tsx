"use client";

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import dynamic from "next/dynamic";
import { GraphNode, GraphLink, GraphData } from "./graph-2d";
import { Button } from "@/components/ui/button";
import { RotateCcw, ZoomIn, ZoomOut, Maximize2, Minimize2 } from "lucide-react";

// Dynamic import to avoid SSR issues with WebGL
// Load react-force-graph-3d which will handle Three.js internally
const ForceGraph3D = dynamic(
  () => {
    // Ensure we're in browser environment
    if (typeof window === "undefined") {
      return Promise.resolve(null);
    }
    
    // Import react-force-graph-3d - it handles Three.js internally
    return import("react-force-graph-3d")
      .then((mod) => {
        const Component = mod.default || mod;
        // Verify Three.js is available
        if (!Component) {
          throw new Error("Failed to load react-force-graph-3d");
        }
        return Component;
      })
      .catch((error) => {
        console.error("Error loading 3D graph library:", error);
        throw error;
      });
  },
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Loading 3D graph...</p>
        </div>
      </div>
    ),
  }
) as any;

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

export interface Graph3DProps {
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

const BRAND_PRIMARY = "#0074d9";
const DEFAULT_NODE_COLOR = BRAND_PRIMARY;
const DEFAULT_LINK_COLOR = BRAND_PRIMARY;
const DEFAULT_LINK_WIDTH = 1;
const DEFAULT_NODE_SIZE = 4;

// ============================================================================
// Component
// ============================================================================

export function Graph3D({
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
}: Graph3DProps) {
  const fgRef = useRef<any>(null);
  const [isMounted, setIsMounted] = useState(false);
  const [isLibraryReady, setIsLibraryReady] = useState(false);
  const hasCenteredRef = useRef(false); // Track if we've centered once
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [hoveredNodePos, setHoveredNodePos] = useState<{ x: number; y: number } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [cameraPosition, setCameraPosition] = useState({ x: 0, y: 0, z: 0 });
  const isDarkMode = useIsDarkMode();

  // Ensure Three.js and library are ready before rendering
  useEffect(() => {
    if (!isMounted) return;

    const checkLibrary = async () => {
      try {
        // Import Three.js first to ensure it's available
        const three = await import("three");
        // Verify Three.js is properly loaded
        if (three && three.VERTEX_SHADER) {
          // Small delay to ensure everything is initialized
          setTimeout(() => {
            setIsLibraryReady(true);
          }, 100);
        } else {
          console.warn("Three.js not properly initialized");
          setIsLibraryReady(true); // Still try to render
        }
      } catch (error) {
        console.error("Error loading Three.js:", error);
        setIsLibraryReady(true); // Still try to render
      }
    };

    checkLibrary();
  }, [isMounted]);

  // Use refs to store callbacks to prevent re-renders
  const onNodeClickRef = useRef(onNodeClick);
  const onNodeHoverRef = useRef(onNodeHover);
  const nodeColorFnRef = useRef(nodeColorFn);
  const linkColorFnRef = useRef(linkColorFn);
  const nodeSizeFnRef = useRef(nodeSizeFn);
  const linkWidthFnRef = useRef(linkWidthFn);

  onNodeClickRef.current = onNodeClick;
  onNodeHoverRef.current = onNodeHover;
  nodeColorFnRef.current = nodeColorFn;
  linkColorFnRef.current = linkColorFn;
  nodeSizeFnRef.current = nodeSizeFn;
  linkWidthFnRef.current = linkWidthFn;

  // Transform and limit data
  const graphData = useMemo(() => {
    let nodes = [...data.nodes];

    // Limit nodes if needed
    if (maxNodes && nodes.length > maxNodes) {
      nodes = nodes.slice(0, maxNodes);
    }

    // Show ALL links between visible nodes
    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = data.links.filter((l) => nodeIds.has(l.source) && nodeIds.has(l.target));

    return { nodes, links };
  }, [data, maxNodes]);

  // Track mounting state
  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  // Calculate node connections for dynamic sizing
  const nodeConnections = useMemo(() => {
    const connections = new Map<string, number>();
    graphData.links.forEach((link) => {
      connections.set(link.source, (connections.get(link.source) || 0) + 1);
      connections.set(link.target, (connections.get(link.target) || 0) + 1);
    });
    return connections;
  }, [graphData.links]);

  // Prepare graph data for 3D force graph
  const graphData3D = useMemo(() => {
    const nodes = graphData.nodes.map((node) => {
      const connections = nodeConnections.get(node.id) || 0;
      const size = nodeSizeFnRef.current
        ? nodeSizeFnRef.current(node)
        : node.size || Math.max(3, Math.min(10, DEFAULT_NODE_SIZE + connections * 0.6));

      return {
        id: node.id,
        name: showLabels ? node.label || node.id.substring(0, 8) : "",
        color: nodeColorFnRef.current
          ? nodeColorFnRef.current(node)
          : node.color || DEFAULT_NODE_COLOR,
        val: size,
        originalNode: node,
      };
    });

    const links = graphData.links.map((link) => ({
      source: link.source,
      target: link.target,
      color: linkColorFnRef.current
        ? linkColorFnRef.current(link)
        : link.color || DEFAULT_LINK_COLOR,
      width: linkWidthFnRef.current
        ? linkWidthFnRef.current(link)
        : link.width || DEFAULT_LINK_WIDTH,
      originalLink: link,
    }));

    return { nodes, links };
  }, [graphData, showLabels, nodeConnections]);

  // Handle node click
  const handleNodeClick = useCallback(
    (node: any) => {
      const originalNode = node.originalNode as GraphNode;
      if (onNodeClickRef.current && originalNode) {
        onNodeClickRef.current(originalNode);
      }
    },
    []
  );

  // Handle node hover with mouse position tracking
  const handleNodeHover = useCallback(
    (node: any, prevNode: any) => {
      if (node) {
        const originalNode = node.originalNode as GraphNode;
        setHoveredNode(originalNode);
        // Use mouse position for tooltip if available, otherwise center
        if (mousePos) {
          setHoveredNodePos(mousePos);
        } else {
          setHoveredNodePos({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
        }
        if (onNodeHoverRef.current && originalNode) {
          onNodeHoverRef.current(originalNode);
        }
      } else {
        setHoveredNode(null);
        setHoveredNodePos(null);
        if (onNodeHoverRef.current) {
          onNodeHoverRef.current(null);
        }
      }
    },
    [mousePos]
  );

  // Track mouse position for tooltip
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Handle link hover
  const handleLinkHover = useCallback((link: any) => {
    // Optional: Add link hover tooltip if needed
  }, []);

  // Control functions
  const handleResetCamera = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.cameraPosition({ x: 0, y: 0, z: 1000 });
      fgRef.current.zoomToFit(400);
    }
  }, []);

  const handleZoomIn = useCallback(() => {
    if (fgRef.current) {
      const distance = fgRef.current.cameraDistance();
      fgRef.current.cameraPosition({ z: Math.max(100, distance * 0.7) });
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (fgRef.current) {
      const distance = fgRef.current.cameraDistance();
      fgRef.current.cameraPosition({ z: Math.min(3000, distance * 1.4) });
    }
  }, []);

  const handleFitView = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  }, []);

  // Center graph on first load - using onEngineStop (proven method)
  // No separate loading state needed - graph renders immediately

  if (!isMounted) {
    return (
      <div
        className="relative w-full rounded-lg overflow-hidden border border-border flex items-center justify-center"
        style={{ height }}
      >
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Loading 3D graph...</p>
        </div>
      </div>
    );
  }


  return (
    <div
      className="relative w-full rounded-lg overflow-hidden border border-border"
      style={{ height }}
    >
      {isMounted && isLibraryReady && ForceGraph3D && graphData3D.nodes.length > 0 && (
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData3D}
          nodeLabel={(node: { originalNode?: GraphNode; id?: string; name?: string }) => {
            const originalNode = node.originalNode as GraphNode;
            return showLabels
              ? originalNode?.label || node.name || originalNode?.id || node.id || ""
              : "";
          }}
          nodeColor={(node: { color?: string }) => node.color || DEFAULT_NODE_COLOR}
          nodeVal={(node: { val?: number }) => node.val || DEFAULT_NODE_SIZE}
          linkColor={(link: { color?: string }) => link.color || DEFAULT_LINK_COLOR}
          linkWidth={(link: { width?: number }) => link.width || DEFAULT_LINK_WIDTH}
          linkDirectionalArrowLength={8}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          linkOpacity={isDarkMode ? 0.6 : 0.5}
          nodeOpacity={0.9}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          onLinkHover={handleLinkHover}
          backgroundColor={isDarkMode ? "#0f1419" : "#f8fafc"}
          showNavInfo={false}
          cooldownTicks={100}
          warmupTicks={60}
          onEngineStop={() => {
            // Auto-center on first engine stop (proven method)
            if (fgRef.current && !hasCenteredRef.current) {
              try {
                fgRef.current.cameraPosition({ x: 0, y: 0, z: 1000 });
                fgRef.current.zoomToFit(400);
                hasCenteredRef.current = true;
              } catch (error) {
                // Silently fail
                hasCenteredRef.current = true;
              }
            }
          }}
          onCameraChange={(camera: any) => {
            if (camera) {
              setCameraPosition({
                x: camera.x || 0,
                y: camera.y || 0,
                z: camera.z || 0,
              });
            }
          }}
        />
      )}

      {/* Node hover tooltip */}
      {hoveredNode && hoveredNodePos && (
        <div
          className="absolute z-30 pointer-events-none"
          style={{
            left: hoveredNodePos.x,
            top: hoveredNodePos.y,
            transform: "translate(-50%, -100%) translateY(-8px)",
          }}
        >
          <div
            className={`px-3 py-2 rounded-lg shadow-lg text-sm max-w-xs ${
              isDarkMode
                ? "bg-gray-800 text-white border border-gray-700"
                : "bg-white text-gray-900 border border-gray-200"
            }`}
          >
            <div className="font-medium mb-1">{hoveredNode.label || hoveredNode.id}</div>
            {hoveredNode.metadata?.text && (
              <div className="text-xs opacity-80 line-clamp-2">
                {hoveredNode.metadata.text}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Control buttons */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-20">
        <Button
          variant="outline"
          size="sm"
          onClick={handleResetCamera}
          className="h-8 w-8 p-0 bg-background/90 backdrop-blur-sm hover:bg-background"
          title="Reset camera position"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleFitView}
          className="h-8 w-8 p-0 bg-background/90 backdrop-blur-sm hover:bg-background"
          title="Fit to view"
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleZoomIn}
          className="h-8 w-8 p-0 bg-background/90 backdrop-blur-sm hover:bg-background"
          title="Zoom in"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleZoomOut}
          className="h-8 w-8 p-0 bg-background/90 backdrop-blur-sm hover:bg-background"
          title="Zoom out"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
      </div>

      {/* Empty state */}
      {graphData.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-muted-foreground">No memories to display</p>
          </div>
        </div>
      )}

      {/* Controls hint */}
      <div className="absolute bottom-4 left-4 text-xs text-muted-foreground/70 z-20 bg-background/90 backdrop-blur-sm px-3 py-2 rounded border border-border/50">
        <div className="font-medium mb-1">Controls</div>
        <div className="space-y-0.5 text-xs">
          <div>🖱️ Drag to rotate • Scroll to zoom</div>
          <div>👆 Click node to view details</div>
          <div>🔘 Use buttons to reset/zoom</div>
        </div>
      </div>
    </div>
  );
}
