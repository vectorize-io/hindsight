"use client";

import { useRef, useEffect, useCallback, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { useTranslations } from "next-intl";
import { prepare, layout, prepareWithSegments, layoutWithLines } from "@chenglou/pretext";
import type { GraphData, GraphNode, GraphLink } from "./graph-data";

// ============================================================================
// Types
// ============================================================================

interface PreparedNode {
  node: GraphNode;
  /** screen x (world coords, before pan/zoom) */
  wx: number;
  wy: number;
  prepared: ReturnType<typeof prepareWithSegments>;
  preparedHeight: ReturnType<typeof prepare>;
  color: string;
  /** Color derived from link count (heat gradient) */
  heatColor: string;
  linkCount: number;
  /**
   * Per-node phase in [0, 2π), derived from the id hash. Desynchronizes the
   * ambient drift + pulse so the field breathes organically instead of in
   * lockstep. Precomputed here so the animation loop stays trig-only.
   */
  phase: number;
}

// ============================================================================
// Props
// ============================================================================

export interface ConstellationProps {
  data: GraphData;
  height?: number;
  onNodeClick?: (node: GraphNode) => void;
  nodeColorFn?: (node: GraphNode) => string;
  linkColorFn?: (link: GraphLink) => string;
  /**
   * Optional override for the on-screen dot radius (in CSS pixels, pre-zoom).
   * When omitted, radius is derived from link count (default star-field behavior).
   * Used by the entities view to scale dots by total co-occurrence weight.
   */
  nodeSizeFn?: (node: GraphNode) => number;
  /**
   * When true, pack labels densely: small deconfliction footprint and no zoom
   * threshold. Appropriate for graphs with short labels (e.g. entity names)
   * where the memory-page truncated-text defaults would hide most of them.
   */
  compactLabels?: boolean;
  /**
   * Optional override for the node heat gradient (0..1 where 1 = hottest).
   * Default is a normalized link-count. Use this to map color to a different
   * dimension — e.g. recency — while keeping size mapped to something else.
   */
  nodeHeatFn?: (node: GraphNode) => number;
  /**
   * Caption for the heat-gradient legend (default: "LINKS"). Use when nodeHeatFn
   * represents something other than connectivity — e.g. "RECENCY".
   */
  heatLegendLabel?: string;
  /**
   * Captions for the heat-gradient endpoints (default: "few" → "many").
   */
  heatLegendEndpoints?: [string, string];
  /**
   * When set, draws a "small dot → big dot" legend entry with this caption.
   * Use this when nodeSizeFn encodes a meaningful dimension (e.g. "source facts",
   * "co-occurrences") so the reader knows what the node size represents.
   */
  sizeLegendLabel?: string;
  /**
   * Optional clustering. When provided, nodes sharing a key are laid out around a
   * common centroid (instead of the default id-hash ring) and wrapped in a
   * translucent "blob", so each group reads as a distinct visual cluster. Used to
   * group observations by scope (exact tag set). Return null to leave a node
   * unclustered (it falls back to the ring layout).
   */
  clusterKeyFn?: (node: GraphNode) => string | null;
  /** Fill/outline color for a cluster key (also tints the cluster's node dots). */
  clusterColorFn?: (key: string) => string;
  /** Short human label for a cluster key, drawn near the blob. */
  clusterLabelFn?: (key: string) => string;
}

// ============================================================================
// Helpers
// ============================================================================

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function hexToRgba(hex: string, alpha: number): string {
  // Handle non-hex formats
  if (!hex.startsWith("#")) return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Monotone-chain convex hull. Returns the hull vertices (CCW) for a set of 2D
// points; used to wrap a cluster's nodes in a translucent blob. Fewer than 3
// points have no polygon — the caller draws a circle instead.
function convexHull(points: Array<[number, number]>): Array<[number, number]> {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * Map a 0..1 value to a perceptually monotonic cool→warm ramp.
 * 0 = cool blue (low / older / few links), 1 = warm orange-red (high / newer / many).
 * Mid passes through a desaturated lavender so brightness stays roughly monotonic
 * — viewers should read the position on the bar at a glance without consulting
 * the legend.
 */
function heatColor(t: number): string {
  const v = Math.max(0, Math.min(1, t));
  const stops = [
    [56, 130, 220], // cool blue
    [170, 130, 200], // muted lavender bridge
    [240, 100, 60], // warm orange-red
  ];
  const seg = v * (stops.length - 1);
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const frac = seg - i;
  const a = stops[i];
  const b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * frac);
  const g = Math.round(a[1] + (b[1] - a[1]) * frac);
  const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
  return `rgb(${r},${g},${bl})`;
}

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return h;
}

/** Fetch a same-origin asset and return it as a base64 data URL (for inlining into the SVG). */
function loadDataUrl(url: string): Promise<string> {
  return fetch(url)
    .then((r) => r.blob())
    .then(
      (blob) =>
        new Promise<string>((resolve, reject) => {
          const fr = new FileReader();
          fr.onload = () => resolve(fr.result as string);
          fr.onerror = reject;
          fr.readAsDataURL(blob);
        })
    );
}

// Dark mode hook
function useIsDarkMode() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const check = () => setIsDark(document.documentElement.classList.contains("dark"));
    check();
    const obs = new MutationObserver(check);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => obs.disconnect();
  }, []);

  return isDark;
}

// ============================================================================
// Constants
// ============================================================================

const FONT_SMALL = '11px Inter, -apple-system, "Segoe UI", sans-serif';
const FONT_BOLD = '600 10px Inter, -apple-system, "Segoe UI", sans-serif';
const MONO = '11px "SF Mono", "Fira Code", Consolas, monospace';

const LINK_TYPE_COLORS: Record<string, string> = {
  semantic: "#0074d9",
  temporal: "#009296",
  entity: "#f59e0b",
  causal: "#8b5cf6",
};

const DEFAULT_NODE_COLOR = "#0074d9";

function toolbarBtnStyle(isDark: boolean): CSSProperties {
  return {
    padding: "6px 10px",
    borderRadius: 6,
    border: `1px solid ${isDark ? "#27272a" : "#e4e4e7"}`,
    background: isDark ? "#18181b" : "#ffffff",
    color: isDark ? "#a1a1aa" : "#71717a",
    fontSize: 12,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 4,
    opacity: 0.7,
    transition: "opacity 0.15s",
  };
}

// ============================================================================
// Component
// ============================================================================

export function Constellation({
  data,
  height = 700,
  onNodeClick,
  nodeColorFn,
  linkColorFn,
  nodeSizeFn,
  nodeHeatFn,
  heatLegendLabel,
  heatLegendEndpoints,
  compactLabels,
  sizeLegendLabel,
  clusterKeyFn,
  clusterColorFn,
  clusterLabelFn,
}: ConstellationProps) {
  const t = useTranslations("constellation");
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const isDark = useIsDarkMode();
  const animRef = useRef<number>(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Interaction state stored in ref for perf (avoid re-renders on every frame)
  const stateRef = useRef({
    panX: 0,
    panY: 0,
    zoom: 0.5,
    targetPanX: 0,
    targetPanY: 0,
    targetZoom: 0.5,
    mouseX: -1,
    mouseY: -1,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    panStartX: 0,
    panStartY: 0,
    hoverIndex: -1,
    prevHoverIndex: -2, // track changes to avoid DOM thrashing
    W: 0,
    H: 0,
    dpr: 1,
  });

  // ----- Prepare data with Pretext -----
  const { preparedNodes, linksByNode, linksWithIndices, clusters } = useMemo(() => {
    if (!data.nodes.length)
      return {
        preparedNodes: [],
        linksByNode: new Map(),
        linksWithIndices: [],
        clusters: [] as Array<{ key: string; members: number[]; color: string; label: string }>,
      };

    const nodeIndexMap = new Map<string, number>();
    data.nodes.forEach((n, i) => nodeIndexMap.set(n.id, i));

    // Count links per node
    const linkCounts = new Map<string, number>();
    for (const link of data.links) {
      linkCounts.set(link.source, (linkCounts.get(link.source) || 0) + 1);
      linkCounts.set(link.target, (linkCounts.get(link.target) || 0) + 1);
    }

    // Find max link count for heat gradient normalization
    let maxLinkCount = 1;
    for (const lc of linkCounts.values()) {
      if (lc > maxLinkCount) maxLinkCount = lc;
    }

    // Clustering precompute: when clusterKeyFn is set, group nodes by key and
    // lay each group out around its own centroid (placed on a wider ring) so
    // groups read as separate clusters. Without it, nodes use the id-hash ring.
    const count = data.nodes.length;
    const clusterKeys = clusterKeyFn ? data.nodes.map((n) => clusterKeyFn(n)) : null;
    const clusterMembers = new Map<string, number[]>();
    if (clusterKeys) {
      clusterKeys.forEach((k, i) => {
        if (k == null) return;
        const arr = clusterMembers.get(k);
        if (arr) arr.push(i);
        else clusterMembers.set(k, [i]);
      });
    }
    const clusterOrder = [...clusterMembers.keys()];
    const clusterCentroid = new Map<string, { cx: number; cy: number; r: number }>();
    const clusterRingR = Math.sqrt(Math.max(count, 1)) * 60;
    clusterOrder.forEach((k, ci) => {
      const a = (ci / Math.max(clusterOrder.length, 1)) * Math.PI * 2;
      const members = clusterMembers.get(k)!;
      const blobR = 22 + Math.sqrt(members.length) * 18;
      clusterCentroid.set(k, {
        cx: Math.cos(a) * clusterRingR,
        cy: Math.sin(a) * clusterRingR,
        r: blobR,
      });
    });

    // Prepare nodes: assign world positions + pretext-prepare text
    const nodes: PreparedNode[] = data.nodes.map((node, i) => {
      const text = node.label || node.id.substring(0, 12);
      const ck = clusterKeys ? clusterKeys[i] : null;
      const color =
        (ck != null && clusterColorFn?.(ck)) ||
        nodeColorFn?.(node) ||
        node.color ||
        DEFAULT_NODE_COLOR;
      const lc = linkCounts.get(node.id) || 0;
      // Use sqrt for a less aggressive curve — avoids everything being red
      const heat = nodeHeatFn
        ? heatColor(Math.max(0, Math.min(1, nodeHeatFn(node))))
        : heatColor(Math.sqrt(lc / maxLinkCount));

      const seed = hashStr(node.id);
      let wx: number;
      let wy: number;
      const centroid = ck != null ? clusterCentroid.get(ck) : undefined;
      if (centroid) {
        // Scatter the node around its cluster centroid in a small disk.
        const members = clusterMembers.get(ck!)!;
        const j = members.indexOf(i);
        const la = (j / Math.max(members.length, 1)) * Math.PI * 2 + ((seed % 100) / 100) * 0.6;
        const lr = centroid.r * (0.25 + 0.75 * ((Math.abs(seed) % 1000) / 1000));
        wx = centroid.cx + Math.cos(la) * lr;
        wy = centroid.cy + Math.sin(la) * lr;
      } else {
        // Position: use hash of id for deterministic placement, spread in a ring
        const angle = (i / count) * Math.PI * 2 + ((seed % 100) / 100) * 0.5;
        const baseRadius = Math.sqrt(count) * 30;
        const radius = baseRadius * 0.3 + ((Math.abs(seed) % 1000) / 1000) * baseRadius * 0.7;
        wx = Math.cos(angle) * radius + ((seed % 200) - 100) * 0.5;
        wy = Math.sin(angle) * radius + (((seed >> 8) % 200) - 100) * 0.5;
      }

      return {
        node,
        wx,
        wy,
        prepared: prepareWithSegments(text, FONT_SMALL),
        preparedHeight: prepare(text, FONT_SMALL),
        color,
        // When clustering, the dot itself is tinted by the cluster (scope) color
        // so the grouping reads at a glance; otherwise it keeps the heat gradient.
        heatColor: centroid ? color : heat,
        linkCount: lc,
        phase: ((Math.abs(seed) % 1000) / 1000) * Math.PI * 2,
      };
    });

    // Cluster descriptors for the blob overlay (color + label + member indices).
    const clusters = clusterOrder.map((key) => ({
      key,
      members: clusterMembers.get(key)!,
      color: clusterColorFn?.(key) || DEFAULT_NODE_COLOR,
      label: clusterLabelFn?.(key) ?? key,
    }));

    // Build link structures
    const linksIdx: Array<{
      a: number;
      b: number;
      color: string;
      type: string;
    }> = [];
    const byNode = new Map<number, number[]>();

    for (let li = 0; li < data.links.length; li++) {
      const link = data.links[li];
      const ai = nodeIndexMap.get(link.source);
      const bi = nodeIndexMap.get(link.target);
      if (ai === undefined || bi === undefined) continue;

      const color =
        linkColorFn?.(link) ||
        link.color ||
        LINK_TYPE_COLORS[link.type || "semantic"] ||
        DEFAULT_NODE_COLOR;

      const idx = linksIdx.length;
      linksIdx.push({ a: ai, b: bi, color, type: link.type || "semantic" });

      if (!byNode.has(ai)) byNode.set(ai, []);
      if (!byNode.has(bi)) byNode.set(bi, []);
      byNode.get(ai)!.push(idx);
      byNode.get(bi)!.push(idx);
    }

    return {
      preparedNodes: nodes,
      linksByNode: byNode,
      linksWithIndices: linksIdx,
      clusters,
    };
  }, [data, nodeColorFn, linkColorFn, nodeHeatFn, clusterKeyFn, clusterColorFn, clusterLabelFn]);

  // ----- Animation loop -----
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const s = stateRef.current;

    // Smooth interpolation
    s.panX = lerp(s.panX, s.targetPanX, 0.12);
    s.panY = lerp(s.panY, s.targetPanY, 0.12);
    s.zoom = lerp(s.zoom, s.targetZoom, 0.12);

    const { W, H, dpr, zoom, panX, panY, mouseX, mouseY, hoverIndex } = s;
    const cx = W / 2 + panX;
    const cy = H / 2 + panY;
    const margin = 60;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(dpr, dpr);

    const bg = isDark ? "#09090b" : "#ffffff";
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Ambient-motion clock (seconds).
    const time = (typeof performance !== "undefined" ? performance.now() : 0) / 1000;
    // Drift amplitude in world units — nodes slowly wander around their home
    // position so the whole field visibly breathes.
    const DRIFT_AMP = 16;

    // Screen positions (with a slow per-node ambient drift baked in, so links —
    // which read straight from screenX/screenY below — follow for free).
    const screenX = new Float32Array(preparedNodes.length);
    const screenY = new Float32Array(preparedNodes.length);
    const visible = new Uint8Array(preparedNodes.length);

    for (let i = 0; i < preparedNodes.length; i++) {
      const n = preparedNodes[i];
      const driftX = DRIFT_AMP * Math.sin(time * 0.6 + n.phase);
      const driftY = DRIFT_AMP * Math.cos(time * 0.5 + n.phase * 1.3);
      const sx = cx + (n.wx + driftX) * zoom;
      const sy = cy + (n.wy + driftY) * zoom;
      screenX[i] = sx;
      screenY[i] = sy;
      visible[i] = sx > -margin && sx < W + margin && sy > -margin && sy < H + margin ? 1 : 0;
    }

    // Hit test for hover
    if (mouseX >= 0 && !s.isDragging) {
      let bestDist = zoom > 1.5 ? 80 : 30;
      let bestIdx = -1;
      for (let i = 0; i < preparedNodes.length; i++) {
        if (!visible[i]) continue;
        const dx = mouseX - screenX[i];
        const dy = mouseY - screenY[i];
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }
      s.hoverIndex = bestIdx;
    }

    const hoveredLinks = new Set<number>();
    if (hoverIndex >= 0) {
      const myLinks = linksByNode.get(hoverIndex) || [];
      for (const li of myLinks) hoveredLinks.add(li);
    }

    // --- Draw links ---
    const maxLinks = 6000;
    let linksDrawn = 0;

    if (hoverIndex >= 0) {
      // Only draw hovered node's links
      for (const li of hoveredLinks) {
        const link = linksWithIndices[li];
        const ax = screenX[link.a];
        const ay = screenY[link.a];
        const bx = screenX[link.b];
        const by = screenY[link.b];

        ctx.strokeStyle = link.color;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.5;

        const midX = (ax + bx) / 2 + (by - ay) * 0.08;
        const midY = (ay + by) / 2 - (bx - ax) * 0.08;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.quadraticCurveTo(midX, midY, bx, by);
        ctx.stroke();

        // A small bead of light travels the curve from the hovered node outward,
        // so connections read as live signal paths rather than static lines.
        {
          // Phase-offset per link so beads don't march in lockstep. Travel runs
          // from the hovered node (u=0) toward its neighbor (u=1).
          const fromHovered = link.a === hoverIndex;
          const raw = (time * 0.22 + (li % 13) / 13) % 1;
          const u = fromHovered ? raw : 1 - raw;
          const iu = 1 - u;
          // Point on the quadratic Bézier at parameter u.
          const px = iu * iu * ax + 2 * iu * u * midX + u * u * bx;
          const py = iu * iu * ay + 2 * iu * u * midY + u * u * by;
          ctx.globalAlpha = 0.9 * (0.4 + 0.6 * Math.sin(u * Math.PI)); // fade at the ends
          ctx.fillStyle = link.color;
          ctx.shadowColor = link.color;
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
          // Restore the stroke state the loop's next iteration expects.
          ctx.globalAlpha = 0.5;
          ctx.lineWidth = 1.5;
        }
        linksDrawn++;
      }
      ctx.globalAlpha = 1;
    } else {
      // Faint, slow breathing across the whole web so idle links feel alive
      // without flickering (one global sine, not per-link — stays calm).
      const shimmer = 1 + 0.18 * Math.sin(time * 0.6);
      const baseAlpha = (0.06 + Math.min(zoom * 0.04, 0.1)) * shimmer;
      ctx.lineWidth = 0.4;

      for (const link of linksWithIndices) {
        if (linksDrawn >= maxLinks) break;
        const ax = screenX[link.a];
        const ay = screenY[link.a];
        const bx = screenX[link.b];
        const by = screenY[link.b];

        if (
          (ax < -margin && bx < -margin) ||
          (ax > W + margin && bx > W + margin) ||
          (ay < -margin && by < -margin) ||
          (ay > H + margin && by > H + margin)
        )
          continue;

        ctx.strokeStyle = link.color;
        ctx.globalAlpha = baseAlpha;

        const midX = (ax + bx) / 2 + (by - ay) * 0.08;
        const midY = (ay + by) / 2 - (bx - ax) * 0.08;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.quadraticCurveTo(midX, midY, bx, by);
        ctx.stroke();
        linksDrawn++;
      }
      ctx.globalAlpha = 1;
    }

    // --- Draw scope cluster blobs (behind nodes) ---
    // Wrap each cluster's nodes in a translucent, outward-padded convex hull so
    // groups read as soft regions. Tiny clusters (<3 nodes) get a circle instead.
    if (clusters.length > 0) {
      ctx.save();
      ctx.lineJoin = "round";
      for (const cluster of clusters) {
        const pts: Array<[number, number]> = [];
        for (const idx of cluster.members) {
          if (idx < screenX.length) pts.push([screenX[idx], screenY[idx]]);
        }
        if (pts.length === 0) continue;

        // Cluster screen centroid (for outward padding + label placement).
        let mx = 0;
        let my = 0;
        for (const [x, y] of pts) {
          mx += x;
          my += y;
        }
        mx /= pts.length;
        my /= pts.length;

        const pad = 26;
        ctx.fillStyle = hexToRgba(cluster.color, isDark ? 0.1 : 0.08);
        ctx.strokeStyle = hexToRgba(cluster.color, 0.35);
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        if (pts.length < 3) {
          // Circle around the 1–2 points.
          let rad = pad;
          for (const [x, y] of pts) rad = Math.max(rad, Math.hypot(x - mx, y - my) + pad);
          ctx.arc(mx, my, rad, 0, Math.PI * 2);
        } else {
          const hull = convexHull(pts);
          for (let j = 0; j < hull.length; j++) {
            // Push each vertex outward from the centroid for a rounded, padded blob.
            const [hx, hy] = hull[j];
            const dx = hx - mx;
            const dy = hy - my;
            const d = Math.hypot(dx, dy) || 1;
            const ex = hx + (dx / d) * pad;
            const ey = hy + (dy / d) * pad;
            if (j === 0) ctx.moveTo(ex, ey);
            else ctx.lineTo(ex, ey);
          }
          ctx.closePath();
        }
        ctx.fill();
        ctx.stroke();

        // Cluster label at the top of the blob.
        if (cluster.label && zoom > 0.4) {
          let topY = my;
          for (const [, y] of pts) topY = Math.min(topY, y);
          ctx.fillStyle = hexToRgba(cluster.color, isDark ? 0.95 : 0.85);
          ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(cluster.label, mx, topY - pad - 4);
        }
      }
      ctx.restore();
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }

    // --- Draw nodes ---
    // Label deconfliction grid. Compact mode shrinks the cell so short labels
    // (e.g. entity names) can pack more densely.
    const GRID_CELL = compactLabels ? 28 : 90;
    const labelGrid = new Set<string>();

    function canPlace(lx: number, ly: number, lw: number, lh: number) {
      const c0 = Math.floor(lx / GRID_CELL);
      const c1 = Math.floor((lx + lw) / GRID_CELL);
      const r0 = Math.floor(ly / GRID_CELL);
      const r1 = Math.floor((ly + lh) / GRID_CELL);
      for (let c = c0; c <= c1; c++)
        for (let r = r0; r <= r1; r++) if (labelGrid.has(`${c},${r}`)) return false;
      return true;
    }

    function claim(lx: number, ly: number, lw: number, lh: number) {
      const c0 = Math.floor(lx / GRID_CELL);
      const c1 = Math.floor((lx + lw) / GRID_CELL);
      const r0 = Math.floor(ly / GRID_CELL);
      const r1 = Math.floor((ly + lh) / GRID_CELL);
      for (let c = c0; c <= c1; c++) for (let r = r0; r <= r1; r++) labelGrid.add(`${c},${r}`);
    }

    let labelsShown = 0;
    let visibleCount = 0;

    // Draw hovered last (on top)
    const order: number[] = [];
    for (let i = 0; i < preparedNodes.length; i++) {
      if (!visible[i]) continue;
      if (i === hoverIndex) continue;
      order.push(i);
    }
    if (hoverIndex >= 0 && visible[hoverIndex]) order.push(hoverIndex);

    for (const i of order) {
      const n = preparedNodes[i];
      const sx = screenX[i];
      const sy = screenY[i];
      visibleCount++;

      const isHovered = i === hoverIndex;
      const isNeighbor =
        hoveredLinks.size > 0 &&
        (linksByNode.get(i) || []).some((li: number) => hoveredLinks.has(li));

      // Size varies slightly by link count — subtle range like star magnitudes.
      // When nodeSizeFn is provided (e.g. entities view), it overrides linkCount
      // sizing so dots can scale by an external weight like co-occurrence count.
      const rawR = nodeSizeFn ? nodeSizeFn(n.node) : 2.5 + Math.min(n.linkCount * 0.15, 2.5);
      // Gentle pulse — each dot "breathes" in size, out of phase with its
      // neighbors, so the field twinkles like a living star map.
      const pulse = 1 + 0.13 * Math.sin(time * 1.05 + n.phase);
      const baseR = rawR * pulse;
      const r = Math.max(1.5, baseR * Math.min(zoom, 2));

      // Opacity varies — fewer links = dimmer, more links = brighter. A brightness
      // twinkle (offset from the size pulse) makes even tiny dots read as alive,
      // where a radius pulse alone would be imperceptible.
      const twinkleAlpha = 0.82 + 0.18 * Math.sin(time * 1.4 + n.phase * 2.1);
      const baseAlpha = (0.45 + Math.min(n.linkCount * 0.03, 0.5)) * twinkleAlpha;

      // Dot — star-like: heat-gradient color, varied size & opacity
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = n.heatColor;
      ctx.globalAlpha = isHovered ? 1 : isNeighbor ? 0.95 : hoverIndex >= 0 ? 0.08 : baseAlpha;

      if (isHovered || isNeighbor) {
        ctx.shadowColor = n.heatColor;
        ctx.shadowBlur = isHovered ? 20 : 10;
      }
      ctx.fill();

      // Soft glow halo for brighter stars (high link count) — the halo twinkles
      // a little (out of phase with the dot's pulse) so hubs feel radiant.
      if (n.linkCount > 3 && !isHovered && hoverIndex < 0) {
        const twinkle = 1 + 0.25 * Math.sin(time * 0.9 + n.phase * 1.7);
        ctx.beginPath();
        ctx.arc(sx, sy, r * 2, 0, Math.PI * 2);
        ctx.fillStyle = n.heatColor;
        ctx.globalAlpha = (0.06 + Math.min(n.linkCount * 0.005, 0.08)) * twinkle;
        ctx.fill();
      }

      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;

      // Label — always use deconfliction grid to prevent overlap
      if (isHovered) {
        // Hovered node always gets a label (skip deconfliction)
        drawLabel(ctx, n, i, sx, sy, r, true, true, isDark, zoom);
        labelsShown++;
      } else if (zoom > 1.5) {
        const cardW = Math.min(200, 80 + zoom * 25);
        const cardH = 50;
        const lx = sx - cardW / 2;
        const ly = sy + r + 4;
        if (canPlace(lx, ly, cardW, cardH)) {
          claim(lx, ly, cardW, cardH);
          drawLabel(ctx, n, i, sx, sy, r, false, isNeighbor, isDark, zoom);
          labelsShown++;
        }
      } else if (compactLabels || zoom > 0.5 || isNeighbor) {
        // Show inline labels with deconfliction — neighbors included but grid-checked.
        // Compact mode uses a tiny bounding box so short labels pack densely.
        const lw = compactLabels ? 60 : 155;
        const lh = compactLabels ? 12 : 16;
        const lx = sx + r + 5;
        const ly = sy - 8;
        if (canPlace(lx, ly, lw, lh)) {
          claim(lx, ly, lw, lh);
          drawLabel(ctx, n, i, sx, sy, r, false, isNeighbor, isDark, zoom);
          labelsShown++;
        }
      }
    }

    // HUD
    ctx.font = MONO;
    ctx.fillStyle = isDark ? "#71717a" : "#71717a";
    ctx.textAlign = "left";
    ctx.fillText(
      t("hudStats", {
        memories: preparedNodes.length,
        visible: visibleCount,
        labels: labelsShown,
        links: linksDrawn,
        zoom: zoom.toFixed(2),
      }),
      12,
      H - 12
    );

    // Legend — link types
    ctx.textAlign = "right";
    let legendX = W - 12;
    ctx.font = FONT_BOLD;
    const linkTypeLabel: Record<string, string> = {
      semantic: t("linkTypeSemantic"),
      temporal: t("linkTypeTemporal"),
      entity: t("linkTypeEntity"),
      causal: t("linkTypeCausal"),
    };
    for (const [type, color] of Object.entries(LINK_TYPE_COLORS).reverse()) {
      const label = linkTypeLabel[type] ?? type;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = isDark ? "#a1a1aa" : "#52525b";
      ctx.fillText(label, legendX, H - 12);
      legendX -= tw + 4;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(legendX, H - 15, 3, 0, Math.PI * 2);
      ctx.fill();
      legendX -= 14;
    }

    // Heat gradient legend (top-left, below instructions). Suppressed while
    // clustering, where node color encodes the cluster (scope), not the gradient.
    if (clusters.length === 0) {
      ctx.textAlign = "left";
      ctx.font = FONT_BOLD;
      ctx.fillStyle = isDark ? "#a1a1aa" : "#52525b";
      ctx.fillText((heatLegendLabel || t("legendLinks")).toUpperCase(), 12, 36);
      const [heatLo, heatHi] = heatLegendEndpoints || [t("legendFew"), t("legendMany")];
      // Size the bar so both endpoint labels fit without overlap (e.g. ISO dates
      // are wider than "few"/"many"). Min 80px keeps the visual weight stable.
      ctx.font = MONO;
      const heatLoW = ctx.measureText(heatLo).width;
      const heatHiW = ctx.measureText(heatHi).width;
      const gradW = Math.max(80, heatLoW + heatHiW + 12);
      for (let gx = 0; gx < gradW; gx++) {
        ctx.fillStyle = heatColor(gx / gradW);
        ctx.fillRect(12 + gx, 42, 1, 6);
      }
      ctx.fillStyle = isDark ? "#a1a1aa" : "#71717a";
      ctx.fillText(heatLo, 12, 60);
      ctx.textAlign = "right";
      ctx.fillText(heatHi, 12 + gradW, 60);
    }

    // Node-size legend — only shown when the caller maps size to a real dimension.
    // Layout: "few • • ● many" so the labels bracket the dots without overlap.
    if (sizeLegendLabel) {
      ctx.textAlign = "left";
      ctx.font = FONT_BOLD;
      ctx.fillStyle = isDark ? "#a1a1aa" : "#52525b";
      ctx.fillText(sizeLegendLabel.toUpperCase(), 12, 82);
      ctx.font = MONO;
      const labelColor = isDark ? "#a1a1aa" : "#71717a";
      ctx.fillStyle = labelColor;
      const sizeFew = t("legendFew");
      const sizeMany = t("legendMany");
      ctx.fillText(sizeFew, 12, 98);
      const fewW = ctx.measureText(sizeFew).width;
      const dotsStart = 12 + fewW + 8;
      const dotColor = isDark ? "#a1a1aa" : "#52525b";
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(dotsStart + 2, 94, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(dotsStart + 14, 94, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(dotsStart + 30, 94, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = labelColor;
      ctx.fillText(sizeMany, dotsStart + 42, 98);
    }

    // Instructions
    ctx.textAlign = "left";
    ctx.font = MONO;
    ctx.fillStyle = isDark ? "#52525b" : "#a1a1aa";
    ctx.fillText(t("instructions"), 12, 16);

    ctx.restore();

    animRef.current = requestAnimationFrame(animate);
  }, [isDark, preparedNodes, linksWithIndices, linksByNode, clusters, t]);

  // ----- Label drawing helper -----
  function drawLabel(
    ctx: CanvasRenderingContext2D,
    n: PreparedNode,
    _i: number,
    sx: number,
    sy: number,
    r: number,
    isHovered: boolean,
    force: boolean,
    dark: boolean,
    zoom: number
  ) {
    if (zoom > 1.5 || isHovered) {
      // Card mode
      const cardW = Math.min(220, 80 + zoom * 25);
      const textW = cardW - 16;
      const { lines } = layoutWithLines(n.prepared, textW, 15);
      const maxLines = isHovered
        ? Math.min(lines.length, 5)
        : Math.min(lines.length, Math.floor(zoom));
      const cardH = 8 + maxLines * 15 + 8;

      const cardX = sx - cardW / 2;
      const cardY = sy + r + 4;

      ctx.fillStyle = isHovered
        ? dark
          ? "#1c1c1e"
          : "#f4f4f5"
        : dark
          ? "rgba(9,9,11,0.92)"
          : "rgba(255,255,255,0.92)";
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, cardH, 6);
      ctx.fill();

      ctx.strokeStyle = isHovered ? n.color : dark ? "rgba(63,63,70,0.5)" : "rgba(212,212,216,0.6)";
      ctx.lineWidth = isHovered ? 1.5 : 0.5;
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, cardH, 6);
      ctx.stroke();

      if (isHovered) {
        ctx.shadowColor = n.color;
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.roundRect(cardX, cardY, cardW, cardH, 6);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      ctx.font = FONT_SMALL;
      ctx.fillStyle = isHovered ? (dark ? "#e4e4e7" : "#18181b") : dark ? "#71717a" : "#71717a";
      ctx.textAlign = "left";
      for (let j = 0; j < maxLines; j++) {
        ctx.fillText(lines[j].text, cardX + 8, cardY + 8 + j * 15 + 11);
      }
    } else {
      // Inline label
      ctx.font = FONT_SMALL;
      ctx.fillStyle = isHovered ? (dark ? "#e4e4e7" : "#18181b") : dark ? "#52525b" : "#a1a1aa";
      ctx.globalAlpha = isHovered ? 1 : force ? 0.85 : Math.min(1, (zoom - 0.3) * 2.5);
      ctx.textAlign = "left";
      const text = n.node.label || n.node.id.substring(0, 12);
      const label = text.length > 45 ? text.slice(0, 45) + "..." : text;
      ctx.fillText(label, sx + r + 5, sy + 4);
      ctx.globalAlpha = 1;
    }
  }

  // ----- Setup & resize -----
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const W = rect.width;
      const H = rect.height;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      stateRef.current.dpr = dpr;
      stateRef.current.W = W;
      stateRef.current.H = H;
    };

    resize();

    // Auto-fit zoom based on node spread
    if (preparedNodes.length > 0) {
      let maxR = 0;
      for (const n of preparedNodes) {
        const d = Math.sqrt(n.wx * n.wx + n.wy * n.wy);
        if (d > maxR) maxR = d;
      }
      const fitZoom =
        maxR > 0 ? Math.min(stateRef.current.W, stateRef.current.H) / (maxR * 2.5) : 0.5;
      stateRef.current.zoom = fitZoom;
      stateRef.current.targetZoom = fitZoom;
    }

    animRef.current = requestAnimationFrame(animate);

    // Events
    const handleResize = () => resize();

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      stateRef.current.targetZoom = Math.max(
        0.03,
        Math.min(8, stateRef.current.targetZoom * factor)
      );
    };

    const updateTooltip = (mx: number, my: number) => {
      const tip = tooltipRef.current;
      if (!tip) return;
      const s = stateRef.current;
      const idx = s.hoverIndex;

      if (idx < 0 || s.isDragging) {
        tip.style.display = "none";
        s.prevHoverIndex = -1;
        return;
      }

      const node = preparedNodes[idx]?.node;
      if (!node) {
        tip.style.display = "none";
        return;
      }

      // Only rebuild innerHTML when the hovered node changes
      if (idx !== s.prevHoverIndex) {
        s.prevHoverIndex = idx;
        const meta = node.metadata as Record<string, any> | undefined;
        const fullText = meta?.text || node.label || node.id;
        const entities: string[] = meta?.entities
          ? String(meta.entities)
              .split(",")
              .map((e: string) => e.trim())
              .filter(Boolean)
          : [];
        const nodeColor = preparedNodes[idx].heatColor;
        const linkCount = preparedNodes[idx].linkCount;
        const factType = meta?.fact_type || node.group || "memory";
        const context = meta?.context && meta.context !== "N/A" ? meta.context : null;
        const tags: string[] = Array.isArray(meta?.tags) ? meta.tags : [];
        const occurredStart = meta?.occurred_start || null;
        const occurredEnd = meta?.occurred_end || null;
        const mentionedAt = meta?.mentioned_at || null;
        const proofCount = meta?.proof_count || null;
        const documentId = meta?.document_id || null;

        const muted = isDark ? "#a1a1aa" : "#71717a";
        const dimmed = isDark ? "#71717a" : "#a1a1aa";
        const labelStyle = `font-size:10px;color:${dimmed};text-transform:uppercase;letter-spacing:0.04em;font-weight:500`;
        const valStyle = `font-size:12px;color:${isDark ? "#e4e4e7" : "#18181b"}`;
        const rowStyle = `display:flex;justify-content:space-between;align-items:baseline;gap:12px`;

        // Type badge + link count
        let html = `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">`;
        html += `<span style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:${nodeColor};background:${nodeColor}18;padding:2px 8px;border-radius:4px">${factType}</span>`;
        html += `<span style="font-size:10px;color:${muted}">${linkCount} link${linkCount !== 1 ? "s" : ""}</span>`;
        html += `</div>`;

        // Text
        html += `<div style="font-size:12px;line-height:1.6;margin-bottom:8px">${fullText}</div>`;

        // Metadata grid
        html += `<div style="display:flex;flex-direction:column;gap:4px;border-top:1px solid ${isDark ? "#27272a" : "#e4e4e7"};padding-top:8px">`;

        if (context) {
          html += `<div style="${rowStyle}"><span style="${labelStyle}">Context</span><span style="${valStyle}">${context}</span></div>`;
        }

        // Format ISO timestamp as "YYYY-MM-DD HH:MM" — keeps the row compact
        // while still surfacing the time-of-day the user asked for.
        const fmtTs = (iso: string) => iso.slice(0, 16).replace("T", " ");

        if (occurredStart) {
          const start = fmtTs(occurredStart);
          const end = occurredEnd ? fmtTs(occurredEnd) : null;
          const occurredDisplay = end && end !== start ? `${start} → ${end}` : start;
          html += `<div style="${rowStyle}"><span style="${labelStyle}">Occurred</span><span style="${valStyle}">${occurredDisplay}</span></div>`;
        }

        if (mentionedAt) {
          html += `<div style="${rowStyle}"><span style="${labelStyle}">Mentioned</span><span style="${valStyle}">${fmtTs(mentionedAt)}</span></div>`;
        }

        if (proofCount && proofCount > 1) {
          html += `<div style="${rowStyle}"><span style="${labelStyle}">Evidence</span><span style="${valStyle}">${proofCount} sources</span></div>`;
        }

        if (documentId) {
          html += `<div style="${rowStyle}"><span style="${labelStyle}">Document</span><span style="font-size:11px;font-family:monospace;color:${muted}">${String(documentId).slice(0, 12)}...</span></div>`;
        }

        html += `</div>`;

        // Entities
        if (entities.length > 0 && !(entities.length === 1 && entities[0] === "None")) {
          html += `<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">`;
          for (const ent of entities) {
            html += `<span style="font-size:10px;background:${isDark ? "#27272a" : "#f4f4f5"};color:${isDark ? "#d4d4d8" : "#3f3f46"};padding:2px 7px;border-radius:4px">${ent}</span>`;
          }
          html += `</div>`;
        }

        // Tags
        if (tags.length > 0) {
          html += `<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">`;
          for (const tag of tags) {
            html += `<span style="font-size:10px;background:${nodeColor}15;color:${nodeColor};padding:2px 7px;border-radius:4px">#${tag}</span>`;
          }
          html += `</div>`;
        }

        // ID
        html += `<div style="margin-top:8px;font-size:9px;font-family:monospace;color:${dimmed}">${node.id}</div>`;

        tip.innerHTML = html;
      }

      tip.style.display = "block";
      const tipW = tip.offsetWidth;
      const tipH = tip.offsetHeight;
      const tx = Math.min(mx + 16, s.W - tipW - 12);
      const ty = Math.min(my + 16, s.H - tipH - 12);
      tip.style.left = `${Math.max(4, tx)}px`;
      tip.style.top = `${Math.max(4, ty)}px`;
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      stateRef.current.mouseX = mx;
      stateRef.current.mouseY = my;

      if (stateRef.current.isDragging) {
        stateRef.current.targetPanX =
          stateRef.current.panStartX + (e.clientX - stateRef.current.dragStartX);
        stateRef.current.targetPanY =
          stateRef.current.panStartY + (e.clientY - stateRef.current.dragStartY);
        canvas.style.cursor = "grabbing";
        if (tooltipRef.current) tooltipRef.current.style.display = "none";
      } else {
        canvas.style.cursor = stateRef.current.hoverIndex >= 0 ? "pointer" : "default";
        updateTooltip(mx, my);
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      stateRef.current.isDragging = true;
      stateRef.current.dragStartX = e.clientX;
      stateRef.current.dragStartY = e.clientY;
      stateRef.current.panStartX = stateRef.current.panX;
      stateRef.current.panStartY = stateRef.current.panY;
    };

    const handleMouseUp = () => {
      // If we didn't drag far, treat as click
      if (stateRef.current.isDragging) {
        const dx = Math.abs(stateRef.current.panX - stateRef.current.panStartX);
        const dy = Math.abs(stateRef.current.panY - stateRef.current.panStartY);
        if (dx < 3 && dy < 3 && stateRef.current.hoverIndex >= 0) {
          const node = preparedNodes[stateRef.current.hoverIndex]?.node;
          if (node && onNodeClick) onNodeClick(node);
        }
      }
      stateRef.current.isDragging = false;
      canvas.style.cursor = stateRef.current.hoverIndex >= 0 ? "pointer" : "default";
    };

    const handleMouseLeave = () => {
      stateRef.current.mouseX = -1;
      stateRef.current.mouseY = -1;
      stateRef.current.isDragging = false;
      stateRef.current.hoverIndex = -1;
      if (tooltipRef.current) tooltipRef.current.style.display = "none";
    };

    window.addEventListener("resize", handleResize);
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mousedown", handleMouseDown);
    canvas.addEventListener("mouseup", handleMouseUp);
    canvas.addEventListener("mouseleave", handleMouseLeave);

    // The canvas also changes size when its container reflows (e.g. the side
    // panel opening/closing) with no window "resize" event. Observe the element
    // so the backing store is re-measured — otherwise CSS stretches the old
    // bitmap and the text/dots look squeezed.
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => resize()) : null;
    ro?.observe(canvas);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", handleResize);
      ro?.disconnect();
      canvas.removeEventListener("wheel", handleWheel);
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mousedown", handleMouseDown);
      canvas.removeEventListener("mouseup", handleMouseUp);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [animate, preparedNodes, onNodeClick, isFullscreen]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  // Esc to exit fullscreen
  useEffect(() => {
    if (!isFullscreen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullscreen(false);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [isFullscreen]);

  // Re-measure canvas when fullscreen changes
  useEffect(() => {
    if (!canvasRef.current) return;
    // Small delay to let the DOM update
    const timer = setTimeout(() => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvasRef.current!.getBoundingClientRect();
      canvasRef.current!.width = rect.width * dpr;
      canvasRef.current!.height = rect.height * dpr;
      stateRef.current.dpr = dpr;
      stateRef.current.W = rect.width;
      stateRef.current.H = rect.height;
    }, 50);
    return () => clearTimeout(timer);
  }, [isFullscreen]);

  // Hindsight logo, fetched once and cached as a base64 data URL so the
  // exported SVG can inline it and stay self-contained when shared.
  const logoDataUrlRef = useRef<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadDataUrl("/logo.png")
      .then((url) => {
        if (!cancelled) logoDataUrlRef.current = url;
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // ----- SVG export (shareable, branded poster of the whole graph) -----
  // Independent of the live camera: fits every node into a fixed high-res
  // poster on a dark "night sky" — soft bottom glow, plus-grid, glowing node
  // bubbles and the Hindsight logo top-left. No labels/legends/footer, so the
  // graph itself is the hero. Hover dimming is omitted; every node shines.
  const buildSvgString = useCallback(
    (logoDataUrl: string | null): string => {
      const W = 1600;
      const H = 1000;
      const f = (v: number) => Math.round(v * 10) / 10;
      const a = (v: number) => Math.round(v * 100) / 100;

      const n = preparedNodes.length;

      // --- Fit the whole graph: bounds of world coords → centered, scaled view ---
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const nd of preparedNodes) {
        if (nd.wx < minX) minX = nd.wx;
        if (nd.wx > maxX) maxX = nd.wx;
        if (nd.wy < minY) minY = nd.wy;
        if (nd.wy > maxY) maxY = nd.wy;
      }
      if (!Number.isFinite(minX)) {
        minX = minY = -1;
        maxX = maxY = 1;
      }
      const padX = 84;
      const padTop = 128; // clears the logo
      const padBottom = 80;
      const availW = W - 2 * padX;
      const availH = H - padTop - padBottom;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const zoom = Math.min(availW / spanX, availH / spanY);
      const cwx = (minX + maxX) / 2;
      const cwy = (minY + maxY) / 2;
      const ox = W / 2;
      const oy = padTop + availH / 2;
      const sx = new Float32Array(n);
      const sy = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        sx[i] = ox + (preparedNodes[i].wx - cwx) * zoom;
        sy[i] = oy + (preparedNodes[i].wy - cwy) * zoom;
      }
      // Always a dark "night sky" poster (matches the share aesthetic).
      const navy = "#070b16";
      const gridStroke = "#8ea3c8";

      const parts: string[] = [];
      parts.push(
        `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`
      );

      // --- Defs: bottom glow, plus-grid, edge vignette, node bloom ---
      parts.push(
        `<defs>` +
          `<radialGradient id="hs-glow" cx="50%" cy="100%" r="62%">` +
          `<stop offset="0" stop-color="#1d4ed8" stop-opacity="0.42"/>` +
          `<stop offset="0.55" stop-color="#1e3a8a" stop-opacity="0.13"/>` +
          `<stop offset="1" stop-color="#1e3a8a" stop-opacity="0"/>` +
          `</radialGradient>` +
          `<radialGradient id="hs-vignette" cx="50%" cy="46%" r="76%">` +
          `<stop offset="0" stop-color="#000000" stop-opacity="0"/>` +
          `<stop offset="0.82" stop-color="#000000" stop-opacity="0"/>` +
          `<stop offset="1" stop-color="#000000" stop-opacity="0.24"/>` +
          `</radialGradient>` +
          `<pattern id="hs-grid" width="30" height="30" patternUnits="userSpaceOnUse">` +
          `<path d="M15 12 V18 M12 15 H18" stroke="${gridStroke}" stroke-width="0.7" opacity="0.06"/>` +
          `</pattern>` +
          `</defs>`
      );

      // --- Background layers ---
      parts.push(`<rect width="${W}" height="${H}" fill="${navy}"/>`);
      parts.push(`<rect width="${W}" height="${H}" fill="url(#hs-grid)"/>`);
      parts.push(`<rect width="${W}" height="${H}" fill="url(#hs-glow)"/>`);

      // --- Links: exactly as the UI canvas draws them in the no-hover state
      // (thin 0.4px colored quadratic curves at a faint, zoom-derived alpha) ---
      const linkParts: string[] = [];
      const maxLinks = 6000;
      let drawn = 0;
      for (const link of linksWithIndices) {
        if (drawn >= maxLinks) break;
        const axp = sx[link.a];
        const ayp = sy[link.a];
        const bxp = sx[link.b];
        const byp = sy[link.b];
        const midX = (axp + bxp) / 2 + (byp - ayp) * 0.08;
        const midY = (ayp + byp) / 2 - (bxp - axp) * 0.08;
        linkParts.push(
          `<path d="M${f(axp)} ${f(ayp)} Q${f(midX)} ${f(midY)} ${f(bxp)} ${f(byp)}" stroke="${link.color}"/>`
        );
        drawn++;
      }
      const linkAlpha = 0.06 + Math.min(zoom * 0.04, 0.1);
      parts.push(
        `<g fill="none" stroke-width="0.4" opacity="${a(linkAlpha)}">${linkParts.join("")}</g>`
      );

      // --- Nodes: exactly as the UI canvas draws them (solid heat dot, opacity
      // by link count, plus a translucent halo for hubs) — minus hover effects ---
      const haloParts: string[] = [];
      const nodeParts: string[] = [];
      for (let i = 0; i < n; i++) {
        const nd = preparedNodes[i];
        const baseR = nodeSizeFn ? nodeSizeFn(nd.node) : 2.5 + Math.min(nd.linkCount * 0.15, 2.5);
        const r = Math.max(1.5, baseR * Math.min(zoom, 2));
        const alpha = 0.45 + Math.min(nd.linkCount * 0.03, 0.5);
        if (nd.linkCount > 3) {
          const halo = 0.06 + Math.min(nd.linkCount * 0.005, 0.08);
          haloParts.push(
            `<circle cx="${f(sx[i])}" cy="${f(sy[i])}" r="${f(r * 2)}" fill="${nd.heatColor}" opacity="${a(halo)}"/>`
          );
        }
        nodeParts.push(
          `<circle cx="${f(sx[i])}" cy="${f(sy[i])}" r="${f(r)}" fill="${nd.heatColor}" opacity="${a(alpha)}"/>`
        );
      }
      parts.push(`<g>${haloParts.join("")}</g>`);
      parts.push(`<g>${nodeParts.join("")}</g>`);

      // --- Edge vignette to sink the corners, then the Hindsight logo ---
      parts.push(`<rect width="${W}" height="${H}" fill="url(#hs-vignette)"/>`);
      if (logoDataUrl) {
        const logoW = 190;
        const logoH = (logoW * 139) / 529; // preserve logo aspect ratio
        parts.push(
          `<image x="48" y="46" width="${logoW}" height="${f(logoH)}" href="${logoDataUrl}" preserveAspectRatio="xMinYMid meet"/>`
        );
      }

      parts.push(`</svg>`);
      return parts.join("");
    },
    [preparedNodes, linksWithIndices, nodeSizeFn]
  );

  const handleExportSvg = useCallback(async () => {
    let logo = logoDataUrlRef.current;
    if (!logo) {
      try {
        logo = await loadDataUrl("/logo.png");
        logoDataUrlRef.current = logo;
      } catch {
        logo = null;
      }
    }
    const svg = buildSvgString(logo);
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "hindsight-constellation.svg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [buildSvgString]);

  return (
    <div
      ref={wrapperRef}
      style={
        isFullscreen
          ? {
              position: "fixed",
              inset: 0,
              zIndex: 50,
              background: isDark ? "#09090b" : "#ffffff",
            }
          : { position: "relative", width: "100%", height: `${height}px` }
      }
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />

      {/* Toolbar */}
      <div
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          zIndex: 21,
          display: "flex",
          gap: 6,
        }}
      >
        {/* Share as SVG */}
        <button
          onClick={handleExportSvg}
          style={toolbarBtnStyle(isDark)}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.opacity = "1";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.opacity = "0.7";
          }}
          title={t("exportSvgTitle")}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7" />
            <polyline points="8 8 12 4 16 8" />
            <line x1="12" y1="4" x2="12" y2="15" />
          </svg>
          {t("exportSvgLabel")}
        </button>

        {/* Fullscreen toggle */}
        <button
          onClick={toggleFullscreen}
          style={toolbarBtnStyle(isDark)}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.opacity = "1";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.opacity = "0.7";
          }}
          title={isFullscreen ? t("exitFullscreenTitle") : t("enterFullscreenTitle")}
        >
          {isFullscreen ? (
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="4 14 10 14 10 20" />
              <polyline points="20 10 14 10 14 4" />
              <line x1="14" y1="10" x2="21" y2="3" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          ) : (
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          )}
          {isFullscreen ? t("exitFullscreenLabel") : t("enterFullscreenLabel")}
        </button>
      </div>

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        style={{
          display: "none",
          position: "absolute",
          pointerEvents: "none",
          zIndex: 22,
          maxWidth: 360,
          padding: "10px 14px",
          borderRadius: 8,
          fontSize: 12,
          lineHeight: 1.5,
          boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
          border: `1px solid ${isDark ? "#27272a" : "#e4e4e7"}`,
          background: isDark ? "#18181b" : "#ffffff",
          color: isDark ? "#e4e4e7" : "#18181b",
        }}
      />
    </div>
  );
}
