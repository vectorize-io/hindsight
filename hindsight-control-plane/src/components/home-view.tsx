"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { client, type KnowledgeNode } from "@/lib/api";
import { bankRoute } from "@/lib/bank-url";
import { Constellation } from "./constellation";
import type { GraphData, GraphNode, GraphLink } from "./graph-data";
import { Button } from "@/components/ui/button";
import { Loader2, Network, FileText, Layers, FilePlus, ArrowRight } from "lucide-react";
import { formatRelativeTime, formatAbsoluteDateTime } from "@/lib/relative-time";
import { MemoryStoreCard, MemoriesActivityChart, type BankStats } from "./bank-stats-view";
import { TreeRow } from "./knowledge-base-view";
import { useFeatures } from "@/lib/features-context";

const FALLBACK_COLOR = "#0074d9";
// Cluster the memory constellation by fact type, matching the memories charts.
const TYPE_COLORS: Record<string, string> = {
  world: "#8b5cf6",
  experience: "#ec4899",
  observation: "#6366f1",
  entity: "#0ea5e9",
};

// Documents have no standard name field (metadata is free-form and varies by
// source), so recent-docs shows the document id.
type DocItem = { id?: string; created_at?: string | null };

function flattenPages(nodes: KnowledgeNode[], out: KnowledgeNode[] = []): KnowledgeNode[] {
  for (const n of nodes) {
    if (n.kind === "page") out.push(n);
    if (n.children?.length) flattenPages(n.children, out);
  }
  return out;
}

function flattenAll(nodes: KnowledgeNode[], out: KnowledgeNode[] = []): KnowledgeNode[] {
  for (const n of nodes) {
    out.push(n);
    if (n.children?.length) flattenAll(n.children, out);
  }
  return out;
}

export function HomeView({
  bankId,
  onNavigate,
}: {
  bankId: string;
  onNavigate: (tab: string) => void;
}) {
  const t = useTranslations("home");
  const tk = useTranslations("knowledgeBase");
  const router = useRouter();
  const { features } = useFeatures();
  const observationsEnabled = features?.observations ?? false;
  const [stats, setStats] = useState<BankStats | null>(null);
  const [graph, setGraph] = useState<{ nodes?: unknown[]; edges?: unknown[] } | null>(null);
  const [roots, setRoots] = useState<KnowledgeNode[]>([]);
  const [pages, setPages] = useState<KnowledgeNode[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [loading, setLoading] = useState(true);

  const toggleFolder = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    // Best-effort in parallel: a slow/failing panel shouldn't blank the whole page.
    Promise.allSettled([
      client.getBankStats(bankId),
      // High cap so the constellation reflects the full memory set (matches the
      // memories views); the graph endpoint returns up to this many units.
      client.getGraph({ bank_id: bankId, limit: 1000 }),
      client.getKnowledgeTree(bankId),
      client.listDocuments({ bank_id: bankId, limit: 6 }),
    ]).then(([s, g, k, d]) => {
      if (cancelled) return;
      if (s.status === "fulfilled") setStats(s.value as BankStats);
      if (g.status === "fulfilled") setGraph(g.value as { nodes?: unknown[]; edges?: unknown[] });
      if (k.status === "fulfilled") {
        const r = (k.value as { roots?: KnowledgeNode[] }).roots || [];
        setRoots(r);
        const flat = flattenPages(r);
        setPages(flat);
        // Expand all folders so the Home card shows the full structure at a glance.
        setExpanded(
          new Set(
            flattenAll(r)
              .filter((n) => n.kind === "folder")
              .map((n) => n.id)
          )
        );
      }
      if (d.status === "fulfilled") setDocs((d.value as { items?: DocItem[] }).items || []);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [bankId]);

  // The dataplane graph is cytoscape-style ({ data: {...} } per node/edge); map
  // it to the Constellation's flat GraphData, clustering by fact type.
  const constellationData = useMemo<GraphData>(() => {
    if (!graph) return { nodes: [], links: [] };
    const nodes: GraphNode[] = (graph.nodes || []).map((raw) => {
      const n =
        (raw as { data?: Record<string, unknown> }).data ?? (raw as Record<string, unknown>);
      return {
        id: String(n.id),
        label: (n.label as string) ?? undefined,
        color: (n.color as string) ?? undefined,
        group: (n.type as string) ?? undefined,
      };
    });
    const links: GraphLink[] = (graph.edges || []).map((raw) => {
      const e =
        (raw as { data?: Record<string, unknown> }).data ?? (raw as Record<string, unknown>);
      return {
        source: String(e.source),
        target: String(e.target),
        weight: typeof e.weight === "number" ? (e.weight as number) : undefined,
      };
    });
    return { nodes, links };
  }, [graph]);

  const nodeWeights = useMemo(() => {
    const w = new Map<string, number>();
    for (const l of constellationData.links) {
      const v = typeof l.weight === "number" && l.weight > 0 ? l.weight : 1;
      w.set(l.source, (w.get(l.source) || 0) + v);
      w.set(l.target, (w.get(l.target) || 0) + v);
    }
    return w;
  }, [constellationData]);
  const maxWeight = useMemo(() => Math.max(1, ...nodeWeights.values()), [nodeWeights]);
  const nodeSizeFn = useCallback(
    (node: GraphNode) => 4 + Math.sqrt((nodeWeights.get(node.id) || 0) / maxWeight) * 9,
    [nodeWeights, maxWeight]
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-7 h-7 text-muted-foreground animate-spin" />
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
      <p className="text-muted-foreground mb-6">
        {t("subtitle")} <span className="font-mono text-foreground">{bankId}</span>
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:h-[520px]">
        {/* Memory constellation — fills the row height so it never leaves a gap. */}
        <div className="lg:col-span-2 rounded-lg border border-border overflow-hidden flex flex-col lg:h-full">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-border shrink-0">
            <Network className="w-4 h-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">{t("constellationTitle")}</h2>
          </div>
          {constellationData.nodes.length > 0 ? (
            <Constellation
              data={constellationData}
              height={464}
              onNodeClick={() => onNavigate("data")}
              nodeSizeFn={nodeSizeFn}
              clusterKeyFn={(node) => node.group ?? null}
              clusterColorFn={(key) => TYPE_COLORS[key] || FALLBACK_COLOR}
              clusterLabelFn={(key) => key}
            />
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
              {t("constellationEmpty")}
            </div>
          )}
        </div>

        {/* Right column: knowledge pages + recent documents (share the row height). */}
        <div className="flex flex-col gap-4 lg:h-full min-h-0">
          <div className="rounded-lg border border-border flex-1 min-h-0 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
              <h2 className="text-sm font-semibold text-foreground">{t("pagesTitle")}</h2>
              {pages.length > 0 && (
                <button
                  onClick={() => onNavigate("knowledge")}
                  className="text-xs text-primary hover:underline flex items-center gap-1"
                >
                  {t("viewAll")} <ArrowRight className="w-3 h-3" />
                </button>
              )}
            </div>
            {pages.length > 0 ? (
              // Same folder/page tree (TOC) as the Pages tab, read-only: shows the
              // structure + freshness badges; clicking a page opens the Knowledge tab.
              <ul className="py-2 flex-1 min-h-0 overflow-y-auto">
                {roots.map((node) => (
                  <TreeRow
                    key={node.id}
                    node={node}
                    depth={0}
                    readOnly
                    expanded={expanded}
                    selectedId={null}
                    onToggle={toggleFolder}
                    onOpenPage={(pageId) =>
                      router.push(
                        bankRoute(bankId, `?view=knowledge&knowledgeTab=pages&page=${pageId}`)
                      )
                    }
                    t={tk}
                  />
                ))}
              </ul>
            ) : (
              <div className="px-4 py-8 text-center">
                <Layers className="w-7 h-7 mx-auto mb-2 text-muted-foreground opacity-60" />
                <p className="text-sm text-muted-foreground mb-3">{t("pagesEmpty")}</p>
                <Button size="sm" onClick={() => onNavigate("knowledge")}>
                  <FilePlus className="w-4 h-4 mr-2" />
                  {t("pagesEmptyCta")}
                </Button>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border flex-1 min-h-0 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
              <h2 className="text-sm font-semibold text-foreground">{t("recentDocsTitle")}</h2>
              {docs.length > 0 && (
                <button
                  onClick={() => onNavigate("documents")}
                  className="text-xs text-primary hover:underline flex items-center gap-1"
                >
                  {t("viewAll")} <ArrowRight className="w-3 h-3" />
                </button>
              )}
            </div>
            {docs.length > 0 ? (
              <ul className="p-2 flex-1 min-h-0 overflow-y-auto">
                {docs.map((d, i) => (
                  <li key={`${d.id}-${i}`}>
                    <button
                      onClick={() => onNavigate("documents")}
                      className="w-full flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-muted text-left"
                    >
                      <span className="inline-flex items-center gap-1.5 min-w-0">
                        <FileText className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                        <span className="text-xs font-mono truncate" title={d.id}>
                          {d.id}
                        </span>
                      </span>
                      {d.created_at && (
                        <span
                          className="text-[11px] text-muted-foreground flex-shrink-0"
                          title={formatAbsoluteDateTime(d.created_at)}
                        >
                          {formatRelativeTime(d.created_at)}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                {t("recentDocsEmpty")}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Memory stats demoted below the constellation + knowledge/docs. */}
      {stats && (
        <div className="mt-4">
          <MemoryStoreCard stats={stats} observationsEnabled={observationsEnabled} />
        </div>
      )}

      {/* Memories by ingested time — reuse the bank-profile activity chart. */}
      <div className="mt-4">
        <MemoriesActivityChart bankId={bankId} observationsEnabled={observationsEnabled} />
      </div>
    </div>
  );
}
