"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { useTranslations } from "next-intl";
import { client, type KnowledgeNode } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FilePlus,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  Loader2,
  Network,
  Trash2,
  X,
} from "lucide-react";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import { CompactMarkdown } from "./compact-markdown";
import { Constellation } from "./constellation";
import type { GraphData, GraphLink, GraphNode } from "./graph-2d";

type ViewMode = "tree" | "graph";
type GraphResponse = Awaited<ReturnType<typeof client.getKnowledgeBaseGraph>>;
type PageDetail = Awaited<ReturnType<typeof client.getKnowledgePage>>;

const FALLBACK_COLOR = "#0074d9";

function flatten(nodes: KnowledgeNode[], out: KnowledgeNode[] = []): KnowledgeNode[] {
  for (const n of nodes) {
    out.push(n);
    if (n.children?.length) flatten(n.children, out);
  }
  return out;
}

export function KnowledgeBaseView() {
  const t = useTranslations("knowledgeBase");
  const { currentBank } = useBank();

  const [roots, setRoots] = useState<KnowledgeNode[]>([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<ViewMode>("tree");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const [selected, setSelected] = useState<PageDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  const [createKind, setCreateKind] = useState<"folder" | "page" | null>(null);
  const [form, setForm] = useState({ name: "", sourceQuery: "", parentId: "" });
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeNode | null>(null);
  const [deleting, setDeleting] = useState(false);

  const loadTree = useCallback(async () => {
    if (!currentBank) return;
    setLoading(true);
    try {
      const result = await client.getKnowledgeTree(currentBank);
      setRoots(result.roots || []);
    } catch {
      // toast handled by interceptor
    } finally {
      setLoading(false);
    }
  }, [currentBank]);

  const loadGraph = useCallback(async () => {
    if (!currentBank) return;
    setGraphLoading(true);
    try {
      setGraph(await client.getKnowledgeBaseGraph(currentBank));
    } catch {
      // toast handled by interceptor
    } finally {
      setGraphLoading(false);
    }
  }, [currentBank]);

  useEffect(() => {
    if (currentBank) {
      setSelected(null);
      setGraph(null);
      loadTree();
    }
  }, [currentBank, loadTree]);

  useEffect(() => {
    if (view === "graph" && currentBank && !graph && !graphLoading) loadGraph();
  }, [view, currentBank, graph, graphLoading, loadGraph]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const allNodes = useMemo(() => flatten(roots), [roots]);
  const folders = useMemo(() => allNodes.filter((n) => n.kind === "folder"), [allNodes]);
  const folderCount = folders.length;
  const pageCount = allNodes.length - folderCount;

  const openPage = useCallback(
    async (pageId: string) => {
      if (!currentBank) return;
      setLoadingDetail(true);
      try {
        setSelected(await client.getKnowledgePage(currentBank, pageId));
      } catch {
        // toast handled by interceptor
      } finally {
        setLoadingDetail(false);
      }
    },
    [currentBank]
  );

  const toggleFolder = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const openCreate = (kind: "folder" | "page", parentId = "") => {
    setForm({ name: "", sourceQuery: "", parentId });
    setCreateKind(kind);
  };

  const handleCreate = async () => {
    if (!currentBank || !createKind || !form.name.trim()) return;
    if (createKind === "page" && !form.sourceQuery.trim()) return;
    setCreating(true);
    try {
      const parent_id = form.parentId || null;
      if (createKind === "folder") {
        await client.createKnowledgeFolder(currentBank, {
          name: form.name.trim(),
          parent_id,
        });
      } else {
        await client.createKnowledgePage(currentBank, {
          name: form.name.trim(),
          source_query: form.sourceQuery.trim(),
          parent_id,
        });
      }
      if (parent_id) setExpanded((prev) => new Set(prev).add(parent_id));
      setCreateKind(null);
      await loadTree();
      setGraph(null);
    } catch {
      // toast handled by interceptor
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!currentBank || !deleteTarget) return;
    setDeleting(true);
    try {
      await client.deleteKnowledgeNode(currentBank, deleteTarget.id);
      if (selected?.id === deleteTarget.id) setSelected(null);
      setDeleteTarget(null);
      await loadTree();
      setGraph(null);
    } catch {
      // toast handled by interceptor
    } finally {
      setDeleting(false);
    }
  };

  const handleExport = async () => {
    if (!currentBank) return;
    setExporting(true);
    try {
      const bundle = await client.exportKnowledgeBase(currentBank);
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentBank}-okf.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      // toast handled by interceptor
    } finally {
      setExporting(false);
    }
  };

  // ── Constellation (graph view) — clustered by parent folder ───────────────
  const typeColors = useMemo(() => {
    const colors = new Map<string, string>();
    for (const n of graph?.nodes ?? []) colors.set(n.data.type, n.data.color);
    return colors;
  }, [graph]);

  const constellationData = useMemo<GraphData>(() => {
    if (!graph) return { nodes: [], links: [] };
    const nodes: GraphNode[] = graph.nodes.map((n) => ({
      id: n.data.id,
      label: n.data.label,
      color: n.data.color,
      group: n.data.type,
    }));
    const links: GraphLink[] = graph.edges.map((e) => ({
      source: e.data.source,
      target: e.data.target,
      color: e.data.color,
      weight: e.data.weight,
    }));
    return { nodes, links };
  }, [graph]);

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
    (node: GraphNode) => 4 + Math.sqrt((nodeWeights.get(node.id) || 0) / maxNodeWeight) * 10,
    [nodeWeights, maxNodeWeight]
  );

  return (
    <div>
      <div className="flex items-start justify-between gap-4 mb-2">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("title")}</h1>
          <p className="text-muted-foreground mt-1">{t("description")}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => openCreate("folder")}>
            <FolderPlus className="w-4 h-4 mr-2" />
            {t("newFolder")}
          </Button>
          <Button variant="outline" size="sm" onClick={() => openCreate("page")}>
            <FilePlus className="w-4 h-4 mr-2" />
            {t("newPage")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exporting || !pageCount}
          >
            {exporting ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Download className="w-4 h-4 mr-2" />
            )}
            {t("exportButton")}
          </Button>
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-muted-foreground">
          {view === "graph"
            ? t("graphCount", { pages: graph?.total_pages ?? 0, links: graph?.total_edges ?? 0 })
            : t("count", { folders: folderCount, pages: pageCount })}
        </div>
        <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
          <button
            onClick={() => setView("tree")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
              view === "tree"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Folder className="w-4 h-4" />
            {t("viewTree")}
          </button>
          <button
            onClick={() => setView("graph")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
              view === "graph"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Network className="w-4 h-4" />
            {t("viewGraph")}
          </button>
        </div>
      </div>

      {view === "tree" ? (
        <div className="border border-border rounded-lg overflow-hidden min-h-[480px]">
          {loading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="w-7 h-7 text-muted-foreground animate-spin" />
            </div>
          ) : roots.length > 0 ? (
            <ul className="py-2">
              {roots.map((node) => (
                <TreeRow
                  key={node.id}
                  node={node}
                  depth={0}
                  expanded={expanded}
                  selectedId={selected?.id ?? null}
                  onToggle={toggleFolder}
                  onOpenPage={openPage}
                  onAddChild={openCreate}
                  onDelete={setDeleteTarget}
                  t={t}
                />
              ))}
            </ul>
          ) : (
            <div className="flex items-center justify-center py-24">
              <div className="text-center max-w-md px-6">
                <FolderOpen className="w-8 h-8 mx-auto mb-3 text-muted-foreground opacity-60" />
                <div className="text-sm text-muted-foreground">{t("empty")}</div>
                <div className="text-xs text-muted-foreground mt-1">{t("emptyHint")}</div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="border border-border rounded-lg overflow-hidden">
          {graphLoading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="w-7 h-7 text-muted-foreground animate-spin" />
            </div>
          ) : constellationData.nodes.length > 0 ? (
            <Constellation
              data={constellationData}
              height={620}
              onNodeClick={(node) => openPage(node.id)}
              nodeSizeFn={nodeSizeFn}
              clusterKeyFn={(node) => node.group ?? null}
              clusterColorFn={(key) => typeColors.get(key) || FALLBACK_COLOR}
              clusterLabelFn={(key) => key}
              sizeLegendLabel={t("sizeLegendLabel")}
              compactLabels
            />
          ) : (
            <div className="flex items-center justify-center py-24">
              <div className="text-sm text-muted-foreground">{t("empty")}</div>
            </div>
          )}
        </div>
      )}

      {/* Page detail panel */}
      {(selected || loadingDetail) && (
        <div className="fixed right-0 top-0 h-screen w-[460px] bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
          <div className="p-5">
            <div className="flex justify-between items-start mb-4 pb-4 border-b border-border">
              <div className="min-w-0 flex-1">
                <h3 className="text-xl font-bold text-card-foreground truncate">
                  {selected?.name ?? t("loadingPage")}
                </h3>
                {selected?.description && (
                  <p className="text-sm text-muted-foreground mt-1 italic">
                    &ldquo;{selected.description}&rdquo;
                  </p>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelected(null)}
                className="h-8 w-8 p-0 flex-shrink-0"
                aria-label={t("close")}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            {loadingDetail && !selected ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-6 h-6 text-muted-foreground animate-spin" />
              </div>
            ) : selected ? (
              <div className="space-y-4">
                {selected.tags.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap text-xs">
                    {selected.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                {selected.timestamp && (
                  <div
                    className="text-xs text-muted-foreground"
                    title={formatAbsoluteDateTime(selected.timestamp)}
                  >
                    {t("updatedLabel")} {formatRelativeTime(selected.timestamp)}
                  </div>
                )}
                {selected.body ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none border-t border-border pt-4">
                    <CompactMarkdown>{selected.body}</CompactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground italic border-t border-border pt-4">
                    {t("noBody")}
                  </p>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Create folder/page dialog */}
      <Dialog open={createKind !== null} onOpenChange={(o) => !o && setCreateKind(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {createKind === "folder" ? t("createFolderTitle") : t("createPageTitle")}
            </DialogTitle>
            <DialogDescription>{t("description")}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">{t("fieldName")}</label>
              <Input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                autoFocus
              />
            </div>
            {createKind === "page" && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">
                  {t("fieldSourceQuery")}
                </label>
                <Textarea
                  value={form.sourceQuery}
                  onChange={(e) => setForm({ ...form, sourceQuery: e.target.value })}
                  className="min-h-[100px]"
                />
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">{t("fieldParent")}</label>
              <Select
                value={form.parentId || "__root__"}
                onValueChange={(v) => setForm({ ...form, parentId: v === "__root__" ? "" : v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__root__">{t("rootFolder")}</SelectItem>
                  {folders.map((f) => (
                    <SelectItem key={f.id} value={f.id}>
                      {f.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateKind(null)} disabled={creating}>
              {t("cancel")}
            </Button>
            <Button
              onClick={handleCreate}
              disabled={
                creating || !form.name.trim() || (createKind === "page" && !form.sourceQuery.trim())
              }
            >
              {creating ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null}
              {t("create")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation */}
      <AlertDialog open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteButton")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteConfirm", { name: deleteTarget?.name ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">{t("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null}
              {t("deleteButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function TreeRow({
  node,
  depth,
  expanded,
  selectedId,
  onToggle,
  onOpenPage,
  onAddChild,
  onDelete,
  t,
}: {
  node: KnowledgeNode;
  depth: number;
  expanded: Set<string>;
  selectedId: string | null;
  onToggle: (id: string) => void;
  onOpenPage: (id: string) => void;
  onAddChild: (kind: "folder" | "page", parentId: string) => void;
  onDelete: (node: KnowledgeNode) => void;
  t: ReturnType<typeof useTranslations>;
}) {
  const isFolder = node.kind === "folder";
  const isOpen = expanded.has(node.id);
  const isActive = selectedId === node.id;

  return (
    <li>
      <div
        className={`group flex items-center gap-1.5 pr-2 py-1.5 cursor-pointer border-l-2 transition-colors ${
          isActive
            ? "bg-primary/10 border-primary text-foreground"
            : "border-transparent hover:bg-muted text-foreground"
        }`}
        style={{ paddingLeft: `${depth * 18 + 10}px` }}
        onClick={() => (isFolder ? onToggle(node.id) : onOpenPage(node.id))}
      >
        {isFolder ? (
          <>
            {isOpen ? (
              <ChevronDown className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
            )}
            {isOpen ? (
              <FolderOpen className="w-4 h-4 flex-shrink-0 text-amber-500" />
            ) : (
              <Folder className="w-4 h-4 flex-shrink-0 text-amber-500" />
            )}
          </>
        ) : (
          <>
            <span className="w-3.5 flex-shrink-0" />
            <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
          </>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm truncate">{node.name}</span>
            {!isFolder && node.managed && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-violet-500/10 text-violet-600 dark:text-violet-400 flex-shrink-0">
                {t("autoBadge")}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
          {isFolder && (
            <>
              <button
                className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                title={t("newFolder")}
                onClick={(e) => {
                  e.stopPropagation();
                  onAddChild("folder", node.id);
                }}
              >
                <FolderPlus className="w-3.5 h-3.5" />
              </button>
              <button
                className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                title={t("newPage")}
                onClick={(e) => {
                  e.stopPropagation();
                  onAddChild("page", node.id);
                }}
              >
                <FilePlus className="w-3.5 h-3.5" />
              </button>
            </>
          )}
          <button
            className="p-1 rounded hover:bg-background text-muted-foreground hover:text-red-600"
            title={t("deleteButton")}
            onClick={(e) => {
              e.stopPropagation();
              onDelete(node);
            }}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      {isFolder && isOpen && node.children?.length > 0 && (
        <ul>
          {node.children.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              selectedId={selectedId}
              onToggle={onToggle}
              onOpenPage={onOpenPage}
              onAddChild={onAddChild}
              onDelete={onDelete}
              t={t}
            />
          ))}
        </ul>
      )}
    </li>
  );
}
