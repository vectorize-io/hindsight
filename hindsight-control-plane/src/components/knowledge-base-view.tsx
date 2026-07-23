"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { useSearchParams } from "next/navigation";
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
  FilePlus,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  Info,
  Loader2,
  Network,
  Trash2,
  X,
} from "lucide-react";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import { CompactMarkdown } from "./compact-markdown";
import { MentalModelDetailModal } from "./mental-model-detail-modal";
import { Constellation } from "./constellation";
import type { GraphData, GraphNode, GraphLink } from "./graph-data";

type PageDetail = Awaited<ReturnType<typeof client.getKnowledgePage>>;

// The synthetic top-level folder representing the bank root. Its id is "" so
// "add child" under it creates a node at the root (parent_id null).
const ROOT_ID = "";
// How often the tree view silently re-polls so page stats (last refresh / sync
// status) stay live while pages refresh in the background.
const AUTO_REFRESH_MS = 12000;

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
  const searchParams = useSearchParams();
  const pageParam = searchParams.get("page");

  const [roots, setRoots] = useState<KnowledgeNode[]>([]);
  const [loading, setLoading] = useState(false);
  // Root folder starts expanded so its contents are visible by default.
  const [expanded, setExpanded] = useState<Set<string>>(new Set([ROOT_ID]));

  // Obsidian-style editor tabs: multiple pages open at once; `activeId` is focused.
  const [tabs, setTabs] = useState<PageDetail[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const selected = useMemo(() => tabs.find((tb) => tb.id === activeId) ?? null, [tabs, activeId]);
  // Provenance: how many source memories grounded the open page's last synthesis.
  // Derived from the backing mental model's reflect_response (not the page endpoint).
  const [supportingCount, setSupportingCount] = useState(0);
  const [selectedMmId, setSelectedMmId] = useState<string | null>(null);
  // Non-null while the provenance dialog (the backing model's based_on) is open.
  const [provenanceMmId, setProvenanceMmId] = useState<string | null>(null);
  // Mirror of open tabs for the auto-refresh interval / openPage without re-arming.
  const tabsRef = useRef<PageDetail[]>([]);
  useEffect(() => {
    tabsRef.current = tabs;
  }, [tabs]);
  // Ensures we auto-open the first page only once per bank (not on every poll,
  // and not fighting the user after they close a page).
  const autoSelectedRef = useRef(false);

  const [createKind, setCreateKind] = useState<"folder" | "page" | null>(null);
  const [form, setForm] = useState({ name: "", sourceQuery: "", parentId: "" });
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeNode | null>(null);
  const [deleting, setDeleting] = useState(false);

  // `silent` skips the loading spinner so the background auto-refresh poll
  // doesn't flicker the tree every tick.
  const loadTree = useCallback(
    async (opts?: { silent?: boolean }) => {
      if (!currentBank) return;
      if (!opts?.silent) setLoading(true);
      try {
        const result = await client.getKnowledgeTree(currentBank);
        setRoots(result.roots || []);
      } catch {
        // toast handled by interceptor
      } finally {
        if (!opts?.silent) setLoading(false);
      }
    },
    [currentBank]
  );

  useEffect(() => {
    if (currentBank) {
      setTabs([]);
      setActiveId(null);
      loadTree();
    }
  }, [currentBank, loadTree]);

  // Auto-refresh: silently re-poll the tree (freshness badges) and refresh every
  // open tab's content on the same tick.
  useEffect(() => {
    if (!currentBank) return;
    const id = setInterval(() => {
      loadTree({ silent: true });
      tabsRef.current.forEach((tb) =>
        client
          .getKnowledgePage(currentBank, tb.id)
          .then((p) => setTabs((prev) => prev.map((x) => (x.id === p.id ? p : x))))
          .catch(() => {})
      );
    }, AUTO_REFRESH_MS);
    return () => clearInterval(id);
  }, [currentBank, loadTree]);

  const allNodes = useMemo(() => flatten(roots), [roots]);
  // A synthetic top-level folder for the bank so the root itself is visible and
  // you can add folders/pages directly under it. Its "" id makes add-child create
  // at the root; it's never deletable (TreeRow hides delete for the root).
  const rootNode = useMemo<KnowledgeNode>(
    () => ({
      id: ROOT_ID,
      kind: "folder",
      name: currentBank || "/",
      parent_id: null,
      mental_model_id: null,
      managed: false,
      description: null,
      tags: [],
      timestamp: null,
      is_stale: null,
      children: roots,
    }),
    [currentBank, roots]
  );
  const folders = useMemo(() => allNodes.filter((n) => n.kind === "folder"), [allNodes]);

  // Graph view: pages linked by shared source memories.
  const [view, setView] = useState<"tree" | "graph">("tree");
  const [graph, setGraph] = useState<Awaited<
    ReturnType<typeof client.getKnowledgeBaseGraph>
  > | null>(null);
  const [memLinks, setMemLinks] = useState<GraphLink[]>([]);
  const [graphLoading, setGraphLoading] = useState(false);
  useEffect(() => {
    if (view !== "graph" || !currentBank) return;
    setGraphLoading(true);
    // Fetch the page→memory constellation + the memory graph; keep only the
    // memory-memory edges whose endpoints are both in the knowledge base.
    Promise.all([
      client.getKnowledgeBaseGraph(currentBank),
      client.getGraph({ bank_id: currentBank, limit: 1000 }) as Promise<{ edges?: unknown[] }>,
    ])
      .then(([kg, mg]) => {
        setGraph(kg);
        const ids = new Set(kg.nodes.map((n) => n.data.id));
        const links: GraphLink[] = (mg.edges ?? [])
          .map(
            (raw) =>
              (raw as { data?: Record<string, unknown> }).data ?? (raw as Record<string, unknown>)
          )
          .filter((e) => ids.has(String(e.source)) && ids.has(String(e.target)))
          .map((e) => ({ source: String(e.source), target: String(e.target) }));
        setMemLinks(links);
      })
      .catch(() => {})
      .finally(() => setGraphLoading(false));
  }, [view, currentBank]);
  // Memories as points; each carries the pages it grounds (for the Venn hulls).
  const constellationData = useMemo<GraphData>(() => {
    if (!graph) return { nodes: [], links: [] };
    const nodes: GraphNode[] = graph.nodes.map((n) => ({
      id: n.data.id,
      label: n.data.label,
      metadata: { pages: n.data.pages },
    }));
    return { nodes, links: memLinks };
  }, [graph, memLinks]);
  // Stable colour per page (each page = one Euler circle).
  const pageColors = useMemo(() => {
    const palette = [
      "#8b5cf6",
      "#ec4899",
      "#0ea5e9",
      "#10b981",
      "#f59e0b",
      "#ef4444",
      "#6366f1",
      "#14b8a6",
      "#f97316",
      "#a855f7",
    ];
    const map = new Map<string, string>();
    const pages = Array.from(new Set((graph?.nodes ?? []).flatMap((n) => n.data.pages))).sort();
    pages.forEach((p, i) => map.set(p, palette[i % palette.length]));
    return map;
  }, [graph]);

  // Sync status for the open page, read from the tree (the page detail response
  // doesn't carry it); updates as the auto-refresh poll refreshes the tree.
  const selectedStale = useMemo(
    () => (selected ? (allNodes.find((n) => n.id === selected.id)?.is_stale ?? null) : null),
    [selected, allNodes]
  );

  // Provenance count: fetch the open page's backing mental model and sum the
  // source memories (world/experience/observation) in its reflect_response —
  // no extra field on the page endpoint.
  useEffect(() => {
    if (!selected || !currentBank) {
      setSupportingCount(0);
      return;
    }
    const mmId = allNodes.find((n) => n.id === selected.id)?.mental_model_id;
    setSelectedMmId(mmId ?? null);
    if (!mmId) {
      setSupportingCount(0);
      return;
    }
    let cancelled = false;
    client
      .getMentalModel(currentBank, mmId)
      .then((mm) => {
        if (cancelled) return;
        const basedOn = mm.reflect_response?.based_on ?? {};
        const count = (["world", "experience", "observation"] as const).reduce(
          (sum, ft) => sum + (basedOn[ft]?.length ?? 0),
          0
        );
        setSupportingCount(count);
      })
      .catch(() => setSupportingCount(0));
    return () => {
      cancelled = true;
    };
  }, [selected, allNodes, currentBank]);

  const openPage = useCallback(
    async (pageId: string) => {
      if (!currentBank) return;
      setActiveId(pageId);
      // Already open → just focus its tab (the poll keeps it fresh).
      if (tabsRef.current.some((tb) => tb.id === pageId)) return;
      setLoadingDetail(true);
      try {
        const page = await client.getKnowledgePage(currentBank, pageId);
        setTabs((prev) =>
          prev.some((tb) => tb.id === page.id)
            ? prev.map((tb) => (tb.id === page.id ? page : tb))
            : [...prev, page]
        );
        setActiveId(page.id);
      } catch {
        // toast handled by interceptor
      } finally {
        setLoadingDetail(false);
      }
    },
    [currentBank]
  );

  const closeTab = useCallback((id: string) => {
    const cur = tabsRef.current;
    const idx = cur.findIndex((tb) => tb.id === id);
    const next = cur.filter((tb) => tb.id !== id);
    setTabs(next);
    // If the closed tab was active, focus its neighbour (right, then left).
    setActiveId((a) => (a === id ? (next[idx]?.id ?? next[idx - 1]?.id ?? null) : a));
  }, []);

  // Deep-link: open the page named in ?page= (e.g. navigated from the Home card).
  useEffect(() => {
    if (pageParam && currentBank) openPage(pageParam);
  }, [pageParam, currentBank, openPage]);

  // Reset the once-per-bank auto-select guard when switching banks.
  useEffect(() => {
    autoSelectedRef.current = false;
  }, [currentBank]);

  // Open the first page on entry so the content pane isn't empty — unless a page
  // is deep-linked or already open. Fires once per bank (guarded), so closing a
  // page doesn't snap it back open.
  useEffect(() => {
    if (autoSelectedRef.current || pageParam || selected || !allNodes.length) return;
    const firstPage = allNodes.find((n) => n.kind === "page");
    if (firstPage) {
      autoSelectedRef.current = true;
      openPage(firstPage.id);
    }
  }, [allNodes, pageParam, selected, openPage]);

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
      closeTab(deleteTarget.id);
      setDeleteTarget(null);
      await loadTree();
    } catch {
      // toast handled by interceptor
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div>
      <div className="flex justify-end mb-3">
        <div className="flex items-center gap-1 bg-muted rounded-lg p-1">
          <button
            onClick={() => setView("tree")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-1.5 transition-all ${
              view === "tree"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <FileText className="w-4 h-4" />
            {t("viewTree")}
          </button>
          <button
            onClick={() => setView("graph")}
            className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-1.5 transition-all ${
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

      {view === "graph" ? (
        <div className="border border-border rounded-lg overflow-hidden h-[calc(100vh-13rem)] min-h-[520px]">
          {graphLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-7 h-7 text-muted-foreground animate-spin" />
            </div>
          ) : constellationData.nodes.length > 0 ? (
            <Constellation
              data={constellationData}
              height={620}
              nodeGroupsFn={(n) => (n.metadata?.pages as string[]) ?? []}
              groupColorFn={(p) => pageColors.get(p) || "#0074d9"}
              groupLabelFn={(p) => p}
            />
          ) : (
            <div className="flex items-center justify-center h-full px-6 text-center text-sm text-muted-foreground">
              {t("graphEmpty")}
            </div>
          )}
        </div>
      ) : (
        /* Obsidian-style workspace: one frame, a file-explorer sidebar + editor pane. */
        <div className="flex items-stretch border border-border rounded-lg overflow-hidden h-[calc(100vh-13rem)] min-h-[520px]">
          <aside className="w-64 flex-shrink-0 bg-muted/30 border-r border-border overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-6 h-6 text-muted-foreground animate-spin" />
              </div>
            ) : (
              <ul className="py-2">
                <TreeRow
                  node={rootNode}
                  depth={0}
                  isRoot
                  expanded={expanded}
                  selectedId={selected?.id ?? null}
                  onToggle={toggleFolder}
                  onOpenPage={openPage}
                  onAddChild={openCreate}
                  onDelete={setDeleteTarget}
                  t={t}
                />
                {roots.length === 0 && (
                  <li
                    className="text-xs text-muted-foreground italic"
                    style={{ paddingLeft: `${1 * 18 + 34}px` }}
                  >
                    {t("emptyHint")}
                  </li>
                )}
              </ul>
            )}
          </aside>

          <main className="flex-1 min-w-0 overflow-y-auto bg-background">
            {/* Editor tabs — open pages, click to focus, × to close. */}
            {tabs.length > 0 && (
              <div className="sticky top-0 z-10 flex items-stretch border-b border-border bg-muted/20 overflow-x-auto">
                {tabs.map((tb) => (
                  <div
                    key={tb.id}
                    onClick={() => setActiveId(tb.id)}
                    className={`group flex items-center gap-2 px-3 py-2 border-r border-border cursor-pointer text-sm whitespace-nowrap ${
                      tb.id === activeId
                        ? "bg-background text-foreground"
                        : "text-muted-foreground hover:bg-background/50"
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate max-w-[160px]">{tb.name}</span>
                    <button
                      className="rounded p-0.5 opacity-0 group-hover:opacity-100 hover:bg-muted hover:text-foreground"
                      onClick={(e) => {
                        e.stopPropagation();
                        closeTab(tb.id);
                      }}
                      aria-label={t("close")}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {loadingDetail && !selected ? (
              <div className="flex items-center justify-center py-24">
                <Loader2 className="w-7 h-7 text-muted-foreground animate-spin" />
              </div>
            ) : selected ? (
              <div className="p-8 max-w-3xl mx-auto">
                <h1 className="text-2xl font-bold text-foreground">{selected.name}</h1>

                {/* Provenance: this wiki isn't written, it's grounded. Links back
                  into the memory substrate the page was synthesized from. */}
                {supportingCount > 0 && selectedMmId && (
                  <button
                    onClick={() => setProvenanceMmId(selectedMmId)}
                    className="mt-1.5 inline-flex items-center gap-1.5 text-xs text-primary hover:underline"
                  >
                    {t("backedBy", { count: supportingCount })}
                  </button>
                )}

                {/* Freshness + tags. The generation prompt (machinery) is tucked
                  behind the expander so the page opens with the knowledge. */}
                <div className="flex items-center gap-2 flex-wrap text-xs mt-2">
                  {selected.timestamp ? (
                    <span
                      className="text-muted-foreground"
                      title={formatAbsoluteDateTime(selected.timestamp)}
                    >
                      {t("updatedLabel")} {formatRelativeTime(selected.timestamp)}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">{t("generating")}</span>
                  )}
                  {selectedStale === false ? (
                    <span className="px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                      {t("inSync")}
                    </span>
                  ) : selectedStale === true ? (
                    <span className="px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400">
                      {t("needsRefresh")}
                    </span>
                  ) : null}
                  {selected.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {selected.description && (
                  <details className="mt-2">
                    <summary className="text-xs text-muted-foreground cursor-pointer list-none inline-flex items-center gap-1 hover:text-foreground [&::-webkit-details-marker]:hidden">
                      <Info className="w-3 h-3" />
                      {t("howDerived")}
                    </summary>
                    <p className="text-xs text-muted-foreground mt-1.5 italic pl-3 border-l-2 border-border">
                      &ldquo;{selected.description}&rdquo;
                    </p>
                  </details>
                )}

                {selected.body ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none border-t border-border mt-5 pt-5">
                    <CompactMarkdown>{selected.body}</CompactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground italic border-t border-border mt-5 pt-5">
                    {t("noBody")}
                  </p>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full min-h-[480px] px-6 text-center">
                <div>
                  <FileText className="w-8 h-8 mx-auto mb-3 text-muted-foreground opacity-60" />
                  <div className="text-sm text-muted-foreground">{t("selectPagePrompt")}</div>
                </div>
              </div>
            )}
          </main>
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

      {/* Provenance dialog — reuses the mental-model detail modal to show the
          backing model's based_on (the memories this page was synthesized from). */}
      {provenanceMmId && (
        <MentalModelDetailModal
          mentalModelId={provenanceMmId}
          onClose={() => setProvenanceMmId(null)}
        />
      )}
    </div>
  );
}

export function TreeRow({
  node,
  depth,
  isRoot = false,
  readOnly = false,
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
  isRoot?: boolean;
  readOnly?: boolean;
  expanded: Set<string>;
  selectedId: string | null;
  onToggle: (id: string) => void;
  onOpenPage: (id: string) => void;
  onAddChild?: (kind: "folder" | "page", parentId: string) => void;
  onDelete?: (node: KnowledgeNode) => void;
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
            {!isFolder &&
              (node.is_stale === false ? (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex-shrink-0">
                  {t("inSync")}
                </span>
              ) : node.is_stale === true ? (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 flex-shrink-0">
                  {t("needsRefresh")}
                </span>
              ) : null)}
          </div>
          {!isFolder && (
            <div
              className="text-xs text-muted-foreground/80 truncate"
              title={node.timestamp ? formatAbsoluteDateTime(node.timestamp) : undefined}
            >
              {node.timestamp
                ? `${t("updatedLabel")} ${formatRelativeTime(node.timestamp)}`
                : t("generating")}
            </div>
          )}
        </div>
        {!readOnly && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
            {isFolder && (
              <>
                <button
                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                  title={t("newFolder")}
                  onClick={(e) => {
                    e.stopPropagation();
                    onAddChild?.("folder", node.id);
                  }}
                >
                  <FolderPlus className="w-3.5 h-3.5" />
                </button>
                <button
                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                  title={t("newPage")}
                  onClick={(e) => {
                    e.stopPropagation();
                    onAddChild?.("page", node.id);
                  }}
                >
                  <FilePlus className="w-3.5 h-3.5" />
                </button>
              </>
            )}
            {!isRoot && (
              <button
                className="p-1 rounded hover:bg-background text-muted-foreground hover:text-red-600"
                title={t("deleteButton")}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete?.(node);
                }}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}
      </div>
      {isFolder && isOpen && node.children?.length > 0 && (
        <ul>
          {node.children.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              readOnly={readOnly}
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
