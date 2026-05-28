"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { client, type TagGroup, type TagsMatch } from "@/lib/api";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import { CompactMarkdown } from "./compact-markdown";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FactType, FactTypeCheckboxGroup } from "@/components/fact-type-filter";
import { toast } from "sonner";
import { Card, CardContent } from "@/components/ui/card";
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
  Plus,
  Sparkles,
  Loader2,
  Trash2,
  Eraser,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  LayoutGrid,
  MoreVertical,
  Pencil,
  FolderOpen,
  FileText,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MentalModelDetailModal } from "./mental-model-detail-modal";
import { TagFilterInput } from "./tag-filter-input";

interface ReflectResponseBasedOnFact {
  id: string;
  text: string;
  type: string;
  context?: string;
}

interface ReflectResponse {
  text: string;
  based_on: Record<string, ReflectResponseBasedOnFact[]>;
}

interface MentalModel {
  id: string;
  bank_id: string;
  name: string;
  source_query: string;
  content: string;
  tags: string[];
  max_tokens: number;
  trigger: {
    mode?: "full" | "delta";
    refresh_after_consolidation: boolean;
    fact_types?: Array<"world" | "experience" | "observation">;
    exclude_mental_models?: boolean;
    exclude_mental_model_ids?: string[];
    tags_match?: TagsMatch;
    tag_groups?: TagGroup[];
    include_chunks?: boolean;
    recall_max_tokens?: number;
    recall_chunks_max_tokens?: number;
  };
  last_refreshed_at: string;
  created_at: string;
  reflect_response?: ReflectResponse;
}

type ViewMode = "dashboard" | "files";

export function MentalModelsView() {
  const t = useTranslations("mentalModels");
  const { currentBank } = useBank();
  const [mentalModels, setMentalModels] = useState<MentalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("files");
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 100;

  const [showCreateMentalModel, setShowCreateMentalModel] = useState(false);
  const [selectedMentalModel, setSelectedMentalModel] = useState<MentalModel | null>(null);
  const [filesSelectedId, setFilesSelectedId] = useState<string | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [tagsMatch, setTagsMatch] = useState<"any" | "all">("any");
  const [showUpdateDialog, setShowUpdateDialog] = useState(false);
  const [mentalModelToUpdate, setMentalModelToUpdate] = useState<MentalModel | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [clearTarget, setClearTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [clearing, setClearing] = useState(false);

  // Tag filtering happens server-side; only the search text is applied locally.
  const filteredMentalModels = mentalModels.filter((m) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      m.id.toLowerCase().includes(query) ||
      m.name.toLowerCase().includes(query) ||
      m.source_query.toLowerCase().includes(query) ||
      m.content.toLowerCase().includes(query)
    );
  });

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const mentalModelsData = await client.listMentalModels(
        currentBank,
        selectedTags.length > 0 ? selectedTags : undefined,
        selectedTags.length > 0 ? tagsMatch : undefined
      );
      setMentalModels(mentalModelsData.items || []);
    } catch (error) {
      console.error("Error loading mental models:", error);
    } finally {
      setLoading(false);
    }
  };

  const [refreshingIds, setRefreshingIds] = useState<Set<string>>(new Set());

  const handleRowRefresh = async (m: MentalModel) => {
    if (!currentBank) return;
    const originalAt = m.last_refreshed_at;
    setRefreshingIds((prev) => new Set(prev).add(m.id));
    try {
      await client.refreshMentalModel(currentBank, m.id);
      const maxAttempts = 120;
      for (let i = 0; i < maxAttempts; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        const updated = await client.getMentalModel(currentBank, m.id);
        if (updated.last_refreshed_at !== originalAt) {
          setMentalModels((prev) => prev.map((x) => (x.id === m.id ? updated : x)));
          if (selectedMentalModel?.id === m.id) setSelectedMentalModel(updated);
          toast.success(t("toastRefreshed"));
          return;
        }
      }
      toast.error(t("toastRefreshTimeout"));
    } catch {
      // Error toast handled by API client interceptor
    } finally {
      setRefreshingIds((prev) => {
        const next = new Set(prev);
        next.delete(m.id);
        return next;
      });
    }
  };

  const handleDelete = async () => {
    if (!currentBank || !deleteTarget) return;

    setDeleting(true);
    try {
      await client.deleteMentalModel(currentBank, deleteTarget.id);
      setMentalModels((prev) => prev.filter((m) => m.id !== deleteTarget.id));
      if (selectedMentalModel?.id === deleteTarget.id) setSelectedMentalModel(null);
      setDeleteTarget(null);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setDeleting(false);
    }
  };

  const handleClear = async () => {
    if (!currentBank || !clearTarget) return;

    setClearing(true);
    try {
      const updated = await client.clearMentalModel(currentBank, clearTarget.id);
      setMentalModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
      if (selectedMentalModel?.id === updated.id) setSelectedMentalModel(updated);
      toast.success("Mental model content cleared");
      setClearTarget(null);
    } catch {
      // Error toast handled by API client interceptor
    } finally {
      setClearing(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadData();
    }
  }, [currentBank, selectedTags, tagsMatch]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedMentalModel(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Reset to first page when search query or tag filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, selectedTags, tagsMatch]);

  if (!currentBank) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <p className="text-muted-foreground">{t("selectBankPrompt")}</p>
        </CardContent>
      </Card>
    );
  }

  // Pagination calculations
  const totalPages = Math.ceil(filteredMentalModels.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedMentalModels = filteredMentalModels.slice(startIndex, endIndex);

  return (
    <div>
      {loading ? (
        <div className="text-center py-12">
          <RefreshCw className="w-8 h-8 mx-auto mb-3 text-muted-foreground animate-spin" />
          <p className="text-muted-foreground">{t("loading")}</p>
        </div>
      ) : (
        <>
          {/* Search + tag filter (single row) */}
          <div className="mb-4 flex items-center gap-3 flex-wrap">
            <Input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t("filterPlaceholder")}
              className="w-80 h-9"
            />
            <TagFilterInput
              value={selectedTags}
              onChange={setSelectedTags}
              fetchSuggestions={async (q) => {
                if (!currentBank) return [];
                const pattern = q ? `${q}*` : undefined;
                const res = await client.listTags(currentBank, pattern, 20, "mental_models");
                return res.items.map((i) => i.tag);
              }}
              matchMode={tagsMatch}
              onMatchModeChange={setTagsMatch}
              className="flex-1 min-w-[260px]"
            />
            <Button onClick={() => setShowCreateMentalModel(true)} size="sm">
              <Plus className="w-4 h-4 mr-2" />
              {t("addMentalModel")}
            </Button>
          </div>

          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-muted-foreground">
              {searchQuery || selectedTags.length > 0
                ? t("countFiltered", {
                    filtered: filteredMentalModels.length,
                    total: mentalModels.length,
                  })
                : t("count", { count: mentalModels.length })}
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
                <button
                  onClick={() => setViewMode("files")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "files"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <FolderOpen className="w-4 h-4" />
                  {t("viewList")}
                </button>
                <button
                  onClick={() => setViewMode("dashboard")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "dashboard"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <LayoutGrid className="w-4 h-4" />
                  {t("viewDashboard")}
                </button>
              </div>
            </div>
          </div>

          {filteredMentalModels.length > 0 ? (
            <>
              {/* Dashboard View - Cards */}
              {viewMode === "dashboard" && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {paginatedMentalModels.map((m) => {
                    return (
                      <Card
                        key={m.id}
                        className={`cursor-pointer transition-all hover:shadow-md hover:border-primary/50 ${
                          selectedMentalModel?.id === m.id ? "border-primary shadow-md" : ""
                        }`}
                        onClick={() => setSelectedMentalModel(m)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex-1 min-w-0">
                              <h3 className="font-semibold text-foreground mb-1 truncate">
                                {m.name}
                              </h3>
                              <div className="flex items-center gap-2">
                                <code className="text-xs font-mono text-muted-foreground truncate">
                                  {m.id}
                                </code>
                                <span
                                  className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${
                                    m.trigger?.refresh_after_consolidation
                                      ? "bg-green-500/10 text-green-600 dark:text-green-400"
                                      : "bg-slate-500/10 text-slate-600 dark:text-slate-400"
                                  }`}
                                >
                                  {m.trigger?.refresh_after_consolidation
                                    ? t("badgeAutoRefresh")
                                    : t("badgeManual")}
                                </span>
                              </div>
                            </div>
                            <RowActionsMenu
                              m={m}
                              refreshing={refreshingIds.has(m.id)}
                              onEdit={(target) => {
                                setMentalModelToUpdate(target);
                                setShowUpdateDialog(true);
                              }}
                              onRefresh={handleRowRefresh}
                              onClear={(target) =>
                                setClearTarget({ id: target.id, name: target.name })
                              }
                              onDelete={(target) =>
                                setDeleteTarget({ id: target.id, name: target.name })
                              }
                              triggerClassName="h-7 w-7 ml-2 flex-shrink-0"
                            />
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                            {m.source_query}
                          </p>
                          <div className="relative mb-3 border-t border-border pt-3 max-h-40 overflow-hidden">
                            <CompactMarkdown>{m.content}</CompactMarkdown>
                            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-card to-transparent" />
                          </div>
                          <div className="flex items-center justify-between text-xs border-t border-border pt-3">
                            <div className="flex items-center gap-2">
                              {m.tags.length > 0 && (
                                <div className="flex gap-1">
                                  {m.tags.slice(0, 2).map((tag) => (
                                    <span
                                      key={tag}
                                      className="px-1.5 py-0.5 rounded text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400"
                                    >
                                      {tag}
                                    </span>
                                  ))}
                                  {m.tags.length > 2 && (
                                    <span className="px-1.5 py-0.5 rounded text-xs bg-muted text-muted-foreground">
                                      +{m.tags.length - 2}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                            <div
                              className="text-muted-foreground"
                              title={formatAbsoluteDateTime(m.last_refreshed_at)}
                            >
                              {formatRelativeTime(m.last_refreshed_at)}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}

              {/* Files View */}
              {viewMode === "files" && (
                <FilesView
                  mentalModels={filteredMentalModels}
                  selectedId={filesSelectedId}
                  onSelect={setFilesSelectedId}
                  onOpenDetail={setSelectedMentalModel}
                  refreshingIds={refreshingIds}
                  onEdit={(target) => {
                    setMentalModelToUpdate(target);
                    setShowUpdateDialog(true);
                  }}
                  onRefresh={handleRowRefresh}
                  onClear={(target) => setClearTarget({ id: target.id, name: target.name })}
                  onDelete={(target) => setDeleteTarget({ id: target.id, name: target.name })}
                />
              )}

              {/* Pagination Controls */}
              {viewMode !== "files" && totalPages > 1 && (
                <div className="flex items-center justify-between mt-3 pt-3 border-t">
                  <div className="text-xs text-muted-foreground">
                    {startIndex + 1}-{Math.min(endIndex, filteredMentalModels.length)} of{" "}
                    {filteredMentalModels.length}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(1)}
                      disabled={currentPage === 1}
                      className="h-7 w-7 p-0"
                    >
                      <ChevronsLeft className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="h-7 w-7 p-0"
                    >
                      <ChevronLeft className="h-3 w-3" />
                    </Button>
                    <span className="text-xs px-2">
                      {currentPage} / {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="h-7 w-7 p-0"
                    >
                      <ChevronRight className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(totalPages)}
                      disabled={currentPage === totalPages}
                      className="h-7 w-7 p-0"
                    >
                      <ChevronsRight className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="p-6 border border-dashed border-border rounded-lg text-center">
              <Sparkles className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                {searchQuery ? t("emptyFilterMatch") : t("emptyNoModels")}
              </p>
            </div>
          )}
        </>
      )}

      <CreateMentalModelDialog
        open={showCreateMentalModel}
        onClose={() => setShowCreateMentalModel(false)}
        onCreated={() => {
          setShowCreateMentalModel(false);
          // Reload the list immediately to show the new mental model
          loadData();
        }}
      />

      <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteDialogTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteDialogDescription", { name: deleteTarget?.name ?? "" })}
              <br />
              <br />
              <span className="text-destructive font-semibold">{t("deleteDialogWarning")}</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">{t("deleteDialogCancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
              {t("deleteDialogConfirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={!!clearTarget} onOpenChange={(open) => !open && setClearTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("clearDialogTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t.rich("clearDialogDescription", {
                name: clearTarget?.name ?? "",
                bold: (chunks) => <span className="font-semibold">{chunks}</span>,
              })}
              <br />
              <br />
              {t("clearDialogReSynth")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">{t("cancelButton")}</AlertDialogCancel>
            <AlertDialogAction onClick={handleClear} disabled={clearing}>
              {clearing ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
              {t("clearDialogConfirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <MentalModelDetailModal
        mentalModelId={selectedMentalModel?.id ?? null}
        onClose={() => setSelectedMentalModel(null)}
        onDelete={(m) => setDeleteTarget({ id: m.id, name: m.name })}
        onClear={(m) => setClearTarget({ id: m.id, name: m.name })}
        onEdit={(m) => {
          setMentalModelToUpdate(m);
          setShowUpdateDialog(true);
        }}
        onRefreshed={(updated) => {
          setMentalModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
          setSelectedMentalModel(updated);
        }}
      />

      {mentalModelToUpdate && (
        <UpdateMentalModelDialog
          open={showUpdateDialog}
          mentalModel={mentalModelToUpdate}
          onClose={() => {
            setShowUpdateDialog(false);
            setMentalModelToUpdate(null);
          }}
          onUpdated={(updated) => {
            setMentalModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
            setSelectedMentalModel(updated);
            setShowUpdateDialog(false);
            setMentalModelToUpdate(null);
          }}
        />
      )}
    </div>
  );
}

function RowActionsMenu({
  m,
  refreshing,
  onEdit,
  onRefresh,
  onClear,
  onDelete,
  triggerClassName,
}: {
  m: MentalModel;
  refreshing: boolean;
  onEdit: (m: MentalModel) => void;
  onRefresh: (m: MentalModel) => void;
  onClear: (m: MentalModel) => void;
  onDelete: (m: MentalModel) => void;
  triggerClassName?: string;
}) {
  const t = useTranslations("mentalModels");
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={`p-0 text-muted-foreground ${triggerClassName ?? "h-8 w-8"}`}
          onClick={(e) => e.stopPropagation()}
          aria-label={t("actionsAriaLabel")}
        >
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
        <DropdownMenuItem onClick={() => onEdit(m)}>
          <Pencil className="h-4 w-4 mr-2" />
          {t("actionEdit")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onRefresh(m)} disabled={refreshing}>
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
          {t("actionRefresh")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onClear(m)}>
          <Eraser className="h-4 w-4 mr-2" />
          {t("actionClearContent")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => onDelete(m)}
          className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          {t("actionDelete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function CreateMentalModelDialog({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
}) {
  const t = useTranslations("mentalModels");
  const { currentBank } = useBank();
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({
    id: "",
    name: "",
    sourceQuery: "",
    maxTokens: "2048",
    tags: "",
    autoRefresh: false,
    mode: "full" as "full" | "delta",
    factTypes: [] as Array<"world" | "experience" | "observation">,
    excludeMentalModels: false,
    excludeMentalModelIds: "",
    tagsMatch: "" as string,
    tagGroups: "",
    // Recall overrides for refresh: "" means inherit bank/global default
    includeChunks: "" as "" | "true" | "false",
    recallMaxTokens: "",
    recallChunksMaxTokens: "",
  });

  const handleCreate = async () => {
    if (!currentBank || !form.name.trim() || !form.sourceQuery.trim()) return;

    setCreating(true);
    try {
      const tags = form.tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const maxTokens = parseInt(form.maxTokens) || 2048;

      // Submit mental model creation - content will be generated in background
      const excludeIds = form.excludeMentalModelIds
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      let tagGroups: TagGroup[] | undefined;
      if (form.tagGroups.trim()) {
        try {
          tagGroups = JSON.parse(form.tagGroups.trim());
        } catch {
          toast.error(t("invalidTagGroupsJson"));
          return;
        }
      }

      const recallMaxTokens = form.recallMaxTokens.trim()
        ? parseInt(form.recallMaxTokens, 10)
        : undefined;
      const recallChunksMaxTokens = form.recallChunksMaxTokens.trim()
        ? parseInt(form.recallChunksMaxTokens, 10)
        : undefined;
      const includeChunks =
        form.includeChunks === "true" ? true : form.includeChunks === "false" ? false : undefined;

      await client.createMentalModel(currentBank, {
        id: form.id.trim() || undefined,
        name: form.name.trim(),
        source_query: form.sourceQuery.trim(),
        tags: tags.length > 0 ? tags : undefined,
        max_tokens: maxTokens,
        trigger: {
          mode: form.mode,
          refresh_after_consolidation: form.autoRefresh,
          fact_types: form.factTypes.length > 0 ? form.factTypes : undefined,
          exclude_mental_models: form.excludeMentalModels || undefined,
          exclude_mental_model_ids: excludeIds.length > 0 ? excludeIds : undefined,
          tags_match: (form.tagsMatch as TagsMatch) || undefined,
          tag_groups: tagGroups,
          include_chunks: includeChunks,
          recall_max_tokens: recallMaxTokens,
          recall_chunks_max_tokens: recallChunksMaxTokens,
        },
      });

      setForm({
        id: "",
        name: "",
        sourceQuery: "",
        maxTokens: "2048",
        tags: "",
        autoRefresh: false,
        mode: "full",
        factTypes: [],
        excludeMentalModels: false,
        excludeMentalModelIds: "",
        tagsMatch: "",
        tagGroups: "",
        includeChunks: "",
        recallMaxTokens: "",
        recallChunksMaxTokens: "",
      });
      onCreated();
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) {
          setForm({
            id: "",
            name: "",
            sourceQuery: "",
            maxTokens: "2048",
            tags: "",
            autoRefresh: false,
            mode: "full",
            factTypes: [],
            excludeMentalModels: false,
            excludeMentalModelIds: "",
            tagsMatch: "",
            tagGroups: "",
            includeChunks: "",
            recallMaxTokens: "",
            recallChunksMaxTokens: "",
          });
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-2xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{t("createDialogTitle")}</DialogTitle>
          <DialogDescription>{t("createDialogDescription")}</DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="general" className="py-2 flex-1 flex flex-col min-h-0 overflow-hidden">
          <TabsList className="w-full">
            <TabsTrigger value="general" className="flex-1">
              {t("tabGeneral")}
            </TabsTrigger>
            <TabsTrigger value="options" className="flex-1">
              {t("tabOptions")}
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-2 px-1.5">
            <TabsContent value="general" className="space-y-4 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">{t("fieldId")}</label>
                <Input
                  value={form.id}
                  onChange={(e) => setForm({ ...form, id: e.target.value })}
                  placeholder={t("fieldIdPlaceholder")}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">{t("fieldName")}</label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder={t("fieldNamePlaceholder")}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">
                  {t("fieldSourceQuery")}
                </label>
                <Textarea
                  value={form.sourceQuery}
                  onChange={(e) => setForm({ ...form, sourceQuery: e.target.value })}
                  placeholder={t("fieldSourceQueryPlaceholder")}
                  className="min-h-[140px] font-mono text-sm leading-6 whitespace-pre-wrap"
                  spellCheck={false}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">{t("fieldMaxTokens")}</label>
                <Input
                  type="number"
                  value={form.maxTokens}
                  onChange={(e) => setForm({ ...form, maxTokens: e.target.value })}
                  placeholder={t("fieldMaxTokensPlaceholder")}
                  min="256"
                  max="8192"
                />
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-6 pt-4">
              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionRefresh")}
                </h3>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="auto-refresh"
                    checked={form.autoRefresh}
                    onCheckedChange={(checked) =>
                      setForm({ ...form, autoRefresh: checked === true })
                    }
                  />
                  <label
                    htmlFor="auto-refresh"
                    className="text-sm font-medium text-foreground cursor-pointer"
                  >
                    {t("optionsAutoRefreshLabel")}
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRefreshModeLabel")}
                  </label>
                  <Select
                    value={form.mode}
                    onValueChange={(value) => setForm({ ...form, mode: value as "full" | "delta" })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full">{t("optionsRefreshModeFull")}</SelectItem>
                      <SelectItem value="delta">{t("optionsRefreshModeDelta")}</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRefreshModeDeltaDescription")}
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionOtherModels")}
                </h3>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="exclude-mental-models"
                    checked={form.excludeMentalModels}
                    onCheckedChange={(checked) =>
                      setForm({ ...form, excludeMentalModels: checked === true })
                    }
                  />
                  <label
                    htmlFor="exclude-mental-models"
                    className="text-sm font-medium text-foreground cursor-pointer"
                  >
                    {t("optionsExcludeAllLabel")}
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsExcludeIdsLabel")}
                  </label>
                  <Input
                    value={form.excludeMentalModelIds}
                    onChange={(e) => setForm({ ...form, excludeMentalModelIds: e.target.value })}
                    placeholder={t("optionsExcludeIdsPlaceholder")}
                  />
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionTags")}
                </h3>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagsLabel")}
                  </label>
                  <Input
                    value={form.tags}
                    onChange={(e) => setForm({ ...form, tags: e.target.value })}
                    placeholder={t("optionsTagsPlaceholder")}
                  />
                  <p className="text-xs text-muted-foreground">{t("optionsTagsDescription")}</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagsMatchLabel")}
                  </label>
                  <Select
                    value={form.tagsMatch}
                    onValueChange={(v) => setForm({ ...form, tagsMatch: v === "default" ? "" : v })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={t("optionsTagsMatchDefaultPlaceholder")} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">{t("optionsTagsMatchDefault")}</SelectItem>
                      <SelectItem value="any">{t("optionsTagsMatchAny")}</SelectItem>
                      <SelectItem value="all">{t("optionsTagsMatchAll")}</SelectItem>
                      <SelectItem value="any_strict">{t("optionsTagsMatchAnyStrict")}</SelectItem>
                      <SelectItem value="all_strict">{t("optionsTagsMatchAllStrict")}</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {t("optionsTagsMatchDescription")}
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagGroupsLabel")}
                  </label>
                  <Textarea
                    value={form.tagGroups}
                    onChange={(e) => setForm({ ...form, tagGroups: e.target.value })}
                    placeholder='e.g., [{"or": [{"tags": ["user:alice"], "match": "all_strict"}, {"tags": ["shared"]}]}]'
                    rows={3}
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsTagGroupsDescription")}
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionRecall")}
                </h3>
                <p className="text-xs text-muted-foreground">{t("optionsRecallDescription")}</p>
                <div className="space-y-3">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsFactTypesLabel")}
                  </label>
                  <FactTypeCheckboxGroup
                    value={form.factTypes}
                    onChange={(v) => setForm({ ...form, factTypes: v as FactType[] })}
                  />
                  <p className="text-xs text-muted-foreground">{t("optionsFactTypesEmpty")}</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsIncludeChunksLabel")}
                  </label>
                  <Select
                    value={form.includeChunks || "default"}
                    onValueChange={(v) =>
                      setForm({
                        ...form,
                        includeChunks: v === "default" ? "" : (v as "true" | "false"),
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">{t("optionsIncludeChunksDefault")}</SelectItem>
                      <SelectItem value="true">{t("optionsIncludeChunksYes")}</SelectItem>
                      <SelectItem value="false">{t("optionsIncludeChunksNo")}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRecallMaxTokensLabel")}
                  </label>
                  <Input
                    type="number"
                    value={form.recallMaxTokens}
                    onChange={(e) => setForm({ ...form, recallMaxTokens: e.target.value })}
                    placeholder={t("optionsRecallMaxTokensPlaceholder")}
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRecallMaxTokensDescription")}
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRecallChunksMaxTokensLabel")}
                  </label>
                  <Input
                    type="number"
                    value={form.recallChunksMaxTokens}
                    onChange={(e) => setForm({ ...form, recallChunksMaxTokens: e.target.value })}
                    placeholder={t("optionsRecallChunksMaxTokensPlaceholder")}
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRecallChunksMaxTokensDescription")}
                  </p>
                </div>
              </section>
            </TabsContent>
          </div>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={creating}>
            {t("cancelButton")}
          </Button>
          <Button
            onClick={handleCreate}
            disabled={creating || !form.name.trim() || !form.sourceQuery.trim()}
          >
            {creating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
                {t("creatingButton")}
              </>
            ) : (
              t("createButton")
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function UpdateMentalModelDialog({
  open,
  mentalModel,
  onClose,
  onUpdated,
}: {
  open: boolean;
  mentalModel: MentalModel;
  onClose: () => void;
  onUpdated: (updated: MentalModel) => void;
}) {
  const t = useTranslations("mentalModels");
  const { currentBank } = useBank();
  const [updating, setUpdating] = useState(false);
  const buildFormState = () => ({
    name: mentalModel.name,
    sourceQuery: mentalModel.source_query,
    maxTokens: String(mentalModel.max_tokens || 2048),
    tags: mentalModel.tags.join(", "),
    autoRefresh: mentalModel.trigger?.refresh_after_consolidation || false,
    mode: (mentalModel.trigger?.mode || "full") as "full" | "delta",
    factTypes:
      (mentalModel.trigger?.fact_types as
        | Array<"world" | "experience" | "observation">
        | undefined) || [],
    excludeMentalModels: mentalModel.trigger?.exclude_mental_models || false,
    excludeMentalModelIds: (mentalModel.trigger?.exclude_mental_model_ids || []).join(", "),
    tagsMatch: (mentalModel.trigger?.tags_match as string) || "",
    tagGroups: mentalModel.trigger?.tag_groups
      ? JSON.stringify(mentalModel.trigger.tag_groups, null, 2)
      : "",
    includeChunks: (mentalModel.trigger?.include_chunks === true
      ? "true"
      : mentalModel.trigger?.include_chunks === false
        ? "false"
        : "") as "" | "true" | "false",
    recallMaxTokens:
      mentalModel.trigger?.recall_max_tokens != null
        ? String(mentalModel.trigger.recall_max_tokens)
        : "",
    recallChunksMaxTokens:
      mentalModel.trigger?.recall_chunks_max_tokens != null
        ? String(mentalModel.trigger.recall_chunks_max_tokens)
        : "",
  });
  const [form, setForm] = useState(buildFormState);

  // Reset form when mental model changes or dialog opens
  useEffect(() => {
    if (open) {
      setForm(buildFormState());
    }
  }, [open, mentalModel]);

  const handleUpdate = async () => {
    if (!currentBank || !form.name.trim() || !form.sourceQuery.trim()) return;

    setUpdating(true);
    try {
      const tags = form.tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const maxTokens = parseInt(form.maxTokens) || 2048;

      const excludeIds = form.excludeMentalModelIds
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      let tagGroups: TagGroup[] | undefined;
      if (form.tagGroups.trim()) {
        try {
          tagGroups = JSON.parse(form.tagGroups.trim());
        } catch {
          toast.error(t("invalidTagGroupsJson"));
          return;
        }
      }

      const recallMaxTokens = form.recallMaxTokens.trim()
        ? parseInt(form.recallMaxTokens, 10)
        : undefined;
      const recallChunksMaxTokens = form.recallChunksMaxTokens.trim()
        ? parseInt(form.recallChunksMaxTokens, 10)
        : undefined;
      const includeChunks =
        form.includeChunks === "true" ? true : form.includeChunks === "false" ? false : undefined;

      const updated = await client.updateMentalModel(currentBank, mentalModel.id, {
        name: form.name.trim(),
        source_query: form.sourceQuery.trim(),
        tags: tags.length > 0 ? tags : undefined,
        max_tokens: maxTokens,
        trigger: {
          mode: form.mode,
          refresh_after_consolidation: form.autoRefresh,
          fact_types: form.factTypes.length > 0 ? form.factTypes : undefined,
          exclude_mental_models: form.excludeMentalModels || undefined,
          exclude_mental_model_ids: excludeIds.length > 0 ? excludeIds : undefined,
          tags_match: (form.tagsMatch as TagsMatch) || undefined,
          tag_groups: tagGroups,
          include_chunks: includeChunks,
          recall_max_tokens: recallMaxTokens,
          recall_chunks_max_tokens: recallChunksMaxTokens,
        },
      });

      onUpdated(updated);
      onClose();
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setUpdating(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{t("updateDialogTitle")}</DialogTitle>
          <DialogDescription>{t("updateDialogDescription")}</DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="general" className="py-2 flex-1 flex flex-col min-h-0 overflow-hidden">
          <TabsList className="w-full">
            <TabsTrigger value="general" className="flex-1">
              {t("tabGeneral")}
            </TabsTrigger>
            <TabsTrigger value="options" className="flex-1">
              {t("tabOptions")}
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-2 px-1.5">
            <TabsContent value="general" className="space-y-4 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">{t("fieldId")}</label>
                <Input value={mentalModel.id} disabled className="bg-muted" />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">{t("fieldName")}</label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder={t("fieldNamePlaceholder")}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">
                  {t("fieldSourceQuery")}
                </label>
                <Textarea
                  value={form.sourceQuery}
                  onChange={(e) => setForm({ ...form, sourceQuery: e.target.value })}
                  placeholder={t("fieldSourceQueryPlaceholder")}
                  className="min-h-[140px] font-mono text-sm leading-6 whitespace-pre-wrap"
                  spellCheck={false}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">{t("fieldMaxTokens")}</label>
                <Input
                  type="number"
                  value={form.maxTokens}
                  onChange={(e) => setForm({ ...form, maxTokens: e.target.value })}
                  placeholder={t("fieldMaxTokensPlaceholder")}
                  min="256"
                  max="8192"
                />
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-6 pt-4">
              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionRefresh")}
                </h3>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="update-auto-refresh"
                    checked={form.autoRefresh}
                    onCheckedChange={(checked) =>
                      setForm({ ...form, autoRefresh: checked === true })
                    }
                  />
                  <label
                    htmlFor="update-auto-refresh"
                    className="text-sm font-medium text-foreground cursor-pointer"
                  >
                    {t("optionsAutoRefreshLabel")}
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRefreshModeLabel")}
                  </label>
                  <Select
                    value={form.mode}
                    onValueChange={(value) => setForm({ ...form, mode: value as "full" | "delta" })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full">{t("optionsRefreshModeFull")}</SelectItem>
                      <SelectItem value="delta">{t("optionsRefreshModeDelta")}</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRefreshModeDeltaDescription")}
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionOtherModels")}
                </h3>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="update-exclude-mental-models"
                    checked={form.excludeMentalModels}
                    onCheckedChange={(checked) =>
                      setForm({ ...form, excludeMentalModels: checked === true })
                    }
                  />
                  <label
                    htmlFor="update-exclude-mental-models"
                    className="text-sm font-medium text-foreground cursor-pointer"
                  >
                    {t("optionsExcludeAllLabel")}
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsExcludeIdsLabel")}
                  </label>
                  <Input
                    value={form.excludeMentalModelIds}
                    onChange={(e) => setForm({ ...form, excludeMentalModelIds: e.target.value })}
                    placeholder={t("optionsExcludeIdsPlaceholder")}
                  />
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionTags")}
                </h3>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagsLabel")}
                  </label>
                  <Input
                    value={form.tags}
                    onChange={(e) => setForm({ ...form, tags: e.target.value })}
                    placeholder={t("optionsTagsPlaceholder")}
                  />
                  <p className="text-xs text-muted-foreground">{t("optionsTagsDescription")}</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagsMatchLabel")}
                  </label>
                  <Select
                    value={form.tagsMatch || "default"}
                    onValueChange={(v) => setForm({ ...form, tagsMatch: v === "default" ? "" : v })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={t("optionsTagsMatchDefaultPlaceholder")} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">{t("optionsTagsMatchDefault")}</SelectItem>
                      <SelectItem value="any">{t("optionsTagsMatchAny")}</SelectItem>
                      <SelectItem value="all">{t("optionsTagsMatchAll")}</SelectItem>
                      <SelectItem value="any_strict">{t("optionsTagsMatchAnyStrict")}</SelectItem>
                      <SelectItem value="all_strict">{t("optionsTagsMatchAllStrict")}</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {t("optionsTagsMatchDescription")}
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsTagGroupsLabel")}
                  </label>
                  <Textarea
                    value={form.tagGroups}
                    onChange={(e) => setForm({ ...form, tagGroups: e.target.value })}
                    placeholder='e.g., [{"or": [{"tags": ["user:alice"], "match": "all_strict"}, {"tags": ["shared"]}]}]'
                    rows={3}
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsTagGroupsDescription")}
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  {t("optionsSectionRecall")}
                </h3>
                <p className="text-xs text-muted-foreground">{t("optionsRecallDescription")}</p>
                <div className="space-y-3">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsFactTypesLabel")}
                  </label>
                  <FactTypeCheckboxGroup
                    value={form.factTypes}
                    onChange={(v) => setForm({ ...form, factTypes: v as FactType[] })}
                  />
                  <p className="text-xs text-muted-foreground">{t("optionsFactTypesEmpty")}</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsIncludeChunksLabel")}
                  </label>
                  <Select
                    value={form.includeChunks || "default"}
                    onValueChange={(v) =>
                      setForm({
                        ...form,
                        includeChunks: v === "default" ? "" : (v as "true" | "false"),
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">{t("optionsIncludeChunksDefault")}</SelectItem>
                      <SelectItem value="true">{t("optionsIncludeChunksYes")}</SelectItem>
                      <SelectItem value="false">{t("optionsIncludeChunksNo")}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRecallMaxTokensLabel")}
                  </label>
                  <Input
                    type="number"
                    value={form.recallMaxTokens}
                    onChange={(e) => setForm({ ...form, recallMaxTokens: e.target.value })}
                    placeholder={t("optionsRecallMaxTokensPlaceholder")}
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRecallMaxTokensDescription")}
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    {t("optionsRecallChunksMaxTokensLabel")}
                  </label>
                  <Input
                    type="number"
                    value={form.recallChunksMaxTokens}
                    onChange={(e) => setForm({ ...form, recallChunksMaxTokens: e.target.value })}
                    placeholder={t("optionsRecallChunksMaxTokensPlaceholder")}
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("optionsRecallChunksMaxTokensDescription")}
                  </p>
                </div>
              </section>
            </TabsContent>
          </div>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={updating}>
            {t("cancelButton")}
          </Button>
          <Button
            onClick={handleUpdate}
            disabled={updating || !form.name.trim() || !form.sourceQuery.trim()}
          >
            {updating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
                {t("updatingButton")}
              </>
            ) : (
              t("updateButton")
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function FilesView({
  mentalModels,
  selectedId,
  onSelect,
  onOpenDetail,
  refreshingIds,
  onEdit,
  onRefresh,
  onClear,
  onDelete,
}: {
  mentalModels: MentalModel[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onOpenDetail: (m: MentalModel) => void;
  refreshingIds: Set<string>;
  onEdit: (m: MentalModel) => void;
  onRefresh: (m: MentalModel) => void;
  onClear: (m: MentalModel) => void;
  onDelete: (m: MentalModel) => void;
}) {
  const t = useTranslations("mentalModels");
  const effectiveId =
    selectedId && mentalModels.some((m) => m.id === selectedId)
      ? selectedId
      : (mentalModels[0]?.id ?? null);
  const selected = mentalModels.find((m) => m.id === effectiveId) ?? null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-0 overflow-hidden min-h-[600px]">
      <aside className="border-r border-border bg-muted/30 overflow-y-auto max-h-[calc(100vh-260px)]">
        <ul className="py-1">
          {mentalModels.map((m) => {
            const isActive = m.id === effectiveId;
            return (
              <li key={m.id}>
                <button
                  onClick={() => onSelect(m.id)}
                  className={`w-full flex items-start gap-2 px-3 py-2 text-left transition-colors border-l-2 ${
                    isActive
                      ? "bg-primary/10 border-primary text-foreground"
                      : "border-transparent text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                  title={`${m.name}\n${m.source_query}`}
                >
                  <FileText className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`truncate text-sm font-medium ${isActive ? "text-foreground" : ""}`}
                      >
                        {m.name}
                      </span>
                      <span
                        className="ml-auto text-[10px] text-muted-foreground flex-shrink-0"
                        title={formatAbsoluteDateTime(m.last_refreshed_at)}
                      >
                        {formatRelativeTime(m.last_refreshed_at)}
                      </span>
                    </div>
                    {m.source_query && (
                      <div className="text-xs text-muted-foreground/80 truncate italic mt-0.5">
                        {m.source_query}
                      </div>
                    )}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      </aside>

      <section className="overflow-y-auto max-h-[calc(100vh-260px)]">
        {selected ? (
          <article className="p-6">
            <header className="mb-4 pb-4 border-b border-border">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <h2 className="text-xl font-semibold text-foreground">{selected.name}</h2>
                  {selected.source_query && (
                    <p className="text-sm text-muted-foreground mt-1 italic">
                      &ldquo;{selected.source_query}&rdquo;
                    </p>
                  )}
                  <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                    <span title={formatAbsoluteDateTime(selected.last_refreshed_at)}>
                      Refreshed {formatRelativeTime(selected.last_refreshed_at)}
                    </span>
                    {selected.tags.length > 0 && (
                      <div className="flex gap-1">
                        {selected.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-1.5 py-0.5 rounded text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  <Button variant="outline" size="sm" onClick={() => onOpenDetail(selected)}>
                    Open
                  </Button>
                  <RowActionsMenu
                    m={selected}
                    refreshing={refreshingIds.has(selected.id)}
                    onEdit={onEdit}
                    onRefresh={onRefresh}
                    onClear={onClear}
                    onDelete={onDelete}
                  />
                </div>
              </div>
            </header>
            {selected.content ? (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <CompactMarkdown>{selected.content}</CompactMarkdown>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground italic">{t("emptyContent")}</p>
            )}
          </article>
        ) : (
          <div className="p-10 text-center text-muted-foreground">
            <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">{t("selectModelPrompt")}</p>
          </div>
        )}
      </section>
    </div>
  );
}
