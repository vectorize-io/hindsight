"use client";

import { useState, useEffect } from "react";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Plus,
  Sparkles,
  Loader2,
  Trash2,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  LayoutGrid,
  List,
  MoreVertical,
  Pencil,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MentalModelDetailModal } from "./mental-model-detail-modal";

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

type ViewMode = "dashboard" | "table";

export function MentalModelsView() {
  const { currentBank } = useBank();
  const [mentalModels, setMentalModels] = useState<MentalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("dashboard");
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 100;

  const [showCreateMentalModel, setShowCreateMentalModel] = useState(false);
  const [selectedMentalModel, setSelectedMentalModel] = useState<MentalModel | null>(null);
  const [showUpdateDialog, setShowUpdateDialog] = useState(false);
  const [mentalModelToUpdate, setMentalModelToUpdate] = useState<MentalModel | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filter mental models based on search query
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
      const mentalModelsData = await client.listMentalModels(currentBank);
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
          toast.success("Mental model refreshed");
          return;
        }
      }
      toast.error("Refresh timeout");
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

  useEffect(() => {
    if (currentBank) {
      loadData();
    }
  }, [currentBank]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedMentalModel(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Reset to first page when search query changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery]);

  if (!currentBank) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <p className="text-muted-foreground">Select a memory bank to view mental models.</p>
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
          <p className="text-muted-foreground">Loading...</p>
        </div>
      ) : (
        <>
          {/* Search filter */}
          <div className="mb-4">
            <Input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter mental models by name, query, or content..."
              className="max-w-md"
            />
          </div>

          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-muted-foreground">
              {searchQuery
                ? `${filteredMentalModels.length} of ${mentalModels.length} mental models`
                : `${mentalModels.length} mental model${mentalModels.length !== 1 ? "s" : ""}`}
            </div>
            <div className="flex items-center gap-3">
              <Button onClick={() => setShowCreateMentalModel(true)} size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Add Mental Model
              </Button>
              <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
                <button
                  onClick={() => setViewMode("dashboard")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "dashboard"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <LayoutGrid className="w-4 h-4" />
                  Dashboard
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
                    viewMode === "table"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <List className="w-4 h-4" />
                  Table
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
                                    ? "Auto Refresh"
                                    : "Manual"}
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

              {/* Table View */}
              {viewMode === "table" && (
                <Table className="table-fixed">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[20%]">ID</TableHead>
                      <TableHead className="w-[20%]">Name</TableHead>
                      <TableHead className="w-[35%]">Source Query</TableHead>
                      <TableHead className="w-[15%]">Last Refreshed</TableHead>
                      <TableHead className="w-[10%]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paginatedMentalModels.map((m) => {
                      return (
                        <TableRow
                          key={m.id}
                          className={`cursor-pointer hover:bg-muted/50 ${
                            selectedMentalModel?.id === m.id ? "bg-primary/10" : ""
                          }`}
                          onClick={() => setSelectedMentalModel(m)}
                        >
                          <TableCell className="py-2">
                            <code className="text-xs font-mono text-muted-foreground truncate block">
                              {m.id}
                            </code>
                          </TableCell>
                          <TableCell className="py-2">
                            <div className="font-medium text-foreground">{m.name}</div>
                          </TableCell>
                          <TableCell className="py-2">
                            <div className="text-sm text-muted-foreground truncate">
                              {m.source_query}
                            </div>
                          </TableCell>
                          <TableCell
                            className="py-2 text-sm text-foreground"
                            title={formatAbsoluteDateTime(m.last_refreshed_at)}
                          >
                            {formatRelativeTime(m.last_refreshed_at)}
                          </TableCell>
                          <TableCell className="py-2">
                            <RowActionsMenu
                              m={m}
                              refreshing={refreshingIds.has(m.id)}
                              onEdit={(target) => {
                                setMentalModelToUpdate(target);
                                setShowUpdateDialog(true);
                              }}
                              onRefresh={handleRowRefresh}
                              onDelete={(target) =>
                                setDeleteTarget({ id: target.id, name: target.name })
                              }
                            />
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}

              {/* Pagination Controls */}
              {totalPages > 1 && (
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
                {searchQuery
                  ? "No mental models match your filter"
                  : "No mental models yet. Create a mental model to generate and save a summary from your memories."}
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
            <AlertDialogTitle>Delete Mental Model</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete{" "}
              <span className="font-semibold">&quot;{deleteTarget?.name}&quot;</span>?
              <br />
              <br />
              <span className="text-destructive font-semibold">This action cannot be undone.</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <MentalModelDetailModal
        mentalModelId={selectedMentalModel?.id ?? null}
        onClose={() => setSelectedMentalModel(null)}
        onDelete={(m) => setDeleteTarget({ id: m.id, name: m.name })}
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
  onDelete,
  triggerClassName,
}: {
  m: MentalModel;
  refreshing: boolean;
  onEdit: (m: MentalModel) => void;
  onRefresh: (m: MentalModel) => void;
  onDelete: (m: MentalModel) => void;
  triggerClassName?: string;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={`p-0 text-muted-foreground ${triggerClassName ?? "h-8 w-8"}`}
          onClick={(e) => e.stopPropagation()}
          aria-label="Actions"
        >
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
        <DropdownMenuItem onClick={() => onEdit(m)}>
          <Pencil className="h-4 w-4 mr-2" />
          Edit
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onRefresh(m)} disabled={refreshing}>
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
          Refresh Manually
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => onDelete(m)}
          className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete
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
          toast.error("Invalid JSON in Tag Groups field");
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
          <DialogTitle>Create Mental Model</DialogTitle>
          <DialogDescription>
            Create a mental model by running a query. The content will be auto-generated and can be
            refreshed later.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="general" className="py-2 flex-1 flex flex-col min-h-0 overflow-hidden">
          <TabsList className="w-full">
            <TabsTrigger value="general" className="flex-1">
              General
            </TabsTrigger>
            <TabsTrigger value="options" className="flex-1">
              Options
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-2 pr-1">
            <TabsContent value="general" className="space-y-4 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">ID</label>
                <Input
                  value={form.id}
                  onChange={(e) => setForm({ ...form, id: e.target.value })}
                  placeholder="e.g., team-communication"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Name *</label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="e.g., Team Communication Preferences"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Source Query *</label>
                <Input
                  value={form.sourceQuery}
                  onChange={(e) => setForm({ ...form, sourceQuery: e.target.value })}
                  placeholder="e.g., How does the team prefer to communicate?"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Max Tokens</label>
                <Input
                  type="number"
                  value={form.maxTokens}
                  onChange={(e) => setForm({ ...form, maxTokens: e.target.value })}
                  placeholder="2048"
                  min="256"
                  max="8192"
                />
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-6 pt-4">
              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Refresh</h3>
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
                    Auto-refresh after consolidation
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Refresh mode</label>
                  <Select
                    value={form.mode}
                    onValueChange={(value) => setForm({ ...form, mode: value as "full" | "delta" })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full">
                        Full — regenerate from scratch each refresh
                      </SelectItem>
                      <SelectItem value="delta">
                        Delta — surgical edits, preserve unchanged content
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Delta mode applies minimal changes to the existing content. Falls back to a full
                    rewrite on the first refresh and whenever the source query changes.
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  Other Mental Models
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
                    Exclude all mental models
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Exclude Mental Model IDs
                  </label>
                  <Input
                    value={form.excludeMentalModelIds}
                    onChange={(e) => setForm({ ...form, excludeMentalModelIds: e.target.value })}
                    placeholder="e.g., model-a, model-b (comma-separated)"
                  />
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Tags</h3>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tags</label>
                  <Input
                    value={form.tags}
                    onChange={(e) => setForm({ ...form, tags: e.target.value })}
                    placeholder="e.g., project-x, team-alpha (comma-separated)"
                  />
                  <p className="text-xs text-muted-foreground">
                    Tags scope the model during reflect <strong>and</strong> filter source memories
                    during refresh (default <code>all_strict</code>: only memories carrying every
                    listed tag are read). If no memories have these tags yet, refresh will produce
                    empty content — backfill tags on memories, or adjust <em>Tags Match</em> /{" "}
                    <em>Tag Groups</em> below.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tags Match</label>
                  <Select
                    value={form.tagsMatch}
                    onValueChange={(v) => setForm({ ...form, tagsMatch: v === "default" ? "" : v })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Default (all_strict when tags set)" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default (all_strict when tags set)</SelectItem>
                      <SelectItem value="any">any — OR matching, includes untagged</SelectItem>
                      <SelectItem value="all">all — AND matching, includes untagged</SelectItem>
                      <SelectItem value="any_strict">
                        any_strict — OR matching, excludes untagged
                      </SelectItem>
                      <SelectItem value="all_strict">
                        all_strict — AND matching, excludes untagged
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Controls how the model&apos;s tags filter memories during refresh.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tag Groups (JSON)</label>
                  <Textarea
                    value={form.tagGroups}
                    onChange={(e) => setForm({ ...form, tagGroups: e.target.value })}
                    placeholder='e.g., [{"or": [{"tags": ["user:alice"], "match": "all_strict"}, {"tags": ["shared"]}]}]'
                    rows={3}
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Compound boolean tag expressions for refresh filtering. Overrides flat tags when
                    set.
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Recall</h3>
                <p className="text-xs text-muted-foreground">
                  Override how the internal recall behaves when this model refreshes. Leave blank to
                  inherit the bank/global default.
                </p>
                <div className="space-y-3">
                  <label className="text-sm font-medium text-foreground">Fact Types</label>
                  <FactTypeCheckboxGroup
                    value={form.factTypes}
                    onChange={(v) => setForm({ ...form, factTypes: v as FactType[] })}
                  />
                  <p className="text-xs text-muted-foreground">Leave empty to include all types.</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Include chunks</label>
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
                      <SelectItem value="default">Default (inherit)</SelectItem>
                      <SelectItem value="true">Yes — include raw chunk text</SelectItem>
                      <SelectItem value="false">No — skip chunks (smaller prompt)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Recall max tokens</label>
                  <Input
                    type="number"
                    value={form.recallMaxTokens}
                    onChange={(e) => setForm({ ...form, recallMaxTokens: e.target.value })}
                    placeholder="Default (inherit)"
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    Token budget for facts returned by recall.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Recall chunks max tokens
                  </label>
                  <Input
                    type="number"
                    value={form.recallChunksMaxTokens}
                    onChange={(e) => setForm({ ...form, recallChunksMaxTokens: e.target.value })}
                    placeholder="Default (inherit)"
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    Token budget for raw chunk text returned by recall.
                  </p>
                </div>
              </section>
            </TabsContent>
          </div>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={creating}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            disabled={creating || !form.name.trim() || !form.sourceQuery.trim()}
          >
            {creating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
                Generating...
              </>
            ) : (
              "Create"
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
          toast.error("Invalid JSON in Tag Groups field");
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
          <DialogTitle>Update Mental Model</DialogTitle>
          <DialogDescription>
            Update the mental model configuration. Changes will take effect immediately.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="general" className="py-2 flex-1 flex flex-col min-h-0 overflow-hidden">
          <TabsList className="w-full">
            <TabsTrigger value="general" className="flex-1">
              General
            </TabsTrigger>
            <TabsTrigger value="options" className="flex-1">
              Options
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-2 pr-1">
            <TabsContent value="general" className="space-y-4 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">ID</label>
                <Input value={mentalModel.id} disabled className="bg-muted" />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Name *</label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="e.g., Team Communication Preferences"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Source Query *</label>
                <Input
                  value={form.sourceQuery}
                  onChange={(e) => setForm({ ...form, sourceQuery: e.target.value })}
                  placeholder="e.g., How does the team prefer to communicate?"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Max Tokens</label>
                <Input
                  type="number"
                  value={form.maxTokens}
                  onChange={(e) => setForm({ ...form, maxTokens: e.target.value })}
                  placeholder="2048"
                  min="256"
                  max="8192"
                />
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-6 pt-4">
              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Refresh</h3>
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
                    Auto-refresh after consolidation
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Refresh mode</label>
                  <Select
                    value={form.mode}
                    onValueChange={(value) => setForm({ ...form, mode: value as "full" | "delta" })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="full">
                        Full — regenerate from scratch each refresh
                      </SelectItem>
                      <SelectItem value="delta">
                        Delta — surgical edits, preserve unchanged content
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Delta mode applies minimal changes to the existing content. Falls back to a full
                    rewrite on the first refresh and whenever the source query changes.
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">
                  Other Mental Models
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
                    Exclude all mental models
                  </label>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Exclude Mental Model IDs
                  </label>
                  <Input
                    value={form.excludeMentalModelIds}
                    onChange={(e) => setForm({ ...form, excludeMentalModelIds: e.target.value })}
                    placeholder="e.g., model-a, model-b (comma-separated)"
                  />
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Tags</h3>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tags</label>
                  <Input
                    value={form.tags}
                    onChange={(e) => setForm({ ...form, tags: e.target.value })}
                    placeholder="e.g., project-x, team-alpha (comma-separated)"
                  />
                  <p className="text-xs text-muted-foreground">
                    Tags scope the model during reflect <strong>and</strong> filter source memories
                    during refresh (default <code>all_strict</code>: only memories carrying every
                    listed tag are read). If no memories have these tags yet, refresh will produce
                    empty content — backfill tags on memories, or adjust <em>Tags Match</em> /{" "}
                    <em>Tag Groups</em> below.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tags Match</label>
                  <Select
                    value={form.tagsMatch || "default"}
                    onValueChange={(v) => setForm({ ...form, tagsMatch: v === "default" ? "" : v })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Default (all_strict when tags set)" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default (all_strict when tags set)</SelectItem>
                      <SelectItem value="any">any — OR matching, includes untagged</SelectItem>
                      <SelectItem value="all">all — AND matching, includes untagged</SelectItem>
                      <SelectItem value="any_strict">
                        any_strict — OR matching, excludes untagged
                      </SelectItem>
                      <SelectItem value="all_strict">
                        all_strict — AND matching, excludes untagged
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Controls how the model&apos;s tags filter memories during refresh.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Tag Groups (JSON)</label>
                  <Textarea
                    value={form.tagGroups}
                    onChange={(e) => setForm({ ...form, tagGroups: e.target.value })}
                    placeholder='e.g., [{"or": [{"tags": ["user:alice"], "match": "all_strict"}, {"tags": ["shared"]}]}]'
                    rows={3}
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Compound boolean tag expressions for refresh filtering. Overrides flat tags when
                    set.
                  </p>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-foreground border-b pb-1">Recall</h3>
                <p className="text-xs text-muted-foreground">
                  Override how the internal recall behaves when this model refreshes. Leave blank to
                  inherit the bank/global default.
                </p>
                <div className="space-y-3">
                  <label className="text-sm font-medium text-foreground">Fact Types</label>
                  <FactTypeCheckboxGroup
                    value={form.factTypes}
                    onChange={(v) => setForm({ ...form, factTypes: v as FactType[] })}
                  />
                  <p className="text-xs text-muted-foreground">Leave empty to include all types.</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Include chunks</label>
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
                      <SelectItem value="default">Default (inherit)</SelectItem>
                      <SelectItem value="true">Yes — include raw chunk text</SelectItem>
                      <SelectItem value="false">No — skip chunks (smaller prompt)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Recall max tokens</label>
                  <Input
                    type="number"
                    value={form.recallMaxTokens}
                    onChange={(e) => setForm({ ...form, recallMaxTokens: e.target.value })}
                    placeholder="Default (inherit)"
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    Token budget for facts returned by recall.
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">
                    Recall chunks max tokens
                  </label>
                  <Input
                    type="number"
                    value={form.recallChunksMaxTokens}
                    onChange={(e) => setForm({ ...form, recallChunksMaxTokens: e.target.value })}
                    placeholder="Default (inherit)"
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground">
                    Token budget for raw chunk text returned by recall.
                  </p>
                </div>
              </section>
            </TabsContent>
          </div>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={updating}>
            Cancel
          </Button>
          <Button
            onClick={handleUpdate}
            disabled={updating || !form.name.trim() || !form.sourceQuery.trim()}
          >
            {updating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
                Updating...
              </>
            ) : (
              "Update"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
