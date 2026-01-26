"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  X,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pencil,
} from "lucide-react";
import { MemoryDetailModal } from "./memory-detail-modal";

interface ReflectResponseBasedOnFact {
  id: string;
  text: string;
  type: string;
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
  last_refreshed_at: string;
  created_at: string;
  reflect_response?: ReflectResponse;
}

export function MentalModelsView() {
  const { currentBank } = useBank();
  const [mentalModels, setMentalModels] = useState<MentalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 100;

  const [showCreateMentalModel, setShowCreateMentalModel] = useState(false);
  const [selectedMentalModel, setSelectedMentalModel] = useState<MentalModel | null>(null);
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

  const handleDelete = async () => {
    if (!currentBank || !deleteTarget) return;

    setDeleting(true);
    try {
      await client.deleteMentalModel(currentBank, deleteTarget.id);
      setMentalModels((prev) => prev.filter((m) => m.id !== deleteTarget.id));
      if (selectedMentalModel?.id === deleteTarget.id) setSelectedMentalModel(null);
      setDeleteTarget(null);
    } catch (error) {
      console.error("Error deleting mental model:", error);
      alert("Error deleting: " + (error as Error).message);
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
            <Button onClick={() => setShowCreateMentalModel(true)} variant="outline" size="sm">
              <Plus className="w-4 h-4 mr-2" />
              Add Mental Model
            </Button>
          </div>

          {filteredMentalModels.length > 0 ? (
            <>
              <div className="border rounded-lg overflow-hidden">
                <Table className="table-fixed">
                  <TableHeader>
                    <TableRow className="bg-muted/50">
                      <TableHead className="w-[20%]">ID</TableHead>
                      <TableHead className="w-[20%]">Name</TableHead>
                      <TableHead className="w-[35%]">Source Query</TableHead>
                      <TableHead className="w-[15%]">Last Refreshed</TableHead>
                      <TableHead className="w-[10%]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paginatedMentalModels.map((m) => {
                      const refreshedDate = new Date(m.last_refreshed_at);
                      const dateDisplay = refreshedDate.toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      });
                      const timeDisplay = refreshedDate.toLocaleTimeString("en-US", {
                        hour: "2-digit",
                        minute: "2-digit",
                        hour12: false,
                      });

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
                          <TableCell className="py-2 text-sm text-foreground">
                            <div>{dateDisplay}</div>
                            <div className="text-xs text-muted-foreground">{timeDisplay}</div>
                          </TableCell>
                          <TableCell className="py-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                setDeleteTarget({ id: m.id, name: m.name });
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

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

      {selectedMentalModel && (
        <MentalModelDetailPanel
          mentalModel={selectedMentalModel}
          onClose={() => setSelectedMentalModel(null)}
          onDelete={() =>
            setDeleteTarget({ id: selectedMentalModel.id, name: selectedMentalModel.name })
          }
          onRefreshed={(updated) => {
            setMentalModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
            setSelectedMentalModel(updated);
          }}
        />
      )}
    </div>
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
  const [form, setForm] = useState({ name: "", sourceQuery: "", maxTokens: "2048", tags: "" });

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
      await client.createMentalModel(currentBank, {
        name: form.name.trim(),
        source_query: form.sourceQuery.trim(),
        tags: tags.length > 0 ? tags : undefined,
        max_tokens: maxTokens,
      });

      setForm({ name: "", sourceQuery: "", maxTokens: "2048", tags: "" });
      onCreated();
    } catch (error) {
      console.error("Error creating mental model:", error);
      alert("Error creating mental model: " + (error as Error).message);
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) {
          setForm({ name: "", sourceQuery: "", maxTokens: "2048", tags: "" });
          onClose();
        }
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Create Mental Model</DialogTitle>
          <DialogDescription>
            Create a mental model by running a query. The content will be auto-generated and can be
            refreshed later.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
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
            <p className="text-xs text-muted-foreground">
              This query will be run to generate the initial content, and re-run when you refresh.
            </p>
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
            <p className="text-xs text-muted-foreground">
              Maximum tokens for the generated response (256-8192).
            </p>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Tags <span className="text-muted-foreground font-normal">(optional)</span>
            </label>
            <Input
              value={form.tags}
              onChange={(e) => setForm({ ...form, tags: e.target.value })}
              placeholder="e.g., project-x, team-alpha (comma-separated)"
            />
          </div>
        </div>

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

function MentalModelDetailPanel({
  mentalModel,
  onClose,
  onDelete,
  onRefreshed,
}: {
  mentalModel: MentalModel;
  onClose: () => void;
  onDelete: () => void;
  onRefreshed: (m: MentalModel) => void;
}) {
  const { currentBank } = useBank();
  const [refreshing, setRefreshing] = useState(false);
  const [viewMemoryId, setViewMemoryId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(mentalModel.name);
  const [saving, setSaving] = useState(false);

  // Reset edit form when mental model changes
  useEffect(() => {
    setEditName(mentalModel.name);
    setIsEditing(false);
  }, [mentalModel.id, mentalModel.name]);

  const handleRefresh = async () => {
    if (!currentBank) return;

    setRefreshing(true);
    const originalRefreshedAt = mentalModel.last_refreshed_at;

    try {
      // Submit the refresh task
      await client.refreshMentalModel(currentBank, mentalModel.id);

      // Poll until last_refreshed_at changes
      const pollInterval = 1000; // 1 second
      const maxAttempts = 120; // 2 minutes max
      let attempts = 0;

      const poll = async (): Promise<void> => {
        attempts++;
        try {
          const updated = await client.getMentalModel(currentBank, mentalModel.id);
          if (updated.last_refreshed_at !== originalRefreshedAt) {
            // Refresh complete
            onRefreshed(updated);
            setRefreshing(false);
            return;
          }
          if (attempts >= maxAttempts) {
            // Timeout
            setRefreshing(false);
            alert("Refresh is taking longer than expected. Check the operations list for status.");
            return;
          }
          // Continue polling
          setTimeout(poll, pollInterval);
        } catch (error) {
          console.error("Error polling mental model:", error);
          setRefreshing(false);
        }
      };

      // Start polling after a short delay
      setTimeout(poll, pollInterval);
    } catch (error) {
      console.error("Error refreshing mental model:", error);
      alert("Error refreshing: " + (error as Error).message);
      setRefreshing(false);
    }
  };

  const handleSave = async () => {
    if (!currentBank || !editName.trim()) return;

    setSaving(true);
    try {
      const updated = await client.updateMentalModel(currentBank, mentalModel.id, {
        name: editName.trim(),
      });
      onRefreshed(updated);
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating mental model:", error);
      alert("Error updating: " + (error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const formatDateTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    })} at ${date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    })}`;
  };

  // Extract all memories from based_on (excluding observations which are shown separately)
  const basedOnFacts = mentalModel.reflect_response?.based_on
    ? Object.entries(mentalModel.reflect_response.based_on)
        .filter(([factType]) => factType !== "observation")
        .flatMap(([factType, facts]) => facts.map((fact) => ({ ...fact, factType })))
    : [];

  // Observations are now in based_on with type=observation
  const observations = mentalModel.reflect_response?.based_on?.observation || [];

  return (
    <div className="fixed right-0 top-0 h-screen w-1/2 bg-card border-l shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
      <div className="p-6">
        <div className="flex justify-between items-start mb-8 pb-5 border-b border-border">
          <div className="flex-1 mr-4">
            {isEditing ? (
              <div className="space-y-2">
                <Input
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="text-lg font-bold"
                  autoFocus
                />
                <div className="flex gap-2">
                  <Button size="sm" onClick={handleSave} disabled={saving || !editName.trim()}>
                    {saving ? <Loader2 className="w-3 h-3 animate-spin mr-1" /> : null}
                    Save
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setEditName(mentalModel.name);
                      setIsEditing(false);
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-bold text-foreground">{mentalModel.name}</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsEditing(true)}
                    className="h-7 w-7 p-0"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground mt-1">{mentalModel.source_query}</p>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing}
              className="h-8"
            >
              {refreshing ? (
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
              ) : (
                <RefreshCw className="w-4 h-4 mr-1" />
              )}
              Refresh
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="space-y-6">
          <div>
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
              Content
            </div>
            <div className="prose prose-base dark:prose-invert max-w-none">
              <ReactMarkdown>{mentalModel.content}</ReactMarkdown>
            </div>
          </div>

          {/* Based On Facts Section */}
          {basedOnFacts.length > 0 && (
            <div className="border-t border-border pt-5">
              <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                Based On ({basedOnFacts.length} {basedOnFacts.length === 1 ? "fact" : "facts"})
              </div>
              <div className="space-y-3">
                {basedOnFacts.map((fact, i) => (
                  <div
                    key={fact.id || i}
                    className="p-4 bg-muted/50 rounded-lg border border-border/50"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          fact.factType === "world"
                            ? "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                            : fact.factType === "experience"
                              ? "bg-green-500/10 text-green-600 dark:text-green-400"
                              : "bg-purple-500/10 text-purple-600 dark:text-purple-400"
                        }`}
                      >
                        {fact.factType}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-6 text-xs"
                        onClick={() => setViewMemoryId(fact.id)}
                      >
                        View
                      </Button>
                    </div>
                    <p className="text-sm text-foreground leading-relaxed">{fact.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Observations Used Section */}
          {observations.length > 0 && (
            <div className="border-t border-border pt-5">
              <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                Observations Used ({observations.length})
              </div>
              <div className="space-y-3">
                {observations.map((obs, i) => (
                  <div
                    key={obs.id || i}
                    className="p-4 bg-muted/50 rounded-lg border border-border/50"
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400">
                        observation
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-6 text-xs"
                        onClick={() => setViewMemoryId(obs.id)}
                      >
                        View
                      </Button>
                    </div>
                    <p className="text-sm text-foreground leading-relaxed">{obs.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No based_on data yet */}
          {!mentalModel.reflect_response && (
            <div className="border-t border-border pt-5">
              <div className="text-xs font-bold text-muted-foreground uppercase mb-3">Based On</div>
              <p className="text-sm text-muted-foreground">
                No source data available. Click &quot;Refresh&quot; to regenerate with source
                tracking.
              </p>
            </div>
          )}

          {mentalModel.tags && mentalModel.tags.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                Tags
              </div>
              <div className="flex flex-wrap gap-2">
                {mentalModel.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 rounded bg-muted text-muted-foreground text-sm"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-6 text-sm text-muted-foreground">
            <span>Created: {formatDateTime(mentalModel.created_at)}</span>
            <span>Refreshed: {formatDateTime(mentalModel.last_refreshed_at)}</span>
          </div>

          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              ID
            </div>
            <code className="text-sm font-mono break-all text-muted-foreground">
              {mentalModel.id}
            </code>
          </div>

          <div className="pt-4 border-t border-border">
            <Button
              variant="outline"
              size="sm"
              onClick={onDelete}
              className="text-muted-foreground hover:text-destructive hover:border-destructive"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </Button>
          </div>
        </div>
      </div>

      {/* Memory Detail Modal */}
      {viewMemoryId && currentBank && (
        <MemoryDetailModal memoryId={viewMemoryId} onClose={() => setViewMemoryId(null)} />
      )}
    </div>
  );
}
