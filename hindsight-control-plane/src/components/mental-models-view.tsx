"use client";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  RefreshCw,
  Lightbulb,
  Sparkles,
  Target,
  Pencil,
  Check,
  X,
  Plus,
  Pin,
  ChevronDown,
  FileText,
  Calendar,
  Tag,
  Users,
  ChevronRight,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { DocumentChunkModal } from "./document-chunk-modal";

type ViewMode = "dashboard" | "table";

interface MentalModelObservation {
  title: string;
  text: string;
  based_on: string[];
}

interface MemoryDetail {
  id: string;
  text: string;
  context: string;
  date: string;
  type: string;
  mentioned_at: string | null;
  occurred_start: string | null;
  occurred_end: string | null;
  entities: string[];
  document_id: string | null;
  chunk_id: string | null;
  tags: string[];
}

interface MentalModel {
  id: string;
  bank_id: string;
  subtype: string;
  name: string;
  description: string;
  observations?: MentalModelObservation[];
  entity_id: string | null;
  links: string[];
  tags?: string[];
  last_updated: string | null;
  created_at: string;
}

// Helper to count total source memories across all observations
function getTotalMemoryCount(model: MentalModel): number {
  return model.observations?.reduce((sum, obs) => sum + (obs.based_on?.length || 0), 0) || 0;
}

export function MentalModelsView() {
  const { currentBank } = useBank();
  const [mentalModels, setMentalModels] = useState<MentalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState<
    "all" | "pinned" | "structural" | "emergent" | "learned" | null
  >(null);
  const [operationStatus, setOperationStatus] = useState<{
    operationId: string;
    type: "all" | "pinned" | "structural" | "emergent" | "learned";
    status: "pending" | "completed" | "failed" | "not_found";
    errorMessage?: string | null;
  } | null>(null);
  const [mission, setMission] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("dashboard");
  const [selectedModel, setSelectedModel] = useState<MentalModel | null>(null);

  // Mission editing state
  const [editingMission, setEditingMission] = useState(false);
  const [editMissionValue, setEditMissionValue] = useState("");
  const [savingMission, setSavingMission] = useState(false);

  // Create mental model state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newModel, setNewModel] = useState({
    name: "",
    description: "",
  });

  // Auto-refresh interval (5 seconds)
  const AUTO_REFRESH_INTERVAL = 5000;
  const currentBankRef = useRef(currentBank);
  currentBankRef.current = currentBank;
  const selectedModelRef = useRef(selectedModel);
  selectedModelRef.current = selectedModel;

  const loadMentalModels = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const [modelsData, profileData] = await Promise.all([
        client.listMentalModels(currentBank),
        client.getBankProfile(currentBank),
      ]);
      setMentalModels(modelsData.items || []);
      setMission(profileData.mission);
      setEditMissionValue(profileData.mission || "");
    } catch (error) {
      console.error("Error loading mental models:", error);
    } finally {
      setLoading(false);
    }
  };

  // Silent refresh for auto-refresh (no loading spinner)
  const silentRefreshModels = async () => {
    const bank = currentBankRef.current;
    if (!bank) return;

    try {
      const modelsData = await client.listMentalModels(bank);
      const items = modelsData.items || [];
      setMentalModels(items);

      // Update selected model if it exists in the refreshed list
      const currentSelected = selectedModelRef.current;
      if (currentSelected) {
        const updatedModel = items.find((m) => m.id === currentSelected.id);
        if (updatedModel) {
          setSelectedModel(updatedModel);
        }
      }
    } catch (error) {
      console.error("Error auto-refreshing mental models:", error);
    }
  };

  const handleSaveMission = async () => {
    if (!currentBank) return;

    setSavingMission(true);
    try {
      await client.setBankMission(currentBank, editMissionValue);
      setMission(editMissionValue || null);
      setEditingMission(false);
    } catch (error) {
      console.error("Error saving mission:", error);
      alert("Error saving mission: " + (error as Error).message);
    } finally {
      setSavingMission(false);
    }
  };

  const handleCancelEditMission = () => {
    setEditMissionValue(mission || "");
    setEditingMission(false);
  };

  const handleRefresh = async (subtype?: "structural" | "emergent" | "pinned" | "learned") => {
    if (!currentBank) return;

    const refreshType = subtype || "all";
    setRefreshing(refreshType);
    setOperationStatus(null);
    try {
      const result = await client.refreshMentalModels(currentBank, subtype);
      // Start polling for operation status
      if (result.operation_id) {
        setOperationStatus({
          operationId: result.operation_id,
          type: refreshType,
          status: "pending",
        });
        pollOperationStatus(result.operation_id, refreshType);
      }
    } catch (error) {
      console.error("Error refreshing mental models:", error);
      alert("Error refreshing mental models: " + (error as Error).message);
    } finally {
      setRefreshing(null);
    }
  };

  const pollOperationStatus = async (
    operationId: string,
    type: "all" | "pinned" | "structural" | "emergent" | "learned"
  ) => {
    const bank = currentBankRef.current;
    if (!bank) return;

    const poll = async () => {
      try {
        const status = await client.getOperationStatus(bank, operationId);
        setOperationStatus({
          operationId,
          type,
          status: status.status,
          errorMessage: status.error_message,
        });

        if (status.status === "pending") {
          // Continue polling every 2 seconds
          setTimeout(poll, 2000);
        } else {
          // Operation completed or failed - refresh models and clear after delay
          if (status.status === "completed") {
            await silentRefreshModels();
          }
          setTimeout(() => setOperationStatus(null), 5000);
        }
      } catch (error) {
        console.error("Error polling operation status:", error);
        // On error, assume completed and clear
        setOperationStatus(null);
      }
    };

    poll();
  };

  const handleCreateModel = async () => {
    if (!currentBank || !newModel.name.trim() || !newModel.description.trim()) return;

    setCreating(true);
    try {
      await client.createMentalModel(currentBank, {
        name: newModel.name.trim(),
        description: newModel.description.trim(),
      });

      // Reset form and reload
      setNewModel({ name: "", description: "" });
      setShowCreateForm(false);
      await loadMentalModels();
    } catch (error) {
      console.error("Error creating mental model:", error);
      alert("Error creating mental model: " + (error as Error).message);
    } finally {
      setCreating(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadMentalModels();

      // Set up auto-refresh interval for mental models only
      const intervalId = setInterval(silentRefreshModels, AUTO_REFRESH_INTERVAL);

      return () => {
        clearInterval(intervalId);
      };
    }
  }, [currentBank]);

  // Close detail panel on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedModel) {
        setSelectedModel(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedModel]);

  if (!currentBank) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <p className="text-muted-foreground">Select a memory bank to view mental models.</p>
        </CardContent>
      </Card>
    );
  }

  // Group mental models by subtype
  const pinnedModels = mentalModels.filter((m) => m.subtype === "pinned");
  const structuralModels = mentalModels.filter((m) => m.subtype === "structural");
  const emergentModels = mentalModels.filter((m) => m.subtype === "emergent");
  const learnedModels = mentalModels.filter((m) => m.subtype === "learned");

  const getSubtypeIcon = (subtype: string) => {
    switch (subtype) {
      case "pinned":
        return <Pin className="w-4 h-4 text-amber-500" />;
      case "structural":
        return <Target className="w-4 h-4 text-blue-500" />;
      case "emergent":
        return <Sparkles className="w-4 h-4 text-emerald-500" />;
      case "learned":
        return <Lightbulb className="w-4 h-4 text-violet-500" />;
      default:
        return <Lightbulb className="w-4 h-4 text-slate-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Mission section - compact inline style */}
      <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-muted/50 border border-border">
        <Target className="w-4 h-4 text-muted-foreground shrink-0" />
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide shrink-0">
          Mission:
        </span>
        {editingMission ? (
          <div className="flex gap-2 items-center flex-1">
            <Input
              value={editMissionValue}
              onChange={(e) => setEditMissionValue(e.target.value)}
              placeholder="e.g., Be a PM for the engineering team..."
              className="flex-1 h-7 text-sm"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSaveMission();
                if (e.key === "Escape") handleCancelEditMission();
              }}
            />
            <Button
              size="icon"
              variant="ghost"
              onClick={handleSaveMission}
              disabled={savingMission}
              className="h-7 w-7"
              title="Save mission"
            >
              <Check className="w-3.5 h-3.5 text-green-600" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              onClick={handleCancelEditMission}
              disabled={savingMission}
              className="h-7 w-7"
              title="Cancel"
            >
              <X className="w-3.5 h-3.5 text-red-600" />
            </Button>
          </div>
        ) : (
          <>
            <span
              className={`text-sm flex-1 ${mission ? "text-foreground" : "text-muted-foreground italic"}`}
            >
              {mission || "No mission set"}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setEditingMission(true)}
              className="h-6 px-2 text-xs"
            >
              <Pencil className="w-3 h-3 mr-1" />
              {mission ? "Edit" : "Set"}
            </Button>
          </>
        )}
      </div>

      {/* Header with count, actions, and view toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <p className="text-sm text-muted-foreground">
            {mentalModels.length} mental model{mentalModels.length !== 1 ? "s" : ""}
          </p>
          <div className="flex gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  size="sm"
                  disabled={!!refreshing || !mission}
                  title={
                    !mission
                      ? "Set a mission first to generate mental models"
                      : "Regenerate mental models"
                  }
                  className="h-8"
                >
                  {refreshing ? (
                    <RefreshCw className="w-4 h-4 mr-1 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4 mr-1" />
                  )}
                  {refreshing ? "Regenerating..." : "Regenerate"}
                  <ChevronDown className="w-3 h-3 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => handleRefresh()}>
                  <Sparkles className="w-4 h-4" />
                  All
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleRefresh("pinned")}>
                  <Pin className="w-4 h-4 text-amber-500" />
                  Pinned
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleRefresh("structural")}>
                  <Target className="w-4 h-4 text-blue-500" />
                  Structural
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleRefresh("emergent")}>
                  <Sparkles className="w-4 h-4 text-emerald-500" />
                  Emergent
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleRefresh("learned")}>
                  <Lightbulb className="w-4 h-4 text-violet-500" />
                  Learned
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* View mode toggle - top right like data-view */}
        <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
          <button
            onClick={() => setViewMode("dashboard")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              viewMode === "dashboard"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Dashboard
          </button>
          <button
            onClick={() => setViewMode("table")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              viewMode === "table"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Table
          </button>
        </div>
      </div>

      {/* Regeneration Progress Banner */}
      {refreshing && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-primary/10 border border-primary/20 animate-in fade-in duration-300">
          <RefreshCw className="w-4 h-4 text-primary animate-spin" />
          <span className="text-sm font-medium text-foreground">
            Scheduling regeneration for{" "}
            {refreshing === "all" ? "all mental models" : `${refreshing} models`}...
          </span>
        </div>
      )}

      {/* Operation Status Banner - shows operation progress */}
      {operationStatus && !refreshing && (
        <div
          className={`flex items-center gap-3 px-4 py-3 rounded-lg animate-in fade-in duration-300 ${
            operationStatus.status === "pending"
              ? "bg-blue-500/10 border border-blue-500/20"
              : operationStatus.status === "completed"
                ? "bg-emerald-500/10 border border-emerald-500/20"
                : "bg-red-500/10 border border-red-500/20"
          }`}
        >
          {operationStatus.status === "pending" ? (
            <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
          ) : operationStatus.status === "completed" ? (
            <Check className="w-4 h-4 text-emerald-500" />
          ) : (
            <X className="w-4 h-4 text-red-500" />
          )}
          <span className="text-sm font-medium text-foreground">
            {operationStatus.status === "pending"
              ? `Regenerating ${operationStatus.type === "all" ? "all mental models" : `${operationStatus.type} models`}...`
              : operationStatus.status === "completed"
                ? `Successfully regenerated ${operationStatus.type === "all" ? "all mental models" : `${operationStatus.type} models`}`
                : `Failed to regenerate ${operationStatus.type === "all" ? "all mental models" : `${operationStatus.type} models`}`}
          </span>
          {operationStatus.status === "pending" && (
            <span className="text-xs text-muted-foreground">Running in background...</span>
          )}
          {operationStatus.errorMessage && (
            <span className="text-xs text-red-500">{operationStatus.errorMessage}</span>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-6 px-2 text-xs"
            onClick={() => setOperationStatus(null)}
          >
            <X className="w-3 h-3" />
          </Button>
        </div>
      )}

      {/* Create Mental Model Dialog */}
      <Dialog open={showCreateForm} onOpenChange={setShowCreateForm}>
        <DialogContent className="sm:max-w-md border-2 border-primary">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Pin className="w-5 h-5 text-primary" />
              Create Pinned Mental Model
            </DialogTitle>
            <DialogDescription>
              Pinned models persist across regeneration and help organize your memory bank.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Name *</label>
              <Input
                value={newModel.name}
                onChange={(e) => setNewModel({ ...newModel, name: e.target.value })}
                placeholder="e.g., Product Roadmap, Team Structure, Q1 Goals"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Description *</label>
              <Textarea
                value={newModel.description}
                onChange={(e) => setNewModel({ ...newModel, description: e.target.value })}
                placeholder="What should this mental model track? e.g., Key product priorities and upcoming features"
                className="min-h-[100px]"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowCreateForm(false);
                setNewModel({ name: "", description: "" });
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateModel}
              disabled={creating || !newModel.name.trim() || !newModel.description.trim()}
            >
              {creating ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Loading state */}
      {loading ? (
        <div className="text-center py-12">
          <RefreshCw className="w-8 h-8 mx-auto mb-3 text-muted-foreground animate-spin" />
          <p className="text-muted-foreground">Loading mental models...</p>
        </div>
      ) : mentalModels.length > 0 ? (
        <>
          {/* Table View */}
          {viewMode === "table" && (
            <div className="border rounded-lg overflow-hidden">
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    <TableHead className="w-[28%]">Name</TableHead>
                    <TableHead className="w-[10%]">Source</TableHead>
                    <TableHead className="w-[32%]">Description</TableHead>
                    <TableHead className="w-[8%] text-center">Obs</TableHead>
                    <TableHead className="w-[10%] text-center">Memories</TableHead>
                    <TableHead className="w-[12%]">Updated</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {mentalModels.map((model) => (
                    <TableRow
                      key={model.id}
                      onClick={() => setSelectedModel(model)}
                      className={`cursor-pointer hover:bg-muted/50 ${
                        selectedModel?.id === model.id ? "bg-primary/10" : ""
                      }`}
                    >
                      <TableCell className="py-2">
                        <div className="flex items-center gap-2">
                          {getSubtypeIcon(model.subtype)}
                          <span className="font-medium text-foreground truncate">{model.name}</span>
                        </div>
                      </TableCell>
                      <TableCell className="py-2">
                        <span
                          className={`text-xs px-1.5 py-0.5 rounded ${
                            model.subtype === "pinned"
                              ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                              : model.subtype === "structural"
                                ? "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                                : model.subtype === "learned"
                                  ? "bg-violet-500/10 text-violet-600 dark:text-violet-400"
                                  : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                          }`}
                        >
                          {model.subtype}
                        </span>
                      </TableCell>
                      <TableCell className="py-2">
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {model.description}
                        </p>
                      </TableCell>
                      <TableCell className="py-2 text-center text-sm text-foreground">
                        {model.observations?.length || 0}
                      </TableCell>
                      <TableCell className="py-2 text-center text-sm text-foreground">
                        {getTotalMemoryCount(model)}
                      </TableCell>
                      <TableCell className="py-2 text-xs text-muted-foreground">
                        {model.last_updated
                          ? new Date(model.last_updated).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                            }) +
                            " " +
                            new Date(model.last_updated).toLocaleTimeString("en-US", {
                              hour: "numeric",
                              minute: "2-digit",
                            })
                          : "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {/* Dashboard View - Grouped by subtype, 2 columns */}
          {/* Order: Structural, Emergent, Pinned, Learned */}
          {viewMode === "dashboard" && (
            <div className="space-y-8">
              {/* Structural Models Section */}
              {structuralModels.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    Structural Models
                    <span className="text-sm font-normal text-muted-foreground">
                      (Mission-derived)
                    </span>
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    {structuralModels.map((model) => (
                      <ModelListCard
                        key={model.id}
                        model={model}
                        selected={selectedModel?.id === model.id}
                        onClick={() => setSelectedModel(model)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Emergent Models Section - Always show */}
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  Emergent Models
                  <span className="text-sm font-normal text-muted-foreground">
                    (Pattern-derived)
                  </span>
                </h3>
                {emergentModels.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3">
                    {emergentModels.map((model) => (
                      <ModelListCard
                        key={model.id}
                        model={model}
                        selected={selectedModel?.id === model.id}
                        onClick={() => setSelectedModel(model)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="p-6 border border-dashed border-border rounded-lg text-center">
                    <Sparkles className="w-6 h-6 mx-auto mb-2 text-emerald-500/50" />
                    <p className="text-sm text-muted-foreground">
                      No emergent models yet. Models are discovered from patterns in your data.
                    </p>
                  </div>
                )}
              </div>

              {/* Pinned Models Section - Always show */}
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                  Pinned Models
                  <span className="text-sm font-normal text-muted-foreground">(User-defined)</span>
                  <Button
                    onClick={() => setShowCreateForm(true)}
                    variant="outline"
                    size="sm"
                    className="ml-auto h-7 text-xs"
                  >
                    <Plus className="w-3 h-3 mr-1" />
                    Add
                  </Button>
                </h3>
                {pinnedModels.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3">
                    {pinnedModels.map((model) => (
                      <ModelListCard
                        key={model.id}
                        model={model}
                        selected={selectedModel?.id === model.id}
                        onClick={() => setSelectedModel(model)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="p-6 border border-dashed border-border rounded-lg text-center">
                    <Pin className="w-6 h-6 mx-auto mb-2 text-amber-500/50" />
                    <p className="text-sm text-muted-foreground">
                      No pinned models yet. Create custom models to track specific topics.
                    </p>
                  </div>
                )}
              </div>

              {/* Learned Models Section - Always show */}
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-violet-500" />
                  Learned Models
                  <span className="text-sm font-normal text-muted-foreground">
                    (Reflection-derived)
                  </span>
                </h3>
                {learnedModels.length > 0 ? (
                  <div className="grid grid-cols-2 gap-3">
                    {learnedModels.map((model) => (
                      <ModelListCard
                        key={model.id}
                        model={model}
                        selected={selectedModel?.id === model.id}
                        onClick={() => setSelectedModel(model)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="p-6 border border-dashed border-border rounded-lg text-center">
                    <Lightbulb className="w-6 h-6 mx-auto mb-2 text-violet-500/50" />
                    <p className="text-sm text-muted-foreground">
                      No learned models yet. Models are created automatically during reflection.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="p-12 text-center">
            <Lightbulb className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-medium mb-2">No Mental Models Yet</h3>
            <p className="text-muted-foreground text-sm max-w-md mx-auto">
              {!mission
                ? "Set a mission above, then use the refresh buttons to create mental models."
                : "Use the refresh buttons to create mental models. Click 'All' for both types, or refresh 'Structural' (from mission) and 'Emergent' (from data) separately."}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Detail Panel - Fixed on Right */}
      {selectedModel && (
        <div className="fixed right-0 top-0 h-screen w-1/2 bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
          <MentalModelDetailPanel
            model={selectedModel}
            onClose={() => setSelectedModel(null)}
            onRegenerated={silentRefreshModels}
          />
        </div>
      )}
    </div>
  );
}

// Compact card for dashboard view (no subtype chip since already grouped)
function ModelListCard({
  model,
  selected,
  onClick,
}: {
  model: MentalModel;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <Card
      className={`cursor-pointer transition-colors ${
        selected ? "bg-primary/10 border-primary" : "hover:bg-muted/50"
      }`}
      onClick={onClick}
    >
      <CardContent className="p-3">
        <div className="flex-1 min-w-0">
          <span className="font-medium text-sm text-foreground">{model.name}</span>
          <p className="text-xs text-muted-foreground line-clamp-2 mt-1">{model.description}</p>
          <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
            <span>
              <span className="font-medium text-foreground">{model.observations?.length || 0}</span>{" "}
              obs
            </span>
            <span>
              <span className="font-medium text-foreground">{getTotalMemoryCount(model)}</span>{" "}
              memories
            </span>
            {model.last_updated && (
              <span>
                {new Date(model.last_updated).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })}{" "}
                {new Date(model.last_updated).toLocaleTimeString("en-US", {
                  hour: "numeric",
                  minute: "2-digit",
                })}
              </span>
            )}
          </div>
          {model.tags && model.tags.length > 0 && (
            <div className="flex items-center gap-1 mt-2 flex-wrap">
              <Tag className="w-3 h-3 text-muted-foreground" />
              {model.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Detail panel for mental model (like MemoryDetailPanel)
function MentalModelDetailPanel({
  model,
  onClose,
  onRegenerated,
}: {
  model: MentalModel;
  onClose: () => void;
  onRegenerated?: () => void;
}) {
  const { currentBank } = useBank();
  const [expandedObservation, setExpandedObservation] = useState<number | null>(null);
  const [loadingFacts, setLoadingFacts] = useState<Record<string, boolean>>({});
  const [factDetails, setFactDetails] = useState<Record<string, MemoryDetail>>({});
  const [factErrors, setFactErrors] = useState<Record<string, string>>({});
  const [regenerateStatus, setRegenerateStatus] = useState<{
    status: "scheduling" | "pending" | "completed" | "failed";
    errorMessage?: string | null;
  } | null>(null);

  // Document/Chunk modal state
  const [modalType, setModalType] = useState<"document" | "chunk" | null>(null);
  const [modalId, setModalId] = useState<string | null>(null);

  const openDocumentModal = (docId: string) => {
    setModalType("document");
    setModalId(docId);
  };

  const openChunkModal = (chunkId: string) => {
    setModalType("chunk");
    setModalId(chunkId);
  };

  const closeModal = () => {
    setModalType(null);
    setModalId(null);
  };

  const handleRegenerate = async () => {
    if (!currentBank) return;

    setRegenerateStatus({ status: "scheduling" });
    try {
      const result = await client.generateMentalModel(currentBank, model.id);
      if (result.operation_id) {
        setRegenerateStatus({ status: "pending" });
        pollRegenerateStatus(result.operation_id);
      }
    } catch (error) {
      console.error("Error regenerating mental model:", error);
      setRegenerateStatus({ status: "failed", errorMessage: (error as Error).message });
      setTimeout(() => setRegenerateStatus(null), 5000);
    }
  };

  const pollRegenerateStatus = async (operationId: string) => {
    if (!currentBank) return;

    const poll = async () => {
      try {
        const status = await client.getOperationStatus(currentBank, operationId);
        if (status.status === "pending") {
          setTimeout(poll, 2000);
        } else if (status.status === "completed") {
          setRegenerateStatus({ status: "completed" });
          onRegenerated?.();
          setTimeout(() => setRegenerateStatus(null), 3000);
        } else {
          setRegenerateStatus({ status: "failed", errorMessage: status.error_message });
          setTimeout(() => setRegenerateStatus(null), 5000);
        }
      } catch (error) {
        console.error("Error polling regenerate status:", error);
        setRegenerateStatus(null);
      }
    };

    poll();
  };

  const loadFactDetails = async (factIds: string[]) => {
    if (!currentBank) return;

    // Filter out already loaded facts
    const toLoad = factIds.filter((id) => !factDetails[id] && !loadingFacts[id]);
    if (toLoad.length === 0) return;

    // Mark as loading
    setLoadingFacts((prev) => {
      const next = { ...prev };
      toLoad.forEach((id) => (next[id] = true));
      return next;
    });

    // Load each fact
    for (const factId of toLoad) {
      try {
        const memory = await client.getMemory(factId, currentBank);
        setFactDetails((prev) => ({ ...prev, [factId]: memory }));
      } catch (error) {
        console.error(`Error loading fact ${factId}:`, error);
        setFactErrors((prev) => ({ ...prev, [factId]: "Failed to load" }));
      } finally {
        setLoadingFacts((prev) => ({ ...prev, [factId]: false }));
      }
    }
  };

  const toggleObservation = (idx: number, factIds: string[]) => {
    if (expandedObservation === idx) {
      setExpandedObservation(null);
    } else {
      setExpandedObservation(idx);
      // Load facts when expanding
      if (factIds.length > 0) {
        loadFactDetails(factIds);
      }
    }
  };

  const getSubtypeIcon = (subtype: string) => {
    switch (subtype) {
      case "pinned":
        return <Pin className="w-5 h-5 text-amber-500" />;
      case "structural":
        return <Target className="w-5 h-5 text-blue-500" />;
      case "emergent":
        return <Sparkles className="w-5 h-5 text-emerald-500" />;
      case "learned":
        return <Lightbulb className="w-5 h-5 text-violet-500" />;
      default:
        return <Lightbulb className="w-5 h-5 text-slate-500" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "world":
        return "bg-blue-500/10 text-blue-600 dark:text-blue-400";
      case "experience":
        return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400";
      case "opinion":
        return "bg-amber-500/10 text-amber-600 dark:text-amber-400";
      default:
        return "bg-slate-500/10 text-slate-600 dark:text-slate-400";
    }
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex justify-between items-start mb-8 pb-5 border-b border-border">
        <div className="flex items-start gap-3">
          {getSubtypeIcon(model.subtype)}
          <div>
            <h3 className="text-xl font-bold text-foreground">{model.name}</h3>
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span
                className={`text-xs px-1.5 py-0.5 rounded ${
                  model.subtype === "pinned"
                    ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                    : model.subtype === "structural"
                      ? "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                      : model.subtype === "learned"
                        ? "bg-violet-500/10 text-violet-600 dark:text-violet-400"
                        : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                }`}
              >
                {model.subtype}
              </span>
              {model.tags && model.tags.length > 0 && (
                <>
                  {model.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground flex items-center gap-1"
                    >
                      <Tag className="w-2.5 h-2.5" />
                      {tag}
                    </span>
                  ))}
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={regenerateStatus?.status === "completed" ? "default" : "outline"}
            size="sm"
            onClick={handleRegenerate}
            disabled={
              !!regenerateStatus &&
              regenerateStatus.status !== "completed" &&
              regenerateStatus.status !== "failed"
            }
            className={`h-8 ${
              regenerateStatus?.status === "completed"
                ? "bg-emerald-500 hover:bg-emerald-600"
                : regenerateStatus?.status === "failed"
                  ? "border-red-500 text-red-500"
                  : ""
            }`}
          >
            {regenerateStatus?.status === "scheduling" || regenerateStatus?.status === "pending" ? (
              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            ) : regenerateStatus?.status === "completed" ? (
              <Check className="w-4 h-4 mr-1" />
            ) : regenerateStatus?.status === "failed" ? (
              <X className="w-4 h-4 mr-1" />
            ) : (
              <RefreshCw className="w-4 h-4 mr-1" />
            )}
            {regenerateStatus?.status === "scheduling"
              ? "Scheduling..."
              : regenerateStatus?.status === "pending"
                ? "Regenerating..."
                : regenerateStatus?.status === "completed"
                  ? "Done!"
                  : regenerateStatus?.status === "failed"
                    ? "Failed"
                    : "Regenerate"}
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Description */}
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Description
          </div>
          <div className="prose prose-base dark:prose-invert max-w-none prose-p:leading-relaxed prose-p:text-foreground">
            <ReactMarkdown>{model.description}</ReactMarkdown>
          </div>
        </div>

        {/* Observations */}
        {model.observations && model.observations.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
              Observations ({model.observations.length})
            </div>
            <div className="space-y-4">
              {model.observations.map((observation, idx) => (
                <div key={idx} className="p-4 bg-muted/50 rounded-lg">
                  {observation.title && (
                    <h4 className="text-base font-semibold mb-2 text-foreground">
                      {observation.title}
                    </h4>
                  )}
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => (
                          <p className="text-sm leading-relaxed my-2 text-foreground">{children}</p>
                        ),
                        ul: ({ children }) => <ul className="my-2 pl-5 list-disc">{children}</ul>,
                        ol: ({ children }) => (
                          <ol className="my-2 pl-5 list-decimal">{children}</ol>
                        ),
                        li: ({ children }) => (
                          <li className="my-1 text-sm text-foreground">{children}</li>
                        ),
                        code: ({ children }) => (
                          <code className="text-sm bg-muted px-1.5 py-0.5 rounded">{children}</code>
                        ),
                        strong: ({ children }) => (
                          <strong className="font-semibold text-foreground">{children}</strong>
                        ),
                      }}
                    >
                      {observation.text}
                    </ReactMarkdown>
                  </div>
                  {observation.based_on && observation.based_on.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-border/50">
                      <button
                        onClick={() => toggleObservation(idx, observation.based_on)}
                        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors group"
                      >
                        <ChevronRight
                          className={`w-3 h-3 transition-transform ${expandedObservation === idx ? "rotate-90" : ""}`}
                        />
                        <FileText className="w-3 h-3" />
                        <span>
                          {observation.based_on.length} source memor
                          {observation.based_on.length !== 1 ? "ies" : "y"}
                        </span>
                        <span className="text-muted-foreground/50 group-hover:text-muted-foreground">
                          {expandedObservation === idx ? "Hide" : "Show"}
                        </span>
                      </button>

                      {expandedObservation === idx && (
                        <div className="mt-3 space-y-2 animate-in slide-in-from-top-2 duration-200">
                          {observation.based_on.map((factId) => {
                            const fact = factDetails[factId];
                            const isLoading = loadingFacts[factId];
                            const error = factErrors[factId];

                            return (
                              <div
                                key={factId}
                                className="p-3 bg-background rounded-md border border-border"
                              >
                                {isLoading ? (
                                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Loading...
                                  </div>
                                ) : error ? (
                                  <div className="text-sm text-red-500">{error}</div>
                                ) : fact ? (
                                  <div className="space-y-2">
                                    <p className="text-sm text-foreground">{fact.text}</p>
                                    <div className="flex flex-wrap items-center gap-2 text-xs">
                                      <span
                                        className={`px-1.5 py-0.5 rounded ${getTypeColor(fact.type)}`}
                                      >
                                        {fact.type}
                                      </span>
                                      {fact.date && (
                                        <span className="flex items-center gap-1 text-muted-foreground">
                                          <Calendar className="w-3 h-3" />
                                          {new Date(fact.date).toLocaleDateString("en-US", {
                                            month: "short",
                                            day: "numeric",
                                            year: "numeric",
                                          })}
                                        </span>
                                      )}
                                      {fact.entities && fact.entities.length > 0 && (
                                        <span className="flex items-center gap-1 text-muted-foreground">
                                          <Users className="w-3 h-3" />
                                          {fact.entities.slice(0, 3).join(", ")}
                                          {fact.entities.length > 3 &&
                                            ` +${fact.entities.length - 3}`}
                                        </span>
                                      )}
                                      {fact.tags && fact.tags.length > 0 && (
                                        <span className="flex items-center gap-1 text-muted-foreground">
                                          <Tag className="w-3 h-3" />
                                          {fact.tags.slice(0, 2).join(", ")}
                                          {fact.tags.length > 2 && ` +${fact.tags.length - 2}`}
                                        </span>
                                      )}
                                    </div>
                                    {fact.context && (
                                      <div className="pt-2 mt-2 border-t border-border/50">
                                        <span className="text-xs text-muted-foreground">
                                          Context:{" "}
                                        </span>
                                        <span className="text-xs text-foreground/80">
                                          {fact.context}
                                        </span>
                                      </div>
                                    )}
                                    <div className="pt-1 flex items-center gap-2">
                                      <code className="text-[10px] font-mono text-muted-foreground/60">
                                        {factId}
                                      </code>
                                      {(fact.document_id || fact.chunk_id) && (
                                        <div className="flex gap-1 ml-auto">
                                          {fact.document_id && (
                                            <button
                                              onClick={(e) => {
                                                e.stopPropagation();
                                                openDocumentModal(fact.document_id!);
                                              }}
                                              className="text-[10px] px-1.5 py-0.5 rounded bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
                                            >
                                              <ExternalLink className="w-2.5 h-2.5" />
                                              Doc
                                            </button>
                                          )}
                                          {fact.chunk_id && (
                                            <button
                                              onClick={(e) => {
                                                e.stopPropagation();
                                                openChunkModal(fact.chunk_id!);
                                              }}
                                              className="text-[10px] px-1.5 py-0.5 rounded bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
                                            >
                                              <ExternalLink className="w-2.5 h-2.5" />
                                              Chunk
                                            </button>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                ) : (
                                  <code className="text-xs font-mono text-muted-foreground">
                                    {factId}
                                  </code>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Observations
            </div>
            <div className="text-2xl font-bold text-foreground">
              {model.observations?.length || 0}
            </div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Source Memories
            </div>
            <div className="text-2xl font-bold text-foreground">{getTotalMemoryCount(model)}</div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
              Last Updated
            </div>
            <div className="text-base font-medium text-foreground">
              {model.last_updated
                ? new Date(model.last_updated).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })
                : "Never"}
            </div>
          </div>
        </div>

        {/* ID */}
        <div className="p-4 bg-muted/50 rounded-lg">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Model ID
          </div>
          <code className="text-sm font-mono break-all text-muted-foreground">{model.id}</code>
        </div>
      </div>

      {/* Document/Chunk Modal */}
      {modalType && modalId && (
        <DocumentChunkModal type={modalType} id={modalId} onClose={closeModal} />
      )}
    </div>
  );
}
