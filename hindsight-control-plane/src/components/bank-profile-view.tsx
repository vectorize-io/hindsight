"use client";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { useRouter } from "next/navigation";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  RefreshCw,
  Save,
  Brain,
  Clock,
  AlertCircle,
  CheckCircle,
  Database,
  Link2,
  FolderOpen,
  Activity,
  Trash2,
  Target,
  AlertTriangle,
  Plus,
  Tag,
  Loader2,
  X,
  MoreVertical,
  Pencil,
} from "lucide-react";

interface DispositionTraits {
  skepticism: number;
  literalism: number;
  empathy: number;
}

interface BankProfile {
  bank_id: string;
  name: string;
  disposition: DispositionTraits;
  mission: string;
}

interface BankStats {
  bank_id: string;
  total_nodes: number;
  total_links: number;
  total_documents: number;
  nodes_by_fact_type: {
    world?: number;
    experience?: number;
    opinion?: number;
  };
  links_by_link_type: {
    temporal?: number;
    semantic?: number;
    entity?: number;
  };
  pending_operations: number;
  failed_operations: number;
  // Consolidation stats
  last_consolidated_at: string | null;
  pending_consolidation: number;
  total_mental_models: number;
}

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  document_id: string | null;
  created_at: string;
  status: string;
  error_message: string | null;
}

interface Directive {
  id: string;
  bank_id: string;
  name: string;
  content: string;
  priority: number;
  is_active: boolean;
  tags: string[];
  created_at: string;
  updated_at: string;
}

const TRAIT_LABELS: Record<
  keyof DispositionTraits,
  { label: string; shortLabel: string; description: string; lowLabel: string; highLabel: string }
> = {
  skepticism: {
    label: "Skepticism",
    shortLabel: "S",
    description: "How skeptical vs trusting when forming observations",
    lowLabel: "Trusting",
    highLabel: "Skeptical",
  },
  literalism: {
    label: "Literalism",
    shortLabel: "L",
    description: "How literally to interpret information when forming observations",
    lowLabel: "Flexible",
    highLabel: "Literal",
  },
  empathy: {
    label: "Empathy",
    shortLabel: "E",
    description: "How much to consider emotional context when forming observations",
    lowLabel: "Detached",
    highLabel: "Empathetic",
  },
};

function DispositionEditor({
  disposition,
  editMode,
  editDisposition,
  onEditChange,
}: {
  disposition: DispositionTraits;
  editMode: boolean;
  editDisposition: DispositionTraits;
  onEditChange: (trait: keyof DispositionTraits, value: number) => void;
}) {
  const data = editMode ? editDisposition : disposition;

  return (
    <div className="space-y-4">
      {(Object.keys(TRAIT_LABELS) as Array<keyof DispositionTraits>).map((trait) => (
        <div key={trait} className="space-y-2">
          <div className="flex justify-between items-center">
            <div>
              <label className="text-sm font-medium text-foreground">
                {TRAIT_LABELS[trait].label}
              </label>
              <p className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].description}</p>
            </div>
            <span className="text-sm font-bold text-primary">{data[trait]}/5</span>
          </div>
          {editMode ? (
            <>
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>{TRAIT_LABELS[trait].lowLabel}</span>
                <span>{TRAIT_LABELS[trait].highLabel}</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={editDisposition[trait]}
                onChange={(e) => onEditChange(trait, parseInt(e.target.value))}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].lowLabel}</span>
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all"
                  style={{ width: `${((data[trait] - 1) / 4) * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].highLabel}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function BankProfileView() {
  const router = useRouter();
  const { currentBank, setCurrentBank, loadBanks } = useBank();
  const { features } = useFeatures();
  const observationsEnabled = features?.observations ?? false;
  const [profile, setProfile] = useState<BankProfile | null>(null);
  const [stats, setStats] = useState<BankStats | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [totalOperations, setTotalOperations] = useState(0);
  const [directives, setDirectives] = useState<Directive[]>([]);
  const [mentalModelsCount, setMentalModelsCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editMode, setEditMode] = useState(false);

  // Directive state
  const [showCreateDirective, setShowCreateDirective] = useState(false);
  const [selectedDirective, setSelectedDirective] = useState<Directive | null>(null);
  const [directiveDeleteTarget, setDirectiveDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deletingDirective, setDeletingDirective] = useState(false);

  // Ref to track editMode for polling (avoids stale closure)
  const editModeRef = useRef(editMode);
  useEffect(() => {
    editModeRef.current = editMode;
  }, [editMode]);

  // Delete state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Clear observations state
  const [showClearObservationsDialog, setShowClearObservationsDialog] = useState(false);
  const [isClearingObservations, setIsClearingObservations] = useState(false);

  // Consolidation state
  const [isConsolidating, setIsConsolidating] = useState(false);

  // Operations filter/pagination state
  const [opsStatusFilter, setOpsStatusFilter] = useState<string | null>(null);
  const [opsLimit] = useState(10);
  const [opsOffset, setOpsOffset] = useState(0);
  const [cancellingOpId, setCancellingOpId] = useState<string | null>(null);

  // Edit state
  const [editMission, setEditMission] = useState("");
  const [editDisposition, setEditDisposition] = useState<DispositionTraits>({
    skepticism: 3,
    literalism: 3,
    empathy: 3,
  });

  const loadOperations = async (
    statusFilter: string | null = opsStatusFilter,
    offset: number = opsOffset
  ) => {
    if (!currentBank) return;
    try {
      const opsData = await client.listOperations(currentBank, {
        status: statusFilter || undefined,
        limit: opsLimit,
        offset,
      });
      setOperations(opsData.operations || []);
      setTotalOperations(opsData.total || 0);
    } catch (error) {
      console.error("Error loading operations:", error);
    }
  };

  const loadData = async (isPolling = false) => {
    if (!currentBank) return;

    // During polling, only refresh stats (not operations to avoid interfering with filters)
    // Use ref to get current value (avoids stale closure in setInterval)
    if (isPolling) {
      try {
        const [statsData, directivesData, mentalModelsData] = await Promise.all([
          client.getBankStats(currentBank),
          client.listDirectives(currentBank),
          client.listMentalModels(currentBank),
        ]);
        setStats(statsData as BankStats);
        setDirectives(directivesData.items || []);
        setMentalModelsCount(mentalModelsData.items?.length || 0);
        // Skip operations refresh during polling to not interfere with filter/pagination state
      } catch (error) {
        console.error("Error refreshing stats:", error);
      }
      return;
    }

    setLoading(true);
    try {
      const [profileData, statsData, directivesData, mentalModelsData] = await Promise.all([
        client.getBankProfile(currentBank),
        client.getBankStats(currentBank),
        client.listDirectives(currentBank),
        client.listMentalModels(currentBank),
      ]);
      setProfile(profileData);
      setStats(statsData as BankStats);
      setDirectives(directivesData.items || []);
      setMentalModelsCount(mentalModelsData.items?.length || 0);
      await loadOperations();

      // Only initialize edit state when not in edit mode
      if (!editModeRef.current) {
        setEditMission(profileData.mission || "");
        setEditDisposition(profileData.disposition);
      }
    } catch (error) {
      console.error("Error loading bank profile:", error);
      alert("Error loading bank profile: " + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!currentBank) return;

    setSaving(true);
    try {
      await client.updateBankProfile(currentBank, {
        mission: editMission,
        disposition: editDisposition,
      });
      await loadData();
      setEditMode(false);
    } catch (error) {
      console.error("Error saving bank profile:", error);
      alert("Error saving bank profile: " + (error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (profile) {
      setEditMission(profile.mission || "");
      setEditDisposition(profile.disposition);
    }
    setEditMode(false);
  };

  const handleDeleteBank = async () => {
    if (!currentBank) return;

    setIsDeleting(true);
    try {
      await client.deleteBank(currentBank);
      setShowDeleteDialog(false);
      setCurrentBank(null);
      await loadBanks();
      router.push("/");
    } catch (error) {
      console.error("Error deleting bank:", error);
      alert("Error deleting bank: " + (error as Error).message);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClearObservations = async () => {
    if (!currentBank) return;

    setIsClearingObservations(true);
    try {
      const result = await client.clearObservations(currentBank);
      setShowClearObservationsDialog(false);
      await loadData();
      alert(result.message || "Observations cleared successfully");
    } catch (error) {
      console.error("Error clearing observations:", error);
      alert("Error clearing observations: " + (error as Error).message);
    } finally {
      setIsClearingObservations(false);
    }
  };

  const handleTriggerConsolidation = async () => {
    if (!currentBank) return;

    setIsConsolidating(true);
    try {
      await client.triggerConsolidation(currentBank);
      // Reload to show the new operation in the list
      await loadData();
      await loadOperations();
    } catch (error) {
      console.error("Error triggering consolidation:", error);
      alert("Error triggering consolidation: " + (error as Error).message);
    } finally {
      setIsConsolidating(false);
    }
  };

  const handleOpsFilterChange = (newFilter: string | null) => {
    setOpsStatusFilter(newFilter);
    setOpsOffset(0); // Reset to first page when filter changes
    loadOperations(newFilter, 0);
  };

  const handleOpsPageChange = (newOffset: number) => {
    setOpsOffset(newOffset);
    loadOperations(opsStatusFilter, newOffset);
  };

  const handleCancelOperation = async (operationId: string) => {
    if (!currentBank) return;

    setCancellingOpId(operationId);
    try {
      await client.cancelOperation(currentBank, operationId);
      await loadOperations();
    } catch (error) {
      console.error("Error cancelling operation:", error);
      alert("Error cancelling operation: " + (error as Error).message);
    } finally {
      setCancellingOpId(null);
    }
  };

  const handleDeleteDirective = async () => {
    if (!currentBank || !directiveDeleteTarget) return;

    setDeletingDirective(true);
    try {
      await client.deleteDirective(currentBank, directiveDeleteTarget.id);
      setDirectives((prev) => prev.filter((d) => d.id !== directiveDeleteTarget.id));
      if (selectedDirective?.id === directiveDeleteTarget.id) setSelectedDirective(null);
      setDirectiveDeleteTarget(null);
    } catch (error) {
      console.error("Error deleting directive:", error);
      alert("Error deleting: " + (error as Error).message);
    } finally {
      setDeletingDirective(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadData();
      // Refresh stats/operations every 5 seconds (isPolling=true to avoid overwriting form)
      const interval = setInterval(() => loadData(true), 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank]);

  // Close directive detail panel on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedDirective(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  if (!currentBank) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <h3 className="text-xl font-semibold mb-2 text-card-foreground">No Bank Selected</h3>
          <p className="text-muted-foreground">
            Please select a memory bank from the dropdown above to view its profile.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (loading && !profile) {
    return (
      <Card>
        <CardContent className="text-center py-10">
          <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
          <div className="text-lg text-muted-foreground">Loading profile...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with actions */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{profile?.name || currentBank}</h2>
          <p className="text-sm text-muted-foreground font-mono">{currentBank}</p>
        </div>
        <div className="flex gap-2">
          {editMode ? (
            <>
              <Button onClick={handleCancel} variant="secondary" disabled={saving}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                {saving ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
            </>
          ) : (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  Actions
                  <MoreVertical className="w-4 h-4 ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onClick={() => setEditMode(true)}>
                  <Pencil className="w-4 h-4 mr-2" />
                  Edit Profile
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={handleTriggerConsolidation}
                  disabled={isConsolidating || !observationsEnabled}
                  title={!observationsEnabled ? "Observations feature is not enabled" : undefined}
                >
                  {isConsolidating ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Brain className="w-4 h-4 mr-2" />
                  )}
                  {isConsolidating ? "Consolidating..." : "Run Consolidation"}
                  {!observationsEnabled && (
                    <span className="ml-auto text-xs text-muted-foreground">Off</span>
                  )}
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setShowClearObservationsDialog(true)}
                  disabled={!observationsEnabled}
                  className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
                  title={!observationsEnabled ? "Observations feature is not enabled" : undefined}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Observations
                  {!observationsEnabled && (
                    <span className="ml-auto text-xs text-muted-foreground">Off</span>
                  )}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => setShowDeleteDialog(true)}
                  className="text-red-600 dark:text-red-400 focus:text-red-700 dark:focus:text-red-300"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Bank
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      {/* Stats Overview - Compact cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-500/20">
                  <Database className="w-5 h-5 text-blue-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Memories</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_nodes}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/20">
                  <Link2 className="w-5 h-5 text-purple-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Links</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_links}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/20">
                  <FolderOpen className="w-5 h-5 text-emerald-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Documents</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_documents}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card
            className={`bg-gradient-to-br ${stats.pending_operations > 0 ? "from-amber-500/10 to-amber-600/5 border-amber-500/20" : "from-slate-500/10 to-slate-600/5 border-slate-500/20"}`}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div
                  className={`p-2 rounded-lg ${stats.pending_operations > 0 ? "bg-amber-500/20" : "bg-slate-500/20"}`}
                >
                  <Activity
                    className={`w-5 h-5 ${stats.pending_operations > 0 ? "text-amber-500 animate-pulse" : "text-slate-500"}`}
                  />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Pending</p>
                  <p className="text-2xl font-bold text-foreground">{stats.pending_operations}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Memory Type Breakdown */}
      {stats && (
        <div className="grid grid-cols-5 gap-3">
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-blue-600 dark:text-blue-400 font-semibold uppercase tracking-wide">
              World Facts
            </p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">
              {stats.nodes_by_fact_type?.world || 0}
            </p>
          </div>
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-purple-600 dark:text-purple-400 font-semibold uppercase tracking-wide">
              Experience
            </p>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">
              {stats.nodes_by_fact_type?.experience || 0}
            </p>
          </div>
          <div
            className={`rounded-xl p-4 text-center ${
              observationsEnabled
                ? "bg-amber-500/10 border border-amber-500/20"
                : "bg-muted/50 border border-muted"
            }`}
            title={!observationsEnabled ? "Observations feature is not enabled" : undefined}
          >
            <p
              className={`text-xs font-semibold uppercase tracking-wide ${
                observationsEnabled ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"
              }`}
            >
              Observations
              {!observationsEnabled && <span className="ml-1 normal-case">(Off)</span>}
            </p>
            <p
              className={`text-2xl font-bold mt-1 ${
                observationsEnabled ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"
              }`}
            >
              {observationsEnabled ? stats.total_mental_models || 0 : "—"}
            </p>
          </div>
          <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-cyan-600 dark:text-cyan-400 font-semibold uppercase tracking-wide">
              Mental Models
            </p>
            <p className="text-2xl font-bold text-cyan-600 dark:text-cyan-400 mt-1">
              {mentalModelsCount}
            </p>
          </div>
          <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-rose-600 dark:text-rose-400 font-semibold uppercase tracking-wide">
              Directives
            </p>
            <p className="text-2xl font-bold text-rose-600 dark:text-rose-400 mt-1">
              {directives.length}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Disposition Chart */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Brain className="w-5 h-5 text-primary" />
              Disposition Profile
            </CardTitle>
            <CardDescription>
              Traits that shape how observations are formed via Reflect
            </CardDescription>
          </CardHeader>
          <CardContent>
            {profile && (
              <DispositionEditor
                disposition={profile.disposition}
                editMode={editMode}
                editDisposition={editDisposition}
                onEditChange={(trait, value) =>
                  setEditDisposition((prev) => ({ ...prev, [trait]: value }))
                }
              />
            )}
          </CardContent>
        </Card>

        {/* Mission */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="w-5 h-5 text-primary" />
              Mission
            </CardTitle>
            <CardDescription>
              Who the agent is and what they&apos;re trying to accomplish. Used for mental models
              and reflect.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {editMode ? (
              <Textarea
                value={editMission}
                onChange={(e) => setEditMission(e.target.value)}
                placeholder="e.g., I am a PM for the engineering team. I help coordinate sprints and track project progress..."
                rows={6}
                className="resize-none"
              />
            ) : (
              <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                {profile?.mission ||
                  "No mission set. Set a mission to derive structural mental models and personalize reflect responses."}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Directives Section */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <AlertTriangle className="w-5 h-5 text-rose-500" />
                Directives
              </CardTitle>
              <CardDescription>Hard rules that must be followed during reflect</CardDescription>
            </div>
            <Button
              onClick={() => setShowCreateDirective(true)}
              variant="outline"
              size="sm"
              className="h-8"
            >
              <Plus className="w-4 h-4 mr-1" />
              Add
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {directives.length > 0 ? (
            <div className="grid grid-cols-2 gap-3">
              {directives.map((d) => (
                <Card
                  key={d.id}
                  className={`cursor-pointer transition-colors border-rose-500/30 ${
                    selectedDirective?.id === d.id
                      ? "bg-rose-500/10 border-rose-500"
                      : "hover:bg-rose-500/5"
                  }`}
                  onClick={() => setSelectedDirective(d)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <span className="font-medium text-sm text-foreground">{d.name}</span>
                        <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                          {d.content}
                        </p>
                        {d.tags && d.tags.length > 0 && (
                          <div className="flex items-center gap-1 mt-2 flex-wrap">
                            <Tag className="w-3 h-3 text-muted-foreground" />
                            {d.tags.map((tag) => (
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
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="p-6 border border-dashed border-rose-500/30 rounded-lg text-center">
              <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-rose-500/50" />
              <p className="text-sm text-muted-foreground">
                No directives yet. Directives are hard rules that must be followed during reflect.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Operations Section */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Activity className="w-5 h-5 text-primary" />
                Background Operations
                <button
                  onClick={() => loadOperations()}
                  className="p-1 rounded hover:bg-muted transition-colors"
                  title="Refresh operations"
                >
                  <RefreshCw className="w-4 h-4 text-muted-foreground hover:text-foreground" />
                </button>
              </CardTitle>
              <CardDescription>
                {totalOperations} operation{totalOperations !== 1 ? "s" : ""}
                {opsStatusFilter ? ` (${opsStatusFilter})` : ""}
              </CardDescription>
            </div>
            <div className="flex gap-1 bg-muted p-1 rounded-lg">
              {[
                { value: null, label: "All" },
                { value: "pending", label: "Pending" },
                { value: "completed", label: "Completed" },
                { value: "failed", label: "Failed" },
              ].map((filter) => (
                <button
                  key={filter.value ?? "all"}
                  onClick={() => handleOpsFilterChange(filter.value)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    opsStatusFilter === filter.value
                      ? "bg-background shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {filter.label}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {operations.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[100px]">ID</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="w-[80px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {operations.map((op) => (
                      <TableRow
                        key={op.id}
                        className={op.status === "failed" ? "bg-red-500/5" : ""}
                      >
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {op.id.substring(0, 8)}
                        </TableCell>
                        <TableCell className="font-medium">{op.task_type}</TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {new Date(op.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          {op.status === "pending" && (
                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                              <Clock className="w-3 h-3" />
                              pending
                            </span>
                          )}
                          {op.status === "failed" && (
                            <span
                              className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20"
                              title={op.error_message ?? undefined}
                            >
                              <AlertCircle className="w-3 h-3" />
                              failed
                            </span>
                          )}
                          {op.status === "completed" && (
                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                              <CheckCircle className="w-3 h-3" />
                              completed
                            </span>
                          )}
                        </TableCell>
                        <TableCell>
                          {op.status === "pending" && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                              onClick={() => handleCancelOperation(op.id)}
                              disabled={cancellingOpId === op.id}
                            >
                              {cancellingOpId === op.id ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                <X className="w-3 h-3 mr-1" />
                              )}
                              {cancellingOpId === op.id ? "" : "Cancel"}
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              {/* Pagination */}
              {totalOperations > opsLimit && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Showing {opsOffset + 1}-{Math.min(opsOffset + opsLimit, totalOperations)} of{" "}
                    {totalOperations}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleOpsPageChange(Math.max(0, opsOffset - opsLimit))}
                      disabled={opsOffset === 0}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleOpsPageChange(opsOffset + opsLimit)}
                      disabled={opsOffset + opsLimit >= totalOperations}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <p className="text-muted-foreground text-center py-8 text-sm">
              No {opsStatusFilter ? `${opsStatusFilter} ` : ""}operations
            </p>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Memory Bank</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  Are you sure you want to delete the memory bank{" "}
                  <span className="font-semibold text-foreground">{currentBank}</span>?
                </p>
                <p className="text-red-600 dark:text-red-400 font-medium">
                  This action cannot be undone. All memories, entities, documents, and the bank
                  profile will be permanently deleted.
                </p>
                {stats && (
                  <p>
                    This will delete {stats.total_nodes} memories, {stats.total_documents}{" "}
                    documents, and {stats.total_links} links.
                  </p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteBank}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Bank
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Clear Observations Confirmation Dialog */}
      <AlertDialog open={showClearObservationsDialog} onOpenChange={setShowClearObservationsDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear Observations</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  Are you sure you want to clear all observations for{" "}
                  <span className="font-semibold text-foreground">{currentBank}</span>?
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  This will delete all consolidated knowledge. Observations will be regenerated the
                  next time consolidation runs.
                </p>
                {stats && stats.total_mental_models > 0 && (
                  <p>This will delete {stats.total_mental_models} observations.</p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isClearingObservations}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClearObservations}
              disabled={isClearingObservations}
              className="bg-amber-500 text-white hover:bg-amber-600"
            >
              {isClearingObservations ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Clearing...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Observations
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Create Directive Dialog */}
      <DirectiveFormDialog
        open={showCreateDirective}
        mode="create"
        onClose={() => setShowCreateDirective(false)}
        onCreated={(d) => {
          setDirectives((prev) => [d, ...prev]);
          setShowCreateDirective(false);
        }}
      />

      {/* Delete Directive Confirmation Dialog */}
      <AlertDialog
        open={!!directiveDeleteTarget}
        onOpenChange={(open) => !open && setDirectiveDeleteTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Directive</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete{" "}
              <span className="font-semibold">&quot;{directiveDeleteTarget?.name}&quot;</span>?
              <br />
              <br />
              <span className="text-destructive font-semibold">This action cannot be undone.</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteDirective}
              disabled={deletingDirective}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deletingDirective ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Directive Detail Panel */}
      {selectedDirective && (
        <DirectiveDetailPanel
          directive={selectedDirective}
          onClose={() => setSelectedDirective(null)}
          onDelete={() =>
            setDirectiveDeleteTarget({
              id: selectedDirective.id,
              name: selectedDirective.name,
            })
          }
          onUpdated={(updated) => {
            setDirectives((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
            setSelectedDirective(updated);
          }}
        />
      )}
    </div>
  );
}

// ============= DIRECTIVE FORM DIALOG (CREATE/EDIT) =============

function DirectiveFormDialog({
  open,
  mode,
  directive,
  onClose,
  onCreated,
  onSaved,
}: {
  open: boolean;
  mode: "create" | "edit";
  directive?: Directive;
  onClose: () => void;
  onCreated?: (d: Directive) => void;
  onSaved?: (d: Directive) => void;
}) {
  const { currentBank } = useBank();
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({ name: "", content: "", tags: "" });

  // Reset form when dialog opens or directive changes
  useEffect(() => {
    if (mode === "edit" && directive) {
      setForm({
        name: directive.name,
        content: directive.content,
        tags: (directive.tags || []).join(", "),
      });
    } else if (mode === "create") {
      setForm({ name: "", content: "", tags: "" });
    }
  }, [open, mode, directive]);

  const handleSubmit = async () => {
    if (!currentBank || !form.name.trim() || !form.content.trim()) return;

    setSubmitting(true);
    try {
      const tags = form.tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      if (mode === "create") {
        const result = await client.createDirective(currentBank, {
          name: form.name.trim(),
          content: form.content.trim(),
          tags: tags.length > 0 ? tags : undefined,
        });
        setForm({ name: "", content: "", tags: "" });
        onCreated?.(result);
      } else if (directive) {
        const result = await client.updateDirective(currentBank, directive.id, {
          name: form.name.trim(),
          content: form.content.trim(),
          tags: tags,
        });
        onSaved?.(result);
        onClose();
      }
    } catch (error) {
      console.error(`Error ${mode === "create" ? "creating" : "updating"} directive:`, error);
      alert(`Error ${mode === "create" ? "creating" : "updating"}: ` + (error as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    if (mode === "create") {
      setForm({ name: "", content: "", tags: "" });
    }
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-rose-500" />
            {mode === "create" ? "Create" : "Edit"} Directive
          </DialogTitle>
          <DialogDescription>
            Directives are hard rules that must be followed during reflect.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Name *</label>
            <Input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g., Competitor Policy"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Rule *</label>
            <Textarea
              value={form.content}
              onChange={(e) => setForm({ ...form, content: e.target.value })}
              placeholder="e.g., Never mention competitor products directly."
              className="min-h-[120px]"
            />
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
          <Button variant="outline" onClick={handleClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={submitting || !form.name.trim() || !form.content.trim()}
            className="bg-rose-500 hover:bg-rose-600"
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
            {mode === "create" ? "Create" : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============= DIRECTIVE DETAIL PANEL =============

function DirectiveDetailPanel({
  directive,
  onClose,
  onDelete,
  onUpdated,
}: {
  directive: Directive;
  onClose: () => void;
  onDelete: () => void;
  onUpdated: (d: Directive) => void;
}) {
  const [showEditModal, setShowEditModal] = useState(false);

  return (
    <div className="fixed right-0 top-0 h-screen w-1/2 bg-card border-l-2 border-rose-500 shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
      <div className="p-6">
        {/* Header */}
        <div className="flex justify-between items-start mb-8 pb-5 border-b border-border">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-500" />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-foreground">{directive.name}</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowEditModal(true)}
                  className="h-7 w-7 p-0"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </Button>
              </div>
              <span className="text-xs px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-600 dark:text-rose-400">
                directive
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              className="h-8 w-8 p-0 text-muted-foreground hover:text-rose-500"
            >
              <Trash2 className="h-4 w-4" />
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
              Rule
            </div>
            <div className="prose prose-base dark:prose-invert max-w-none">
              <ReactMarkdown>{directive.content}</ReactMarkdown>
            </div>
          </div>

          {/* Tags */}
          {directive.tags && directive.tags.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                Tags
              </div>
              <div className="flex flex-wrap gap-2">
                {directive.tags.map((tag) => (
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

          {/* ID */}
          <div>
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              ID
            </div>
            <code className="text-sm font-mono break-all text-muted-foreground">
              {directive.id}
            </code>
          </div>
        </div>
      </div>

      {/* Edit Modal */}
      <DirectiveFormDialog
        open={showEditModal}
        mode="edit"
        directive={directive}
        onClose={() => setShowEditModal(false)}
        onSaved={onUpdated}
      />
    </div>
  );
}
