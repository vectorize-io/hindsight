"use client";

import { useState, useEffect, useRef } from "react";
import { useTranslations } from "next-intl";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRouter } from "next/navigation";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
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
  total_observations: number;
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

const TRAIT_KEYS: Array<keyof DispositionTraits> = ["skepticism", "literalism", "empathy"];

const TRAIT_SHORT_LABELS: Record<keyof DispositionTraits, string> = {
  skepticism: "S",
  literalism: "L",
  empathy: "E",
};

function useTraitLabels() {
  const t = useTranslations("bankProfile");
  return {
    skepticism: {
      label: t("skepticismLabel"),
      shortLabel: TRAIT_SHORT_LABELS.skepticism,
      description: t("skepticismDescription"),
      lowLabel: t("skepticismLow"),
      highLabel: t("skepticismHigh"),
    },
    literalism: {
      label: t("literalismLabel"),
      shortLabel: TRAIT_SHORT_LABELS.literalism,
      description: t("literalismDescription"),
      lowLabel: t("literalismLow"),
      highLabel: t("literalismHigh"),
    },
    empathy: {
      label: t("empathyLabel"),
      shortLabel: TRAIT_SHORT_LABELS.empathy,
      description: t("empathyDescription"),
      lowLabel: t("empathyLow"),
      highLabel: t("empathyHigh"),
    },
  } as const;
}

export function BankProfileView({ hideReflectFields = false }: { hideReflectFields?: boolean }) {
  const router = useRouter();
  const t = useTranslations("bankProfile");
  const tCommon = useTranslations("common");
  const tBank = useTranslations("bank");
  const traitLabels = useTraitLabels();
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
  const [showDispositionDialog, setShowDispositionDialog] = useState(false);
  const [showMissionDialog, setShowMissionDialog] = useState(false);

  // Directive state
  const [showCreateDirective, setShowCreateDirective] = useState(false);
  const [selectedDirective, setSelectedDirective] = useState<Directive | null>(null);
  const [directiveDeleteTarget, setDirectiveDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deletingDirective, setDeletingDirective] = useState(false);

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
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoading(false);
    }
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
      // Error toast is shown automatically by the API client interceptor
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
      toast.success("Success", {
        description: result.message || "Observations cleared successfully",
      });
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
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
      // Error toast is shown automatically by the API client interceptor
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
      // Error toast is shown automatically by the API client interceptor
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
      // Error toast is shown automatically by the API client interceptor
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
          <h3 className="text-xl font-semibold mb-2 text-card-foreground">{t("noBankSelected")}</h3>
          <p className="text-muted-foreground">{t("noBankSelectedDescription")}</p>
        </CardContent>
      </Card>
    );
  }

  if (loading && !profile) {
    return (
      <Card>
        <CardContent className="text-center py-10">
          <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
          <div className="text-lg text-muted-foreground">{t("loadingProfile")}</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {!hideReflectFields && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Disposition Chart */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Brain className="w-5 h-5 text-primary" />
                    {t("dispositionProfileTitle")}
                  </CardTitle>
                  <CardDescription>{t("editDispositionDescription")}</CardDescription>
                </div>
                <Button onClick={() => setShowDispositionDialog(true)} variant="ghost" size="sm">
                  <Pencil className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {profile && (
                <div className="space-y-4">
                  {TRAIT_KEYS.map((trait) => (
                    <div key={trait} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div>
                          <label className="text-sm font-medium text-foreground">
                            {traitLabels[trait].label}
                          </label>
                          <p className="text-xs text-muted-foreground">
                            {traitLabels[trait].description}
                          </p>
                        </div>
                        <span className="text-sm font-bold text-primary">
                          {profile.disposition[trait]}/5
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {traitLabels[trait].lowLabel}
                        </span>
                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full transition-all"
                            style={{ width: `${((profile.disposition[trait] - 1) / 4) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {traitLabels[trait].highLabel}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Mission */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Target className="w-5 h-5 text-primary" />
                    {t("missionTitle")}
                  </CardTitle>
                  <CardDescription>{t("editMissionDescription")}</CardDescription>
                </div>
                <Button onClick={() => setShowMissionDialog(true)} variant="ghost" size="sm">
                  <Pencil className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                {profile?.mission || t("noMissionSet")}
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Directives Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-rose-500" />
              {t("directivesTitle")}
            </h3>
            <p className="text-sm text-muted-foreground">{t("directivesDescription")}</p>
          </div>
          <Button
            onClick={() => setShowCreateDirective(true)}
            variant="outline"
            size="sm"
            className="h-8"
          >
            <Plus className="w-4 h-4 mr-1" />
            {t("addDirective")}
          </Button>
        </div>
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
                      <p className="text-xs text-muted-foreground line-clamp-2 mt-1">{d.content}</p>
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
            <p className="text-sm text-muted-foreground">{t("noDirectivesMessage")}</p>
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{tBank("deleteMemoryBank")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {tBank.rich("deleteBankPrompt", {
                    bankName: () => (
                      <span className="font-semibold text-foreground">{currentBank}</span>
                    ),
                  })}
                </p>
                <p className="text-red-600 dark:text-red-400 font-medium">
                  {tBank("deleteBankWarning")}
                </p>
                {stats && (
                  <p>
                    {tBank("deleteWillDeleteDetails", {
                      memories: stats.total_nodes,
                      documents: stats.total_documents,
                      links: stats.total_links,
                    })}
                  </p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteBank}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  {tBank("deleting")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {tBank("deleteBank")}
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
            <AlertDialogTitle>{tBank("clearObservationsTitle")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {tBank.rich("clearObservationsPrompt", {
                    bankName: () => (
                      <span className="font-semibold text-foreground">{currentBank}</span>
                    ),
                  })}
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  {tBank("clearObservationsWarning")}
                </p>
                {stats && stats.total_observations > 0 && (
                  <p>
                    {tBank("deleteWillDeleteObservations", { count: stats.total_observations })}
                  </p>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isClearingObservations}>
              {tCommon("cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClearObservations}
              disabled={isClearingObservations}
              className="bg-amber-500 text-white hover:bg-amber-600"
            >
              {isClearingObservations ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {tBank("clearing")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {tBank("clearObservations")}
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
            <AlertDialogTitle>{t("deleteDirectiveTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteDirectiveConfirm", { name: `"${directiveDeleteTarget?.name ?? ""}"` })}
              <br />
              <br />
              <span className="text-destructive font-semibold">{t("deleteDirectiveWarning")}</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex-row justify-end space-x-2">
            <AlertDialogCancel className="mt-0">{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteDirective}
              disabled={deletingDirective}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deletingDirective ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
              {tCommon("delete")}
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

      {/* Disposition Edit Dialog */}
      {showDispositionDialog && profile && (
        <DispositionEditDialog
          disposition={profile.disposition}
          onClose={() => setShowDispositionDialog(false)}
          onSaved={async () => {
            await loadData();
            setShowDispositionDialog(false);
          }}
        />
      )}

      {/* Mission Edit Dialog */}
      {showMissionDialog && profile && (
        <MissionEditDialog
          mission={profile.mission || ""}
          onClose={() => setShowMissionDialog(false)}
          onSaved={async () => {
            await loadData();
            setShowMissionDialog(false);
          }}
        />
      )}
    </div>
  );
}

// ============= DISPOSITION EDIT DIALOG =============

function DispositionEditDialog({
  disposition,
  onClose,
  onSaved,
}: {
  disposition: DispositionTraits;
  onClose: () => void;
  onSaved: () => void;
}) {
  const t = useTranslations("bankProfile");
  const traitLabels = useTraitLabels();
  const { currentBank } = useBank();
  const [saving, setSaving] = useState(false);
  const [editDisposition, setEditDisposition] = useState<DispositionTraits>(disposition);

  const handleSave = async () => {
    if (!currentBank) return;

    setSaving(true);
    try {
      await client.updateBankConfig(currentBank, {
        disposition_skepticism: editDisposition.skepticism,
        disposition_literalism: editDisposition.literalism,
        disposition_empathy: editDisposition.empathy,
      });
      onSaved();
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("editDispositionTitle")}</DialogTitle>
          <DialogDescription>{t("editDispositionDescription")}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {TRAIT_KEYS.map((trait) => (
            <div key={trait} className="space-y-2">
              <div className="flex justify-between items-center">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    {traitLabels[trait].label}
                  </label>
                  <p className="text-xs text-muted-foreground">{traitLabels[trait].description}</p>
                </div>
                <span className="text-sm font-bold text-primary">{editDisposition[trait]}/5</span>
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>{traitLabels[trait].lowLabel}</span>
                <span>{traitLabels[trait].highLabel}</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={editDisposition[trait]}
                onChange={(e) =>
                  setEditDisposition((prev) => ({ ...prev, [trait]: parseInt(e.target.value) }))
                }
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>
          ))}
        </div>

        <DialogFooter>
          <Button onClick={onClose} variant="outline" disabled={saving}>
            {t("cancel")}
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                {t("saving")}
              </>
            ) : (
              t("saveChanges")
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============= MISSION EDIT DIALOG =============

function MissionEditDialog({
  mission,
  onClose,
  onSaved,
}: {
  mission: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const t = useTranslations("bankProfile");
  const { currentBank } = useBank();
  const [saving, setSaving] = useState(false);
  const [editMission, setEditMission] = useState(mission);

  const handleSave = async () => {
    if (!currentBank) return;

    setSaving(true);
    try {
      await client.updateBankConfig(currentBank, {
        reflect_mission: editMission || null,
      });
      onSaved();
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("editMissionTitle")}</DialogTitle>
          <DialogDescription>{t("editMissionDescription")}</DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-4">
          <Textarea
            value={editMission}
            onChange={(e) => setEditMission(e.target.value)}
            placeholder={t("missionPlaceholder")}
            rows={8}
            className="resize-none"
          />
        </div>

        <DialogFooter>
          <Button onClick={onClose} variant="outline" disabled={saving}>
            {t("cancel")}
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                {t("saving")}
              </>
            ) : (
              t("saveChanges")
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
  const t = useTranslations("bankProfile");
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
      // Error toast is shown automatically by the API client interceptor
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
            {mode === "create" ? t("directiveFormCreateTitle") : t("directiveFormEditTitle")}
          </DialogTitle>
          <DialogDescription>{t("directiveFormDescription")}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">{t("directiveNameLabel")}</label>
            <Input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder={t("directiveNamePlaceholder")}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">{t("directiveRuleLabel")}</label>
            <Textarea
              value={form.content}
              onChange={(e) => setForm({ ...form, content: e.target.value })}
              placeholder={t("directiveRulePlaceholder")}
              className="min-h-[120px]"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              {t("directiveTagsLabel")}{" "}
              <span className="text-muted-foreground font-normal">
                {t("directiveTagsOptional")}
              </span>
            </label>
            <Input
              value={form.tags}
              onChange={(e) => setForm({ ...form, tags: e.target.value })}
              placeholder={t("directiveTagsPlaceholder")}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={submitting}>
            {t("cancel")}
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={submitting || !form.name.trim() || !form.content.trim()}
            className="bg-rose-500 hover:bg-rose-600"
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
            {mode === "create" ? t("create") : t("save")}
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
  const t = useTranslations("bankProfile");
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
                {t("directiveDetailBadge")}
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
              {t("directiveDetailRuleLabel")}
            </div>
            <div className="prose prose-base dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{directive.content}</ReactMarkdown>
            </div>
          </div>

          {/* Tags */}
          {directive.tags && directive.tags.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                {t("directiveDetailTagsLabel")}
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
              {t("directiveDetailIdLabel")}
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
