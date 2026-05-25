"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";
import { BankSelector } from "@/components/bank-selector";
import { Sidebar } from "@/components/sidebar";
import { DataView } from "@/components/data-view";
import { DocumentsView } from "@/components/documents-view";
import { EntitiesView } from "@/components/entities-view";
import { ThinkView } from "@/components/think-view";
import { SearchDebugView } from "@/components/search-debug-view";
import { BankProfileView } from "@/components/bank-profile-view";
import { BankConfigView } from "@/components/bank-config-view";
import { BankStatsView } from "@/components/bank-stats-view";
import { BankOperationsView } from "@/components/bank-operations-view";
import { MentalModelsView } from "@/components/mental-models-view";
import { WebhooksView } from "@/components/webhooks-view";
import { AuditLogsView } from "@/components/audit-logs-view";
import { useFeatures } from "@/lib/features-context";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import { Brain, Download, Trash2, Loader2, MoreVertical, RotateCcw } from "lucide-react";

type NavItem = "recall" | "reflect" | "data" | "documents" | "entities" | "profile";
type DataSubTab = "world" | "experience" | "observations" | "mental-models";
type BankConfigTab = "general" | "configuration" | "webhooks" | "audit-logs";

export default function BankPage() {
  const { t } = useTranslation();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { features } = useFeatures();
  const { currentBank: bankId, setCurrentBank, loadBanks } = useBank();

  const view = (searchParams.get("view") || "profile") as NavItem;
  const subTab = (searchParams.get("subTab") || "world") as DataSubTab;
  const bankConfigTab = (searchParams.get("bankConfigTab") || "general") as BankConfigTab;
  const observationsEnabled = features?.observations ?? false;
  const bankConfigEnabled = features?.bank_config_api ?? false;

  // Bank actions state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showClearObservationsDialog, setShowClearObservationsDialog] = useState(false);
  const [isClearingObservations, setIsClearingObservations] = useState(false);
  const [isConsolidating, setIsConsolidating] = useState(false);
  const [isRecoveringConsolidation, setIsRecoveringConsolidation] = useState(false);
  const [showResetConfigDialog, setShowResetConfigDialog] = useState(false);
  const [isResettingConfig, setIsResettingConfig] = useState(false);

  const handleTabChange = (tab: NavItem) => {
    if (!bankId) return;
    router.push(bankRoute(bankId, `?view=${tab}`));
  };

  const handleDataSubTabChange = (newSubTab: DataSubTab) => {
    if (!bankId) return;
    router.push(bankRoute(bankId, `?view=data&subTab=${newSubTab}`));
  };

  const handleBankConfigTabChange = (newTab: BankConfigTab) => {
    if (!bankId) return;
    router.push(bankRoute(bankId, `?view=profile&bankConfigTab=${newTab}`));
  };

  const handleDeleteBank = async () => {
    if (!bankId) return;

    setIsDeleting(true);
    try {
      await client.deleteBank(bankId);
      setShowDeleteDialog(false);
      setCurrentBank(null);
      await loadBanks();
      router.push("/");
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClearObservations = async () => {
    if (!bankId) return;

    setIsClearingObservations(true);
    try {
      const result = await client.clearObservations(bankId);
      setShowClearObservationsDialog(false);
      toast.success(t("common.success"), {
        description: result.message || t("dialogs.clearObservations.success"),
      });
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsClearingObservations(false);
    }
  };

  const handleResetConfig = async () => {
    if (!bankId) return;
    setIsResettingConfig(true);
    try {
      await client.resetBankConfig(bankId);
      setShowResetConfigDialog(false);
    } catch {
      // Error toast shown by API client interceptor
    } finally {
      setIsResettingConfig(false);
    }
  };

  const handleTriggerConsolidation = async () => {
    if (!bankId) return;

    setIsConsolidating(true);
    try {
      await client.triggerConsolidation(bankId);
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsConsolidating(false);
    }
  };

  const handleRecoverConsolidation = async () => {
    if (!bankId) return;

    setIsRecoveringConsolidation(true);
    try {
      const result = await client.recoverConsolidation(bankId);
      toast.success(t("bankActions.recoveredFailedMemory", { count: result.retried_count }));
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsRecoveringConsolidation(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <BankSelector />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar currentTab={view} onTabChange={handleTabChange} />

        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {/* Bank Configuration Tab */}
            {view === "profile" && (
              <div>
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h1 className="text-3xl font-bold mb-2 text-foreground">
                      {t("pages.bankConfiguration.title")}
                    </h1>
                    <p className="text-muted-foreground">
                      {t("pages.bankConfiguration.description")}
                    </p>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        {t("common.actions.actions")}
                        <MoreVertical className="w-4 h-4 ml-2" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem
                        onClick={async () => {
                          if (!bankId) return;
                          try {
                            const manifest = await client.exportBankTemplate(bankId);
                            const json = JSON.stringify(manifest, null, 2);
                            await navigator.clipboard.writeText(json);
                            toast.success(t("common.toasts.templateCopied"));
                          } catch {
                            toast.error(t("common.toasts.templateExportFailed"));
                          }
                        }}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        {t("bankActions.exportTemplate")}
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onClick={handleTriggerConsolidation}
                        disabled={isConsolidating || !observationsEnabled}
                        title={
                          !observationsEnabled
                            ? t("bankActions.observationsDisabledTitle")
                            : undefined
                        }
                      >
                        {isConsolidating ? (
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                          <Brain className="w-4 h-4 mr-2" />
                        )}
                        {isConsolidating
                          ? t("bankActions.consolidating")
                          : t("bankActions.runConsolidation")}
                        {!observationsEnabled && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {t("common.actions.off")}
                          </span>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={handleRecoverConsolidation}
                        disabled={isRecoveringConsolidation || !observationsEnabled}
                        title={
                          !observationsEnabled
                            ? t("bankActions.observationsDisabledTitle")
                            : undefined
                        }
                      >
                        {isRecoveringConsolidation ? (
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                          <RotateCcw className="w-4 h-4 mr-2" />
                        )}
                        {isRecoveringConsolidation
                          ? t("bankActions.recovering")
                          : t("bankActions.recoverConsolidation")}
                        {!observationsEnabled && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {t("common.actions.off")}
                          </span>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => setShowClearObservationsDialog(true)}
                        disabled={!observationsEnabled}
                        className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
                        title={
                          !observationsEnabled
                            ? t("bankActions.observationsDisabledTitle")
                            : undefined
                        }
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        {t("bankActions.clearObservations")}
                        {!observationsEnabled && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {t("common.actions.off")}
                          </span>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onClick={() => setShowResetConfigDialog(true)}
                        disabled={!bankConfigEnabled}
                        className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
                        title={
                          !bankConfigEnabled ? t("bankActions.bankConfigDisabledTitle") : undefined
                        }
                      >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        {t("bankActions.resetConfiguration")}
                        {!bankConfigEnabled && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {t("common.actions.off")}
                          </span>
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onClick={() => setShowDeleteDialog(true)}
                        className="text-red-600 dark:text-red-400 focus:text-red-700 dark:focus:text-red-300"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        {t("bankActions.deleteBank")}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Sub-tabs */}
                <div className="mb-6 border-b border-border">
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleBankConfigTabChange("general")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        bankConfigTab === "general"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("bankConfigTabs.general")}
                      {bankConfigTab === "general" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    {bankConfigEnabled && (
                      <button
                        onClick={() => handleBankConfigTabChange("configuration")}
                        className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                          bankConfigTab === "configuration"
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {t("bankConfigTabs.configuration")}
                        {bankConfigTab === "configuration" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                        )}
                      </button>
                    )}
                    <button
                      onClick={() => handleBankConfigTabChange("webhooks")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        bankConfigTab === "webhooks"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("bankConfigTabs.webhooks")}
                      {bankConfigTab === "webhooks" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    <button
                      onClick={() => handleBankConfigTabChange("audit-logs")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        bankConfigTab === "audit-logs"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("bankConfigTabs.auditLogs")}
                      {bankConfigTab === "audit-logs" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Tab content */}
                <div>
                  {bankConfigTab === "general" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.bankConfiguration.generalDescription")}
                      </p>
                      <div className="space-y-6">
                        <BankStatsView />
                        <BankOperationsView />
                        <BankProfileView hideReflectFields />
                      </div>
                    </div>
                  )}
                  {bankConfigTab === "configuration" && bankConfigEnabled && (
                    <div className="space-y-6">
                      <BankConfigView />
                    </div>
                  )}
                  {bankConfigTab === "webhooks" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.bankConfiguration.webhooksDescription")}
                      </p>
                      <WebhooksView />
                    </div>
                  )}
                  {bankConfigTab === "audit-logs" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.bankConfiguration.auditLogsDescription")}
                      </p>
                      <AuditLogsView />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Recall Tab */}
            {view === "recall" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">
                  {t("pages.recall.title")}
                </h1>
                <p className="text-muted-foreground mb-6">{t("pages.recall.description")}</p>
                <SearchDebugView />
              </div>
            )}

            {/* Reflect Tab */}
            {view === "reflect" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">
                  {t("pages.reflect.title")}
                </h1>
                <p className="text-muted-foreground mb-6">{t("pages.reflect.description")}</p>
                <ThinkView />
              </div>
            )}

            {/* Data/Memories Tab */}
            {view === "data" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">
                  {t("pages.memories.title")}
                </h1>
                <p className="text-muted-foreground mb-6">{t("pages.memories.description")}</p>

                <div className="mb-6 border-b border-border">
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleDataSubTabChange("world")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        subTab === "world"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("dataTabs.worldFacts")}
                      {subTab === "world" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDataSubTabChange("experience")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        subTab === "experience"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("dataTabs.experience")}
                      {subTab === "experience" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDataSubTabChange("observations")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        subTab === "observations"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("dataTabs.observations")}
                      {!observationsEnabled && (
                        <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                          {t("common.actions.off")}
                        </span>
                      )}
                      {subTab === "observations" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDataSubTabChange("mental-models")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        subTab === "mental-models"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t("dataTabs.mentalModels")}
                      {subTab === "mental-models" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                  </div>
                </div>

                <div>
                  {subTab === "world" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.memories.worldDescription")}
                      </p>
                      <DataView key="world" factType="world" />
                    </div>
                  )}
                  {subTab === "experience" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.memories.experienceDescription")}
                      </p>
                      <DataView key="experience" factType="experience" />
                    </div>
                  )}
                  {subTab === "observations" &&
                    (observationsEnabled ? (
                      <div>
                        <p className="text-sm text-muted-foreground mb-4">
                          {t("pages.memories.observationsDescription")}
                        </p>
                        <DataView key="observations" factType="observation" />
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-16 text-center">
                        <div className="text-muted-foreground mb-2">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="48"
                            height="48"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Z" />
                            <path d="M12 8v4" />
                            <path d="M12 16h.01" />
                          </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-foreground mb-1">
                          {t("disabledStates.observationsTitle")}
                        </h3>
                        <p className="text-sm text-muted-foreground max-w-md">
                          {t("disabledStates.observationsBody")}{" "}
                          <code className="px-1 py-0.5 bg-muted rounded text-xs">
                            HINDSIGHT_API_ENABLE_OBSERVATIONS=true
                          </code>{" "}
                          {t("disabledStates.observationsEnableSuffix")}
                        </p>
                      </div>
                    ))}
                  {subTab === "mental-models" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("pages.memories.mentalModelsDescription")}
                      </p>
                      <MentalModelsView key="mental-models" />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Documents Tab */}
            {view === "documents" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">
                  {t("pages.documents.title")}
                </h1>
                <p className="text-muted-foreground mb-6">{t("pages.documents.description")}</p>
                <DocumentsView />
              </div>
            )}

            {/* Entities Tab */}
            {view === "entities" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">
                  {t("pages.entities.title")}
                </h1>
                <p className="text-muted-foreground mb-6">{t("pages.entities.description")}</p>
                <EntitiesView />
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Delete Bank Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("dialogs.deleteBank.title")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t("dialogs.deleteBank.confirm")}{" "}
                  <span className="font-semibold text-foreground">{bankId}</span>?
                </p>
                <p className="text-red-600 dark:text-red-400 font-medium">
                  {t("dialogs.deleteBank.warning")}
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>
              {t("common.actions.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteBank}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t("common.actions.deleting")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {t("bankActions.deleteBank")}
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Reset Configuration Confirmation Dialog */}
      <AlertDialog open={showResetConfigDialog} onOpenChange={setShowResetConfigDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("dialogs.resetConfiguration.title")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t("dialogs.resetConfiguration.confirm")}{" "}
                  <span className="font-semibold text-foreground">{bankId}</span>?
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  {t("dialogs.resetConfiguration.warning")}
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isResettingConfig}>
              {t("common.actions.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleResetConfig} disabled={isResettingConfig}>
              {isResettingConfig ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t("bankActions.resetting")}
                </>
              ) : (
                <>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  {t("bankActions.resetConfiguration")}
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
            <AlertDialogTitle>{t("dialogs.clearObservations.title")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t("dialogs.clearObservations.confirm")}{" "}
                  <span className="font-semibold text-foreground">{bankId}</span>?
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  {t("dialogs.clearObservations.warning")}
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isClearingObservations}>
              {t("common.actions.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClearObservations}
              disabled={isClearingObservations}
              className="bg-amber-500 text-white hover:bg-amber-600"
            >
              {isClearingObservations ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t("dialogs.clearObservations.clearing")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {t("bankActions.clearObservations")}
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
