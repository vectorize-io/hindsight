"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { toast } from "sonner";
import { BankSelector } from "@/components/bank-selector";
import { Sidebar } from "@/components/sidebar";
import { DataView } from "@/components/data-view";
import { DocumentsView } from "@/components/documents-view";
import { EntitiesView } from "@/components/entities-view";
import { KnowledgeBaseView } from "@/components/knowledge-base-view";
import { HomeView } from "@/components/home-view";
import { ThinkView } from "@/components/think-view";
import { SearchDebugView } from "@/components/search-debug-view";
import { BankProfileView } from "@/components/bank-profile-view";
import { BankConfigView } from "@/components/bank-config-view";
import { MemoryDefenseSection } from "@/components/memory-defense-section";
import { BankStatsView } from "@/components/bank-stats-view";
import { BankOperationsView } from "@/components/bank-operations-view";
import { MentalModelsView } from "@/components/mental-models-view";
import { WebhooksView } from "@/components/webhooks-view";
import { AuditLogsView } from "@/components/audit-logs-view";
import { LLMRequestsView } from "@/components/llm-requests-view";
import { FeatureNotEnabled } from "@/components/feature-not-enabled";
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
import {
  Brain,
  Download,
  Trash2,
  MoreVertical,
  Pencil,
  RotateCcw,
  Activity,
  FlaskConical,
} from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { LlmHealthDialog } from "@/components/llm-health-dialog";
import { ExtractDialog } from "@/components/extract-dialog";

type NavItem =
  "home" | "recall" | "reflect" | "data" | "documents" | "entities" | "knowledge" | "profile";
type DataSubTab = "world" | "experience" | "observations";
type KnowledgeTab = "pages" | "models";
type BankConfigTab =
  "general" | "memory-defense" | "configuration" | "webhooks" | "audit-logs" | "llm-requests";

export default function BankPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const t = useTranslations("bank");
  const tCommon = useTranslations("common");
  const { features } = useFeatures();
  const { currentBank: bankId, setCurrentBank, loadBanks } = useBank();

  const view = (searchParams.get("view") || "home") as NavItem;
  const subTab = (searchParams.get("subTab") || "world") as DataSubTab;
  const knowledgeTab = (searchParams.get("knowledgeTab") || "pages") as KnowledgeTab;
  const bankConfigTab = (searchParams.get("bankConfigTab") || "general") as BankConfigTab;
  const bankConfigEnabled = features?.bank_config_api ?? false;
  const llmTraceEnabled = features?.llm_trace ?? false;
  const llmHealthEnabled = features?.bank_llm_health ?? false;

  // `audit_log_enabled` and `enable_observations` are hierarchical
  // (env -> tenant -> bank): a bank can opt in even when the deployment default
  // is off. The /version feature flags only report the global default, so gate
  // these tabs on the bank's *resolved* config instead. Fall back to the global
  // flag when the bank config API is disabled (per-bank overrides can't exist
  // then) or the field is unavailable.
  const [bankAuditLogEnabled, setBankAuditLogEnabled] = useState<boolean | null>(null);
  const [bankObservationsEnabled, setBankObservationsEnabled] = useState<boolean | null>(null);
  useEffect(() => {
    if (!bankId || !bankConfigEnabled) {
      setBankAuditLogEnabled(null);
      setBankObservationsEnabled(null);
      return;
    }
    let cancelled = false;
    client
      .getBankConfig(bankId)
      .then((r) => {
        if (cancelled) return;
        const audit = r.config?.audit_log_enabled;
        const observations = r.config?.enable_observations;
        setBankAuditLogEnabled(typeof audit === "boolean" ? audit : null);
        setBankObservationsEnabled(typeof observations === "boolean" ? observations : null);
      })
      .catch(() => {
        if (cancelled) return;
        setBankAuditLogEnabled(null);
        setBankObservationsEnabled(null);
      });
    return () => {
      cancelled = true;
    };
  }, [bankId, bankConfigEnabled]);
  const auditLogEnabled = bankAuditLogEnabled ?? features?.audit_log ?? false;
  const observationsEnabled = bankObservationsEnabled ?? features?.observations ?? false;

  // Bank actions state
  const [showLlmHealthDialog, setShowLlmHealthDialog] = useState(false);
  const [showExtractDialog, setShowExtractDialog] = useState(false);
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

  const handleKnowledgeTabChange = (tab: KnowledgeTab) => {
    if (!bankId) return;
    router.push(bankRoute(bankId, `?view=knowledge&knowledgeTab=${tab}`));
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
    } catch (error) {
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
      toast.success(t("observationsCleared"), {
        description: result.message || t("observationsClearedDefault"),
      });
    } catch (error) {
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
    } catch (error) {
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
      toast.success(t("recoveredMemories", { count: result.retried_count }));
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsRecoveringConsolidation(false);
    }
  };

  return (
    // Pinned to the viewport (h-screen, not min-h-screen) so `main` below is
    // the only scroll container: with min-h-screen the page itself grew past
    // the viewport and scrolled the header and sidebar out of view.
    <div className="h-screen overflow-hidden bg-background flex flex-col">
      <div className="shrink-0">
        <BankSelector />
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Sidebar currentTab={view} onTabChange={handleTabChange} />

        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="p-6">
            {/* Content width staircase. Without a cap the views ran the full
                width of the window, which on a 16" display left body text and
                table rows spanning ~1700px. The Kit specifies a flat 1024px;
                the steps at xl/2xl are a Hindsight extension so large displays
                don't pay ~350px gutters for that cap. One wrapper covers
                every view because they all sit under this container. */}
            <div className="max-w-[1024px] xl:max-w-[1280px] 2xl:max-w-[1440px] mx-auto w-full">
              {/* Bank Configuration Tab */}
              {view === "profile" && (
                <div>
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h1 className="text-3xl font-bold mb-2 text-foreground">
                        {t("bankConfiguration")}
                      </h1>
                      <p className="text-muted-foreground">{t("bankConfigurationDescription")}</p>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm">
                          {t("actions")}
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
                              toast.success(t("templateCopied"));
                            } catch {
                              toast.error(t("failedToExportTemplate"));
                            }
                          }}
                        >
                          <Download className="w-4 h-4 mr-2" />
                          {t("exportTemplate")}
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => setShowExtractDialog(true)}>
                          <FlaskConical className="w-4 h-4 mr-2" />
                          {t("dryRunExtraction")}
                        </DropdownMenuItem>
                        {llmHealthEnabled && (
                          <DropdownMenuItem onClick={() => setShowLlmHealthDialog(true)}>
                            <Activity className="w-4 h-4 mr-2" />
                            {t("health")}
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={handleTriggerConsolidation}
                          disabled={isConsolidating || !observationsEnabled}
                          title={
                            !observationsEnabled ? "Observations feature is not enabled" : undefined
                          }
                        >
                          {isConsolidating ? (
                            <Spinner size="sm" className="mr-2" />
                          ) : (
                            <Brain className="w-4 h-4 mr-2" />
                          )}
                          {isConsolidating ? t("consolidating") : t("runConsolidation")}
                          {!observationsEnabled && (
                            <span className="ml-auto text-xs text-muted-foreground">Off</span>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={handleRecoverConsolidation}
                          disabled={isRecoveringConsolidation || !observationsEnabled}
                          title={
                            !observationsEnabled ? "Observations feature is not enabled" : undefined
                          }
                        >
                          {isRecoveringConsolidation ? (
                            <Spinner size="sm" className="mr-2" />
                          ) : (
                            <RotateCcw className="w-4 h-4 mr-2" />
                          )}
                          {isRecoveringConsolidation ? t("recovering") : t("recoverConsolidation")}
                          {!observationsEnabled && (
                            <span className="ml-auto text-xs text-muted-foreground">Off</span>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => setShowClearObservationsDialog(true)}
                          disabled={!observationsEnabled}
                          className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
                          title={
                            !observationsEnabled ? "Observations feature is not enabled" : undefined
                          }
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          {t("clearObservations")}
                          {!observationsEnabled && (
                            <span className="ml-auto text-xs text-muted-foreground">Off</span>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => setShowResetConfigDialog(true)}
                          disabled={!bankConfigEnabled}
                          className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
                          title={!bankConfigEnabled ? "Bank Config API is disabled" : undefined}
                        >
                          <RotateCcw className="w-4 h-4 mr-2" />
                          {t("resetConfiguration")}
                          {!bankConfigEnabled && (
                            <span className="ml-auto text-xs text-muted-foreground">Off</span>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => setShowDeleteDialog(true)}
                          className="text-red-600 dark:text-red-400 focus:text-red-700 dark:focus:text-red-300"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          {t("deleteBank")}
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
                        {t("general")}
                        {bankConfigTab === "general" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                      {bankConfigEnabled && (
                        <button
                          onClick={() => handleBankConfigTabChange("memory-defense")}
                          className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                            bankConfigTab === "memory-defense"
                              ? "text-primary"
                              : "text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {t("memoryDefense")}
                          {bankConfigTab === "memory-defense" && (
                            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                          )}
                        </button>
                      )}
                      {bankConfigEnabled && (
                        <button
                          onClick={() => handleBankConfigTabChange("configuration")}
                          className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                            bankConfigTab === "configuration"
                              ? "text-primary"
                              : "text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {t("configuration")}
                          {bankConfigTab === "configuration" && (
                            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
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
                        {t("webhooks")}
                        {bankConfigTab === "webhooks" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
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
                        {t("auditLogs")}
                        {!auditLogEnabled && (
                          <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            Off
                          </span>
                        )}
                        {bankConfigTab === "audit-logs" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                      <button
                        onClick={() => handleBankConfigTabChange("llm-requests")}
                        className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                          bankConfigTab === "llm-requests"
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {t("llmRequests")}
                        {!llmTraceEnabled && (
                          <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            Off
                          </span>
                        )}
                        {bankConfigTab === "llm-requests" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Tab content */}
                  <div>
                    {bankConfigTab === "general" && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-4">
                          {t("overviewAndOperations")}
                        </p>
                        <div className="space-y-6">
                          <BankStatsView />
                          <BankOperationsView />
                          <BankProfileView hideReflectFields />
                        </div>
                      </div>
                    )}
                    {bankConfigTab === "memory-defense" && bankConfigEnabled && bankId && (
                      <div className="space-y-6">
                        <MemoryDefenseSection bankId={bankId} />
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
                          {t("webhooksDescription")}
                        </p>
                        <WebhooksView />
                      </div>
                    )}
                    {bankConfigTab === "audit-logs" &&
                      (auditLogEnabled ? (
                        <div>
                          <p className="text-sm text-muted-foreground mb-4">
                            {t("auditLogsDescription")}
                          </p>
                          <AuditLogsView />
                        </div>
                      ) : (
                        <FeatureNotEnabled
                          title={t("auditLogsNotEnabled")}
                          description={t.rich("auditLogsDisabledMessage", {
                            envVar: () => (
                              <code className="px-1 py-0.5 bg-muted rounded text-xs">
                                HINDSIGHT_API_AUDIT_LOG_ENABLED=true
                              </code>
                            ),
                          })}
                        />
                      ))}
                    {bankConfigTab === "llm-requests" &&
                      (llmTraceEnabled ? (
                        <div>
                          <p className="text-sm text-muted-foreground mb-4">
                            {t("llmRequestsDescription")}
                          </p>
                          <LLMRequestsView />
                        </div>
                      ) : (
                        <FeatureNotEnabled
                          title={t("llmRequestsNotEnabled")}
                          description={t.rich("llmRequestsDisabledMessage", {
                            envVar: () => (
                              <code className="px-1 py-0.5 bg-muted rounded text-xs">
                                HINDSIGHT_API_LLM_TRACE_ENABLED=true
                              </code>
                            ),
                          })}
                        />
                      ))}
                  </div>
                </div>
              )}

              {/* Recall Tab */}
              {view === "recall" && (
                <div>
                  <h1 className="text-3xl font-bold mb-2 text-foreground">{t("recallAnalyzer")}</h1>
                  <p className="text-muted-foreground mb-6">{t("recallAnalyzerDescription")}</p>
                  <SearchDebugView />
                </div>
              )}

              {/* Reflect Tab */}
              {view === "reflect" && (
                <div>
                  <h1 className="text-3xl font-bold mb-2 text-foreground">{t("reflect")}</h1>
                  <p className="text-muted-foreground mb-6">{t("reflectDescription")}</p>
                  <ThinkView />
                </div>
              )}

              {/* Data/Memories Tab */}
              {view === "data" && (
                <div>
                  <h1 className="text-3xl font-bold mb-2 text-foreground">{t("memories")}</h1>
                  <p className="text-muted-foreground mb-6">{t("memoriesDescription")}</p>

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
                        {t("worldFacts")}
                        {subTab === "world" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
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
                        {t("experience")}
                        {subTab === "experience" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
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
                        {t("observations")}
                        {!observationsEnabled && (
                          <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                            Off
                          </span>
                        )}
                        {subTab === "observations" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div>
                    {subTab === "world" && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-4">
                          {t("worldFactsDescription")}
                        </p>
                        <DataView key="world" factType="world" />
                      </div>
                    )}
                    {subTab === "experience" && (
                      <div>
                        <p className="text-sm text-muted-foreground mb-4">
                          {t("experienceDescription")}
                        </p>
                        <DataView key="experience" factType="experience" />
                      </div>
                    )}
                    {subTab === "observations" &&
                      (observationsEnabled ? (
                        <div>
                          <p className="text-sm text-muted-foreground mb-4">
                            {t("observationsDescription")}
                          </p>
                          <DataView key="observations" factType="observation" />
                        </div>
                      ) : (
                        <FeatureNotEnabled
                          title={t("observationsNotEnabled")}
                          description={t.rich("observationsDisabledMessage", {
                            envVar: () => (
                              <code className="px-1 py-0.5 bg-muted rounded text-xs">
                                HINDSIGHT_API_ENABLE_OBSERVATIONS=true
                              </code>
                            ),
                          })}
                        />
                      ))}
                  </div>
                </div>
              )}

              {/* Documents Tab — DocumentsView renders its own title row so the
                Export/Import Actions menu can sit beside the heading. */}
              {view === "documents" && (
                <div>
                  <DocumentsView />
                </div>
              )}

              {/* Entities Tab */}
              {view === "entities" && (
                <div>
                  <h1 className="text-3xl font-bold mb-2 text-foreground">{t("entities")}</h1>
                  <p className="text-muted-foreground mb-6">{t("entitiesDescription")}</p>
                  <EntitiesView />
                </div>
              )}

              {/* Knowledge Tab — Pages (knowledge base) + Mental Models sub-tabs. */}
              {view === "knowledge" && (
                <div>
                  <h1 className="text-3xl font-bold mb-2 text-foreground">{t("knowledge")}</h1>
                  <p className="text-muted-foreground mb-4">{t("knowledgeDescription")}</p>

                  <div className="mb-4 border-b border-border">
                    <div className="flex gap-1">
                      <button
                        onClick={() => handleKnowledgeTabChange("pages")}
                        className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                          knowledgeTab === "pages"
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {t("pages")}
                        {knowledgeTab === "pages" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                      <button
                        onClick={() => handleKnowledgeTabChange("models")}
                        className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                          knowledgeTab === "models"
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {t("mentalModels")}
                        {knowledgeTab === "models" && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
                        )}
                      </button>
                    </div>
                  </div>

                  {knowledgeTab === "pages" && <KnowledgeBaseView />}
                  {knowledgeTab === "models" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {t("mentalModelsDescription")}
                      </p>
                      <MentalModelsView key="mental-models" />
                    </div>
                  )}
                </div>
              )}

              {/* Home Tab — bank dashboard. */}
              {view === "home" && bankId && (
                <HomeView bankId={bankId} onNavigate={(tab) => handleTabChange(tab as NavItem)} />
              )}
            </div>
          </div>
        </main>
      </div>

      {/* LLM connectivity check */}
      {bankId && (
        <LlmHealthDialog
          bankId={bankId}
          open={showLlmHealthDialog}
          onOpenChange={setShowLlmHealthDialog}
        />
      )}

      {/* Dry-run extraction */}
      <ExtractDialog open={showExtractDialog} onOpenChange={setShowExtractDialog} />

      {/* Delete Bank Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteMemoryBank")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t.rich("deleteBankPrompt", {
                    bankName: () => <span className="font-semibold text-foreground">{bankId}</span>,
                  })}
                </p>
                <p className="text-red-600 dark:text-red-400 font-medium">
                  {t("deleteBankWarning")}
                </p>
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
                  <Spinner size="sm" className="mr-2" />
                  {t("deleting")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {t("deleteBank")}
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
            <AlertDialogTitle>{t("resetConfigTitle")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t.rich("resetConfigPrompt", {
                    bankName: () => <span className="font-semibold text-foreground">{bankId}</span>,
                  })}
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  {t("resetConfigWarning")}
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isResettingConfig}>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={handleResetConfig} disabled={isResettingConfig}>
              {isResettingConfig ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  {t("resetting")}
                </>
              ) : (
                <>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  {t("resetConfiguration")}
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
            <AlertDialogTitle>{t("clearObservationsTitle")}</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  {t.rich("clearObservationsPrompt", {
                    bankName: () => <span className="font-semibold text-foreground">{bankId}</span>,
                  })}
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  {t("clearObservationsWarning")}
                </p>
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
                  <Spinner size="sm" className="mr-2" />
                  {t("clearing")}
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  {t("clearObservations")}
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
