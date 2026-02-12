"use client";

import { useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
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
import { useFeatures } from "@/lib/features-context";
import { useBank } from "@/lib/bank-context";
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
import { Brain, Trash2, Loader2, MoreVertical, Pencil } from "lucide-react";

type NavItem = "recall" | "reflect" | "data" | "documents" | "entities" | "profile";
type DataSubTab = "world" | "experience" | "observations" | "mental-models";
type BankConfigTab = "general" | "configuration";

export default function BankPage() {
  const params = useParams();
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

  const handleTabChange = (tab: NavItem) => {
    router.push(`/banks/${bankId}?view=${tab}`);
  };

  const handleDataSubTabChange = (newSubTab: DataSubTab) => {
    router.push(`/banks/${bankId}?view=data&subTab=${newSubTab}`);
  };

  const handleBankConfigTabChange = (newTab: BankConfigTab) => {
    router.push(`/banks/${bankId}?view=profile&bankConfigTab=${newTab}`);
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
      console.error("Error deleting bank:", error);
      alert("Error deleting bank: " + (error as Error).message);
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
      alert(result.message || "Observations cleared successfully");
    } catch (error) {
      console.error("Error clearing observations:", error);
      alert("Error clearing observations: " + (error as Error).message);
    } finally {
      setIsClearingObservations(false);
    }
  };

  const handleTriggerConsolidation = async () => {
    if (!bankId) return;

    setIsConsolidating(true);
    try {
      await client.triggerConsolidation(bankId);
    } catch (error) {
      console.error("Error triggering consolidation:", error);
      alert("Error triggering consolidation: " + (error as Error).message);
    } finally {
      setIsConsolidating(false);
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
                    <h1 className="text-3xl font-bold mb-2 text-foreground">Bank Configuration</h1>
                    <p className="text-muted-foreground">
                      Manage bank settings, profile, and operations.
                    </p>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        Actions
                        <MoreVertical className="w-4 h-4 ml-2" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem
                        onClick={handleTriggerConsolidation}
                        disabled={isConsolidating || !observationsEnabled}
                        title={
                          !observationsEnabled ? "Observations feature is not enabled" : undefined
                        }
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
                        title={
                          !observationsEnabled ? "Observations feature is not enabled" : undefined
                        }
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
                      General
                      {bankConfigTab === "general" && (
                        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                      )}
                    </button>
                    <button
                      onClick={() => handleBankConfigTabChange("configuration")}
                      className={`px-6 py-3 font-semibold text-sm transition-all relative ${
                        bankConfigTab === "configuration"
                          ? "text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Configuration
                      {bankConfigTab === "configuration" && (
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
                        Overview statistics and background operations for this memory bank.
                      </p>
                      <div className="space-y-6">
                        <BankStatsView />
                        <BankOperationsView />
                      </div>
                    </div>
                  )}
                  {bankConfigTab === "configuration" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-6">
                        Configure disposition traits, mission, directives, and behavioral settings
                        for this bank.
                      </p>
                      <div className="space-y-6">
                        <BankProfileView />
                        {bankConfigEnabled && <BankConfigView />}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Recall Tab */}
            {view === "recall" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">Recall</h1>
                <p className="text-muted-foreground mb-6">
                  Analyze memory recall with detailed trace information and retrieval methods.
                </p>
                <SearchDebugView />
              </div>
            )}

            {/* Reflect Tab */}
            {view === "reflect" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">Reflect</h1>
                <p className="text-muted-foreground mb-6">
                  Run an agentic loop that autonomously gathers evidence and reasons through the
                  lens of the bank&apos;s disposition to generate contextual responses.
                </p>
                <ThinkView />
              </div>
            )}

            {/* Data/Memories Tab */}
            {view === "data" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">Memories</h1>
                <p className="text-muted-foreground mb-6">
                  View and explore different types of memories stored in this memory bank.
                </p>

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
                      World Facts
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
                      Experience
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
                      Observations
                      {!observationsEnabled && (
                        <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                          Off
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
                      Mental Models
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
                        Objective facts about the world received from external sources.
                      </p>
                      <DataView key="world" factType="world" />
                    </div>
                  )}
                  {subTab === "experience" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        The bank&apos;s own actions, interactions, and first-person experiences.
                      </p>
                      <DataView key="experience" factType="experience" />
                    </div>
                  )}
                  {subTab === "observations" &&
                    (observationsEnabled ? (
                      <div>
                        <p className="text-sm text-muted-foreground mb-4">
                          Consolidated knowledge synthesized from facts — patterns, preferences, and
                          learnings that emerge from accumulated evidence.
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
                          Observations Not Enabled
                        </h3>
                        <p className="text-sm text-muted-foreground max-w-md">
                          Observations consolidation is disabled on this server. Set{" "}
                          <code className="px-1 py-0.5 bg-muted rounded text-xs">
                            HINDSIGHT_API_ENABLE_OBSERVATIONS=true
                          </code>{" "}
                          to enable.
                        </p>
                      </div>
                    ))}
                  {subTab === "mental-models" && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-4">
                        User-curated summaries generated from queries — reusable knowledge snapshots
                        that can be refreshed as memories evolve.
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
                <h1 className="text-3xl font-bold mb-2 text-foreground">Documents</h1>
                <p className="text-muted-foreground mb-6">
                  Manage documents and retain new memories.
                </p>
                <DocumentsView />
              </div>
            )}

            {/* Entities Tab */}
            {view === "entities" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">Entities</h1>
                <p className="text-muted-foreground mb-6">
                  Explore entities (people, organizations, places) mentioned in memories.
                </p>
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
            <AlertDialogTitle>Delete Memory Bank</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  Are you sure you want to delete the memory bank{" "}
                  <span className="font-semibold text-foreground">{bankId}</span>?
                </p>
                <p className="text-red-600 dark:text-red-400 font-medium">
                  This action cannot be undone. All memories, entities, documents, and the bank
                  profile will be permanently deleted.
                </p>
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
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
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
                  <span className="font-semibold text-foreground">{bankId}</span>?
                </p>
                <p className="text-amber-600 dark:text-amber-400 font-medium">
                  This will delete all consolidated knowledge. Observations will be regenerated the
                  next time consolidation runs.
                </p>
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
    </div>
  );
}
