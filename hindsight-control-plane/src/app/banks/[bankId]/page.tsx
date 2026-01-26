"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { BankSelector } from "@/components/bank-selector";
import { Sidebar } from "@/components/sidebar";
import { DataView } from "@/components/data-view";
import { DocumentsView } from "@/components/documents-view";
import { EntitiesView } from "@/components/entities-view";
import { ThinkView } from "@/components/think-view";
import { SearchDebugView } from "@/components/search-debug-view";
import { BankProfileView } from "@/components/bank-profile-view";
import { MentalModelsView } from "@/components/mental-models-view";
import { useFeatures } from "@/lib/features-context";

type NavItem = "recall" | "reflect" | "data" | "documents" | "entities" | "profile";
type DataSubTab = "world" | "experience" | "observations" | "mental-models";

export default function BankPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { features } = useFeatures();

  const bankId = params.bankId as string;
  const view = (searchParams.get("view") || "profile") as NavItem;
  const subTab = (searchParams.get("subTab") || "world") as DataSubTab;
  const observationsEnabled = features?.observations ?? false;

  const handleTabChange = (tab: NavItem) => {
    router.push(`/banks/${bankId}?view=${tab}`);
  };

  const handleDataSubTabChange = (newSubTab: DataSubTab) => {
    router.push(`/banks/${bankId}?view=data&subTab=${newSubTab}`);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <BankSelector />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar currentTab={view} onTabChange={handleTabChange} />

        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {/* Profile Tab */}
            {view === "profile" && (
              <div>
                <h1 className="text-3xl font-bold mb-2 text-foreground">Bank Profile</h1>
                <p className="text-muted-foreground mb-6">
                  View and edit the memory bank profile, disposition traits, and background
                  information.
                </p>
                <BankProfileView />
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
                  Query the memory bank and generate a response with optional disposition-aware
                  reasoning.
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
                  {subTab === "world" && <DataView key="world" factType="world" />}
                  {subTab === "experience" && <DataView key="experience" factType="experience" />}
                  {subTab === "observations" &&
                    (observationsEnabled ? (
                      <DataView key="observations" factType="observation" />
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
                  {subTab === "mental-models" && <MentalModelsView key="mental-models" />}
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
    </div>
  );
}
