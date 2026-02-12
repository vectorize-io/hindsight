"use client";

import { useState, useEffect } from "react";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { client } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Database, Link2, FolderOpen, Activity, Clock } from "lucide-react";

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
  last_consolidated_at: string | null;
  pending_consolidation: number;
  total_mental_models: number;
}

export function BankStatsView() {
  const { currentBank } = useBank();
  const { features } = useFeatures();
  const observationsEnabled = features?.observations ?? false;
  const [stats, setStats] = useState<BankStats | null>(null);
  const [mentalModelsCount, setMentalModelsCount] = useState(0);
  const [directivesCount, setDirectivesCount] = useState(0);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const [statsData, mentalModelsData, directivesData] = await Promise.all([
        client.getBankStats(currentBank),
        client.listMentalModels(currentBank),
        client.listDirectives(currentBank),
      ]);
      setStats(statsData as BankStats);
      setMentalModelsCount(mentalModelsData.items?.length || 0);
      setDirectivesCount(directivesData.items?.length || 0);
    } catch (error) {
      console.error("Error loading bank stats:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadData();
      // Refresh stats every 5 seconds
      const interval = setInterval(loadData, 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank]);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center py-12">
        <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="space-y-6">
      {/* Stats Overview - Compact cards */}
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

      {/* Memory Type Breakdown */}
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
            {directivesCount}
          </p>
        </div>
      </div>
    </div>
  );
}
