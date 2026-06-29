"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import Link from "next/link";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Database,
  Activity,
  FileText,
  RefreshCw,
  Loader2,
  ChevronRight,
  Shield,
  Clock,
  CheckCircle2,
  AlertCircle,
  Search,
  Banknote,
  ArrowRight,
  HardDrive,
} from "lucide-react";

interface BankSummary {
  bank_id: string;
  name?: string;
  mission?: string;
}

interface BankStats {
  totalFacts?: number;
  totalMemories?: number;
  totalChunks?: number;
  totalDocuments?: number;
}

interface Operation {
  id: string;
  type: string;
  status: "completed" | "running" | "failed" | "pending";
  created_at?: string;
  started_at?: string;
}

export default function MemoriesPage() {
  const t = useTranslations("operator");
  const params = useParams();
  const locale = (params?.locale as string) || "en";

  const [banks, setBanks] = useState<BankSummary[]>([]);
  const [bankStatsMap, setBankStatsMap] = useState<Record<string, BankStats>>({});
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loadingBanks, setLoadingBanks] = useState(true);
  const [loadingOps, setLoadingOps] = useState(true);

  const loadBanks = async () => {
    setLoadingBanks(true);
    try {
      const res = await fetch("/api/banks");
      const data = await res.json();
      const bankList: BankSummary[] = data.banks || [];
      setBanks(bankList);

      // Fetch stats for each bank
      const statsMap: Record<string, BankStats> = {};
      await Promise.all(
        bankList.map(async (bank: BankSummary) => {
          try {
            const sRes = await fetch(`/api/stats/${encodeURIComponent(bank.bank_id)}`);
            if (sRes.ok) {
              const sData = await sRes.json();
              statsMap[bank.bank_id] = {
                totalFacts: sData.total_facts ?? sData.totalFacts ?? sData.nodes ?? 0,
                totalMemories: sData.total_memories ?? sData.totalMemories ?? 0,
                totalDocuments: sData.total_documents ?? sData.totalDocuments ?? 0,
                totalChunks: sData.total_chunks ?? sData.totalChunks ?? 0,
              };
            }
          } catch {
            // Stats unavailable for this bank
          }
        })
      );
      setBankStatsMap(statsMap);
    } catch (e) {
      console.error("Failed to load banks:", e);
      setBanks([]);
    } finally {
      setLoadingBanks(false);
    }
  };

  const loadOperations = async () => {
    setLoadingOps(true);
    try {
      const res = await fetch("/api/system/operations?limit=10");
      const data = await res.json();
      setOperations(data.operations || data.data || []);
    } catch {
      setOperations([]);
    } finally {
      setLoadingOps(false);
    }
  };

  useEffect(() => {
    loadBanks();
    loadOperations();
  }, []);

  // Compute aggregate stats
  const totalFacts = Object.values(bankStatsMap).reduce(
    (sum, s) => sum + (s.totalFacts ?? 0),
    0
  );
  const totalDocuments = Object.values(bankStatsMap).reduce(
    (sum, s) => sum + (s.totalDocuments ?? 0),
    0
  );
  const activeOps = operations.filter(
    (o) => o.status === "running" || o.status === "pending"
  ).length;

  const opStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-3 h-3 text-green-500" />;
      case "running":
        return <Loader2 className="w-3 h-3 text-blue-500 animate-spin" />;
      case "failed":
        return <AlertCircle className="w-3 h-3 text-red-500" />;
      default:
        return <Clock className="w-3 h-3 text-muted-foreground" />;
    }
  };

  const opStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400 border-green-200 dark:border-green-800";
      case "running":
        return "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400 border-blue-200 dark:border-blue-800";
      case "failed":
        return "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400 border-red-200 dark:border-red-800";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Brain className="h-6 w-6 text-primary" />
              {t("memories.title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {t("descriptions.memories")}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => { loadBanks(); loadOperations(); }} disabled={loadingBanks}>
            <RefreshCw className={`h-4 w-4 mr-1 ${loadingBanks ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Database className="h-4 w-4" /> {t("memories.statTotalBanks")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {loadingBanks ? <Loader2 className="h-5 w-5 animate-spin" /> : banks.length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Brain className="h-4 w-4" /> {t("memories.statTotalFacts")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {loadingBanks ? <Loader2 className="h-5 w-5 animate-spin" /> : totalFacts.toLocaleString()}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> {t("memories.statActiveOps")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${activeOps > 0 ? "text-blue-600" : ""}`}>
                {loadingOps ? <Loader2 className="h-5 w-5 animate-spin" /> : activeOps}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <FileText className="h-4 w-4" /> {t("memories.statDocuments")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {loadingBanks ? <Loader2 className="h-5 w-5 animate-spin" /> : totalDocuments.toLocaleString()}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Bank grid */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <HardDrive className="h-5 w-5 text-primary" />
                {t("memories.memoryEngines")}
              </h2>
              <Link href={`/${locale}/dashboard`}>
                <Button variant="ghost" size="sm" className="text-xs">
                  {t("memories.browseAllBanks")} <ArrowRight className="w-3 h-3 ml-1" />
                </Button>
              </Link>
            </div>

            {loadingBanks ? (
              <div className="flex items-center justify-center py-16 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> {t("memories.loading")}
              </div>
            ) : banks.length === 0 ? (
              <Card>
                <CardContent className="flex items-center justify-center py-12 text-muted-foreground">
                  <div className="text-center space-y-2">
                    <Database className="h-10 w-10 mx-auto opacity-30" />
                    <p>{t("memories.noBanks")}</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {banks.map((bank) => {
                  const stats = bankStatsMap[bank.bank_id];
                  return (
                    <Link
                      key={bank.bank_id}
                      href={`/${locale}/banks/${encodeURIComponent(bank.bank_id)}`}
                    >
                      <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-sm font-medium flex items-center gap-2">
                              <Database className="h-4 w-4 text-primary" />
                              {bank.name || bank.bank_id}
                            </CardTitle>
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          </div>
                          {bank.mission && (
                            <CardDescription className="text-xs line-clamp-2">
                              {bank.mission}
                            </CardDescription>
                          )}
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Brain className="w-3 h-3" />
                              {stats?.totalFacts ?? "—"} facts
                            </span>
                            <span className="flex items-center gap-1">
                              <FileText className="w-3 h-3" />
                              {stats?.totalDocuments ?? "—"} docs
                            </span>
                            <Badge
                              variant="outline"
                              className="text-[10px] h-5 border-green-200 text-green-700 bg-green-50 dark:border-green-800 dark:text-green-400 dark:bg-green-950/30"
                            >
                              <Shield className="w-2.5 h-2.5 mr-1" />
                              {t("memories.operational")}
                            </Badge>
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>

          {/* Recent operations */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              {t("memories.operations")}
            </h2>
            <Card>
              <CardContent className="p-4 space-y-2">
                {loadingOps ? (
                  <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading...
                  </div>
                ) : operations.length === 0 ? (
                  <div className="text-sm text-muted-foreground text-center py-8">
                    {t("memories.noOps")}
                  </div>
                ) : (
                  operations.slice(0, 8).map((op) => (
                    <div
                      key={op.id}
                      className="flex items-center justify-between py-2 border-b border-border last:border-0"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        {opStatusIcon(op.status)}
                        <span className="text-xs font-mono truncate">
                          {op.type || op.id.substring(0, 16)}
                        </span>
                      </div>
                      <Badge
                        variant="outline"
                        className={`text-[10px] h-5 px-1.5 ${opStatusColor(op.status)}`}
                      >
                        {op.status}
                      </Badge>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </OperatorShell>
  );
}
