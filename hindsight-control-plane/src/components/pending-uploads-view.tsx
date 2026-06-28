"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, FileUp, Loader2, CheckCircle2, XCircle, Clock, AlertCircle } from "lucide-react";
import { useBank } from "@/lib/bank-context";

interface UploadOperation {
  id: string;
  task_type: string;
  items_count: number;
  document_id: string | null;
  created_at: string;
  updated_at?: string;
  status: "pending" | "processing" | "completed" | "failed";
  error_message: string | null;
  retry_count: number;
}

interface OperationsResponse {
  operations: UploadOperation[];
  total: number;
}

function formatDuration(start: string, end?: string): string {
  const startDate = new Date(start);
  const endDate = end ? new Date(end) : new Date();
  const ms = endDate.getTime() - startDate.getTime();
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes}m ${secs}s`;
}

export function PendingUploadsView() {
  const t = useTranslations("uploads");
  const { currentBank } = useBank();
  const [operations, setOperations] = useState<UploadOperation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchUploads = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const bankId = currentBank || "default";
      const response = await fetch(`/api/system/operations?bank_id=${bankId}&limit=50`);
      const data: OperationsResponse = await response.json();
      const fileOps = (data.operations || []).filter(
        (op) => op.task_type === "file_convert_retain" || op.task_type === "retain"
      );
      setOperations(fileOps);
    } catch (err) {
      console.error("Failed to fetch upload operations:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [currentBank]);

  useEffect(() => {
    fetchUploads();
    const interval = setInterval(() => fetchUploads(), 5000);
    return () => clearInterval(interval);
  }, [fetchUploads]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "processing":
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-500" />;
      case "pending":
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "processing":
        return t("statusProcessing");
      case "completed":
        return t("statusCompleted");
      case "failed":
        return t("statusFailed");
      case "pending":
        return t("statusPending");
      default:
        return status;
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "processing":
        return "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400";
      case "completed":
        return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400";
      case "failed":
        return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400";
      case "pending":
        return "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const activeOps = operations.filter((op) => op.status === "processing" || op.status === "pending");
  const completedOps = operations.filter((op) => op.status === "completed");
  const failedOps = operations.filter((op) => op.status === "failed");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">{t("title")}</h2>
          <p className="text-sm text-muted-foreground">{t("description")}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => fetchUploads(true)}
          disabled={refreshing}
          className="gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
          {t("refresh")}
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{t("active")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{activeOps.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{t("completed")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-600">{completedOps.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{t("failed")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{failedOps.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{t("total")}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{operations.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Active/Pending Uploads */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-sm text-muted-foreground">{t("loading")}</span>
        </div>
      ) : operations.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <FileUp className="w-12 h-12 text-muted-foreground/40 mb-4" />
            <p className="text-sm text-muted-foreground">{t("emptyState")}</p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileUp className="w-4 h-4" />
              {t("recentUploads")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {operations.map((op) => (
                <div
                  key={op.id}
                  className={`p-4 rounded-lg border transition-colors ${
                    op.status === "failed"
                      ? "border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/20"
                      : op.status === "processing"
                      ? "border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950/20"
                      : "border-border bg-card"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(op.status)}
                      <span className="font-mono text-xs text-muted-foreground">
                        {op.id.substring(0, 12)}...
                      </span>
                      {op.document_id && (
                        <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                          {op.document_id}
                        </span>
                      )}
                    </div>
                    <Badge
                      variant="outline"
                      className={`text-xs ${getStatusBadgeClass(op.status)}`}
                    >
                      {getStatusLabel(op.status)}
                    </Badge>
                  </div>

                  {/* Progress bar for active uploads */}
                  {op.status === "processing" && (
                    <div className="mt-2">
                      <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                        <span>{t("itemsCount", { count: op.items_count })}</span>
                        <span>{formatDuration(op.created_at)}</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-blue-500 h-full rounded-full animate-pulse"
                          style={{ width: `${Math.min(100, (op.items_count > 0 ? 50 : 10))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Error message */}
                  {op.error_message && (
                    <div className="flex items-start gap-2 mt-2 p-2 rounded bg-red-100 dark:bg-red-900/20 text-xs text-red-700 dark:text-red-400">
                      <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{op.error_message}</span>
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                    <span>{t("started")}: {new Date(op.created_at).toLocaleString()}</span>
                    {op.retry_count > 0 && (
                      <span>{t("retries", { count: op.retry_count })}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
