"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { client, type OperationProgress } from "@/lib/api";
import { Button } from "@/components/ui/button";
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
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RefreshCw,
  Clock,
  AlertCircle,
  CheckCircle,
  Loader2,
  X,
  RotateCcw,
  Code,
  Ban,
} from "lucide-react";

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  document_id: string | null;
  created_at: string;
  updated_at?: string | null;
  status: string;
  error_message: string | null;
  progress?: OperationProgress | null;
}

interface ChildOperationStatus {
  operation_id: string;
  status: string;
  sub_batch_index: number | null;
  items_count: number | null;
  error_message: string | null;
}

type OperationDetails =
  | {
      operation_id: string;
      status: string;
      operation_type: string | null;
      created_at: string | null;
      updated_at: string | null;
      completed_at: string | null;
      error_message: string | null;
      progress?: OperationProgress | null;
      result_metadata?: {
        items_count?: number;
        total_tokens?: number;
        num_sub_batches?: number;
        is_parent?: boolean;
        [key: string]: any;
      } | null;
      child_operations?: ChildOperationStatus[] | null;
      task_payload?: Record<string, unknown> | null;
      error?: never; // Not present in success case
    }
  | {
      error: string; // Error state when loading fails
      operation_id?: never;
      status?: never;
      operation_type?: never;
      created_at?: never;
      updated_at?: never;
      completed_at?: never;
      error_message?: never;
      progress?: never;
      result_metadata?: never;
      child_operations?: never;
      task_payload?: never;
    };

const OPERATION_TYPE_VALUES = [
  "all",
  "retain",
  "consolidation",
  "refresh_mental_model",
  "file_convert_retain",
  "webhook_delivery",
  "graph_maintenance",
] as const;

const STATUS_FILTER_VALUES = [
  null,
  "pending",
  "processing",
  "completed",
  "failed",
  "cancelled",
] as const;

export function BankOperationsView() {
  const t = useTranslations("bankOperations");
  const { currentBank } = useBank();
  const [operations, setOperations] = useState<Operation[]>([]);
  const [totalOperations, setTotalOperations] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [taskTypeFilter, setTaskTypeFilter] = useState<string | null>(null);
  const [limit] = useState(10);
  const [offset, setOffset] = useState(0);
  const [cancellingOpId, setCancellingOpId] = useState<string | null>(null);
  const [retryingOpId, setRetryingOpId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState<OperationDetails | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [loadingPayload, setLoadingPayload] = useState(false);
  const [payloadLoadedFor, setPayloadLoadedFor] = useState<string | null>(null);
  // Ticks once a second so the "last heartbeat" relative time counts up live between
  // the 5s data polls — a heartbeat that keeps aging without the snapshot advancing is
  // the signal a job is stuck.
  const [nowMs, setNowMs] = useState(() => Date.now());

  const statusLabels: Record<string, string> = {
    all: t("status.all"),
    pending: t("status.pending"),
    processing: t("status.processing"),
    completed: t("status.completed"),
    failed: t("status.failed"),
    cancelled: t("status.cancelled"),
  };

  const operationTypeLabels: Record<string, string> = {
    all: t("operationType.all"),
    retain: t("operationType.retain"),
    consolidation: t("operationType.consolidation"),
    refresh_mental_model: t("operationType.refreshMentalModel"),
    file_convert_retain: t("operationType.fileConvertRetain"),
    webhook_delivery: t("operationType.webhookDelivery"),
    graph_maintenance: t("operationType.graphMaintenance"),
  };

  const formatStatus = (status: string | null | undefined) =>
    status ? (statusLabels[status] ?? status) : t("unknown");

  const formatOperationType = (operationType: string | null | undefined) =>
    operationType ? (operationTypeLabels[operationType] ?? operationType) : t("notAvailable");

  const renderStatusBadge = (status: string | null | undefined, title?: string | null) => {
    const label = formatStatus(status);

    if (status === "pending") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
          <Clock className="w-3 h-3" />
          {label}
        </span>
      );
    }

    if (status === "processing") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
          <Loader2 className="w-3 h-3 animate-spin" />
          {label}
        </span>
      );
    }

    if (status === "failed") {
      return (
        <span
          className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20"
          title={title ?? undefined}
        >
          <AlertCircle className="w-3 h-3" />
          {label}
        </span>
      );
    }

    if (status === "completed") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
          <CheckCircle className="w-3 h-3" />
          {label}
        </span>
      );
    }

    if (status === "cancelled") {
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-gray-500/10 text-gray-600 dark:text-gray-400 border border-gray-500/20">
          <Ban className="w-3 h-3" />
          {label}
        </span>
      );
    }

    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground border border-border">
        {label}
      </span>
    );
  };

  // Compact "time since the snapshot was written", recomputed against the 1s tick so it
  // counts up live. Seconds granularity (unlike the minute-level bankStats helper) because
  // a heartbeat that hasn't moved for even ~30s on an active job is already suspicious.
  const formatHeartbeat = (atIso: string): string => {
    const at = new Date(atIso).getTime();
    if (Number.isNaN(at)) return "";
    const secs = Math.max(0, Math.round((nowMs - at) / 1000));
    if (secs < 5) return t("heartbeat.justNow");
    if (secs < 60) return t("heartbeat.secondsAgo", { secs });
    if (secs < 3600) return t("heartbeat.minutesAgo", { mins: Math.floor(secs / 60) });
    return t("heartbeat.hoursAgo", { hours: Math.floor(secs / 3600) });
  };

  // Render the last-known progress snapshot for a running operation. The worker writes
  // it per LLM batch / sub-batch; a snapshot that keeps advancing (and a heartbeat that
  // stays fresh) means a healthy long-running job, a frozen one means it may be stuck.
  // `compact` keeps it to a single inline row for the table; the full form (with stage
  // label, counters and a labelled heartbeat) is used in the details dialog.
  const renderProgress = (
    progress: OperationProgress | null | undefined,
    opts?: { compact?: boolean }
  ) => {
    if (!progress) return null;
    const stageLabel = progress.stage.replace(/[._]/g, " ");
    const hasCounts =
      typeof progress.processed === "number" &&
      typeof progress.total === "number" &&
      progress.total > 0;
    const pct = hasCounts
      ? Math.min(100, Math.round((progress.processed! / progress.total!) * 100))
      : null;
    const heartbeat = progress.at ? formatHeartbeat(progress.at) : null;

    if (opts?.compact) {
      // Single line so the table row stays short: bar + count + heartbeat age. The stage
      // name and absolute timestamp move to the tooltip (and the details dialog).
      return (
        <div
          className="flex items-center gap-1.5 text-[11px] text-muted-foreground whitespace-nowrap"
          title={`${stageLabel}${progress.at ? ` — ${new Date(progress.at).toLocaleString()}` : ""}`}
        >
          {pct !== null && (
            <span className="h-1 w-14 shrink-0 rounded-full bg-muted overflow-hidden">
              <span
                className="block h-full rounded-full bg-blue-500/70"
                style={{ width: `${pct}%` }}
              />
            </span>
          )}
          {hasCounts && (
            <span className="font-mono">
              {progress.processed}/{progress.total}
            </span>
          )}
          {heartbeat && <span className="text-muted-foreground/70">· {heartbeat}</span>}
        </div>
      );
    }

    return (
      <div className="mt-1 space-y-1">
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span className="capitalize">{stageLabel}</span>
          {hasCounts && (
            <span className="font-mono">
              {progress.processed}/{progress.total}
            </span>
          )}
        </div>
        {pct !== null && (
          <div className="h-1 w-24 rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-blue-500/70" style={{ width: `${pct}%` }} />
          </div>
        )}
        {progress.at && (
          <div
            className="flex items-center gap-1 text-[10px] text-muted-foreground/80"
            title={new Date(progress.at).toLocaleString()}
          >
            <Clock className="w-2.5 h-2.5" />
            <span>
              {t("field.lastHeartbeat")} · {formatHeartbeat(progress.at)}
            </span>
          </div>
        )}
      </div>
    );
  };

  const loadOperations = useCallback(
    async (
      newStatusFilter: string | null = statusFilter,
      newOffset: number = offset,
      newTaskTypeFilter: string | null = taskTypeFilter
    ) => {
      if (!currentBank) return;

      setLoading(true);
      try {
        const opsData = await client.listOperations(currentBank, {
          status: newStatusFilter || undefined,
          type: newTaskTypeFilter || undefined,
          limit,
          offset: newOffset,
          excludeParents: true,
        });
        setOperations(opsData.operations || []);
        setTotalOperations(opsData.total || 0);
        // Refresh the clock on every poll so relative times (e.g. the "Updated" column)
        // stay accurate even when nothing is processing and the 1s ticker is idle.
        setNowMs(Date.now());
      } catch (error) {
        console.error("Error loading operations:", error);
      } finally {
        setLoading(false);
      }
    },
    [currentBank, statusFilter, offset, taskTypeFilter, limit]
  );

  const handleFilterChange = (newFilter: string | null) => {
    setStatusFilter(newFilter);
    setOffset(0);
    loadOperations(newFilter, 0, taskTypeFilter);
  };

  const handleTaskTypeFilterChange = (newTaskType: string | null) => {
    setTaskTypeFilter(newTaskType);
    setOffset(0);
    loadOperations(statusFilter, 0, newTaskType);
  };

  const handlePageChange = (newOffset: number) => {
    setOffset(newOffset);
    loadOperations(statusFilter, newOffset, taskTypeFilter);
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

  const handleRetryOperation = async (operationId: string) => {
    if (!currentBank) return;

    setRetryingOpId(operationId);
    try {
      await client.retryOperation(currentBank, operationId);
      await loadOperations();
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setRetryingOpId(null);
    }
  };

  const handleOperationClick = async (operationId: string) => {
    if (!currentBank) return;

    setLoadingDetails(true);
    setDialogOpen(true);
    setPayloadLoadedFor(null);
    try {
      const details = await client.getOperationStatus(currentBank, operationId);
      setSelectedOperation(details);
    } catch (error) {
      console.error("Error loading operation details:", error);
      setSelectedOperation({ error: t("operationDetailsLoadError") });
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleLoadRaw = async () => {
    if (!currentBank || !selectedOperation?.operation_id) return;

    setLoadingPayload(true);
    try {
      const opId = selectedOperation.operation_id;
      const details = await client.getOperationStatus(currentBank, opId, {
        includePayload: true,
      });
      setSelectedOperation(details);
      setPayloadLoadedFor(opId);
    } catch (error) {
      console.error("Error loading raw payload:", error);
    } finally {
      setLoadingPayload(false);
    }
  };

  // Poll faster while work is in flight so the progress bar / heartbeat feel live, and
  // back off to a calmer cadence when everything is terminal.
  const hasProcessing = operations.some((op) => op.status === "processing");
  useEffect(() => {
    if (currentBank) {
      loadOperations(statusFilter, offset, taskTypeFilter);
      const interval = setInterval(
        () => loadOperations(statusFilter, offset, taskTypeFilter),
        hasProcessing ? 2000 : 5000
      );
      return () => clearInterval(interval);
    }
  }, [currentBank, statusFilter, offset, taskTypeFilter, hasProcessing, loadOperations]);

  // Only tick the heartbeat clock while something is actually running — no point
  // re-rendering every second when every operation is in a terminal state.
  useEffect(() => {
    if (!hasProcessing) return;
    const tick = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [hasProcessing]);

  // Flash a row when it transitions into a terminal state, so a completion that lands
  // on a poll reads as a deliberate change rather than a silent badge swap. We diff the
  // status each load against the previous one and briefly mark the changed ids.
  const prevStatusRef = useRef<Record<string, string>>({});
  const [flashIds, setFlashIds] = useState<Set<string>>(new Set());
  useEffect(() => {
    const prev = prevStatusRef.current;
    const justFinished = operations
      .filter(
        (op) =>
          prev[op.id] &&
          prev[op.id] !== op.status &&
          (op.status === "completed" || op.status === "failed" || op.status === "cancelled")
      )
      .map((op) => op.id);
    prevStatusRef.current = Object.fromEntries(operations.map((op) => [op.id, op.status]));
    if (justFinished.length === 0) return;
    setFlashIds((s) => new Set([...s, ...justFinished]));
    const timer = setTimeout(
      () =>
        setFlashIds((s) => {
          const next = new Set(s);
          justFinished.forEach((id) => next.delete(id));
          return next;
        }),
      1200
    );
    return () => clearTimeout(timer);
  }, [operations]);

  if (!currentBank) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">{t("title")}</h3>
            <button
              onClick={() => loadOperations()}
              className="p-1 rounded hover:bg-muted transition-colors"
              title={t("refreshOperations")}
              disabled={loading}
            >
              <RefreshCw
                className={`w-4 h-4 text-muted-foreground hover:text-foreground ${loading ? "animate-spin" : ""}`}
              />
            </button>
          </div>
          <p className="text-sm text-muted-foreground">
            {t("operationCount", { count: totalOperations })}
            {statusFilter ? ` (${formatStatus(statusFilter)})` : ""}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select
            value={taskTypeFilter ?? "all"}
            onValueChange={(val) => handleTaskTypeFilterChange(val === "all" ? null : val)}
          >
            <SelectTrigger className="h-9 w-[180px] text-sm">
              <SelectValue placeholder={t("allTypes")} />
            </SelectTrigger>
            <SelectContent>
              {OPERATION_TYPE_VALUES.map((value) => (
                <SelectItem key={value} value={value}>
                  <div>
                    <div>{operationTypeLabels[value]}</div>
                    {value !== "all" && (
                      <div className="text-xs text-muted-foreground font-mono">{value}</div>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex gap-1 bg-muted p-1 rounded-lg">
            {STATUS_FILTER_VALUES.map((filter) => (
              <button
                key={filter ?? "all"}
                onClick={() => handleFilterChange(filter)}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  statusFilter === filter
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {filter ? formatStatus(filter) : statusLabels.all}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div>
        {operations.length > 0 ? (
          <>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">{t("table.id")}</TableHead>
                    <TableHead>{t("table.type")}</TableHead>
                    <TableHead>{t("table.created")}</TableHead>
                    <TableHead>{t("field.updated")}</TableHead>
                    {/* Fixed width so the row doesn't reflow when the inline progress
                        appears/disappears as an operation starts or finishes. */}
                    <TableHead className="w-[300px]">{t("table.status")}</TableHead>
                    {/* Fixed width + always-present label so the column doesn't grow
                        when a pending/failed row's Cancel/Retry button appears. */}
                    <TableHead className="w-[110px]">{t("table.actions")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {operations.map((op) => (
                    <TableRow
                      key={op.id}
                      className={`cursor-pointer transition-colors duration-700 hover:bg-muted/50 ${
                        flashIds.has(op.id)
                          ? op.status === "completed"
                            ? "bg-emerald-500/15"
                            : "bg-red-500/15"
                          : op.status === "failed"
                            ? "bg-red-500/5"
                            : ""
                      }`}
                      onClick={() => handleOperationClick(op.id)}
                    >
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {op.id.substring(0, 8)}
                      </TableCell>
                      <TableCell className="font-medium" title={op.task_type}>
                        {formatOperationType(op.task_type)}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {new Date(op.created_at).toLocaleString()}
                      </TableCell>
                      <TableCell
                        className="text-sm text-muted-foreground"
                        title={op.updated_at ? new Date(op.updated_at).toLocaleString() : undefined}
                      >
                        {op.updated_at ? formatHeartbeat(op.updated_at) : "—"}
                      </TableCell>
                      <TableCell className="w-[300px]">
                        <div className="flex items-center gap-2 whitespace-nowrap">
                          {renderStatusBadge(op.status, op.error_message)}
                          {op.status === "processing" &&
                            renderProgress(op.progress, { compact: true })}
                        </div>
                      </TableCell>
                      <TableCell className="w-[110px] whitespace-nowrap">
                        {op.status === "pending" && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCancelOperation(op.id);
                            }}
                            disabled={cancellingOpId === op.id}
                          >
                            {cancellingOpId === op.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <X className="w-3 h-3 mr-1" />
                            )}
                            {cancellingOpId === op.id ? "" : t("action.cancel")}
                          </Button>
                        )}
                        {(op.status === "failed" || op.status === "cancelled") && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs text-muted-foreground hover:text-blue-600 dark:hover:text-blue-400"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRetryOperation(op.id);
                            }}
                            disabled={retryingOpId === op.id}
                          >
                            {retryingOpId === op.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <RotateCcw className="w-3 h-3 mr-1" />
                            )}
                            {retryingOpId === op.id ? "" : t("action.retry")}
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {/* Pagination */}
            {totalOperations > limit && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <p className="text-sm text-muted-foreground">
                  {t("pagination", {
                    start: offset + 1,
                    end: Math.min(offset + limit, totalOperations),
                    total: totalOperations,
                  })}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(Math.max(0, offset - limit))}
                    disabled={offset === 0}
                  >
                    {t("paginationPrevious")}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(offset + limit)}
                    disabled={offset + limit >= totalOperations}
                  >
                    {t("paginationNext")}
                  </Button>
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-muted-foreground text-center py-8 text-sm">
            {statusFilter
              ? t("emptyStateWithStatus", { status: formatStatus(statusFilter) })
              : t("emptyState")}
          </p>
        )}
      </div>

      {/* Operation Details Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("operationDetails")}</DialogTitle>
            <DialogDescription>
              {selectedOperation?.operation_id && (
                <span className="font-mono text-xs">{selectedOperation.operation_id}</span>
              )}
            </DialogDescription>
          </DialogHeader>
          {loadingDetails ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : selectedOperation ? (
            <div className="space-y-4">
              {selectedOperation.error ? (
                <div className="text-red-600 dark:text-red-400">{selectedOperation.error}</div>
              ) : (
                <>
                  {/* Basic Info */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">
                        {t("field.status")}
                      </div>
                      <div className="mt-1">{renderStatusBadge(selectedOperation.status)}</div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">
                        {t("field.type")}
                      </div>
                      <div
                        className="mt-1 font-mono text-sm"
                        title={selectedOperation.operation_type ?? undefined}
                      >
                        {formatOperationType(selectedOperation.operation_type)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">
                        {t("field.created")}
                      </div>
                      <div className="mt-1 text-sm">
                        {selectedOperation.created_at
                          ? new Date(selectedOperation.created_at).toLocaleString()
                          : t("notAvailable")}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">
                        {t("field.updated")}
                      </div>
                      <div className="mt-1 text-sm">
                        {selectedOperation.updated_at
                          ? new Date(selectedOperation.updated_at).toLocaleString()
                          : t("notAvailable")}
                      </div>
                    </div>
                    {selectedOperation.completed_at && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">
                          {t("field.completed")}
                        </div>
                        <div className="mt-1 text-sm">
                          {new Date(selectedOperation.completed_at).toLocaleString()}
                        </div>
                      </div>
                    )}
                    {selectedOperation.result_metadata?.items_count !== undefined && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">
                          {t("totalItems")}
                        </div>
                        <div className="mt-1 text-sm">
                          {selectedOperation.result_metadata.items_count}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Progress snapshot — only meaningful while the operation is in
                      flight. For a terminal operation the status badge + completed_at are
                      the truth; a leftover heartbeat would misleadingly read as in-progress. */}
                  {selectedOperation.status === "processing" && selectedOperation.progress && (
                    <div>
                      <div className="text-sm font-medium text-muted-foreground mb-1">
                        {t("field.progress")}
                      </div>
                      {renderProgress(selectedOperation.progress)}
                      {selectedOperation.operation_type === "consolidation" && (
                        <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground/70">
                          {t("progressEstimateNote")}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Action buttons */}
                  {(selectedOperation.status === "pending" ||
                    selectedOperation.status === "failed" ||
                    selectedOperation.status === "cancelled") && (
                    <div className="flex gap-2">
                      {selectedOperation.status === "pending" && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs"
                          onClick={() => handleCancelOperation(selectedOperation.operation_id)}
                          disabled={cancellingOpId === selectedOperation.operation_id}
                        >
                          {cancellingOpId === selectedOperation.operation_id ? (
                            <Loader2 className="w-3 h-3 animate-spin mr-1" />
                          ) : (
                            <X className="w-3 h-3 mr-1" />
                          )}
                          {t("action.cancel")}
                        </Button>
                      )}
                      {(selectedOperation.status === "failed" ||
                        selectedOperation.status === "cancelled") && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs"
                          onClick={() => handleRetryOperation(selectedOperation.operation_id)}
                          disabled={retryingOpId === selectedOperation.operation_id}
                        >
                          {retryingOpId === selectedOperation.operation_id ? (
                            <Loader2 className="w-3 h-3 animate-spin mr-1" />
                          ) : (
                            <RotateCcw className="w-3 h-3 mr-1" />
                          )}
                          {t("action.retry")}
                        </Button>
                      )}
                    </div>
                  )}

                  {/* Metadata */}
                  {selectedOperation.result_metadata &&
                    Object.keys(selectedOperation.result_metadata).length > 0 && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground mb-2">
                          {t("metadata")}
                        </div>
                        <pre className="rounded-lg border bg-muted/30 p-3 text-xs font-mono overflow-x-auto max-h-96 whitespace-pre-wrap break-words">
                          {JSON.stringify(selectedOperation.result_metadata, null, 2)}
                        </pre>
                      </div>
                    )}

                  {/* Error Message */}
                  {selectedOperation.error_message && (
                    <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                      <div className="text-sm font-medium text-red-600 dark:text-red-400 mb-1">
                        {t("error")}
                      </div>
                      <div className="text-sm text-red-600/80 dark:text-red-400/80 font-mono">
                        {selectedOperation.error_message}
                      </div>
                    </div>
                  )}

                  {/* Child Operations (for parent operations) */}
                  {selectedOperation.child_operations &&
                    selectedOperation.child_operations.length > 0 && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground mb-2">
                          {t("subBatchesCount", {
                            count:
                              selectedOperation.result_metadata?.num_sub_batches ||
                              selectedOperation.child_operations.length,
                          })}
                        </div>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead className="w-[60px]">{t("table.index")}</TableHead>
                              <TableHead className="w-[100px]">{t("table.id")}</TableHead>
                              <TableHead className="w-[80px]">{t("table.items")}</TableHead>
                              <TableHead>{t("table.status")}</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {selectedOperation.child_operations.map((child) => (
                              <TableRow key={child.operation_id}>
                                <TableCell className="text-sm">{child.sub_batch_index}</TableCell>
                                <TableCell className="font-mono text-xs text-muted-foreground">
                                  {child.operation_id.substring(0, 8)}
                                </TableCell>
                                <TableCell className="text-sm">{child.items_count}</TableCell>
                                <TableCell>
                                  {renderStatusBadge(child.status, child.error_message)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    )}

                  {/* Raw payload */}
                  {(() => {
                    const loadedThisOp = payloadLoadedFor === selectedOperation.operation_id;
                    const hasPayload = !!selectedOperation.task_payload;
                    const isParent = !!selectedOperation.result_metadata?.is_parent;
                    return (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <div className="text-sm font-medium text-muted-foreground">
                            {t("rawPayload")}
                          </div>
                          {!loadedThisOp && (
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 text-xs"
                              onClick={handleLoadRaw}
                              disabled={loadingPayload}
                            >
                              {loadingPayload ? (
                                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                              ) : (
                                <Code className="w-3 h-3 mr-1" />
                              )}
                              {t("loadRaw")}
                            </Button>
                          )}
                        </div>
                        {hasPayload ? (
                          <pre className="rounded-lg border bg-muted/30 p-3 text-xs font-mono overflow-x-auto max-h-96 whitespace-pre-wrap break-words">
                            {JSON.stringify(selectedOperation.task_payload, null, 2)}
                          </pre>
                        ) : loadedThisOp ? (
                          <p className="text-xs text-muted-foreground">
                            {isParent ? t("rawPayloadParentHelp") : t("rawPayloadEmpty")}
                          </p>
                        ) : (
                          <p className="text-xs text-muted-foreground">{t("rawPayloadHelp")}</p>
                        )}
                      </div>
                    );
                  })()}
                </>
              )}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
