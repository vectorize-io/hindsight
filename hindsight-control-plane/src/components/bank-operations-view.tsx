"use client";

import { useState, useEffect } from "react";
import { useBank } from "@/lib/bank-context";
import { client } from "@/lib/api";
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
import { RefreshCw, Clock, AlertCircle, CheckCircle, Loader2, X } from "lucide-react";

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  document_id: string | null;
  created_at: string;
  status: string;
  error_message: string | null;
}

export function BankOperationsView() {
  const { currentBank } = useBank();
  const [operations, setOperations] = useState<Operation[]>([]);
  const [totalOperations, setTotalOperations] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [limit] = useState(10);
  const [offset, setOffset] = useState(0);
  const [cancellingOpId, setCancellingOpId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState<any | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const loadOperations = async (
    newStatusFilter: string | null = statusFilter,
    newOffset: number = offset
  ) => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const opsData = await client.listOperations(currentBank, {
        status: newStatusFilter || undefined,
        limit,
        offset: newOffset,
      });
      setOperations(opsData.operations || []);
      setTotalOperations(opsData.total || 0);
    } catch (error) {
      console.error("Error loading operations:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (newFilter: string | null) => {
    setStatusFilter(newFilter);
    setOffset(0);
    loadOperations(newFilter, 0);
  };

  const handlePageChange = (newOffset: number) => {
    setOffset(newOffset);
    loadOperations(statusFilter, newOffset);
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

  const handleOperationClick = async (operationId: string) => {
    if (!currentBank) return;

    setLoadingDetails(true);
    setDialogOpen(true);
    try {
      const details = await client.getOperationStatus(currentBank, operationId);
      setSelectedOperation(details);
    } catch (error) {
      console.error("Error loading operation details:", error);
      setSelectedOperation({ error: "Failed to load operation details" });
    } finally {
      setLoadingDetails(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadOperations();
      // Refresh operations every 5 seconds
      const interval = setInterval(() => loadOperations(), 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank]);

  if (!currentBank) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Background Operations</h3>
            <button
              onClick={() => loadOperations()}
              className="p-1 rounded hover:bg-muted transition-colors"
              title="Refresh operations"
              disabled={loading}
            >
              <RefreshCw
                className={`w-4 h-4 text-muted-foreground hover:text-foreground ${loading ? "animate-spin" : ""}`}
              />
            </button>
          </div>
          <p className="text-sm text-muted-foreground">
            {totalOperations} operation{totalOperations !== 1 ? "s" : ""}
            {statusFilter ? ` (${statusFilter})` : ""}
          </p>
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
              onClick={() => handleFilterChange(filter.value)}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                statusFilter === filter.value
                  ? "bg-background shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>
      <div>
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
                      className={`cursor-pointer hover:bg-muted/50 ${op.status === "failed" ? "bg-red-500/5" : ""}`}
                      onClick={() => handleOperationClick(op.id)}
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
            {totalOperations > limit && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <p className="text-sm text-muted-foreground">
                  Showing {offset + 1}-{Math.min(offset + limit, totalOperations)} of{" "}
                  {totalOperations}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(Math.max(0, offset - limit))}
                    disabled={offset === 0}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(offset + limit)}
                    disabled={offset + limit >= totalOperations}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-muted-foreground text-center py-8 text-sm">
            No {statusFilter ? `${statusFilter} ` : ""}operations
          </p>
        )}
      </div>

      {/* Operation Details Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Operation Details</DialogTitle>
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
                      <div className="text-sm font-medium text-muted-foreground">Status</div>
                      <div className="mt-1">
                        {selectedOperation.status === "pending" && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                            <Clock className="w-3 h-3" />
                            pending
                          </span>
                        )}
                        {selectedOperation.status === "failed" && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20">
                            <AlertCircle className="w-3 h-3" />
                            failed
                          </span>
                        )}
                        {selectedOperation.status === "completed" && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                            <CheckCircle className="w-3 h-3" />
                            completed
                          </span>
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">Type</div>
                      <div className="mt-1 font-mono text-sm">
                        {selectedOperation.operation_type}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">Created</div>
                      <div className="mt-1 text-sm">
                        {selectedOperation.created_at
                          ? new Date(selectedOperation.created_at).toLocaleString()
                          : "N/A"}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">Updated</div>
                      <div className="mt-1 text-sm">
                        {selectedOperation.updated_at
                          ? new Date(selectedOperation.updated_at).toLocaleString()
                          : "N/A"}
                      </div>
                    </div>
                    {selectedOperation.completed_at && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">Completed</div>
                        <div className="mt-1 text-sm">
                          {new Date(selectedOperation.completed_at).toLocaleString()}
                        </div>
                      </div>
                    )}
                    {selectedOperation.result_metadata?.items_count !== undefined && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">Total Items</div>
                        <div className="mt-1 text-sm">
                          {selectedOperation.result_metadata.items_count}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Error Message */}
                  {selectedOperation.error_message && (
                    <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                      <div className="text-sm font-medium text-red-600 dark:text-red-400 mb-1">
                        Error
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
                          Sub-batches (
                          {selectedOperation.result_metadata?.num_sub_batches ||
                            selectedOperation.child_operations.length}
                          )
                        </div>
                        <div className="rounded-lg border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="w-[60px]">Index</TableHead>
                                <TableHead className="w-[100px]">ID</TableHead>
                                <TableHead className="w-[80px]">Items</TableHead>
                                <TableHead>Status</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {selectedOperation.child_operations.map((child: any) => (
                                <TableRow key={child.operation_id}>
                                  <TableCell className="text-sm">{child.sub_batch_index}</TableCell>
                                  <TableCell className="font-mono text-xs text-muted-foreground">
                                    {child.operation_id.substring(0, 8)}
                                  </TableCell>
                                  <TableCell className="text-sm">{child.items_count}</TableCell>
                                  <TableCell>
                                    {child.status === "pending" && (
                                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400">
                                        <Clock className="w-3 h-3" />
                                        pending
                                      </span>
                                    )}
                                    {child.status === "failed" && (
                                      <span
                                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400"
                                        title={child.error_message ?? undefined}
                                      >
                                        <AlertCircle className="w-3 h-3" />
                                        failed
                                      </span>
                                    )}
                                    {child.status === "completed" && (
                                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                                        <CheckCircle className="w-3 h-3" />
                                        completed
                                      </span>
                                    )}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    )}
                </>
              )}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
