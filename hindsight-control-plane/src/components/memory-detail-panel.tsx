"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { TagList } from "@/components/ui/tag-list";
import { Copy, Check, X, Loader2, Calendar, History, Activity } from "lucide-react";
import { DocumentChunkModal } from "./document-chunk-modal";
import { MemoryDetailModal } from "./memory-detail-modal";
import { TraceDialog } from "./llm-requests-view";
import { client, LLMRequestEntry } from "@/lib/api";

interface MemoryDetailPanelProps {
  memory: any;
  onClose: () => void;
  compact?: boolean;
  inPanel?: boolean;
  bankId?: string;
}

interface TraceRun {
  traceId: string;
  entry: LLMRequestEntry;
  calls: number;
  tokens: number;
  status: string;
  start: string | null;
  // "created": this run produced the memory (it's in metadata.memory_ids);
  // "used": this run consumed it as a consolidation source.
  relation: "created" | "used";
}

function metaHasId(
  metadata: Record<string, unknown> | null | undefined,
  key: string,
  id: string
): boolean {
  const raw = metadata?.[key];
  return Array.isArray(raw) && raw.includes(id);
}

// The operation runs touching this memory: the one that produced it ("Created
// by" — retain for facts, consolidation for observations) and the consolidation
// runs that consumed it as a source ("Used by"). Hidden when tracing is disabled
// or nothing was recorded.
function MemoryTraceRef({ bankId, memoryId }: { bankId: string; memoryId: string }) {
  const t = useTranslations("memoryDetailPanel");
  const [runs, setRuns] = useState<TraceRun[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [dialogEntry, setDialogEntry] = useState<LLMRequestEntry | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await client.listLLMRequests(bankId, {
          memory_id: memoryId,
          group: true,
          limit: 50,
        });
        if (cancelled) return;
        const byTrace = new Map<string, LLMRequestEntry[]>();
        for (const it of data.items || []) {
          const key = it.trace_id || it.id;
          if (!byTrace.has(key)) byTrace.set(key, []);
          byTrace.get(key)!.push(it);
        }
        const list: TraceRun[] = [...byTrace.entries()].map(([traceId, rows]) => ({
          traceId,
          entry: rows[0],
          calls: rows.length,
          tokens: rows.reduce((s, r) => s + (r.total_tokens ?? 0), 0),
          status: rows.some((r) => r.status === "error") ? "error" : "success",
          start:
            rows
              .map((r) => r.started_at)
              .filter(Boolean)
              .sort()[0] ?? null,
          // Produced this memory if it's in memory_ids; otherwise it was consumed
          // as a source (the filter only returns runs matching one of the two).
          relation: metaHasId(rows[0].metadata, "memory_ids", memoryId) ? "created" : "used",
        }));
        list.sort((a, b) => (b.start || "").localeCompare(a.start || ""));
        setRuns(list);
      } catch {
        // Tracing may be disabled or the endpoint unavailable — stay hidden.
      } finally {
        if (!cancelled) setLoaded(true);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [bankId, memoryId]);

  if (!loaded || runs.length === 0) return null;

  const createdRuns = runs.filter((r) => r.relation === "created");
  const usedRuns = runs.filter((r) => r.relation === "used");

  const renderRun = (run: TraceRun) => (
    <button
      key={run.traceId}
      type="button"
      onClick={() => setDialogEntry(run.entry)}
      className="w-full flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-sm text-left hover:bg-muted/50"
    >
      <span className="inline-flex items-center gap-2 min-w-0">
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${run.status === "error" ? "bg-red-500" : "bg-green-500"}`}
        />
        <span className="font-mono text-xs">{run.entry.operation || "—"}</span>
        <span className="text-muted-foreground text-xs truncate">
          {run.start ? new Date(run.start).toLocaleString() : ""}
        </span>
      </span>
      <span className="text-muted-foreground text-xs font-mono shrink-0">
        {t("tracedBySummary", { calls: run.calls, tokens: run.tokens.toLocaleString() })}
      </span>
    </button>
  );

  return (
    <div className="border-t border-border pt-5 space-y-4">
      {createdRuns.length > 0 && (
        <div>
          <div className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase mb-3">
            <Activity className="h-3.5 w-3.5" />
            {t("tracedByTitle")}
          </div>
          <div className="space-y-1">{createdRuns.map(renderRun)}</div>
        </div>
      )}
      {usedRuns.length > 0 && (
        <div>
          <div className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase mb-3">
            <Activity className="h-3.5 w-3.5" />
            {t("consolidatedByTitle")}
          </div>
          <div className="space-y-1">{usedRuns.map(renderRun)}</div>
        </div>
      )}
      <TraceDialog
        bankId={bankId}
        entry={dialogEntry}
        open={!!dialogEntry}
        onOpenChange={(o) => !o && setDialogEntry(null)}
      />
    </div>
  );
}

export function MemoryDetailPanel({
  memory,
  onClose,
  compact = false,
  inPanel = false,
  bankId,
}: MemoryDetailPanelProps) {
  const t = useTranslations("memoryDetailPanel");
  const tModal = useTranslations("memoryDetailModal");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [modalType, setModalType] = useState<"document" | "chunk" | null>(null);
  const [modalId, setModalId] = useState<string | null>(null);
  const [fullMemory, setFullMemory] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [sourceMemoryModalId, setSourceMemoryModalId] = useState<string | null>(null);
  const [historyModalOpen, setHistoryModalOpen] = useState(false);

  // Fetch full memory data when panel opens
  // For mental models, use getMentalModel to get source memories
  useEffect(() => {
    const memoryId = memory?.id || memory?.node_id;
    if (!memoryId || !bankId) {
      setFullMemory(null);
      return;
    }

    setLoading(true);

    // Use getMemory for all memory types - it now returns source_memories for mental models
    client
      .getMemory(memoryId, bankId)
      .then((data) => {
        setFullMemory(data);
      })
      .catch((err) => {
        console.error("Failed to fetch memory details:", err);
        // Fall back to showing the partial data we have
        setFullMemory(null);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [memory?.id, memory?.node_id, memory?.fact_type, memory?.type, bankId]);

  // Use full memory data if available, otherwise fall back to the partial data passed in
  const displayMemory = fullMemory || memory;
  const isObservation =
    displayMemory?.fact_type === "observation" || displayMemory?.type === "observation";

  // Determine the display title based on memory type
  const getMemoryTypeTitle = () => {
    const factType = displayMemory?.fact_type || displayMemory?.type;
    if (factType === "observation") return tModal("typeObservation");
    if (factType === "world") return tModal("typeWorldFact");
    if (factType === "experience") return tModal("typeExperience");
    return t("title");
  };
  const memoryTypeTitle = getMemoryTypeTitle();

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(text);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const openDocumentModal = (docId: string) => {
    setModalType("document");
    setModalId(docId);
  };

  const openChunkModal = (chunkId: string) => {
    setModalType("chunk");
    setModalId(chunkId);
  };

  const closeModal = () => {
    setModalType(null);
    setModalId(null);
  };

  if (!memory) return null;

  // Handle both 'id' and 'node_id' (trace results use node_id)
  const memoryId = displayMemory.id || displayMemory.node_id;

  const labelSize = compact ? "text-[10px]" : "text-xs";
  const textSize = compact ? "text-xs" : "text-sm";

  // Panel mode: no outer border/bg, larger padding, prominent close button
  if (inPanel) {
    return (
      <>
        <div className="p-5">
          {/* Header with close button */}
          <div className="flex justify-between items-center mb-6 pb-4 border-b border-border">
            <h3 className="text-xl font-bold text-foreground">{memoryTypeTitle}</h3>
            <Button variant="secondary" size="sm" onClick={onClose} className="h-8 w-8 p-0">
              <X className="h-5 w-5" />
            </Button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">{t("loadingDetails")}</span>
            </div>
          ) : (
            <div className="space-y-5">
              {/* Text */}
              <div>
                <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                  {t("sectionFullText")}
                </div>
                <div className="text-sm whitespace-pre-wrap leading-relaxed text-foreground">
                  {displayMemory.text}
                </div>
              </div>

              {/* Context (not shown for observations) */}
              {displayMemory.context && !isObservation && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    {t("sectionContext")}
                  </div>
                  <div className="text-sm text-foreground">{displayMemory.context}</div>
                </div>
              )}

              {/* Dates */}
              {displayMemory.occurred_start && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    {t("sectionOccurred")}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-foreground">
                    <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    <span>
                      {new Date(displayMemory.occurred_start).toLocaleString()}
                      {displayMemory.occurred_end &&
                        displayMemory.occurred_end !== displayMemory.occurred_start && (
                          <>
                            <span className="text-muted-foreground mx-1">→</span>
                            {new Date(displayMemory.occurred_end).toLocaleString()}
                          </>
                        )}
                    </span>
                  </div>
                </div>
              )}

              {displayMemory.mentioned_at && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    {t("sectionMentioned")}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-foreground">
                    <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    <span>{new Date(displayMemory.mentioned_at).toLocaleString()}</span>
                  </div>
                </div>
              )}

              {/* Entities */}
              {displayMemory.entities &&
                (Array.isArray(displayMemory.entities)
                  ? displayMemory.entities.length > 0
                  : displayMemory.entities) && (
                  <div>
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                      {t("sectionEntities")}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {(Array.isArray(displayMemory.entities)
                        ? displayMemory.entities
                        : String(displayMemory.entities).split(", ")
                      ).map((entity: any, i: number) => {
                        const entityText =
                          typeof entity === "string"
                            ? entity
                            : entity?.name || JSON.stringify(entity);
                        return (
                          <span
                            key={i}
                            className="text-sm px-3 py-1.5 rounded-full bg-primary/10 text-primary font-medium"
                          >
                            {entityText}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

              {/* Tags */}
              <TagList tags={displayMemory.tags} size="md" showLabel />

              {/* Source Memories (for observations) */}
              {displayMemory.source_memories && displayMemory.source_memories.length > 0 && (
                <div className="border-t border-border pt-5">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                    {t("sectionSourceMemories", { count: displayMemory.source_memories.length })}
                  </div>
                  <div className="space-y-3">
                    {displayMemory.source_memories.map((source: any, i: number) => (
                      <div
                        key={source.id || i}
                        className="p-4 bg-muted/50 rounded-lg border border-border/50"
                      >
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <span
                            className={`px-2 py-0.5 rounded text-xs flex-shrink-0 ${
                              source.type === "experience"
                                ? "bg-green-500/10 text-green-600"
                                : "bg-blue-500/10 text-blue-600"
                            }`}
                          >
                            {source.type}
                          </span>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-xs"
                            onClick={() => setSourceMemoryModalId(source.id)}
                          >
                            {t("sourceViewButton")}
                          </Button>
                        </div>
                        <p className="text-sm text-foreground mb-3">{source.text}</p>
                        {source.context && (
                          <p className="text-xs text-muted-foreground mb-3 italic">
                            {t("sourceContextPrefix", { context: source.context })}
                          </p>
                        )}
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="p-2 bg-background/50 rounded">
                            <div className="text-muted-foreground mb-0.5">
                              {t("sourceOccurred")}
                            </div>
                            <div className="font-medium">
                              {source.occurred_start
                                ? new Date(source.occurred_start).toLocaleString()
                                : t("notAvailable")}
                            </div>
                          </div>
                          <div className="p-2 bg-background/50 rounded">
                            <div className="text-muted-foreground mb-0.5">
                              {t("sourceMentioned")}
                            </div>
                            <div className="font-medium">
                              {source.mentioned_at
                                ? new Date(source.mentioned_at).toLocaleString()
                                : t("notAvailable")}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Document/Chunk buttons */}
              {(displayMemory.document_id || displayMemory.chunk_id) && (
                <div className="flex gap-3 pt-2">
                  {displayMemory.document_id && (
                    <Button
                      onClick={() => openDocumentModal(displayMemory.document_id)}
                      variant="secondary"
                      className="flex-1"
                    >
                      {t("viewDocumentButton")}
                    </Button>
                  )}
                  {displayMemory.chunk_id && (
                    <Button
                      onClick={() => openChunkModal(displayMemory.chunk_id)}
                      variant="secondary"
                      className="flex-1"
                    >
                      {t("viewChunkButton")}
                    </Button>
                  )}
                </div>
              )}

              {/* View History button (observations only) */}
              {isObservation && (
                <div className="border-t border-border pt-5">
                  <Button
                    variant="outline"
                    className="w-full flex items-center gap-2"
                    onClick={() => setHistoryModalOpen(true)}
                  >
                    <History className="h-4 w-4" />
                    {t("viewHistory")}
                  </Button>
                </div>
              )}

              {/* Memory ID */}
              {memoryId && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    {t("sectionMemoryId")}
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="text-xs font-mono text-muted-foreground">{memoryId}</code>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-5 w-5 p-0"
                      onClick={() => copyToClipboard(memoryId)}
                    >
                      {copiedId === memoryId ? (
                        <Check className="h-3 w-3 text-green-600" />
                      ) : (
                        <Copy className="h-3 w-3 text-muted-foreground" />
                      )}
                    </Button>
                  </div>
                </div>
              )}

              {/* Producing trace (retain/consolidation that created this memory) */}
              {memoryId && bankId && <MemoryTraceRef bankId={bankId} memoryId={memoryId} />}
            </div>
          )}
        </div>

        {/* Document/Chunk Modal */}
        {modalType && modalId && (
          <DocumentChunkModal type={modalType} id={modalId} onClose={closeModal} />
        )}

        {/* Source Memory Modal */}
        <MemoryDetailModal
          memoryId={sourceMemoryModalId}
          onClose={() => setSourceMemoryModalId(null)}
        />

        {/* History Modal */}
        {historyModalOpen && memoryId && bankId && (
          <MemoryDetailModal
            memoryId={memoryId}
            onClose={() => setHistoryModalOpen(false)}
            initialTab="history"
          />
        )}
      </>
    );
  }

  // Original compact/default mode
  const padding = compact ? "p-3" : "p-4";
  const titleSize = compact ? "text-sm" : "text-lg";
  const gap = compact ? "space-y-2" : "space-y-4";

  return (
    <>
      <div
        className={`bg-card border-2 border-primary rounded-lg ${padding} sticky top-4 max-h-[calc(100vh-120px)] overflow-y-auto`}
      >
        <div className="flex justify-between items-start mb-4">
          <h3 className={`${titleSize} font-bold text-card-foreground`}>{memoryTypeTitle}</h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className={compact ? "h-6 w-6 p-0" : "h-8 w-8 p-0"}
          >
            <X className={compact ? "h-3 w-3" : "h-4 w-4"} />
          </Button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">{t("loading")}</span>
          </div>
        ) : (
          <div className={gap}>
            {/* Text */}
            <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
              <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                {t("sectionFullText")}
              </div>
              <div className={`${textSize} whitespace-pre-wrap`}>{displayMemory.text}</div>
            </div>

            {/* Context */}
            {displayMemory.context && (
              <div>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  {t("sectionContext")}
                </div>
                <div className={textSize}>{displayMemory.context}</div>
              </div>
            )}

            {/* Dates */}
            {displayMemory.occurred_start && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  {t("sectionOccurred")}
                </div>
                <div className={`flex items-center gap-2 ${textSize}`}>
                  <Calendar
                    className={`${compact ? "h-3 w-3" : "h-4 w-4"} text-muted-foreground flex-shrink-0`}
                  />
                  <span>
                    {new Date(displayMemory.occurred_start).toLocaleString()}
                    {displayMemory.occurred_end &&
                      displayMemory.occurred_end !== displayMemory.occurred_start && (
                        <>
                          <span className="text-muted-foreground mx-1">→</span>
                          {new Date(displayMemory.occurred_end).toLocaleString()}
                        </>
                      )}
                  </span>
                </div>
              </div>
            )}

            {displayMemory.mentioned_at && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  {t("sectionMentioned")}
                </div>
                <div className={`flex items-center gap-2 ${textSize}`}>
                  <Calendar
                    className={`${compact ? "h-3 w-3" : "h-4 w-4"} text-muted-foreground flex-shrink-0`}
                  />
                  <span>{new Date(displayMemory.mentioned_at).toLocaleString()}</span>
                </div>
              </div>
            )}

            {/* Entities */}
            {displayMemory.entities &&
              (Array.isArray(displayMemory.entities)
                ? displayMemory.entities.length > 0
                : displayMemory.entities) && (
                <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                  <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-2`}>
                    {t("sectionEntities")}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {(Array.isArray(displayMemory.entities)
                      ? displayMemory.entities
                      : String(displayMemory.entities).split(", ")
                    ).map((entity: any, i: number) => {
                      const entityText =
                        typeof entity === "string"
                          ? entity
                          : entity?.name || JSON.stringify(entity);
                      return (
                        <span
                          key={i}
                          className={`${compact ? "text-[10px] px-1.5 py-0.5" : "text-xs px-2 py-1"} rounded bg-secondary text-secondary-foreground`}
                        >
                          {entityText}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

            {/* Tags */}
            {displayMemory.tags && displayMemory.tags.length > 0 && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-2`}>
                  {t("sectionTags")}
                </div>
                <TagList tags={displayMemory.tags} size={compact ? "xs" : "sm"} />
              </div>
            )}

            {/* Document/Chunk buttons */}
            {(displayMemory.document_id || displayMemory.chunk_id) && (
              <div className={`flex gap-2 ${compact ? "pt-1" : ""}`}>
                {displayMemory.document_id && (
                  <Button
                    onClick={() => openDocumentModal(displayMemory.document_id)}
                    size="sm"
                    variant="secondary"
                    className={`flex-1 ${compact ? "h-7 text-xs" : ""}`}
                  >
                    {t("viewDocumentButton")}
                  </Button>
                )}
                {displayMemory.chunk_id && (
                  <Button
                    onClick={() => openChunkModal(displayMemory.chunk_id)}
                    size="sm"
                    variant="secondary"
                    className={`flex-1 ${compact ? "h-7 text-xs" : ""}`}
                  >
                    {t("viewChunkButton")}
                  </Button>
                )}
              </div>
            )}

            {/* Source Memories (for mental models) */}
            {displayMemory.source_memories && displayMemory.source_memories.length > 0 && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-2`}>
                  {t("sectionSourceMemories", { count: displayMemory.source_memories.length })}
                </div>
                <div className="space-y-2">
                  {displayMemory.source_memories.map((source: any, i: number) => (
                    <div
                      key={source.id || i}
                      className="p-2 bg-background/50 rounded border border-border/50"
                    >
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] flex-shrink-0 ${
                            source.type === "experience"
                              ? "bg-green-500/10 text-green-600"
                              : "bg-blue-500/10 text-blue-600"
                          }`}
                        >
                          {source.type}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-5 text-[10px] px-2"
                          onClick={() => setSourceMemoryModalId(source.id)}
                        >
                          {t("sourceViewButton")}
                        </Button>
                      </div>
                      <p className={`${textSize} mb-1`}>{source.text}</p>
                      {source.context && (
                        <p className="text-[10px] text-muted-foreground italic">
                          {t("sourceContextPrefix", { context: source.context })}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Memory ID */}
            {memoryId && (
              <div>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  {t("sectionMemoryId")}
                </div>
                <div className="flex items-center gap-2">
                  <code
                    className={`${compact ? "text-[9px]" : "text-xs"} font-mono text-muted-foreground`}
                  >
                    {memoryId}
                  </code>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`${compact ? "h-4 w-4" : "h-5 w-5"} p-0`}
                    onClick={() => copyToClipboard(memoryId)}
                  >
                    {copiedId === memoryId ? (
                      <Check className={`${compact ? "h-2.5 w-2.5" : "h-3 w-3"} text-green-600`} />
                    ) : (
                      <Copy
                        className={`${compact ? "h-2.5 w-2.5" : "h-3 w-3"} text-muted-foreground`}
                      />
                    )}
                  </Button>
                </div>
              </div>
            )}

            {/* Producing trace (retain/consolidation that created this memory) */}
            {memoryId && bankId && <MemoryTraceRef bankId={bankId} memoryId={memoryId} />}
          </div>
        )}
      </div>

      {/* Document/Chunk Modal */}
      {modalType && modalId && (
        <DocumentChunkModal type={modalType} id={modalId} onClose={closeModal} />
      )}

      {/* Source Memory Modal */}
      <MemoryDetailModal
        memoryId={sourceMemoryModalId}
        onClose={() => setSourceMemoryModalId(null)}
      />
    </>
  );
}
