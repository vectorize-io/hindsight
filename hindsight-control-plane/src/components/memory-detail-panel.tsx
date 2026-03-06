"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { TagList } from "@/components/ui/tag-list";
import { Copy, Check, X, Loader2, Calendar } from "lucide-react";
import { DocumentChunkModal } from "./document-chunk-modal";
import { MemoryDetailModal } from "./memory-detail-modal";
import { client } from "@/lib/api";
import { ObservationHistoryView } from "@/components/observation-history-view";

interface MemoryDetailPanelProps {
  memory: any;
  onClose: () => void;
  compact?: boolean;
  inPanel?: boolean;
  bankId?: string;
}

export function MemoryDetailPanel({
  memory,
  onClose,
  compact = false,
  inPanel = false,
  bankId,
}: MemoryDetailPanelProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [modalType, setModalType] = useState<"document" | "chunk" | null>(null);
  const [modalId, setModalId] = useState<string | null>(null);
  const [fullMemory, setFullMemory] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [sourceMemoryModalId, setSourceMemoryModalId] = useState<string | null>(null);

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
    if (factType === "observation") return "Observation";
    if (factType === "world") return "World Fact";
    if (factType === "experience") return "Experience";
    return "Memory Details";
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
              <span className="ml-2 text-muted-foreground">Loading memory details...</span>
            </div>
          ) : (
            <div className="space-y-5">
              {/* Text */}
              <div>
                <div className="text-xs font-bold text-muted-foreground uppercase mb-2">Text</div>
                <div className="text-sm whitespace-pre-wrap leading-relaxed text-foreground">
                  {displayMemory.text}
                </div>
              </div>

              {/* Context (not shown for observations) */}
              {displayMemory.context && !isObservation && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Context
                  </div>
                  <div className="text-sm text-foreground">{displayMemory.context}</div>
                </div>
              )}

              {/* Dates */}
              {displayMemory.occurred_start && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Occurred
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
                    Mentioned
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
                      Entities
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
                    Source Memories ({displayMemory.source_memories.length})
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
                            View
                          </Button>
                        </div>
                        <p className="text-sm text-foreground mb-3">{source.text}</p>
                        {source.context && (
                          <p className="text-xs text-muted-foreground mb-3 italic">
                            Context: {source.context}
                          </p>
                        )}
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="p-2 bg-background/50 rounded">
                            <div className="text-muted-foreground mb-0.5">Occurred</div>
                            <div className="font-medium">
                              {source.occurred_start
                                ? new Date(source.occurred_start).toLocaleString()
                                : "N/A"}
                            </div>
                          </div>
                          <div className="p-2 bg-background/50 rounded">
                            <div className="text-muted-foreground mb-0.5">Mentioned</div>
                            <div className="font-medium">
                              {source.mentioned_at
                                ? new Date(source.mentioned_at).toLocaleString()
                                : "N/A"}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* History (observations only) */}
              {isObservation && (
                <div className="border-t border-border pt-5">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                    History
                    {displayMemory.history?.length > 0 ? ` (${displayMemory.history.length})` : ""}
                  </div>
                  {displayMemory.history && displayMemory.history.length > 0 ? (
                    <ObservationHistoryView
                      history={displayMemory.history}
                      current={{
                        text: displayMemory.text,
                        tags: displayMemory.tags || [],
                        occurred_start: displayMemory.occurred_start || null,
                        occurred_end: displayMemory.occurred_end || null,
                        mentioned_at: displayMemory.mentioned_at || null,
                      }}
                    />
                  ) : (
                    <p className="text-sm text-muted-foreground italic">No history recorded yet.</p>
                  )}
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
                      View Document
                    </Button>
                  )}
                  {displayMemory.chunk_id && (
                    <Button
                      onClick={() => openChunkModal(displayMemory.chunk_id)}
                      variant="secondary"
                      className="flex-1"
                    >
                      View Chunk
                    </Button>
                  )}
                </div>
              )}

              {/* Memory ID */}
              {memoryId && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Memory ID
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
            <span className="ml-2 text-sm text-muted-foreground">Loading...</span>
          </div>
        ) : (
          <div className={gap}>
            {/* Text */}
            <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
              <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                Text
              </div>
              <div className={`${textSize} whitespace-pre-wrap`}>{displayMemory.text}</div>
            </div>

            {/* Context */}
            {displayMemory.context && (
              <div>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Context
                </div>
                <div className={textSize}>{displayMemory.context}</div>
              </div>
            )}

            {/* Dates */}
            {displayMemory.occurred_start && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Occurred
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
                  Mentioned
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
                    Entities
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
                  Tags
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
                    View Document
                  </Button>
                )}
                {displayMemory.chunk_id && (
                  <Button
                    onClick={() => openChunkModal(displayMemory.chunk_id)}
                    size="sm"
                    variant="secondary"
                    className={`flex-1 ${compact ? "h-7 text-xs" : ""}`}
                  >
                    View Chunk
                  </Button>
                )}
              </div>
            )}

            {/* Source Memories (for mental models) */}
            {displayMemory.source_memories && displayMemory.source_memories.length > 0 && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-2`}>
                  Source Memories ({displayMemory.source_memories.length})
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
                          View
                        </Button>
                      </div>
                      <p className={`${textSize} mb-1`}>{source.text}</p>
                      {source.context && (
                        <p className="text-[10px] text-muted-foreground italic">
                          Context: {source.context}
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
                  Memory ID
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
