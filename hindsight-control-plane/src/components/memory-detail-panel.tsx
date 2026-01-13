"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Copy, Check, X, Loader2 } from "lucide-react";
import { DocumentChunkModal } from "./document-chunk-modal";
import { client } from "@/lib/api";

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

  // Fetch full memory data when panel opens
  useEffect(() => {
    const memoryId = memory?.id || memory?.node_id;
    if (!memoryId || !bankId) {
      setFullMemory(null);
      return;
    }

    setLoading(true);
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
  }, [memory?.id, memory?.node_id, bankId]);

  // Use full memory data if available, otherwise fall back to the partial data passed in
  const displayMemory = fullMemory || memory;

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
            <div>
              <h3 className="text-xl font-bold text-foreground">Memory Details</h3>
              <p className="text-sm text-muted-foreground mt-1">Full memory content and metadata</p>
            </div>
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
              {/* Full Text */}
              <div>
                <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                  Full Text
                </div>
                <div className="text-sm whitespace-pre-wrap leading-relaxed text-foreground">
                  {displayMemory.text}
                </div>
              </div>

              {/* Context */}
              {displayMemory.context && (
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Context
                  </div>
                  <div className="text-sm text-foreground">{displayMemory.context}</div>
                </div>
              )}

              {/* Dates */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Occurred
                  </div>
                  <div className="text-sm font-medium text-foreground">
                    {displayMemory.occurred_start
                      ? new Date(displayMemory.occurred_start).toLocaleString()
                      : "N/A"}
                  </div>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Mentioned
                  </div>
                  <div className="text-sm font-medium text-foreground">
                    {displayMemory.mentioned_at
                      ? new Date(displayMemory.mentioned_at).toLocaleString()
                      : "N/A"}
                  </div>
                </div>
              </div>

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
              {displayMemory.tags && displayMemory.tags.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-3">Tags</div>
                  <div className="flex flex-wrap gap-2">
                    {displayMemory.tags.map((tag: string, i: number) => (
                      <span
                        key={i}
                        className="text-sm px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* ID */}
              {memoryId && (
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Memory ID
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="text-xs font-mono break-all flex-1 text-muted-foreground">
                      {memoryId}
                    </code>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 flex-shrink-0"
                      onClick={() => copyToClipboard(memoryId)}
                    >
                      {copiedId === memoryId ? (
                        <Check className="h-4 w-4 text-green-600" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
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
            </div>
          )}
        </div>

        {/* Document/Chunk Modal */}
        {modalType && modalId && (
          <DocumentChunkModal type={modalType} id={modalId} onClose={closeModal} />
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
          <div>
            <h3 className={`${titleSize} font-bold text-card-foreground`}>Memory Details</h3>
            {!compact && (
              <p className="text-sm text-muted-foreground">Full memory content and metadata</p>
            )}
          </div>
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
            {/* Full Text */}
            <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
              <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                Full Text
              </div>
              <div className={`${textSize} whitespace-pre-wrap`}>{displayMemory.text}</div>
            </div>

            {/* Context */}
            {displayMemory.context && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Context
                </div>
                <div className={textSize}>{displayMemory.context}</div>
              </div>
            )}

            {/* Dates */}
            <div className="grid grid-cols-2 gap-2">
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Occurred
                </div>
                <div className={textSize}>
                  {displayMemory.occurred_start
                    ? new Date(displayMemory.occurred_start).toLocaleString()
                    : "N/A"}
                </div>
              </div>
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Mentioned
                </div>
                <div className={textSize}>
                  {displayMemory.mentioned_at
                    ? new Date(displayMemory.mentioned_at).toLocaleString()
                    : "N/A"}
                </div>
              </div>
            </div>

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
                <div className="flex flex-wrap gap-1">
                  {displayMemory.tags.map((tag: string, i: number) => (
                    <span
                      key={i}
                      className={`${compact ? "text-[10px] px-1.5 py-0.5" : "text-xs px-2 py-1"} rounded bg-amber-500/10 text-amber-600 dark:text-amber-400`}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* ID */}
            {memoryId && (
              <div className={`${compact ? "p-2" : "p-3"} bg-muted rounded-lg`}>
                <div className={`${labelSize} font-bold text-muted-foreground uppercase mb-1`}>
                  Memory ID
                </div>
                <div className="flex items-center gap-2">
                  <span className={`${compact ? "text-[10px]" : "text-sm"} font-mono break-all`}>
                    {memoryId}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 flex-shrink-0"
                    onClick={() => copyToClipboard(memoryId)}
                  >
                    {copiedId === memoryId ? (
                      <Check className="h-3 w-3 text-green-600" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
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
          </div>
        )}
      </div>

      {/* Document/Chunk Modal */}
      {modalType && modalId && (
        <DocumentChunkModal type={modalType} id={modalId} onClose={closeModal} />
      )}
    </>
  );
}
