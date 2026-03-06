"use client";

import { useState, useEffect } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Calendar, Users, FileText, Layers, Tag, History } from "lucide-react";
import { TagList } from "@/components/ui/tag-list";
import { Button } from "@/components/ui/button";
import { ObservationHistoryView, type HistoryEntry } from "@/components/observation-history-view";

interface SourceMemory {
  id: string;
  text: string;
  context: string | null;
  type: string;
  occurred_start: string | null;
  mentioned_at: string | null;
}

interface MemoryDetail {
  id: string;
  text: string;
  context: string;
  date: string;
  type: string;
  mentioned_at: string | null;
  occurred_start: string | null;
  occurred_end: string | null;
  entities: string[];
  document_id: string | null;
  chunk_id: string | null;
  tags: string[];
  observation_scopes: string | string[][] | null;
  source_memories?: SourceMemory[];
  history?: HistoryEntry[];
}

interface MemoryDetailModalProps {
  memoryId: string | null;
  onClose: () => void;
}

export function MemoryDetailModal({ memoryId, onClose }: MemoryDetailModalProps) {
  const { currentBank } = useBank();
  const [memory, setMemory] = useState<MemoryDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("memory");

  // Document and chunk data
  const [document, setDocument] = useState<any>(null);
  const [chunk, setChunk] = useState<any>(null);
  const [loadingDocument, setLoadingDocument] = useState(false);
  const [loadingChunk, setLoadingChunk] = useState(false);

  // Source memory modal (for viewing source memories of observations)
  const [sourceMemoryModalId, setSourceMemoryModalId] = useState<string | null>(null);

  // Load memory details
  useEffect(() => {
    if (!memoryId || !currentBank) return;

    const loadMemory = async () => {
      setLoading(true);
      setError(null);
      setMemory(null);
      setDocument(null);
      setChunk(null);
      setActiveTab("memory");

      try {
        const data = await client.getMemory(memoryId, currentBank);
        setMemory(data);
      } catch (err) {
        console.error("Error loading memory:", err);
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    loadMemory();
  }, [memoryId, currentBank]);

  // Load document when tab is selected
  useEffect(() => {
    if (activeTab !== "document" || !memory?.document_id || !currentBank || document) return;

    const loadDocument = async () => {
      setLoadingDocument(true);
      try {
        const data = await client.getDocument(memory.document_id!, currentBank);
        setDocument(data);
      } catch (err) {
        console.error("Error loading document:", err);
      } finally {
        setLoadingDocument(false);
      }
    };

    loadDocument();
  }, [activeTab, memory?.document_id, currentBank, document]);

  // Load chunk when tab is selected
  useEffect(() => {
    if (activeTab !== "chunk" || !memory?.chunk_id || chunk) return;

    const loadChunk = async () => {
      setLoadingChunk(true);
      try {
        const data = await client.getChunk(memory.chunk_id!);
        setChunk(data);
      } catch (err) {
        console.error("Error loading chunk:", err);
      } finally {
        setLoadingChunk(false);
      }
    };

    loadChunk();
  }, [activeTab, memory?.chunk_id, chunk]);

  const isOpen = memoryId !== null;

  // Determine the display title based on memory type
  const getMemoryTypeTitle = () => {
    if (memory?.type === "observation") return "Observation";
    if (memory?.type === "world") return "World Fact";
    if (memory?.type === "experience") return "Experience";
    return "Memory Details";
  };

  const isObservation = memory?.type === "observation";

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle>{memory ? getMemoryTypeTitle() : "Memory Details"}</DialogTitle>
          </DialogHeader>

          {loading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-20">
              <div className="text-center text-destructive">
                <div className="text-sm">Error: {error}</div>
              </div>
            </div>
          ) : memory ? (
            isObservation ? (
              /* Observation view - tabs for Info and History */
              <Tabs
                value={activeTab}
                onValueChange={setActiveTab}
                className="flex-1 flex flex-col overflow-hidden"
              >
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="memory" className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    Observation
                  </TabsTrigger>
                  <TabsTrigger value="history" className="flex items-center gap-1.5">
                    <History className="w-3.5 h-3.5" />
                    History
                    {memory.history && memory.history.length > 0
                      ? ` (${memory.history.length})`
                      : ""}
                  </TabsTrigger>
                </TabsList>

                <div className="flex-1 overflow-y-auto mt-4">
                  <TabsContent value="memory" className="mt-0 space-y-4">
                    {/* Text */}
                    <div>
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                        Text
                      </div>
                      <p className="text-sm text-foreground leading-relaxed">{memory.text}</p>
                    </div>

                    {/* Dates */}
                    {memory.occurred_start && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                          Occurred
                        </div>
                        <div className="flex items-center gap-2 text-sm text-foreground">
                          <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          <span>
                            {new Date(memory.occurred_start).toLocaleString()}
                            {memory.occurred_end &&
                              memory.occurred_end !== memory.occurred_start && (
                                <>
                                  <span className="text-muted-foreground mx-1">→</span>
                                  {new Date(memory.occurred_end).toLocaleString()}
                                </>
                              )}
                          </span>
                        </div>
                      </div>
                    )}

                    {memory.mentioned_at && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                          Mentioned
                        </div>
                        <div className="flex items-center gap-2 text-sm text-foreground">
                          <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          <span>{new Date(memory.mentioned_at).toLocaleString()}</span>
                        </div>
                      </div>
                    )}

                    {/* Entities */}
                    {memory.entities && memory.entities.length > 0 && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          Entities
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {memory.entities.map((entity, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs"
                            >
                              {entity}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tags */}
                    <TagList tags={memory.tags} showLabel />

                    {/* Observation Scopes */}
                    {memory.observation_scopes && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                          <Tag className="w-3 h-3" />
                          Observation Scopes
                        </div>
                        {typeof memory.observation_scopes === "string" ? (
                          <span className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded">
                            {memory.observation_scopes}
                          </span>
                        ) : (
                          <div className="space-y-1.5">
                            {(memory.observation_scopes as string[][]).map((scope, i) => (
                              <TagList key={i} tags={scope} />
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Source Memories */}
                    {memory.source_memories && memory.source_memories.length > 0 && (
                      <div className="border-t border-border pt-4">
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-3">
                          Source Memories ({memory.source_memories.length})
                        </div>
                        <div className="space-y-3">
                          {memory.source_memories.map((source, i) => (
                            <div
                              key={source.id || i}
                              className="p-3 bg-muted/50 rounded-lg border border-border/50"
                            >
                              <div className="flex items-start justify-between gap-2 mb-2">
                                <span
                                  className={`px-2 py-0.5 rounded text-xs flex-shrink-0 ${
                                    source.type === "experience"
                                      ? "bg-green-500/10 text-green-600 dark:text-green-400"
                                      : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
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
                              <p className="text-sm text-foreground mb-2">{source.text}</p>
                              {source.context && (
                                <p className="text-xs text-muted-foreground mb-2 italic">
                                  Context: {source.context}
                                </p>
                              )}
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                {source.occurred_start && (
                                  <div className="p-2 bg-background/50 rounded">
                                    <div className="text-muted-foreground mb-0.5">Occurred</div>
                                    <div className="font-medium">
                                      {new Date(source.occurred_start).toLocaleString()}
                                    </div>
                                  </div>
                                )}
                                {source.mentioned_at && (
                                  <div className="p-2 bg-background/50 rounded">
                                    <div className="text-muted-foreground mb-0.5">Mentioned</div>
                                    <div className="font-medium">
                                      {new Date(source.mentioned_at).toLocaleString()}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ID */}
                    <div>
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                        Memory ID
                      </div>
                      <code className="text-xs font-mono text-muted-foreground break-all">
                        {memory.id}
                      </code>
                    </div>
                  </TabsContent>

                  <TabsContent value="history" className="mt-0">
                    {memory.history && memory.history.length > 0 ? (
                      <ObservationHistoryView
                        history={memory.history}
                        current={{
                          text: memory.text,
                          tags: memory.tags,
                          occurred_start: memory.occurred_start,
                          occurred_end: memory.occurred_end,
                          mentioned_at: memory.mentioned_at,
                        }}
                      />
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        No history recorded yet.
                      </p>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            ) : (
              /* World/Experience view - with tabs */
              <Tabs
                value={activeTab}
                onValueChange={setActiveTab}
                className="flex-1 flex flex-col overflow-hidden"
              >
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="memory" className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    {memory.type === "world" ? "World Fact" : "Experience"}
                  </TabsTrigger>
                  <TabsTrigger
                    value="chunk"
                    disabled={!memory.chunk_id}
                    className="flex items-center gap-1.5"
                  >
                    <Layers className="w-3.5 h-3.5" />
                    Chunk
                  </TabsTrigger>
                  <TabsTrigger
                    value="document"
                    disabled={!memory.document_id}
                    className="flex items-center gap-1.5"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Document
                  </TabsTrigger>
                </TabsList>

                <div className="flex-1 overflow-y-auto mt-4">
                  <TabsContent value="memory" className="mt-0 space-y-4">
                    {/* Memory text */}
                    <div>
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                        Text
                      </div>
                      <p className="text-sm text-foreground leading-relaxed">{memory.text}</p>
                    </div>

                    {/* Context */}
                    {memory.context && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                          Context
                        </div>
                        <div className="text-sm text-foreground">{memory.context}</div>
                      </div>
                    )}

                    {/* Dates */}
                    {memory.occurred_start && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                          Occurred
                        </div>
                        <div className="flex items-center gap-2 text-sm text-foreground">
                          <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          <span>
                            {new Date(memory.occurred_start).toLocaleString()}
                            {memory.occurred_end &&
                              memory.occurred_end !== memory.occurred_start && (
                                <>
                                  <span className="text-muted-foreground mx-1">→</span>
                                  {new Date(memory.occurred_end).toLocaleString()}
                                </>
                              )}
                          </span>
                        </div>
                      </div>
                    )}

                    {memory.mentioned_at && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                          Mentioned
                        </div>
                        <div className="flex items-center gap-2 text-sm text-foreground">
                          <Calendar className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          <span>{new Date(memory.mentioned_at).toLocaleString()}</span>
                        </div>
                      </div>
                    )}

                    {/* Entities */}
                    {memory.entities && memory.entities.length > 0 && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          Entities
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {memory.entities.map((entity, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs"
                            >
                              {entity}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tags */}
                    <TagList tags={memory.tags} showLabel />

                    {/* Observation Scopes */}
                    {memory.observation_scopes && (
                      <div>
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                          <Tag className="w-3 h-3" />
                          Observation Scopes
                        </div>
                        {typeof memory.observation_scopes === "string" ? (
                          <span className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded">
                            {memory.observation_scopes}
                          </span>
                        ) : (
                          <div className="space-y-1.5">
                            {(memory.observation_scopes as string[][]).map((scope, i) => (
                              <TagList key={i} tags={scope} />
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* ID */}
                    <div>
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                        Memory ID
                      </div>
                      <code className="text-xs font-mono text-muted-foreground break-all">
                        {memory.id}
                      </code>
                    </div>
                  </TabsContent>

                  <TabsContent value="chunk" className="mt-0 space-y-4">
                    {loadingChunk ? (
                      <div className="flex items-center justify-center py-12">
                        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : chunk ? (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                              Chunk Index
                            </div>
                            <div className="text-sm text-foreground">{chunk.chunk_index}</div>
                          </div>
                          {chunk.chunk_text && (
                            <div className="p-3 bg-muted rounded-lg">
                              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                                Text Length
                              </div>
                              <div className="text-sm text-foreground">
                                {chunk.chunk_text.length.toLocaleString()} chars
                              </div>
                            </div>
                          )}
                        </div>

                        {chunk.chunk_text && (
                          <div>
                            <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                              Chunk Text
                            </div>
                            <div className="p-4 bg-muted rounded-lg border border-border max-h-[300px] overflow-y-auto">
                              <pre className="text-sm whitespace-pre-wrap font-mono text-foreground">
                                {chunk.chunk_text}
                              </pre>
                            </div>
                          </div>
                        )}

                        <div className="p-3 bg-muted rounded-lg">
                          <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                            Chunk ID
                          </div>
                          <code className="text-xs font-mono text-muted-foreground break-all">
                            {chunk.chunk_id}
                          </code>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-12 text-muted-foreground">
                        No chunk data available
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="document" className="mt-0 space-y-4">
                    {loadingDocument ? (
                      <div className="flex items-center justify-center py-12">
                        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : document ? (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          {document.created_at && (
                            <div className="p-3 bg-muted rounded-lg">
                              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                                Created
                              </div>
                              <div className="text-sm text-foreground">
                                {new Date(document.created_at).toLocaleString()}
                              </div>
                            </div>
                          )}
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                              Memory Units
                            </div>
                            <div className="text-sm text-foreground">
                              {document.memory_unit_count}
                            </div>
                          </div>
                        </div>

                        {document.original_text && (
                          <>
                            <div className="p-3 bg-muted rounded-lg">
                              <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                                Text Length
                              </div>
                              <div className="text-sm text-foreground">
                                {document.original_text.length.toLocaleString()} chars
                              </div>
                            </div>

                            <div>
                              <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                                Original Text
                              </div>
                              <div className="p-4 bg-muted rounded-lg border border-border max-h-[300px] overflow-y-auto">
                                <pre className="text-sm whitespace-pre-wrap font-mono text-foreground">
                                  {document.original_text}
                                </pre>
                              </div>
                            </div>
                          </>
                        )}

                        {document.tags && document.tags.length > 0 && (
                          <div className="p-3 bg-muted rounded-lg">
                            <TagList tags={document.tags} showLabel />
                          </div>
                        )}

                        <div className="p-3 bg-muted rounded-lg">
                          <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                            Document ID
                          </div>
                          <code className="text-xs font-mono text-muted-foreground break-all">
                            {document.id}
                          </code>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-12 text-muted-foreground">
                        No document data available
                      </div>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            )
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Nested modal for viewing source memories */}
      {sourceMemoryModalId && (
        <MemoryDetailModal
          memoryId={sourceMemoryModalId}
          onClose={() => setSourceMemoryModalId(null)}
        />
      )}
    </>
  );
}
