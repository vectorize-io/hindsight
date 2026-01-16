"use client";

import { useState, useEffect } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Calendar, Tag, Users, FileText, Layers } from "lucide-react";

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

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Memory Details</DialogTitle>
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
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col overflow-hidden"
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="memory" className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" />
                Memory
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
                <div className="p-4 bg-muted rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Memory Text
                  </div>
                  <p className="text-sm text-foreground leading-relaxed">{memory.text}</p>
                </div>

                {/* Metadata grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                      Type
                    </div>
                    <div className="text-sm text-foreground capitalize">{memory.type}</div>
                  </div>
                  {memory.context && (
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
                        Context
                      </div>
                      <div className="text-sm text-foreground">{memory.context}</div>
                    </div>
                  )}
                </div>

                {/* Dates */}
                {(memory.mentioned_at || memory.occurred_start) && (
                  <div className="grid grid-cols-2 gap-3">
                    {memory.mentioned_at && (
                      <div className="p-3 bg-muted rounded-lg">
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-1 flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          Mentioned At
                        </div>
                        <div className="text-sm text-foreground">
                          {new Date(memory.mentioned_at).toLocaleString()}
                        </div>
                      </div>
                    )}
                    {memory.occurred_start && (
                      <div className="p-3 bg-muted rounded-lg">
                        <div className="text-xs font-bold text-muted-foreground uppercase mb-1 flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          Occurred
                        </div>
                        <div className="text-sm text-foreground">
                          {new Date(memory.occurred_start).toLocaleDateString()}
                          {memory.occurred_end && memory.occurred_end !== memory.occurred_start && (
                            <> - {new Date(memory.occurred_end).toLocaleDateString()}</>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Entities */}
                {memory.entities && memory.entities.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                      <Users className="w-3 h-3" />
                      Entities
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {memory.entities.map((entity, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-background rounded text-xs text-foreground"
                        >
                          {entity}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Tags */}
                {memory.tags && memory.tags.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                      <Tag className="w-3 h-3" />
                      Tags
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {memory.tags.map((tag, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* IDs */}
                <div className="p-3 bg-muted rounded-lg">
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
                        <div className="text-sm text-foreground">{document.memory_unit_count}</div>
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
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
