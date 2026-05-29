"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { DataView } from "./data-view";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  X,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pencil,
  Check,
  RefreshCw,
  MoreVertical,
  FileText,
  Settings,
  Layers,
  ChevronDown,
  Network,
  Eye,
} from "lucide-react";

const ITEMS_PER_PAGE = 50;

function formatRelativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const seconds = Math.floor((now - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function MetadataBadges({ metadata }: { metadata: Record<string, any> }) {
  const entries = Object.entries(metadata);
  if (entries.length === 0) return <span>-</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {entries.slice(0, 3).map(([k, v]) => (
        <span
          key={k}
          className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 font-medium"
        >
          {k}={String(v)}
        </span>
      ))}
      {entries.length > 3 && (
        <span className="text-xs px-2 py-0.5 text-muted-foreground">+{entries.length - 3}</span>
      )}
    </div>
  );
}

/* ── Shared helper components (match mental-model-detail-modal pattern) ── */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground mb-1">
      {children}
    </div>
  );
}

function InfoCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/20 overflow-hidden">
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-border bg-muted/40 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
      </div>
      <div className="p-4 space-y-4">{children}</div>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  );
}

const COMPOSITION_COLORS = {
  world: "#8b5cf6",
  experience: "#ec4899",
  observation: "#6366f1",
};

function MemoryComposition({
  nodesByFactType,
}: {
  nodesByFactType: { world: number; experience: number; observation: number } | undefined;
}) {
  const t = useTranslations("dataView");
  const counts = nodesByFactType ?? { world: 0, experience: 0, observation: 0 };
  const total = counts.world + counts.experience + counts.observation;
  const items = [
    { name: "World", value: counts.world, color: COMPOSITION_COLORS.world },
    { name: "Experience", value: counts.experience, color: COMPOSITION_COLORS.experience },
    { name: "Observations", value: counts.observation, color: COMPOSITION_COLORS.observation },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
          {t("memoryComposition")}
        </h4>
        <span className="text-xs text-muted-foreground tabular-nums">{total.toLocaleString()}</span>
      </div>
      {total === 0 ? (
        <div className="text-xs text-muted-foreground py-2">{t("noMemoriesYet")}</div>
      ) : (
        <>
          <div className="h-1.5 flex w-full rounded-full overflow-hidden bg-muted">
            {items
              .filter((d) => d.value > 0)
              .map((d) => (
                <div
                  key={d.name}
                  className="h-full"
                  style={{ width: `${(d.value / total) * 100}%`, backgroundColor: d.color }}
                  title={`${d.name}: ${d.value.toLocaleString()}`}
                />
              ))}
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm">
            {items.map((d) => (
              <div key={d.name} className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-[2px]" style={{ backgroundColor: d.color }} />
                  <span className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium">
                    {d.name}
                  </span>
                </div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-base font-semibold tabular-nums">
                    {d.value.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {total > 0 ? `${((d.value / total) * 100).toFixed(0)}%` : "0%"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

type ChunkFactType = "world" | "experience" | "observation";

function ChunkMemoriesHeader({
  chunkFactType,
  setChunkFactType,
}: {
  chunkFactType: ChunkFactType;
  setChunkFactType: (ft: ChunkFactType) => void;
}) {
  return (
    <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border bg-muted/30">
      {(["world", "experience", "observation"] as const).map((ft) => (
        <button
          key={ft}
          onClick={() => setChunkFactType(ft)}
          className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
            chunkFactType === ft
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {ft === "observation" ? "Obs" : ft.charAt(0).toUpperCase() + ft.slice(1)}
        </button>
      ))}
    </div>
  );
}

function ChunkRow({ chunk }: { chunk: any }) {
  const [expanded, setExpanded] = useState(false);
  const [memoriesExpanded, setMemoriesExpanded] = useState(false);
  const [chunkFactType, setChunkFactType] = useState<ChunkFactType>("world");
  const previewLength = 150;
  const text = chunk.chunk_text ?? "";
  const preview = text.length > previewLength ? text.slice(0, previewLength) + "..." : text;

  return (
    <div>
      <button
        className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-muted/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform ${expanded ? "" : "-rotate-90"}`}
        />
        <span className="text-xs font-mono text-muted-foreground shrink-0">
          #{chunk.chunk_index}
        </span>
        <span className="text-[11px] text-muted-foreground/60 shrink-0">
          {text.length.toLocaleString()} chars
        </span>
        {!expanded && <span className="text-xs text-foreground/50 truncate">{preview}</span>}
      </button>
      {expanded &&
        (memoriesExpanded ? (
          /* Full-width memories view with controls (text hidden) */
          <div style={{ height: "500px" }} className="flex flex-col border-t border-border">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/30 border-b border-border">
              <ChunkMemoriesHeader
                chunkFactType={chunkFactType}
                setChunkFactType={setChunkFactType}
              />
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setMemoriesExpanded(false)}
                className="h-6 px-2 text-xs gap-1"
              >
                <Eye className="w-3 h-3" />
                Compact
              </Button>
            </div>
            <div className="flex-1 min-h-0">
              <DataView
                key={`${chunk.chunk_id}-${chunkFactType}-full`}
                factType={chunkFactType}
                chunkId={chunk.chunk_id}
              />
            </div>
          </div>
        ) : (
          /* Split view: left text, right compact memories */
          <div className="grid grid-cols-2 divide-x divide-border" style={{ height: "350px" }}>
            <div className="overflow-y-auto">
              <pre className="px-4 py-3 text-[11px] leading-5 text-foreground/80 whitespace-pre-wrap font-mono">
                {text}
              </pre>
            </div>
            <div className="flex flex-col overflow-hidden">
              <ChunkMemoriesHeader
                chunkFactType={chunkFactType}
                setChunkFactType={setChunkFactType}
              />
              <div className="flex-1 min-h-0">
                <DataView
                  key={`${chunk.chunk_id}-${chunkFactType}-compact`}
                  factType={chunkFactType}
                  chunkId={chunk.chunk_id}
                  compact
                  onExpandToggle={() => setMemoriesExpanded(true)}
                />
              </div>
            </div>
          </div>
        ))}
    </div>
  );
}

export function DocumentsView() {
  const t = useTranslations("documentsView");
  const tCommon = useTranslations("common");
  const { currentBank } = useBank();
  const [documents, setDocuments] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [total, setTotal] = useState(0);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);
  const offset = (currentPage - 1) * ITEMS_PER_PAGE;

  // Document view panel state
  const [selectedDocument, setSelectedDocument] = useState<any>(null);
  const [loadingDocument, setLoadingDocument] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);

  // Tag editing state
  const [editingTags, setEditingTags] = useState(false);
  const [tagInput, setTagInput] = useState("");
  const [savingTags, setSavingTags] = useState(false);

  // Content editing state
  const [editingContent, setEditingContent] = useState(false);
  const [contentInput, setContentInput] = useState("");
  const [savingContent, setSavingContent] = useState(false);

  // Chunks state
  const [chunks, setChunks] = useState<any[]>([]);
  const [chunksTotal, setChunksTotal] = useState(0);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunksLoaded, setChunksLoaded] = useState(false);

  // Reprocess state
  const [reprocessing, setReprocessing] = useState(false);
  const [reprocessResult, setReprocessResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  // Delete confirmation dialog state
  const [documentToDelete, setDocumentToDelete] = useState<{
    id: string;
    memoryCount?: number;
  } | null>(null);
  const [deleteResult, setDeleteResult] = useState<{ success: boolean; message: string } | null>(
    null
  );

  const loadDocuments = async (page: number = 1) => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const pageOffset = (page - 1) * ITEMS_PER_PAGE;
      const data: any = await client.listDocuments({
        bank_id: currentBank,
        q: searchQuery,
        limit: ITEMS_PER_PAGE,
        offset: pageOffset,
      });
      setDocuments(data.items || []);
      setTotal(data.total || 0);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoading(false);
    }
  };

  // Handle page change
  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
    loadDocuments(newPage);
  };

  const viewDocumentText = async (documentId: string) => {
    if (!currentBank) return;

    setLoadingDocument(true);
    setSelectedDocument({ id: documentId }); // Set placeholder to show loading
    setEditingTags(false);
    setTagInput("");
    setEditingContent(false);
    setContentInput("");
    setChunks([]);
    setChunksTotal(0);
    setChunksLoaded(false);

    try {
      const doc: any = await client.getDocument(documentId, currentBank);
      setSelectedDocument(doc);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
      setSelectedDocument(null);
    } finally {
      setLoadingDocument(false);
    }
  };

  const loadChunks = async (documentId: string) => {
    if (!currentBank) return;

    setLoadingChunks(true);
    try {
      const data: any = await client.listDocumentChunks({
        document_id: documentId,
        bank_id: currentBank,
        limit: 100,
      });
      setChunks(data.items || []);
      setChunksTotal(data.total || 0);
      setChunksLoaded(true);
    } catch (error) {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setLoadingChunks(false);
    }
  };

  const reprocessDocument = async () => {
    if (!currentBank || !selectedDocument) return;

    setReprocessing(true);
    try {
      const result = await client.reprocessDocument(selectedDocument.id, currentBank);
      setReprocessResult({
        success: true,
        message: `Reprocessing started (operation: ${result.operation_id})`,
      });
    } catch (error) {
      setReprocessResult({
        success: false,
        message: "Error reprocessing document: " + (error as Error).message,
      });
    } finally {
      setReprocessing(false);
    }
  };

  const confirmDeleteDocument = async () => {
    if (!currentBank || !documentToDelete) return;

    const documentId = documentToDelete.id;
    setDeletingDocumentId(documentId);
    setDocumentToDelete(null);

    try {
      const result = await client.deleteDocument(documentId, currentBank);
      setDeleteResult({
        success: true,
        message: t("toastDeletedDocumentAndUnits", { count: result.memory_units_deleted }),
      });

      // Close panel if this document was selected
      if (selectedDocument?.id === documentId) {
        setSelectedDocument(null);
      }

      // Reload documents list at current page
      loadDocuments(currentPage);
    } catch (error) {
      console.error("Error deleting document:", error);
      setDeleteResult({
        success: false,
        message: t("toastErrorDeletingDocument") + (error as Error).message,
      });
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const requestDeleteDocument = (documentId: string, memoryCount?: number) => {
    setDocumentToDelete({ id: documentId, memoryCount });
  };

  const startEditTags = () => {
    setTagInput((selectedDocument?.tags ?? []).join(", "));
    setEditingTags(true);
  };

  const cancelEditTags = () => {
    setEditingTags(false);
    setTagInput("");
  };

  const startEditContent = () => {
    setContentInput(selectedDocument?.original_text ?? "");
    setEditingContent(true);
  };

  const cancelEditContent = () => {
    setEditingContent(false);
    setContentInput("");
  };

  const saveDocumentContent = async () => {
    if (!currentBank || !selectedDocument) return;

    const newContent = contentInput;
    if (!newContent.trim()) return;

    const retainParams = selectedDocument.retain_params ?? {};
    const item: Parameters<typeof client.retain>[0]["items"][number] = {
      content: newContent,
      document_id: selectedDocument.id,
    };
    if (retainParams.context) item.context = retainParams.context;
    if (retainParams.event_date) item.timestamp = retainParams.event_date;
    if (retainParams.metadata && Object.keys(retainParams.metadata).length > 0) {
      item.metadata = retainParams.metadata;
    }
    if (selectedDocument.tags && selectedDocument.tags.length > 0) {
      item.tags = selectedDocument.tags;
    }

    setSavingContent(true);
    try {
      await client.retain({
        bank_id: currentBank,
        items: [item],
        async: false,
      });
      // Refresh the document and the list
      const doc: any = await client.getDocument(selectedDocument.id, currentBank);
      setSelectedDocument(doc);
      setEditingContent(false);
      setContentInput("");
      loadDocuments(currentPage);
    } catch (error) {
      console.error("Error updating document content:", error);
    } finally {
      setSavingContent(false);
    }
  };

  const saveDocumentTags = async () => {
    if (!currentBank || !selectedDocument) return;

    const newTags = tagInput
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    setSavingTags(true);
    try {
      await client.updateDocument(selectedDocument.id, currentBank, newTags);
      setSelectedDocument({ ...selectedDocument, tags: newTags });
      // Update tags in the documents list too
      setDocuments((prev) =>
        prev.map((d) => (d.id === selectedDocument.id ? { ...d, tags: newTags } : d))
      );
      setEditingTags(false);
      setTagInput("");
    } catch (error) {
      console.error("Error updating document tags:", error);
    } finally {
      setSavingTags(false);
    }
  };

  // Auto-load documents when component mounts or bank changes
  useEffect(() => {
    if (currentBank) {
      setCurrentPage(1);
      loadDocuments(1);
    }
  }, [currentBank]);

  // Reload when search query changes (with debounce)
  useEffect(() => {
    if (!currentBank) return;

    const timeoutId = setTimeout(() => {
      setCurrentPage(1);
      loadDocuments(1);
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  return (
    <div>
      {/* Documents List Section */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">⏳</div>
            <div className="text-sm text-muted-foreground">{t("loadingDocuments")}</div>
          </div>
        </div>
      ) : documents.length > 0 ? (
        <>
          <div className="mb-4 text-sm text-muted-foreground">{t("totalDocuments", { total })}</div>
          {/* Documents Table */}
          <div className="w-full">
            <div className="px-5 mb-4">
              <Input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={t("searchPlaceholder")}
                className="max-w-2xl"
              />
            </div>

            <div className="overflow-x-auto px-5 pb-5">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("colDocumentId")}</TableHead>
                    <TableHead>{t("colCreated")}</TableHead>
                    <TableHead>{t("colUpdated")}</TableHead>
                    <TableHead>{t("colTags")}</TableHead>
                    <TableHead>{t("colMetadata")}</TableHead>
                    <TableHead>{t("colSize")}</TableHead>
                    <TableHead>{t("colMemoryUnits")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {documents.length > 0 ? (
                    documents.map((doc) => (
                      <TableRow
                        key={doc.id}
                        className={`cursor-pointer hover:bg-muted/50 ${selectedDocument?.id === doc.id ? "bg-primary/10" : ""}`}
                        onClick={() => viewDocumentText(doc.id)}
                      >
                        <TableCell className="text-card-foreground font-mono text-xs break-all">
                          {doc.id}
                        </TableCell>
                        <TableCell
                          className="text-card-foreground"
                          title={doc.created_at ? new Date(doc.created_at).toLocaleString() : ""}
                        >
                          {doc.created_at ? formatRelativeTime(doc.created_at) : "N/A"}
                        </TableCell>
                        <TableCell
                          className="text-card-foreground"
                          title={doc.updated_at ? new Date(doc.updated_at).toLocaleString() : ""}
                        >
                          {doc.updated_at ? formatRelativeTime(doc.updated_at) : "N/A"}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.tags && doc.tags.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {doc.tags.slice(0, 3).map((tag: string, i: number) => (
                                <span
                                  key={i}
                                  className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium"
                                >
                                  {tag}
                                </span>
                              ))}
                              {doc.tags.length > 3 && (
                                <span className="text-xs px-2 py-0.5 text-muted-foreground">
                                  +{doc.tags.length - 3}
                                </span>
                              )}
                            </div>
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.document_metadata &&
                          Object.keys(doc.document_metadata).length > 0 ? (
                            <MetadataBadges metadata={doc.document_metadata} />
                          ) : (
                            "-"
                          )}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {formatBytes(doc.text_length || 0)}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.memory_unit_count}
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center">
                        {t("clickLoadDocumentsToView")}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-3 pt-3 border-t px-5">
                <div className="text-xs text-muted-foreground">
                  {offset + 1}-{Math.min(offset + ITEMS_PER_PAGE, total)} of {total}
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(1)}
                    disabled={currentPage === 1 || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronsLeft className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1 || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronLeft className="h-3 w-3" />
                  </Button>
                  <span className="text-xs px-2">
                    {currentPage} / {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronRight className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(totalPages)}
                    disabled={currentPage === totalPages || loading}
                    className="h-7 w-7 p-0"
                  >
                    <ChevronsRight className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">📄</div>
            <div className="text-sm text-muted-foreground">{t("noDocumentsFound")}</div>
          </div>
        </div>
      )}

      {/* Document Detail Dialog */}
      <Dialog open={!!selectedDocument} onOpenChange={(open) => !open && setSelectedDocument(null)}>
        <DialogContent className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden">
          <DialogHeader className="pr-10">
            <DialogTitle className="flex items-center gap-2">
              <span className="truncate font-mono text-sm">
                {selectedDocument?.id ?? "Document"}
              </span>
            </DialogTitle>
          </DialogHeader>

          {loadingDocument ? (
            <div className="flex items-center justify-center flex-1">
              <div className="text-center">
                <div className="text-4xl mb-2">⏳</div>
                <div className="text-sm text-muted-foreground">{t("loadingDocument")}</div>
              </div>
            </div>
          ) : selectedDocument ? (
            <Tabs defaultValue="general" className="flex-1 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between gap-2">
                <TabsList className="grid grid-cols-3 w-full max-w-md">
                  <TabsTrigger value="general" className="flex items-center gap-1.5">
                    <Settings className="w-3.5 h-3.5" />
                    General
                  </TabsTrigger>
                  <TabsTrigger value="content" className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    Content
                  </TabsTrigger>
                  <TabsTrigger
                    value="chunks"
                    className="flex items-center gap-1.5"
                    onClick={() => {
                      if (!chunksLoaded && selectedDocument?.id) {
                        loadChunks(selectedDocument.id);
                      }
                    }}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    Chunks{chunksLoaded ? ` (${chunksTotal})` : ""}
                  </TabsTrigger>
                </TabsList>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 shrink-0"
                      disabled={reprocessing}
                      aria-label={tCommon("actions")}
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={reprocessDocument} disabled={reprocessing}>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Reprocess
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onClick={() =>
                        requestDeleteDocument(
                          selectedDocument.id,
                          selectedDocument.memory_unit_count
                        )
                      }
                      className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {t("deleteButton")}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex-1 overflow-y-auto mt-4">
                {/* Content Tab */}
                <TabsContent value="content" className="mt-0">
                  {selectedDocument.original_text !== undefined &&
                    (editingContent ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-end mb-2">
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={saveDocumentContent}
                              disabled={savingContent || !contentInput.trim()}
                              className="h-7 px-3 gap-1 text-xs"
                            >
                              {savingContent ? (
                                <span className="animate-spin">⏳</span>
                              ) : (
                                <Check className="h-3 w-3" />
                              )}
                              {t("saveButton")}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={cancelEditContent}
                              disabled={savingContent}
                              className="h-7 px-3 gap-1 text-xs"
                            >
                              <X className="h-3 w-3" />
                              {t("cancelButton")}
                            </Button>
                          </div>
                        </div>
                        <textarea
                          value={contentInput}
                          onChange={(e) => setContentInput(e.target.value)}
                          className="w-full min-h-[400px] max-h-[600px] p-4 bg-muted/50 rounded-lg border border-border text-sm font-mono leading-relaxed text-card-foreground resize-y"
                          autoFocus
                        />
                        <p className="text-xs text-muted-foreground">{t("saveHint")}</p>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-border bg-muted/30 overflow-hidden">
                        <div className="flex items-center justify-between gap-2 px-4 py-2 border-b border-border bg-muted/50 text-xs text-muted-foreground">
                          <div className="flex items-center gap-1.5">
                            <FileText className="w-3.5 h-3.5" />
                            <span className="font-semibold uppercase tracking-wide">
                              {t("storedContent")}
                            </span>
                            <span className="text-muted-foreground/70">
                              &middot;{" "}
                              {selectedDocument.original_text?.length?.toLocaleString() ?? 0} chars
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={startEditContent}
                            className="h-6 px-2 gap-1 text-xs"
                          >
                            <Pencil className="h-3 w-3" />
                            {t("editButton")}
                          </Button>
                        </div>
                        <pre className="p-4 text-[11px] leading-5 text-foreground/80 whitespace-pre-wrap font-mono">
                          {selectedDocument.original_text}
                        </pre>
                      </div>
                    ))}
                </TabsContent>

                {/* General Tab */}
                <TabsContent value="general" className="mt-0">
                  <div className="space-y-4">
                    {/* Memories constellation — first */}
                    <Tabs defaultValue="world" className="flex flex-col">
                      <TabsList className="w-fit">
                        <TabsTrigger value="world">World</TabsTrigger>
                        <TabsTrigger value="experience">Experience</TabsTrigger>
                        <TabsTrigger value="observation">Observations</TabsTrigger>
                      </TabsList>
                      <div className="mt-2">
                        <TabsContent value="world" className="mt-0">
                          <DataView factType="world" documentId={selectedDocument.id} compact />
                        </TabsContent>
                        <TabsContent value="experience" className="mt-0">
                          <DataView
                            factType="experience"
                            documentId={selectedDocument.id}
                            compact
                          />
                        </TabsContent>
                        <TabsContent value="observation" className="mt-0">
                          <DataView
                            factType="observation"
                            documentId={selectedDocument.id}
                            compact
                          />
                        </TabsContent>
                      </div>
                    </Tabs>

                    {/* Info cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InfoCard title="Document" icon={<FileText className="w-3.5 h-3.5" />}>
                        {selectedDocument.created_at && (
                          <MetadataRow
                            label={t("labelCreated")}
                            value={new Date(selectedDocument.created_at).toLocaleString()}
                          />
                        )}
                        {selectedDocument.updated_at && (
                          <MetadataRow
                            label={t("labelUpdated")}
                            value={new Date(selectedDocument.updated_at).toLocaleString()}
                          />
                        )}
                        {selectedDocument.original_text && (
                          <MetadataRow
                            label={t("colSize")}
                            value={formatBytes(new Blob([selectedDocument.original_text]).size)}
                          />
                        )}
                        <MetadataRow
                          label={t("labelTags")}
                          value={
                            editingTags ? (
                              <div className="flex items-center gap-2">
                                <Input
                                  value={tagInput}
                                  onChange={(e) => setTagInput(e.target.value)}
                                  placeholder={t("tagsInputPlaceholder")}
                                  className="text-sm h-7 w-64"
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") saveDocumentTags();
                                    if (e.key === "Escape") cancelEditTags();
                                  }}
                                  autoFocus
                                />
                                <Button
                                  size="sm"
                                  onClick={saveDocumentTags}
                                  disabled={savingTags}
                                  className="h-7 w-7 p-0"
                                >
                                  {savingTags ? (
                                    <span className="animate-spin text-xs">⏳</span>
                                  ) : (
                                    <Check className="h-3 w-3" />
                                  )}
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={cancelEditTags}
                                  disabled={savingTags}
                                  className="h-7 w-7 p-0"
                                >
                                  <X className="h-3 w-3" />
                                </Button>
                              </div>
                            ) : (
                              <div className="flex items-center gap-2">
                                {selectedDocument.tags && selectedDocument.tags.length > 0 ? (
                                  <div className="flex flex-wrap gap-1.5">
                                    {selectedDocument.tags.map((tag: string, i: number) => (
                                      <span
                                        key={i}
                                        className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium"
                                      >
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                ) : (
                                  <span className="text-sm text-muted-foreground italic">none</span>
                                )}
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={startEditTags}
                                  className="h-6 w-6 p-0"
                                >
                                  <Pencil className="h-3 w-3" />
                                </Button>
                              </div>
                            )
                          }
                        />
                        {selectedDocument.retain_params?.context && (
                          <MetadataRow
                            label="Context"
                            value={selectedDocument.retain_params.context}
                          />
                        )}
                        {selectedDocument.retain_params?.event_date && (
                          <MetadataRow
                            label={t("labelEventDate")}
                            value={new Date(
                              selectedDocument.retain_params.event_date
                            ).toLocaleString()}
                          />
                        )}
                        {selectedDocument.retain_params?.metadata &&
                          Object.keys(selectedDocument.retain_params.metadata).length > 0 && (
                            <MetadataRow
                              label="Metadata"
                              value={
                                <MetadataBadges
                                  metadata={selectedDocument.retain_params.metadata}
                                />
                              }
                            />
                          )}
                      </InfoCard>

                      <InfoCard
                        title={t("memoryCompositionTitle")}
                        icon={<Network className="w-3.5 h-3.5" />}
                      >
                        <MemoryComposition nodesByFactType={selectedDocument.nodes_by_fact_type} />
                      </InfoCard>
                    </div>
                  </div>
                </TabsContent>

                {/* Chunks Tab */}
                <TabsContent value="chunks" className="mt-0">
                  {loadingChunks ? (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <div className="text-4xl mb-2">⏳</div>
                        <div className="text-sm text-muted-foreground">{t("loadingChunks")}</div>
                      </div>
                    </div>
                  ) : chunks.length > 0 ? (
                    <div className="rounded-lg border border-border overflow-hidden divide-y divide-border">
                      {chunks.map((chunk) => (
                        <ChunkRow key={chunk.chunk_id} chunk={chunk} />
                      ))}
                    </div>
                  ) : chunksLoaded ? (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <div className="text-4xl mb-2">📄</div>
                        <div className="text-sm text-muted-foreground">{t("noChunksFound")}</div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center py-20">
                      <div className="text-center">
                        <div className="text-sm text-muted-foreground">
                          {t("clickChunksTabToLoad")}
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </div>
            </Tabs>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={!!documentToDelete}
        onOpenChange={(open) => !open && setDocumentToDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteDialogTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteDialogDescription")}{" "}
              <span className="font-mono font-semibold">&quot;{documentToDelete?.id}&quot;</span>?
              <br />
              <br />
              {t("deleteDialogWillDelete")}{" "}
              {documentToDelete?.memoryCount !== undefined ? (
                <span className="font-semibold">
                  {t("deleteDialogMemoryUnits", { count: documentToDelete.memoryCount })}
                </span>
              ) : (
                t("deleteDialogAllMemoryUnits")
              )}{" "}
              {t("deleteDialogExtracted")}
              <br />
              <br />
              <span className="text-destructive font-semibold">{t("deleteDialogCannotUndo")}</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("cancelButton")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteDocument}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("deleteButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Result Dialog */}
      <AlertDialog open={!!deleteResult} onOpenChange={(open) => !open && setDeleteResult(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteResult?.success ? t("deleteResultSuccessTitle") : t("deleteResultErrorTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>{deleteResult?.message}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setDeleteResult(null)}>
              {t("okButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Reprocess Result Dialog */}
      <AlertDialog
        open={!!reprocessResult}
        onOpenChange={(open) => !open && setReprocessResult(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {reprocessResult?.success ? "Reprocessing Started" : t("deleteResultErrorTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>{reprocessResult?.message}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setReprocessResult(null)}>
              {t("okButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
