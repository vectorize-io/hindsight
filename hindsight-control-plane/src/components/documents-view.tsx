"use client";

import { useState, useEffect } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
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
import { X, Trash2, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

const ITEMS_PER_PAGE = 50;

export function DocumentsView() {
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
      console.error("Error loading documents:", error);
      alert("Error loading documents: " + (error as Error).message);
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

    try {
      const doc: any = await client.getDocument(documentId, currentBank);
      setSelectedDocument(doc);
    } catch (error) {
      console.error("Error loading document:", error);
      alert("Error loading document: " + (error as Error).message);
      setSelectedDocument(null);
    } finally {
      setLoadingDocument(false);
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
        message: `Deleted document and ${result.memory_units_deleted} memory units.`,
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
        message: "Error deleting document: " + (error as Error).message,
      });
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const requestDeleteDocument = (documentId: string, memoryCount?: number) => {
    setDocumentToDelete({ id: documentId, memoryCount });
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
            <div className="text-sm text-muted-foreground">Loading documents...</div>
          </div>
        </div>
      ) : documents.length > 0 ? (
        <>
          <div className="mb-4 text-sm text-muted-foreground">
            {total} {total === 1 ? "document" : "documents"}
          </div>
          {/* Documents Table */}
          <div className="w-full">
            <div className="px-5 mb-4">
              <Input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search documents (ID)..."
                className="max-w-2xl"
              />
            </div>

            <div className="overflow-x-auto px-5 pb-5">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document ID</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Tags</TableHead>
                    <TableHead>Context</TableHead>
                    <TableHead>Text Length</TableHead>
                    <TableHead>Memory Units</TableHead>
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
                        <TableCell title={doc.id} className="text-card-foreground">
                          {doc.id.length > 30 ? doc.id.substring(0, 30) + "..." : doc.id}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.created_at ? new Date(doc.created_at).toLocaleString() : "N/A"}
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
                          {doc.retain_params?.context || "-"}
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.text_length?.toLocaleString()} chars
                        </TableCell>
                        <TableCell className="text-card-foreground">
                          {doc.memory_unit_count}
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center">
                        Click "Load Documents" to view data
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
            <div className="text-sm text-muted-foreground">No documents found</div>
          </div>
        </div>
      )}

      {/* Document Detail Panel - Fixed on Right */}
      {documents.length > 0 && selectedDocument && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l-2 border-primary shadow-2xl z-50 overflow-y-auto animate-in slide-in-from-right duration-300 ease-out">
          <div className="p-5">
            {/* Header with close button */}
            <div className="flex justify-between items-center mb-6 pb-4 border-b border-border">
              <div>
                <h3 className="text-xl font-bold text-foreground">Document Details</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Original document text and metadata
                </p>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setSelectedDocument(null)}
                className="h-9 px-3 gap-2"
              >
                <X className="h-4 w-4" />
                Close
              </Button>
            </div>

            {loadingDocument ? (
              <div className="flex items-center justify-center py-20">
                <div className="text-center">
                  <div className="text-4xl mb-2">⏳</div>
                  <div className="text-sm text-muted-foreground">Loading document...</div>
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {/* Document ID */}
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                    Document ID
                  </div>
                  <div className="text-sm font-mono break-all text-card-foreground">
                    {selectedDocument.id}
                  </div>
                </div>

                {/* Created & Memory Units */}
                {selectedDocument.created_at && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                        Created
                      </div>
                      <div className="text-sm font-medium text-card-foreground">
                        {new Date(selectedDocument.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                        Memory Units
                      </div>
                      <div className="text-sm font-medium text-card-foreground">
                        {selectedDocument.memory_unit_count}
                      </div>
                    </div>
                  </div>
                )}

                {/* Text Length */}
                {selectedDocument.original_text && (
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                      Text Length
                    </div>
                    <div className="text-sm font-medium text-card-foreground">
                      {selectedDocument.original_text.length.toLocaleString()} characters
                    </div>
                  </div>
                )}

                {/* Retain Parameters */}
                {selectedDocument.retain_params && (
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                      Retain Parameters
                    </div>
                    <div className="text-sm space-y-2 text-card-foreground">
                      {selectedDocument.retain_params.context && (
                        <div>
                          <span className="font-semibold">Context:</span>{" "}
                          {selectedDocument.retain_params.context}
                        </div>
                      )}
                      {selectedDocument.retain_params.event_date && (
                        <div>
                          <span className="font-semibold">Event Date:</span>{" "}
                          {new Date(selectedDocument.retain_params.event_date).toLocaleString()}
                        </div>
                      )}
                      {selectedDocument.retain_params.metadata && (
                        <div className="mt-2">
                          <span className="font-semibold">Metadata:</span>
                          <pre className="mt-1 text-xs bg-background p-2 rounded text-card-foreground">
                            {JSON.stringify(selectedDocument.retain_params.metadata, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Tags */}
                {selectedDocument.tags && selectedDocument.tags.length > 0 && (
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                      Tags
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {selectedDocument.tags.map((tag: string, i: number) => (
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

                {/* Delete Button */}
                <div className="pt-2 border-t border-border">
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() =>
                      requestDeleteDocument(selectedDocument.id, selectedDocument.memory_unit_count)
                    }
                    className="w-full gap-2"
                    disabled={deletingDocumentId === selectedDocument.id}
                  >
                    {deletingDocumentId === selectedDocument.id ? (
                      <span className="animate-spin">⏳</span>
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                    Delete Document
                  </Button>
                </div>

                {/* Original Text */}
                {selectedDocument.original_text && (
                  <div>
                    <div className="text-xs font-bold text-muted-foreground uppercase mb-2">
                      Original Text
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg border border-border max-h-[400px] overflow-y-auto">
                      <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed text-card-foreground">
                        {selectedDocument.original_text}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={!!documentToDelete}
        onOpenChange={(open) => !open && setDocumentToDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete document{" "}
              <span className="font-mono font-semibold">&quot;{documentToDelete?.id}&quot;</span>?
              <br />
              <br />
              This will also delete{" "}
              {documentToDelete?.memoryCount !== undefined ? (
                <span className="font-semibold">{documentToDelete.memoryCount} memory units</span>
              ) : (
                "all memory units"
              )}{" "}
              extracted from this document.
              <br />
              <br />
              <span className="text-destructive font-semibold">This action cannot be undone.</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteDocument}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Result Dialog */}
      <AlertDialog open={!!deleteResult} onOpenChange={(open) => !open && setDeleteResult(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteResult?.success ? "Document Deleted" : "Error"}
            </AlertDialogTitle>
            <AlertDialogDescription>{deleteResult?.message}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setDeleteResult(null)}>OK</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
