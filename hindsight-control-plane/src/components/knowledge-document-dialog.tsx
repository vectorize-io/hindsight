"use client";

// One document of a knowledge bank, in the same dialog the memory bank documents
// view opens: same size, tabs and actions menu, with the passages as the main tab —
// page images in a grid (with a viewer), an image, one player per audio or video
// window, or text passages.

import * as React from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  FileQuestion,
  FileText,
  Layers,
  MoreVertical,
  Settings,
  Tag,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import {
  documentLabel,
  formatBytes,
  isInFlight,
  isPicturePassage,
  proxiedMediaUrl,
  type KnowledgeDocumentDetail,
  type KnowledgePassage,
} from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { InfoCard, MetadataRow } from "@/components/documents-view";
import { CompactMarkdown } from "@/components/compact-markdown";
import {
  KindIcon,
  PassageLocation,
  PassageMedia,
  StatusChip,
} from "@/components/knowledge-passage";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { TagList } from "@/components/ui/facet-chip";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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

const POLL_MS = 3000;

/** The page/image viewer: one passage at a time, with prev/next and arrow keys. */
function ImageViewer({
  passages,
  index,
  label,
  onIndexChange,
}: {
  passages: KnowledgePassage[];
  index: number | null;
  label: string;
  onIndexChange: (index: number | null) => void;
}) {
  const t = useTranslations("knowledgeBanks.detail");
  const current = index != null ? passages[index] : null;
  const go = (delta: number) => {
    if (index == null) return;
    const next = index + delta;
    if (next >= 0 && next < passages.length) onIndexChange(next);
  };
  return (
    <Dialog open={current != null} onOpenChange={(o) => !o && onIndexChange(null)}>
      <DialogContent
        className="max-w-[min(1100px,calc(100vw-1rem))] p-4"
        aria-describedby={undefined}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") go(1);
          if (e.key === "ArrowLeft") go(-1);
        }}
      >
        {current && (
          <>
            <DialogTitle className="pr-8 text-sm">
              {label} · <PassageLocation passage={current} />
            </DialogTitle>
            <PassageMedia passage={current} alt={label} />
            {passages.length > 1 && (
              <div className="flex items-center justify-between">
                <Button variant="outline" size="sm" onClick={() => go(-1)} disabled={index === 0}>
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  {t("previous")}
                </Button>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {(index ?? 0) + 1} / {passages.length}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => go(1)}
                  disabled={index === passages.length - 1}
                >
                  {t("next")}
                  <ChevronRight className="ml-1 h-4 w-4" />
                </Button>
              </div>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Passages({
  doc,
  highlightPassage,
}: {
  doc: KnowledgeDocumentDetail;
  highlightPassage: number | null;
}) {
  const t = useTranslations("knowledgeBanks.detail");
  const [viewerIndex, setViewerIndex] = React.useState<number | null>(null);
  const label = documentLabel(doc);
  const images = doc.passages.filter(isPicturePassage);
  const others = doc.passages.filter((p) => !isPicturePassage(p));

  // Scroll a search hit's passage into view once the passages are on screen.
  React.useEffect(() => {
    if (highlightPassage == null) return;
    document
      .getElementById(`passage-${highlightPassage}`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlightPassage, doc.passages.length]);

  if (isInFlight(doc.status)) {
    return (
      <div className="flex items-center gap-3 rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-sm text-blue-700 dark:text-blue-300">
        <Spinner size="sm" />
        {t("ingesting")}
      </div>
    );
  }
  if (doc.passages.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">{t("noPassages")}</p>;
  }
  return (
    <>
      {images.length > 0 && (
        <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {images.map((p, i) => (
            <button
              key={p.index}
              id={`passage-${p.index}`}
              type="button"
              onClick={() => setViewerIndex(i)}
              className={cn(
                "group overflow-hidden rounded-lg border border-border bg-card text-left transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:ring-primary",
                highlightPassage === p.index && "ring-2 ring-primary"
              )}
            >
              <img
                src={proxiedMediaUrl(p.passage_url!)}
                alt={`${label} — ${p.page ?? p.index + 1}`}
                loading="lazy"
                className="aspect-[3/4] w-full bg-white object-contain"
              />
              <div className="border-t border-border px-2.5 py-1.5 text-xs text-muted-foreground group-hover:text-foreground">
                <PassageLocation passage={p} />
              </div>
            </button>
          ))}
        </div>
      )}
      {others.length > 0 && (
        <ol className="space-y-3">
          {others.map((p) => (
            <li
              key={p.index}
              id={`passage-${p.index}`}
              className={cn(
                "rounded-lg border border-border bg-muted/20 p-4",
                highlightPassage === p.index && "ring-2 ring-primary"
              )}
            >
              <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-medium tabular-nums">
                  <PassageLocation passage={p} />
                </span>
                <span className="font-mono">#{p.index}</span>
              </div>
              <PassageMedia passage={p} alt={label} />
            </li>
          ))}
        </ol>
      )}
      <ImageViewer
        passages={images}
        index={viewerIndex}
        label={label}
        onIndexChange={setViewerIndex}
      />
    </>
  );
}

// File types a browser renders in a frame (its own PDF viewer, or a page). Anything
// else in the "document" media type (Office files) can only be downloaded.
const FRAMEABLE = /^(application\/pdf|text\/html)/;
const FRAMEABLE_NAME = /\.(pdf|html?)$/i;

/** The original upload, shown the way the browser can: a frame, an image, a player or text. */
function OriginalPreview({ doc }: { doc: KnowledgeDocumentDetail }) {
  const t = useTranslations("knowledgeBanks.detail");
  const url = proxiedMediaUrl(doc.document_url);
  const [text, setText] = React.useState<string | null>(null);
  const isText = doc.media_type === "text";

  React.useEffect(() => {
    if (!isText) return;
    let live = true;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
      .then((body) => live && setText(body))
      .catch(() => live && setText(""));
    return () => {
      live = false;
    };
  }, [url, isText]);

  if (doc.media_type === "image") {
    return (
      <img src={url} alt={documentLabel(doc)} className="mx-auto max-h-[70vh] object-contain" />
    );
  }
  if (doc.media_type === "audio") return <audio controls src={url} className="w-full" />;
  if (doc.media_type === "video") {
    return (
      <video controls playsInline src={url} className="mx-auto max-h-[70vh] rounded-md bg-black" />
    );
  }
  if (isText) {
    if (text === null) return <Spinner size="md" />;
    return /\.(md|markdown)$/i.test(doc.filename ?? "") ? (
      <div className="prose prose-sm dark:prose-invert max-w-none rounded-lg border border-border bg-muted/20 p-6">
        <CompactMarkdown>{text}</CompactMarkdown>
      </div>
    ) : (
      <pre className="whitespace-pre-wrap break-words rounded-lg border border-border bg-muted/20 p-4 font-mono text-xs">
        {text}
      </pre>
    );
  }
  if (FRAMEABLE.test(doc.content_type) || FRAMEABLE_NAME.test(doc.filename ?? "")) {
    return (
      <iframe
        src={url}
        title={documentLabel(doc)}
        className="h-[calc(92vh-11rem)] w-full rounded-lg border border-border bg-white"
      />
    );
  }
  return (
    <div className="flex flex-col items-center gap-3 py-12 text-center text-sm text-muted-foreground">
      <FileQuestion className="h-10 w-10" />
      {t("originalUnsupported")}
      <Button variant="outline" size="sm" asChild>
        <a href={url} download={doc.filename ?? undefined}>
          <Download className="mr-1.5 h-4 w-4" />
          {t("download")}
        </a>
      </Button>
    </div>
  );
}

export function KnowledgeDocumentDialog({
  bankId,
  documentId,
  highlightPassage,
  onClose,
  onDeleted,
}: {
  bankId: string;
  /** The open document, or null for a closed dialog. */
  documentId: string | null;
  /** Passage index to scroll to and mark (from a search hit). */
  highlightPassage: number | null;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const t = useTranslations("knowledgeBanks.detail");
  const tCommon = useTranslations("common");
  const [doc, setDoc] = React.useState<KnowledgeDocumentDetail | null>(null);
  const [notFound, setNotFound] = React.useState(false);
  const [confirmDelete, setConfirmDelete] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);

  const load = React.useCallback(async () => {
    if (!documentId) return;
    try {
      setDoc(await client.getKnowledgeDocument(bankId, documentId));
      setNotFound(false);
    } catch (e) {
      if ((e as { status?: number }).status === 404) setNotFound(true);
    }
  }, [bankId, documentId]);

  React.useEffect(() => {
    setDoc(null);
    setNotFound(false);
    load();
  }, [load]);

  const inFlight = doc ? isInFlight(doc.status) : false;
  React.useEffect(() => {
    if (!inFlight) return;
    const timer = setInterval(load, POLL_MS);
    return () => clearInterval(timer);
  }, [inFlight, load]);

  const remove = async () => {
    if (!documentId) return;
    setDeleting(true);
    try {
      await client.deleteKnowledgeDocument(bankId, documentId);
      toast.success(t("deleted"));
      setConfirmDelete(false);
      onDeleted();
    } catch {
      // toasted by the API client
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <Dialog open={documentId != null} onOpenChange={(open) => !open && onClose()}>
        <DialogContent
          className="w-[95vw] max-w-[95vw] h-[92vh] sm:max-w-[95vw] flex flex-col overflow-hidden"
          aria-describedby={undefined}
        >
          <DialogHeader className="pr-10">
            <DialogTitle className="flex items-center gap-2">
              {doc && <KindIcon kind={doc.media_type} />}
              <span className="truncate font-mono text-sm">{documentId}</span>
              {doc && <StatusChip status={doc.status} />}
            </DialogTitle>
          </DialogHeader>

          {notFound ? (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <FileQuestion className="mb-3 h-10 w-10 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">{t("notFound")}</p>
            </div>
          ) : !doc ? (
            <div className="flex items-center justify-center flex-1">
              <div className="text-center">
                <Spinner size="xl" variant="jump" className="mx-auto mb-2" />
                <div className="text-sm text-muted-foreground">{t("loading")}</div>
              </div>
            </div>
          ) : (
            <Tabs defaultValue="passages" className="flex-1 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between gap-2">
                <TabsList className="grid grid-cols-3 w-full max-w-md">
                  <TabsTrigger value="general" className="flex items-center gap-1.5">
                    <Settings className="w-3.5 h-3.5" />
                    {t("tabGeneral")}
                  </TabsTrigger>
                  <TabsTrigger value="original" className="flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5" />
                    {t("tabOriginal")}
                  </TabsTrigger>
                  <TabsTrigger value="passages" className="flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5" />
                    {t("tabPassages", { count: doc.passage_count })}
                  </TabsTrigger>
                </TabsList>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 shrink-0"
                      aria-label={tCommon("actions")}
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem asChild>
                      <a
                        href={proxiedMediaUrl(doc.document_url)}
                        download={doc.filename ?? undefined}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        {t("download")}
                      </a>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onClick={() => setConfirmDelete(true)}
                      className="text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-500/10"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {t("delete")}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex-1 overflow-y-auto mt-4">
                <TabsContent value="general" className="mt-0">
                  <div className="space-y-4">
                    {doc.status === "failed" && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>{t("failedTitle")}</AlertTitle>
                        <AlertDescription className="whitespace-pre-wrap break-words">
                          {doc.error || t("failedNoReason")}
                        </AlertDescription>
                      </Alert>
                    )}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <InfoCard
                        title={t("documentCard")}
                        icon={<FileText className="w-3.5 h-3.5" />}
                      >
                        {doc.filename && <MetadataRow label={t("filename")} value={doc.filename} />}
                        <MetadataRow label={t("contentType")} value={doc.content_type} />
                        <MetadataRow label={t("size")} value={formatBytes(doc.size_bytes)} />
                        <MetadataRow
                          label={t("passages")}
                          value={inFlight ? "—" : doc.passage_count}
                        />
                        {doc.created_at && (
                          <MetadataRow
                            label={t("created")}
                            value={new Date(doc.created_at).toLocaleString()}
                          />
                        )}
                        {doc.updated_at && (
                          <MetadataRow
                            label={t("updated")}
                            value={new Date(doc.updated_at).toLocaleString()}
                          />
                        )}
                      </InfoCard>
                      <InfoCard title={t("tagsCard")} icon={<Tag className="w-3.5 h-3.5" />}>
                        <MetadataRow
                          label={t("tags")}
                          value={doc.tags.length > 0 ? <TagList tags={doc.tags} /> : <span>-</span>}
                        />
                      </InfoCard>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="original" className="mt-0">
                  <OriginalPreview doc={doc} />
                </TabsContent>
                <TabsContent value="passages" className="mt-0">
                  {doc.status === "failed" ? (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>{t("failedTitle")}</AlertTitle>
                      <AlertDescription className="whitespace-pre-wrap break-words">
                        {doc.error || t("failedNoReason")}
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <Passages doc={doc} highlightPassage={highlightPassage} />
                  )}
                </TabsContent>
              </div>
            </Tabs>
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog open={confirmDelete} onOpenChange={(o) => !deleting && setConfirmDelete(o)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteBody", { name: documentId ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                remove();
              }}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? (
                <Spinner size="sm" className="mr-2" />
              ) : (
                <Trash2 className="mr-2 h-4 w-4" />
              )}
              {t("delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
