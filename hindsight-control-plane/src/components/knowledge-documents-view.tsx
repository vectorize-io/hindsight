"use client";

// A knowledge bank's documents: filterable, paginated, live while documents ingest,
// and a drop target for new files anywhere on the view.

import * as React from "react";
import { useTranslations } from "next-intl";
import {
  Download,
  FilePlus2,
  Files,
  LayoutGrid,
  List,
  Plus,
  ScatterChart,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import {
  MEDIA_TYPES,
  DOCUMENT_STATUSES,
  documentLabel,
  firstPassageUrl,
  formatBytes,
  isInFlight,
  KNOWLEDGE_UPLOAD_EVENT,
  proxiedMediaUrl,
  type MediaType,
  type DocumentStatus,
  type KnowledgeDocument,
} from "@/lib/knowledge-banks";
import { formatRelativeTime } from "@/lib/relative-time";
import { cn } from "@/lib/utils";
import { KindIcon, StatusMark } from "@/components/knowledge-passage";
import { KnowledgeConstellation } from "@/components/knowledge-constellation";
import {
  KnowledgeAddDocumentDialog,
  type AddDocumentTab,
} from "@/components/knowledge-write-dialogs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { TagChip } from "@/components/ui/facet-chip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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

const PAGE_SIZE = 25;

/** The first passage's text, fetched only for the text cards actually on screen. */
function TextSnippet({ bankId, documentId }: { bankId: string; documentId: string }) {
  const [text, setText] = React.useState<string | null>(null);
  React.useEffect(() => {
    let live = true;
    client
      .getKnowledgeDocument(bankId, documentId)
      .then((d) => live && setText(d.passages[0]?.text ?? ""))
      .catch(() => live && setText(""));
    return () => {
      live = false;
    };
  }, [bankId, documentId]);
  return (
    <p className="line-clamp-6 h-full whitespace-pre-wrap break-words p-3 text-left text-xs leading-relaxed text-muted-foreground">
      {text}
    </p>
  );
}

/** A document as a card: a preview of its content on top, then its name and facts. */
function DocumentCard({
  bankId,
  doc,
  onOpen,
}: {
  bankId: string;
  doc: KnowledgeDocument;
  onOpen: () => void;
}) {
  const label = documentLabel(doc);
  const ready = doc.status === "ready" && doc.passage_count > 0;
  const preview = proxiedMediaUrl(firstPassageUrl(doc));
  let thumb: React.ReactNode = <KindIcon kind={doc.media_type} className="h-10 w-10 opacity-70" />;
  // Only PDF pages and images have a picture: Office and HTML files are split as text.
  const picture = doc.media_type === "image" || (doc.media_type === "document" && isPdf(doc));
  if (ready && picture) {
    thumb = (
      <img src={preview} alt="" loading="lazy" className="h-full w-full bg-white object-contain" />
    );
  } else if (ready && doc.media_type === "video") {
    // The first window's clip, paused on its first frame; the dialog plays it.
    thumb = (
      <video
        src={preview}
        preload="metadata"
        muted
        className="h-full w-full bg-black object-contain"
      />
    );
  } else if (ready && doc.media_type === "text") {
    thumb = <TextSnippet bankId={bankId} documentId={doc.document_id} />;
  }
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex min-w-0 flex-col overflow-hidden rounded-lg border border-border bg-card text-left text-sm transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:ring-primary"
    >
      <div className="flex h-32 w-full items-center justify-center overflow-hidden border-b border-border bg-muted/30">
        {thumb}
      </div>
      <div className="flex min-w-0 flex-col gap-1.5 p-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <KindIcon kind={doc.media_type} />
          <span className="min-w-0 flex-1 truncate text-[13px] font-medium" title={label}>
            {label}
          </span>
          <StatusMark status={doc.status} error={doc.error} />
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground tabular-nums">
          <span>{formatBytes(doc.size_bytes)}</span>
          <span>{doc.updated_at ? formatRelativeTime(doc.updated_at) : ""}</span>
        </div>
        {doc.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {doc.tags.slice(0, 3).map((tag) => (
              <TagChip key={tag} tag={tag} size="xs" />
            ))}
          </div>
        )}
      </div>
    </button>
  );
}
const POLL_MS = 3000;
const ALL = "all";
const CONSTELLATION_HEIGHT = 600;

/** List first (the default), as the memory bank's views sit side by side. */
export const DOCUMENT_LAYOUTS = ["list", "grid", "constellation"] as const;
export type DocumentsLayout = (typeof DOCUMENT_LAYOUTS)[number];

export function KnowledgeDocumentsView({
  bankId,
  onOpenDocument,
  onChanged,
  requestedKind,
  requestedStatus,
  requestedLayout,
}: {
  bankId: string;
  onOpenDocument: (documentId: string, passage?: number) => void;
  /** Something was written, deleted, or finished ingesting: the bank's counts moved. */
  onChanged: () => void;
  /** A link asked for these filters or this layout (e.g. a Home card); applied when they change. */
  requestedKind?: MediaType;
  requestedStatus?: DocumentStatus;
  requestedLayout?: DocumentsLayout;
}) {
  const t = useTranslations("knowledgeBanks.documents");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const tStatuses = useTranslations("knowledgeBanks.statuses");
  const tCommon = useTranslations("common");
  const tAdd = useTranslations("addDocument");

  const [docs, setDocs] = React.useState<KnowledgeDocument[] | null>(null);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [queryDraft, setQueryDraft] = React.useState("");
  const [query, setQuery] = React.useState("");
  const [kind, setKind] = React.useState<MediaType | typeof ALL>(ALL);
  const [status, setStatus] = React.useState<DocumentStatus | typeof ALL>(ALL);
  const [loading, setLoading] = React.useState(false);
  const [addOpen, setAddOpen] = React.useState(false);
  const [addTab, setAddTab] = React.useState<AddDocumentTab>("text");
  const [droppedFiles, setDroppedFiles] = React.useState<File[]>([]);
  const openAdd = React.useCallback((tab: AddDocumentTab, files: File[] = []) => {
    setAddTab(tab);
    setDroppedFiles(files);
    setAddOpen(true);
  }, []);
  const [dragDepth, setDragDepth] = React.useState(0);
  const [toDelete, setToDelete] = React.useState<KnowledgeDocument | null>(null);
  const [layout, setLayout] = React.useState<DocumentsLayout>("list");
  const [deleting, setDeleting] = React.useState(false);

  // The view stays mounted while hidden, so a link's filters land on the live state.
  React.useEffect(() => {
    if (!requestedKind && !requestedStatus && !requestedLayout) return;
    setKind(requestedKind ?? ALL);
    setStatus(requestedStatus ?? ALL);
    if (requestedLayout) setLayout(requestedLayout);
    setQueryDraft("");
    setPage(1);
  }, [requestedKind, requestedStatus, requestedLayout]);

  React.useEffect(() => {
    // The header's "Add Document" opens on Text, as the memory bank one does.
    const open = () => openAdd("text");
    window.addEventListener(KNOWLEDGE_UPLOAD_EVENT, open);
    return () => window.removeEventListener(KNOWLEDGE_UPLOAD_EVENT, open);
  }, [openAdd]);

  // Typing filters after a pause, not on every keystroke.
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(queryDraft.trim());
      setPage(1);
    }, 250);
    return () => clearTimeout(timer);
  }, [queryDraft]);

  const seq = React.useRef(0);
  const docsRef = React.useRef<KnowledgeDocument[] | null>(null);
  const load = React.useCallback(
    async ({ quiet = false }: { quiet?: boolean } = {}) => {
      const mine = ++seq.current;
      if (!quiet) setLoading(true);
      try {
        const res = await client.listKnowledgeDocuments(
          bankId,
          {
            limit: PAGE_SIZE,
            offset: (page - 1) * PAGE_SIZE,
            q: query || undefined,
            media_type: kind === ALL ? undefined : kind,
            status: status === ALL ? undefined : status,
          },
          { quiet }
        );
        if (mine !== seq.current) return;
        // A document that just finished (or failed) moved the bank's counts.
        const settled = docsRef.current?.some(
          (p) =>
            isInFlight(p.status) &&
            res.documents.some((d) => d.document_id === p.document_id && !isInFlight(d.status))
        );
        docsRef.current = res.documents;
        setDocs(res.documents);
        setTotal(res.total);
        if (settled) onChanged();
      } catch {
        if (mine === seq.current) setDocs((prev) => prev ?? []);
      } finally {
        if (mine === seq.current) setLoading(false);
      }
    },
    [bankId, page, query, kind, status, onChanged]
  );

  React.useEffect(() => {
    load();
  }, [load]);

  const inFlight = docs?.some((d) => isInFlight(d.status)) ?? false;
  React.useEffect(() => {
    if (!inFlight) return;
    const timer = setInterval(() => load({ quiet: true }), POLL_MS);
    return () => clearInterval(timer);
  }, [inFlight, load]);

  const afterWrite = () => {
    load({ quiet: true });
    onChanged();
  };

  const remove = async () => {
    if (!toDelete) return;
    setDeleting(true);
    try {
      await client.deleteKnowledgeDocument(bankId, toDelete.document_id);
      toast.success(t("deleted", { name: documentLabel(toDelete) }));
      setToDelete(null);
      afterWrite();
    } catch {
      // toasted by the API client
    } finally {
      setDeleting(false);
    }
  };

  const filtered = query !== "" || kind !== ALL || status !== ALL;

  return (
    <div
      className="relative"
      // Drop files anywhere on the view. A depth counter, because dragenter/leave
      // fire for every child the pointer crosses.
      onDragEnter={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDragDepth((d) => d + 1);
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) e.preventDefault();
      }}
      onDragLeave={() => setDragDepth((d) => Math.max(0, d - 1))}
      onDrop={(e) => {
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        setDragDepth(0);
        openAdd("upload", Array.from(e.dataTransfer.files));
      }}
    >
      {dragDepth > 0 && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-primary bg-primary/10 backdrop-blur-[1px]">
          <div className="flex items-center gap-2 rounded-lg bg-card px-4 py-3 text-sm font-medium shadow">
            <Upload className="h-4 w-4 text-primary" />
            {t("dropToUpload")}
          </div>
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[12rem] flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={queryDraft}
            onChange={(e) => setQueryDraft(e.target.value)}
            placeholder={t("filterPlaceholder")}
            className="pl-9"
            aria-label={t("filterPlaceholder")}
          />
        </div>
        <Select
          value={kind}
          onValueChange={(v) => {
            setKind(v as MediaType | typeof ALL);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-full sm:w-36" aria-label={t("kindFilter")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>{t("allKinds")}</SelectItem>
            {MEDIA_TYPES.map((k) => (
              <SelectItem key={k} value={k}>
                <span className="inline-flex items-center gap-2">
                  <KindIcon kind={k} />
                  {tKinds(k)}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={status}
          onValueChange={(v) => {
            setStatus(v as DocumentStatus | typeof ALL);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-full sm:w-36" aria-label={t("statusFilter")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>{t("allStatuses")}</SelectItem>
            {DOCUMENT_STATUSES.map((s) => (
              <SelectItem key={s} value={s}>
                {tStatuses(s)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex gap-2 sm:ml-auto">
          {/* List, cards or constellation: the memory bank's view switch. */}
          <div className="flex gap-1 rounded-lg bg-muted p-1" role="group" aria-label={t("layout")}>
            {DOCUMENT_LAYOUTS.map((mode) => {
              const Icon = LAYOUT_ICON[mode];
              const label = t(LAYOUT_LABEL[mode]);
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setLayout(mode)}
                  aria-pressed={layout === mode}
                  title={label}
                  aria-label={label}
                  className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-sm font-medium transition-all sm:px-3 ${
                    layout === mode
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </button>
              );
            })}
          </div>
          <Button size="sm" className="h-9" onClick={() => openAdd("upload")}>
            <Plus className="mr-1.5 h-4 w-4" />
            {tAdd("addDocumentButton")}
          </Button>
        </div>
      </div>

      {docs === null ? (
        <div className="space-y-2" aria-hidden>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-12 animate-pulse rounded-lg bg-muted/60" />
          ))}
        </div>
      ) : docs.length === 0 ? (
        <div className="flex flex-col items-center rounded-xl border border-dashed border-border bg-card px-6 py-16 text-center">
          <Files className="mb-4 h-10 w-10 text-muted-foreground" />
          <h3 className="text-lg font-semibold">{filtered ? t("noMatches") : t("emptyTitle")}</h3>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            {filtered ? t("noMatchesBody") : t("emptyBody")}
          </p>
          {!filtered && (
            <div className="mt-6 flex gap-2">
              <Button variant="outline" onClick={() => openAdd("text")}>
                <FilePlus2 className="mr-1.5 h-4 w-4" />
                {t("addText")}
              </Button>
              <Button onClick={() => openAdd("upload")}>
                <Upload className="mr-1.5 h-4 w-4" />
                {t("upload")}
              </Button>
            </div>
          )}
        </div>
      ) : (
        <>
          {layout === "constellation" ? (
            <div className="overflow-hidden rounded-[16px] border border-border bg-card">
              <KnowledgeConstellation
                bankId={bankId}
                docs={docs}
                height={CONSTELLATION_HEIGHT}
                onOpenDocument={onOpenDocument}
              />
            </div>
          ) : layout === "grid" ? (
            <div
              className={cn(
                "grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 transition-opacity duration-150",
                loading && "opacity-50"
              )}
            >
              {docs.map((doc) => (
                <DocumentCard
                  key={doc.document_id}
                  bankId={bankId}
                  doc={doc}
                  onOpen={() => onOpenDocument(doc.document_id)}
                />
              ))}
            </div>
          ) : (
            <Table className={cn("transition-opacity duration-150", loading && "opacity-50")}>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("columnDocument")}</TableHead>
                  <TableHead className="w-24 text-right">{t("columnPassages")}</TableHead>
                  <TableHead className="w-24 text-right">{t("columnSize")}</TableHead>
                  <TableHead className="w-48">{t("columnTags")}</TableHead>
                  <TableHead className="w-32">{t("columnUpdated")}</TableHead>
                  <TableHead className="w-20" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {docs.map((doc) => {
                  const label = documentLabel(doc);
                  return (
                    <TableRow
                      key={doc.document_id}
                      className="cursor-pointer"
                      onClick={() => onOpenDocument(doc.document_id)}
                    >
                      <TableCell>
                        <div className="flex min-w-0 items-center gap-2.5">
                          <KindIcon kind={doc.media_type} />
                          <div className="min-w-0">
                            <button
                              type="button"
                              className="block max-w-full truncate text-left font-medium text-card-foreground hover:underline focus-visible:underline"
                              title={label}
                              onClick={(e) => {
                                e.stopPropagation();
                                onOpenDocument(doc.document_id);
                              }}
                            >
                              {label}
                            </button>
                            {label !== doc.document_id && (
                              <div
                                className="truncate font-mono text-[11px] text-muted-foreground/70"
                                title={doc.document_id}
                              >
                                {doc.document_id}
                              </div>
                            )}
                          </div>
                          <StatusMark status={doc.status} error={doc.error} />
                        </div>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {isInFlight(doc.status) ? "—" : doc.passage_count}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-muted-foreground">
                        {formatBytes(doc.size_bytes)}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {doc.tags.slice(0, 3).map((tag) => (
                            <TagChip key={tag} tag={tag} size="xs" />
                          ))}
                          {doc.tags.length > 3 && (
                            <span className="text-[11px] text-muted-foreground">
                              +{doc.tags.length - 3}
                            </span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-sm text-muted-foreground">
                        {doc.updated_at ? formatRelativeTime(doc.updated_at) : "—"}
                      </TableCell>
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <div className="flex justify-end gap-0.5">
                          <a
                            href={proxiedMediaUrl(doc.document_url)}
                            download={doc.filename ?? undefined}
                            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
                            title={t("download")}
                            aria-label={t("download")}
                          >
                            <Download className="h-4 w-4" />
                          </a>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                            title={t("delete")}
                            aria-label={t("delete")}
                            onClick={() => setToDelete(doc)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={total}
            disabled={loading}
            onPageChange={setPage}
          />
        </>
      )}

      <KnowledgeAddDocumentDialog
        bankId={bankId}
        open={addOpen}
        initialTab={addTab}
        initialFiles={droppedFiles}
        onOpenChange={setAddOpen}
        onWritten={afterWrite}
      />
      <AlertDialog
        open={toDelete !== null}
        onOpenChange={(o) => !o && !deleting && setToDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("deleteBody", { name: toDelete ? documentLabel(toDelete) : "" })}
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
              {t("delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

const LAYOUT_ICON = { list: List, grid: LayoutGrid, constellation: ScatterChart } as const;
const LAYOUT_LABEL = {
  list: "viewList",
  grid: "viewGrid",
  constellation: "viewConstellation",
} as const;

function isPdf(doc: KnowledgeDocument): boolean {
  return (
    doc.content_type === "application/pdf" || (doc.filename ?? "").toLowerCase().endsWith(".pdf")
  );
}
