"use client";

// Adding documents to a knowledge bank. Same dialog as a memory bank's "Add Document"
// (bank-selector.tsx): the same shell, Text / Upload Files tabs, file rows that expand
// into Document and Tags sub-tabs, and footer — minus what a knowledge bank has no use
// for (context, event date, entities, metadata). Tags are the only labels.

import * as React from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FileText,
  Upload,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import { formatBytes, packUploads, parseTags, uploadKnowledgeFiles } from "@/lib/knowledge-banks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

/** File types the API splits into passages; the picker offers these first. */
const ACCEPTED_FILE_TYPES =
  "application/pdf,image/*,audio/*,video/*,text/*,.md,.markdown,.rst,.json,.jsonl,.csv,.tsv,.yaml,.yml,.xml,.html,.htm,.epub,.docx,.pptx,.xlsx,.xls";

// The underline sub-tabs of the memory dialog's per-file and advanced sections.
const SUB_TAB =
  "rounded-none h-full px-4 text-xs font-medium bg-transparent shadow-none text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px";
const SUB_TAB_LIST =
  "w-full border-b border-border bg-transparent h-8 p-0 gap-0 justify-start rounded-none";
const FIELD_LABEL = "font-bold block mb-1 text-sm text-foreground";

interface StagedFile {
  key: string;
  file: File;
  documentId: string;
  tags: string;
  expanded: boolean;
  subTab: "document" | "tags";
  progress: number;
  state: "staged" | "uploading" | "done" | "failed";
  error: string | null;
}

let stagedSeq = 0;
function stage(file: File): StagedFile {
  stagedSeq += 1;
  return {
    key: `${stagedSeq}-${file.name}`,
    file,
    documentId: "",
    tags: "",
    expanded: false,
    subTab: "tags",
    progress: 0,
    state: "staged",
    error: null,
  };
}

// The files go up together, as few requests as fit, so one upload is one parent operation
// with a child per file. Requests stay under the proxy's 100 MB default body cap, which
// truncates silently; a file bigger than that goes alone. This used to be one request per
// file (3 at a time) so one bad file could not fail the rest; the server checks every file
// before storing any, so now a bad file fails its batch, and the batch's rows show why.
const UPLOAD_BATCH_BYTES = 90 * 1024 * 1024;

export type AddDocumentTab = "text" | "upload";

export function KnowledgeAddDocumentDialog({
  bankId,
  open,
  initialTab,
  initialFiles,
  onOpenChange,
  onWritten,
}: {
  bankId: string;
  open: boolean;
  initialTab: AddDocumentTab;
  /** Files dropped on the documents view before the dialog opened. */
  initialFiles: File[];
  onOpenChange: (open: boolean) => void;
  onWritten: () => void;
}) {
  const tAdd = useTranslations("addDocument");
  const t = useTranslations("knowledgeBanks.addDocument");
  const [tab, setTab] = React.useState<AddDocumentTab>(initialTab);
  const [busy, setBusy] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = React.useState(false);

  // Text tab
  const [text, setText] = React.useState("");
  const [documentId, setDocumentId] = React.useState("");
  const [filename, setFilename] = React.useState("");
  const [tags, setTags] = React.useState("");
  const [textSubTab, setTextSubTab] = React.useState<"document" | "tags">("tags");

  // Upload tab
  const [files, setFiles] = React.useState<StagedFile[]>([]);
  const [allTags, setAllTags] = React.useState("");

  React.useEffect(() => {
    if (!open) return;
    setTab(initialTab);
    setFiles(initialFiles.map(stage));
    setAllTags("");
    setText("");
    setDocumentId("");
    setFilename("");
    setTags("");
  }, [open, initialTab, initialFiles]);

  const add = (list: FileList | File[]) =>
    setFiles((prev) => [...prev, ...Array.from(list).map(stage)]);
  const update = (key: string, patch: Partial<StagedFile>) =>
    setFiles((prev) => prev.map((f) => (f.key === key ? { ...f, ...patch } : f)));

  const pending = files.filter((f) => f.state === "staged" || f.state === "failed");

  /** Writes the shared tags onto every file not yet uploaded, merged with its own. */
  const applyTagsToAll = () => {
    const shared = parseTags(allTags);
    if (shared.length === 0) return;
    setFiles((prev) =>
      prev.map((f) =>
        f.state === "done"
          ? f
          : { ...f, tags: [...new Set([...parseTags(f.tags), ...shared])].join(", ") }
      )
    );
  };

  const uploadAll = async () => {
    setBusy(true);
    let ok = 0;
    let unchanged = 0;
    let failed = 0;
    let firstError: string | null = null;
    for (const batch of packUploads(pending, (f) => f.file.size, UPLOAD_BATCH_BYTES)) {
      for (const f of batch) update(f.key, { state: "uploading", progress: 0, error: null });
      try {
        const result = await uploadKnowledgeFiles(
          bankId,
          batch.map((f) => f.file),
          batch.map((f) => ({
            tags: parseTags(f.tags),
            ...(f.documentId.trim() ? { document_id: f.documentId.trim() } : {}),
          })),
          (progress) => {
            for (const f of batch) update(f.key, { progress });
          }
        );
        for (const f of batch) update(f.key, { state: "done", progress: 1 });
        unchanged += result.unchanged_document_ids.length;
        ok += batch.length - result.unchanged_document_ids.length;
      } catch (e) {
        const reason = (e as Error).message;
        for (const f of batch) update(f.key, { state: "failed", error: reason });
        failed += batch.length;
        firstError ??= reason;
      }
    }
    setBusy(false);
    if (ok > 0) {
      toast.success(t("uploaded", { count: ok }));
      onWritten();
    }
    // An identical file is not ingested again: say so, rather than "uploaded".
    if (unchanged > 0) toast.info(t("unchanged", { count: unchanged }));
    // The server's reason in the toast too (e.g. "knowledge embedder unavailable"), not
    // only on each file's row, so a failure that hit every file reads at a glance.
    if (failed === 0) onOpenChange(false);
    else toast.error(t("someFailed", { count: failed }), { description: firstError ?? undefined });
  };

  const writeText = async () => {
    setBusy(true);
    try {
      const result = await client.writeKnowledgeTextDocument(bankId, {
        text,
        ...(filename.trim() ? { filename: filename.trim() } : {}),
        ...(documentId.trim() ? { document_id: documentId.trim() } : {}),
        tags: parseTags(tags),
      });
      if (result.unchanged_document_ids.length > 0) toast.info(t("writtenUnchanged"));
      else toast.success(t("written"));
      onOpenChange(false);
      onWritten();
    } catch {
      // toasted by the API client
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !busy && onOpenChange(o)}>
      <DialogContent className="sm:max-w-[750px] max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{tAdd("dialogTitle")}</DialogTitle>
          <p className="text-sm text-muted-foreground">
            {t("dialogSubtitle")}
            <span className="font-semibold">{bankId}</span>
          </p>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto flex-1 px-1 -mx-1">
          <Tabs value={tab} onValueChange={(v) => setTab(v as AddDocumentTab)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="text" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                {tAdd("tabText")}
              </TabsTrigger>
              <TabsTrigger value="upload" className="flex items-center gap-2">
                <Upload className="h-4 w-4" />
                {tAdd("tabUploadFiles")}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="text" className="mt-3 space-y-4">
              <div>
                <label htmlFor="kb-text-body" className={FIELD_LABEL}>
                  {tAdd("contentLabel")}
                </label>
                <Textarea
                  id="kb-text-body"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={tAdd("contentPlaceholder")}
                  className="min-h-[180px]"
                  autoFocus
                />
                <p className="text-xs text-muted-foreground mt-1">{t("textHelp")}</p>
              </div>
              <Tabs
                value={textSubTab}
                onValueChange={(v) => setTextSubTab(v as "document" | "tags")}
              >
                <TabsList className={SUB_TAB_LIST}>
                  <TabsTrigger value="tags" className={SUB_TAB}>
                    {tAdd("tagsLabel")}
                  </TabsTrigger>
                  <TabsTrigger value="document" className={SUB_TAB}>
                    {t("documentTab")}
                  </TabsTrigger>
                </TabsList>
                <div className="pt-3">
                  <TabsContent value="tags" className="mt-0">
                    <label htmlFor="kb-text-tags" className={FIELD_LABEL}>
                      {tAdd("tagsLabel")}
                    </label>
                    <Input
                      id="kb-text-tags"
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                      placeholder={tAdd("tagsPlaceholder")}
                    />
                    <p className="text-xs text-muted-foreground mt-1">{t("tagsHelp")}</p>
                  </TabsContent>
                  <TabsContent value="document" className="mt-0">
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <div className="min-w-0">
                        <label htmlFor="kb-text-id" className={FIELD_LABEL}>
                          {tAdd("documentIdLabel")}
                        </label>
                        <Input
                          id="kb-text-id"
                          value={documentId}
                          onChange={(e) => setDocumentId(e.target.value)}
                          placeholder={tAdd("documentIdPlaceholder")}
                          autoComplete="off"
                        />
                      </div>
                      <div className="min-w-0">
                        <label htmlFor="kb-text-filename" className={FIELD_LABEL}>
                          {t("filenameLabel")}
                        </label>
                        <Input
                          id="kb-text-filename"
                          value={filename}
                          onChange={(e) => setFilename(e.target.value)}
                          placeholder={t("filenamePlaceholder")}
                        />
                      </div>
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </TabsContent>

            <TabsContent value="upload" className="mt-3">
              <input
                ref={inputRef}
                type="file"
                multiple
                accept={ACCEPTED_FILE_TYPES}
                className="hidden"
                id="kb-file-upload"
                data-testid="kb-upload-input"
                onChange={(e) => {
                  if (e.target.files?.length) add(e.target.files);
                  e.target.value = "";
                }}
              />
              <label
                htmlFor="kb-file-upload"
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files.length) add(e.dataTransfer.files);
                }}
                className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                  dragOver
                    ? "border-primary bg-accent/50"
                    : "border-muted-foreground/25 hover:border-primary/50 hover:bg-accent/50"
                }`}
              >
                <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                <span className="text-sm text-muted-foreground">{tAdd("clickToSelectFiles")}</span>
                <span className="text-xs text-muted-foreground/70 mt-1 px-4 text-center">
                  {t("uploadHint")}
                </span>
              </label>

              {files.length > 0 && (
                <>
                  {/* Quick path: the same tags on every file, merged with their own. */}
                  <div className="mt-3 flex items-end gap-2">
                    <div className="min-w-0 flex-1">
                      <label htmlFor="kb-all-tags" className={FIELD_LABEL}>
                        {t("allTagsLabel")}
                      </label>
                      <Input
                        id="kb-all-tags"
                        value={allTags}
                        onChange={(e) => setAllTags(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            applyTagsToAll();
                          }
                        }}
                        placeholder={tAdd("fileTagsPlaceholder")}
                        className="h-8 text-sm"
                        disabled={busy}
                      />
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8"
                      onClick={applyTagsToAll}
                      disabled={busy || parseTags(allTags).length === 0}
                    >
                      {t("applyToAll")}
                    </Button>
                  </div>

                  <div className="mt-3 space-y-1">
                    {files.map((f) => {
                      const tagCount = parseTags(f.tags).length;
                      return (
                        <div key={f.key} className="bg-muted rounded-md overflow-hidden">
                          <div className="flex items-center gap-1 px-2 py-2">
                            <button
                              type="button"
                              className="flex items-center gap-1.5 min-w-0 flex-1 text-left hover:opacity-75 transition-opacity"
                              onClick={() => update(f.key, { expanded: !f.expanded })}
                              title={t("editFile")}
                            >
                              {f.expanded ? (
                                <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                              ) : (
                                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                              )}
                              <FileText
                                className={`h-4 w-4 shrink-0 ${
                                  tagCount > 0 || f.documentId
                                    ? "text-primary"
                                    : "text-muted-foreground"
                                }`}
                              />
                              <span className="text-sm truncate">{f.file.name}</span>
                              <span className="text-xs text-muted-foreground shrink-0">
                                ({formatBytes(f.file.size)})
                              </span>
                              {tagCount > 0 && (
                                <span className="ml-1 shrink-0 truncate text-xs text-primary">
                                  {parseTags(f.tags)
                                    .map((tag) => `#${tag}`)
                                    .join(" ")}
                                </span>
                              )}
                            </button>
                            {f.state === "done" && (
                              <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
                            )}
                            {f.state === "failed" && (
                              <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
                            )}
                            {f.state !== "uploading" && f.state !== "done" && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0 shrink-0"
                                onClick={() =>
                                  setFiles((prev) => prev.filter((x) => x.key !== f.key))
                                }
                                aria-label={t("remove", { name: f.file.name })}
                                disabled={busy}
                              >
                                <X className="h-4 w-4" />
                              </Button>
                            )}
                          </div>

                          {(f.state === "uploading" || f.state === "done") && (
                            <div className="mx-2 mb-2 h-1.5 overflow-hidden rounded-full bg-background">
                              <div
                                className="h-full rounded-full bg-primary transition-[width] duration-200"
                                style={{ width: `${Math.round(f.progress * 100)}%` }}
                              />
                            </div>
                          )}
                          {f.error && (
                            <p className="px-3 pb-2 text-xs text-red-600 dark:text-red-400">
                              {f.error}
                            </p>
                          )}

                          {f.expanded && f.state !== "done" && (
                            <div className="border-t border-border/50">
                              <Tabs
                                value={f.subTab}
                                onValueChange={(v) =>
                                  update(f.key, { subTab: v as "document" | "tags" })
                                }
                              >
                                <TabsList className={SUB_TAB_LIST}>
                                  <TabsTrigger value="tags" className={SUB_TAB}>
                                    {tAdd("tagsLabel")}
                                  </TabsTrigger>
                                  <TabsTrigger value="document" className={SUB_TAB}>
                                    {t("documentTab")}
                                  </TabsTrigger>
                                </TabsList>
                                <div className="px-3 py-3">
                                  <TabsContent value="tags" className="mt-0">
                                    <label className={FIELD_LABEL}>{tAdd("tagsLabel")}</label>
                                    <Input
                                      value={f.tags}
                                      onChange={(e) => update(f.key, { tags: e.target.value })}
                                      placeholder={tAdd("fileTagsPlaceholder")}
                                      aria-label={t("tagsFor", { name: f.file.name })}
                                      className="h-8 text-sm"
                                      disabled={busy}
                                    />
                                    <p className="text-xs text-muted-foreground mt-1">
                                      {t("tagsHelp")}
                                    </p>
                                  </TabsContent>
                                  <TabsContent value="document" className="mt-0">
                                    <label className={FIELD_LABEL}>{tAdd("documentIdLabel")}</label>
                                    <Input
                                      value={f.documentId}
                                      onChange={(e) =>
                                        update(f.key, { documentId: e.target.value })
                                      }
                                      placeholder={tAdd("fileIdPlaceholder")}
                                      className="h-8 text-sm"
                                      disabled={busy}
                                    />
                                  </TabsContent>
                                </div>
                              </Tabs>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </TabsContent>
          </Tabs>
        </div>

        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)} disabled={busy}>
            {tAdd("cancel")}
          </Button>
          {tab === "text" ? (
            <Button onClick={writeText} disabled={busy || text.trim() === ""}>
              {busy ? tAdd("addingDocument") : tAdd("addDocumentSubmit")}
            </Button>
          ) : (
            <Button onClick={uploadAll} disabled={busy || pending.length === 0}>
              {busy ? tAdd("uploading") : tAdd("uploadFiles", { count: pending.length })}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
