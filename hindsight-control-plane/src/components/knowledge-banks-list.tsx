"use client";

// The knowledge banks overview: every knowledge bank with what is in it, plus
// create and delete. A row opens the bank's documents.

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { AlertTriangle, Library, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import {
  MEDIA_TYPES,
  formatBytes,
  knowledgeBankRoute,
  type KnowledgeBank,
} from "@/lib/knowledge-banks";
import { formatRelativeTime } from "@/lib/relative-time";
import { KindIcon } from "@/components/knowledge-passage";
import { BankKindSwitch } from "@/components/bank-kind-switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { ToneChip } from "@/components/ui/facet-chip";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function CreateKnowledgeBankDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useTranslations("knowledgeBanks.create");
  const tCommon = useTranslations("common");
  const router = useRouter();
  const [bankId, setBankId] = React.useState("");
  const [creating, setCreating] = React.useState(false);

  const id = bankId.trim();
  // The API owns the id rules (knowledge and memory banks share one id space, so it
  // also refuses an id a memory bank already has); here an id only has to be present.
  const idValid = id !== "";

  const create = async () => {
    if (!idValid || creating) return;
    setCreating(true);
    try {
      await client.createKnowledgeBank(id);
      toast.success(t("created", { bank: id }));
      onOpenChange(false);
      setBankId("");
      // Straight into the new bank: the next thing anyone does is add documents.
      router.push(knowledgeBankRoute(id));
    } catch {
      // The API client already toasted the reason (e.g. the id is taken).
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("title")}</DialogTitle>
          <DialogDescription>{t("description")}</DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            create();
          }}
        >
          <div className="space-y-1.5">
            <Label htmlFor="kb-create-id">{t("idLabel")}</Label>
            <Input
              id="kb-create-id"
              autoFocus
              placeholder="product-docs"
              value={bankId}
              onChange={(e) => setBankId(e.target.value)}
              autoComplete="off"
            />
            <p className="text-xs text-muted-foreground">{t("idHelp")}</p>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {tCommon("cancel")}
            </Button>
            <Button type="submit" disabled={!idValid || creating}>
              {creating && <Spinner size="sm" className="mr-2" />}
              {t("submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function DeleteKnowledgeBankDialog({
  bank,
  onOpenChange,
  onDeleted,
}: {
  bank: KnowledgeBank | null;
  onOpenChange: (open: boolean) => void;
  onDeleted: () => void;
}) {
  const t = useTranslations("knowledgeBanks.delete");
  const tCommon = useTranslations("common");
  const [deleting, setDeleting] = React.useState(false);

  const remove = async () => {
    if (!bank) return;
    setDeleting(true);
    try {
      await client.deleteKnowledgeBank(bank.bank_id);
      toast.success(t("deleted", { bank: bank.bank_id }));
      onOpenChange(false);
      onDeleted();
    } catch {
      // toasted by the API client
    } finally {
      setDeleting(false);
    }
  };

  return (
    <AlertDialog open={bank !== null} onOpenChange={(open) => !deleting && onOpenChange(open)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          {/* Same confirm as deleting a memory bank: the prompt, then the warning. */}
          <AlertDialogTitle>{t("title")}</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                {t.rich("prompt", {
                  bank: () => (
                    <span className="font-semibold text-foreground">{bank?.bank_id}</span>
                  ),
                })}
              </p>
              <p className="text-red-600 dark:text-red-400 font-medium">
                {t("body", {
                  documents: bank?.document_count ?? 0,
                  passages: bank?.passage_count ?? 0,
                })}
              </p>
            </div>
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
            {t("submit")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** "3 PDF · 1 audio" as icons with counts. */
function KindCounts({ counts }: { counts: KnowledgeBank["documents_by_media_type"] }) {
  const t = useTranslations("knowledgeBanks.kinds");
  const present = MEDIA_TYPES.filter((kind) => (counts[kind] ?? 0) > 0);
  if (present.length === 0) return <span className="text-xs text-muted-foreground/60">—</span>;
  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
      {present.map((kind) => (
        <span
          key={kind}
          className="inline-flex items-center gap-1 text-xs tabular-nums text-muted-foreground"
          title={`${counts[kind]} ${t(kind)}`}
        >
          <KindIcon kind={kind} className="h-3.5 w-3.5" />
          {counts[kind]}
        </span>
      ))}
    </div>
  );
}

/** Pending / failed counts as chips; nothing when everything is ready. */
function IngestCounts({ bank }: { bank: KnowledgeBank }) {
  const t = useTranslations("knowledgeBanks.list");
  if (bank.pending_count === 0 && bank.failed_count === 0) {
    return bank.document_count > 0 ? (
      <ToneChip tone="emerald" label={t("allReady")} />
    ) : (
      <span className="text-xs text-muted-foreground/60">—</span>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {bank.pending_count > 0 && (
        <ToneChip
          tone="amber"
          icon={<Spinner size="xs" />}
          label={t("pendingCount", { count: bank.pending_count })}
        />
      )}
      {bank.failed_count > 0 && (
        <ToneChip
          tone="red"
          icon={<AlertTriangle className="h-3 w-3" />}
          label={t("failedCount", { count: bank.failed_count })}
        />
      )}
    </div>
  );
}

function ListSkeleton() {
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="h-14 animate-pulse rounded-lg bg-muted/60" />
      ))}
    </div>
  );
}

export function KnowledgeBanksList() {
  const t = useTranslations("knowledgeBanks.list");
  const router = useRouter();
  const [banks, setBanks] = React.useState<KnowledgeBank[] | null>(null);
  const [failed, setFailed] = React.useState(false);
  const [creating, setCreating] = React.useState(false);
  const [deleting, setDeleting] = React.useState<KnowledgeBank | null>(null);

  const load = React.useCallback(async () => {
    try {
      setBanks(await client.listAllKnowledgeBanks());
      setFailed(false);
    } catch {
      setFailed(true);
      setBanks((prev) => prev ?? []);
    }
  }, []);

  React.useEffect(() => {
    load();
  }, [load]);

  // Ingestion runs in the background: while any bank has documents in flight, keep
  // its counts moving instead of freezing them at the moment the page opened.
  const anyPending = banks?.some((b) => b.pending_count > 0) ?? false;
  React.useEffect(() => {
    if (!anyPending) return;
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [anyPending, load]);

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6">
      <BankKindSwitch
        value="knowledge"
        onChange={(kind) => kind === "memory" && router.push("/dashboard")}
      />
      <div className="mb-6 mt-4 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-foreground">{t("title")}</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{t("subtitle")}</p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button variant="ghost" size="sm" onClick={load} title={t("refresh")}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button size="sm" onClick={() => setCreating(true)}>
            <Plus className="mr-1.5 h-4 w-4" />
            {t("create")}
          </Button>
        </div>
      </div>

      {banks === null ? (
        <ListSkeleton />
      ) : banks.length === 0 ? (
        <div className="flex flex-col items-center rounded-xl border border-dashed border-border bg-card px-6 py-16 text-center">
          <Library className="mb-4 h-10 w-10 text-muted-foreground" />
          <h2 className="text-lg font-semibold text-card-foreground">
            {failed ? t("loadFailed") : t("emptyTitle")}
          </h2>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            {failed ? t("loadFailedBody") : t("emptyBody")}
          </p>
          <Button className="mt-6" onClick={failed ? load : () => setCreating(true)}>
            {failed ? (
              <RefreshCw className="mr-1.5 h-4 w-4" />
            ) : (
              <Plus className="mr-1.5 h-4 w-4" />
            )}
            {failed ? t("retry") : t("create")}
          </Button>
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("columnBank")}</TableHead>
              <TableHead className="w-28 text-right">{t("columnDocuments")}</TableHead>
              <TableHead className="w-28 text-right">{t("columnPassages")}</TableHead>
              <TableHead className="w-24 text-right">{t("columnSize")}</TableHead>
              <TableHead className="w-56">{t("columnKinds")}</TableHead>
              <TableHead className="w-36">{t("columnStatus")}</TableHead>
              <TableHead className="w-32">{t("columnCreated")}</TableHead>
              <TableHead className="w-12" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {banks.map((bank, index) => (
              <TableRow
                key={bank.bank_id}
                onClick={() => router.push(knowledgeBankRoute(bank.bank_id))}
                className="cursor-pointer animate-list-row-enter"
                style={{ animationDelay: `${Math.min(index, 10) * 18}ms` }}
              >
                <TableCell>
                  <Link
                    href={knowledgeBankRoute(bank.bank_id)}
                    onClick={(e) => e.stopPropagation()}
                    className="font-medium text-card-foreground hover:underline focus-visible:underline"
                  >
                    {bank.bank_id}
                  </Link>
                </TableCell>
                <TableCell className="text-right tabular-nums">{bank.document_count}</TableCell>
                <TableCell className="text-right tabular-nums">{bank.passage_count}</TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">
                  {formatBytes(bank.total_bytes)}
                </TableCell>
                <TableCell>
                  <KindCounts counts={bank.documents_by_media_type} />
                </TableCell>
                <TableCell>
                  <IngestCounts bank={bank} />
                </TableCell>
                <TableCell className="whitespace-nowrap text-sm text-muted-foreground">
                  {bank.created_at ? formatRelativeTime(bank.created_at) : "—"}
                </TableCell>
                <TableCell onClick={(e) => e.stopPropagation()}>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                    title={t("deleteBank")}
                    aria-label={t("deleteBank")}
                    onClick={() => setDeleting(bank)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <CreateKnowledgeBankDialog open={creating} onOpenChange={setCreating} />
      <DeleteKnowledgeBankDialog
        bank={deleting}
        onOpenChange={(open) => !open && setDeleting(null)}
        onDeleted={load}
      />
    </div>
  );
}
