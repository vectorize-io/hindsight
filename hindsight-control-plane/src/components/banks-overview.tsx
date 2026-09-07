"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { Trash2, Search, X, Plus } from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import { useBank, type BankInfo } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";
import { formatRelativeTime } from "@/lib/relative-time";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { Spinner } from "@/components/ui/spinner";
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

const ITEMS_PER_PAGE = 50;

/** How many bank names the confirmation dialog spells out before collapsing the rest. */
const CONFIRM_LIST_LIMIT = 8;

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 10_000 ? 0 : 1)}k`;
  return n.toString();
}

/** The list endpoint's rows are untyped JSON; fill the optional stats so the table can
 *  assume numbers and nulls (a missing `fact_count` would poison the bar's max). */
function toBankInfo(bank: Record<string, unknown>): BankInfo {
  return {
    bank_id: String(bank.bank_id),
    name: (bank.name as string) ?? null,
    mission: (bank.mission as string) ?? null,
    created_at: (bank.created_at as string) ?? null,
    updated_at: (bank.updated_at as string) ?? null,
    fact_count: (bank.fact_count as number) ?? 0,
    last_document_at: (bank.last_document_at as string) ?? null,
    last_write_at: (bank.last_write_at as string) ?? null,
  };
}

/** Last write, not last ingestion: appends to an existing document bump `last_write_at` only. */
function lastActivityOf(bank: BankInfo): string | null {
  return bank.last_write_at || bank.last_document_at;
}

interface DeleteFailure {
  bankId: string;
  message: string;
}

interface DeleteProgress {
  total: number;
  done: number;
  failures: DeleteFailure[];
  /** Set once the loop has finished, so the dialog can switch from progress to result. */
  finished: boolean;
}

export function BanksOverview() {
  const t = useTranslations("banksOverview");
  const tCommon = useTranslations("common");
  const tNavBank = useTranslations("nav.bank");
  const router = useRouter();
  // The overview pages discretely while the header selector scrolls infinitely, so it
  // cannot share the selector's cumulative list — it fetches its own page. `loadBanks`
  // is still used to keep that selector in sync after a bulk delete.
  const { currentBank, setCurrentBank, loadBanks } = useBank();

  const [banks, setBanks] = React.useState<BankInfo[]>([]);
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(true);
  const [page, setPage] = React.useState(1);
  const [search, setSearch] = React.useState("");
  const [searchDraft, setSearchDraft] = React.useState("");

  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [confirmOpen, setConfirmOpen] = React.useState(false);
  const [confirmText, setConfirmText] = React.useState("");
  const [progress, setProgress] = React.useState<DeleteProgress | null>(null);

  // Search runs server-side (the list is paginated), so the input holds a draft that is
  // debounced into a query. A new query always restarts at page 1.
  React.useEffect(() => {
    if (searchDraft === search) return;
    const timer = setTimeout(() => {
      setSearch(searchDraft);
      setPage(1);
    }, 250);
    return () => clearTimeout(timer);
  }, [searchDraft, search]);

  // Every fetch bumps this; a response whose stamp is stale is dropped, so paging fast
  // or typing during a fetch can't leave an earlier page's rows on screen.
  const requestSeq = React.useRef(0);

  const fetchPage = React.useCallback(async (targetPage: number, query: string) => {
    const seq = ++requestSeq.current;
    setLoading(true);
    try {
      const response = await client.listBanks({
        q: query || undefined,
        limit: ITEMS_PER_PAGE,
        offset: (targetPage - 1) * ITEMS_PER_PAGE,
      });
      if (seq !== requestSeq.current) return;
      setBanks((response.banks || []).map(toBankInfo));
      setTotal(response.total ?? 0);
    } catch (error) {
      if (seq !== requestSeq.current) return;
      console.error("Error loading banks:", error);
    } finally {
      if (seq === requestSeq.current) setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchPage(page, search);
  }, [fetchPage, page, search]);

  // Only rows on screen can be selected, so a page change or a new query must drop the
  // selection — otherwise invisible banks stay armed for deletion.
  React.useEffect(() => {
    setSelected(new Set());
  }, [page, search]);

  const maxFactCount = React.useMemo(() => Math.max(1, ...banks.map((b) => b.fact_count)), [banks]);

  const selectedBanks = banks.filter((b) => selected.has(b.bank_id));
  const allSelected = banks.length > 0 && selectedBanks.length === banks.length;
  const isDeleting = progress !== null && !progress.finished;

  const toggleBank = (bankId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(bankId)) next.delete(bankId);
      else next.add(bankId);
      return next;
    });
  };

  const toggleAll = () => {
    setSelected(allSelected ? new Set() : new Set(banks.map((b) => b.bank_id)));
  };

  const runBulkDelete = async () => {
    const targets = selectedBanks.map((b) => b.bank_id);
    if (targets.length === 0) return;

    // Deliberately sequential. `delete_bank` drops the bank's partial vector indexes
    // with DROP INDEX CONCURRENTLY on the shared `memory_units` relation; concurrent
    // drops wait on each other's virtual xids and deadlock, so a fan-out here would
    // fail banks at random.
    setProgress({ total: targets.length, done: 0, failures: [], finished: false });
    const failures: DeleteFailure[] = [];
    for (const bankId of targets) {
      try {
        await client.deleteBank(bankId, { suppressErrorToast: true });
        if (bankId === currentBank) setCurrentBank(null);
      } catch (error) {
        failures.push({
          bankId,
          message: error instanceof Error ? error.message : String(error),
        });
      }
      setProgress((prev) => (prev ? { ...prev, done: prev.done + 1, failures } : prev));
    }

    const deletedCount = targets.length - failures.length;
    // The current page may have emptied out; step back rather than showing a blank one.
    const remaining = total - deletedCount;
    const lastPage = Math.max(1, Math.ceil(remaining / ITEMS_PER_PAGE));
    const nextPage = Math.min(page, lastPage);
    if (nextPage === page) await fetchPage(page, search);
    else setPage(nextPage);
    // The header selector holds its own copy of the list.
    await loadBanks();
    // Only the banks that survived stay armed, so a retry doesn't re-delete.
    setSelected(new Set(failures.map((f) => f.bankId)));
    setProgress({ total: targets.length, done: targets.length, failures, finished: true });

    if (failures.length === 0) {
      setConfirmOpen(false);
      setProgress(null);
      toast.success(t("bulkDeleteDone", { count: deletedCount }));
    } else {
      toast.warning(t("bulkDeletePartial", { deleted: deletedCount, failed: failures.length }));
    }
  };

  const openBank = (bankId: string) => {
    setCurrentBank(bankId);
    // No view param: the bank page defaults to its home view.
    router.push(bankRoute(bankId));
  };

  const confirmWord = t("bulkDeleteConfirmWord");
  const confirmed = confirmText.trim().toLowerCase() === confirmWord.toLowerCase();

  // A server with no banks at all gets the greeting it had before this page existed —
  // a search box and an empty table say nothing to someone who has yet to create one.
  // A search that matches nothing still shows the table UI, so the query can be edited.
  if (!loading && total === 0 && !search) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-80px)] bg-muted/20">
        <div className="max-w-md rounded-lg border-2 border-border bg-card p-10 text-center shadow-lg">
          {/* Tumbling Hindsight mascot — loops forever as a friendly greeting. */}
          <Spinner size="xl" variant="jump" className="mx-auto mb-4" />
          <h3 className="mb-3 text-2xl font-bold text-card-foreground">{t("welcomeTitle")}</h3>
          <p className="mb-6 text-muted-foreground">{t("welcomeBody")}</p>
          <Button onClick={() => window.dispatchEvent(new CustomEvent("hindsight:create-bank"))}>
            <Plus className="mr-1.5 h-4 w-4" />
            {tNavBank("create")}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-8">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{t("title")}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t("subtitle")}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          // The create dialog (with its template import) lives in the header selector;
          // this asks that component to open it rather than shipping a second copy.
          onClick={() => window.dispatchEvent(new CustomEvent("hindsight:create-bank"))}
        >
          <Plus className="mr-1.5 h-4 w-4" />
          {tNavBank("create")}
        </Button>
      </div>

      <div className="mb-4 flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            placeholder={t("search")}
            className="pl-9"
          />
        </div>
        {selected.size > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground tabular-nums">
              {t("selectedCount", { count: selected.size })}
            </span>
            <Button variant="ghost" size="sm" onClick={() => setSelected(new Set())}>
              <X className="mr-1 h-3.5 w-3.5" />
              {tCommon("clear")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setConfirmText("");
                setProgress(null);
                setConfirmOpen(true);
              }}
              className="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" />
              {t("deleteSelected")}
            </Button>
          </div>
        )}
      </div>

      {loading && banks.length === 0 ? (
        <div className="flex items-center justify-center gap-2 py-16 text-muted-foreground">
          <Spinner size="sm" />
          <span>{tCommon("loading")}</span>
        </div>
      ) : banks.length === 0 ? (
        <div className="rounded-xl border border-border bg-card py-16 text-center text-muted-foreground">
          {t("noSearchResults")}
        </div>
      ) : (
        <>
          <Table
            className={cn(
              "transition-opacity duration-150 motion-reduce:transition-none",
              loading && "opacity-40"
            )}
          >
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">
                  <Checkbox
                    checked={allSelected}
                    onCheckedChange={toggleAll}
                    aria-label={t("selectAll")}
                  />
                </TableHead>
                <TableHead>{t("columnBank")}</TableHead>
                <TableHead className="w-48">{t("columnMemories")}</TableHead>
                <TableHead className="w-40">{t("columnLastActivity")}</TableHead>
                <TableHead className="w-40">{t("columnCreated")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {banks.map((bank, index) => {
                const lastActivity = lastActivityOf(bank);
                const barPct = (bank.fact_count / maxFactCount) * 100;
                return (
                  <TableRow
                    key={bank.bank_id}
                    onClick={() => openBank(bank.bank_id)}
                    className="cursor-pointer animate-list-row-enter"
                    style={{ animationDelay: `${Math.min(index, 10) * 18}ms` }}
                  >
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Checkbox
                        checked={selected.has(bank.bank_id)}
                        onCheckedChange={() => toggleBank(bank.bank_id)}
                        aria-label={t("selectBank", { bank: bank.name || bank.bank_id })}
                      />
                    </TableCell>
                    <TableCell>
                      <div className="font-medium text-card-foreground">
                        {bank.name || bank.bank_id}
                      </div>
                      {bank.name && bank.name !== bank.bank_id && (
                        <div className="truncate text-xs text-muted-foreground">{bank.bank_id}</div>
                      )}
                    </TableCell>
                    <TableCell>
                      {bank.fact_count > 0 ? (
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-primary/60"
                              style={{ width: `${barPct}%` }}
                            />
                          </div>
                          <span className="tabular-nums text-xs text-muted-foreground">
                            {formatCompact(bank.fact_count)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-xs italic text-muted-foreground/60">
                          {t("emptyBank")}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {lastActivity ? formatRelativeTime(lastActivity) : t("never")}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {bank.created_at ? formatRelativeTime(bank.created_at) : "—"}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>

          <Pagination
            page={page}
            pageSize={ITEMS_PER_PAGE}
            total={total}
            disabled={loading || isDeleting}
            onPageChange={setPage}
          />
        </>
      )}

      <AlertDialog
        open={confirmOpen}
        onOpenChange={(open) => {
          // The loop keeps running if the dialog is torn down mid-flight, so block it.
          if (isDeleting) return;
          setConfirmOpen(open);
          if (!open) setProgress(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("bulkDeleteTitle", { count: selectedBanks.length })}
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3 text-sm text-muted-foreground">
                {progress?.finished && progress.failures.length > 0 ? (
                  <>
                    <p>{t("bulkDeleteFailures")}</p>
                    <ul className="max-h-48 space-y-1 overflow-y-auto rounded-md bg-muted/40 p-2 font-mono text-xs">
                      {progress.failures.map((failure) => (
                        <li key={failure.bankId}>
                          <span className="text-foreground">{failure.bankId}</span> —{" "}
                          {failure.message}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <>
                    <p>{t("bulkDeletePrompt")}</p>
                    <ul className="max-h-48 space-y-1 overflow-y-auto rounded-md bg-muted/40 p-2 font-mono text-xs">
                      {selectedBanks.slice(0, CONFIRM_LIST_LIMIT).map((bank) => (
                        <li key={bank.bank_id} className="truncate text-foreground">
                          {bank.bank_id}
                        </li>
                      ))}
                      {selectedBanks.length > CONFIRM_LIST_LIMIT && (
                        <li>
                          {t("bulkDeleteMore", {
                            count: selectedBanks.length - CONFIRM_LIST_LIMIT,
                          })}
                        </li>
                      )}
                    </ul>
                    <p className="font-medium text-red-600 dark:text-red-400">
                      {t("bulkDeleteWarning")}
                    </p>
                    <p className="text-xs">{t("bulkDeleteSequentialNote")}</p>
                    {isDeleting ? (
                      <p className="flex items-center gap-2 tabular-nums">
                        <Spinner size="sm" />
                        {t("bulkDeleteProgress", { done: progress.done, total: progress.total })}
                      </p>
                    ) : (
                      <div className="space-y-1.5">
                        <label
                          htmlFor="bulk-delete-confirm"
                          className="block text-xs font-medium text-foreground"
                        >
                          {t("bulkDeleteConfirmLabel", { word: confirmWord })}
                        </label>
                        <Input
                          id="bulk-delete-confirm"
                          value={confirmText}
                          onChange={(e) => setConfirmText(e.target.value)}
                          autoComplete="off"
                        />
                      </div>
                    )}
                  </>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>
              {progress?.finished ? tCommon("close") : tCommon("cancel")}
            </AlertDialogCancel>
            {!progress?.finished && (
              <AlertDialogAction
                onClick={(e) => {
                  // The dialog closes on action by default; the loop needs it to stay
                  // open to show progress, and to survive to the failure summary.
                  e.preventDefault();
                  runBulkDelete();
                }}
                disabled={!confirmed || isDeleting}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {isDeleting ? (
                  <Spinner size="sm" className="mr-2" />
                ) : (
                  <Trash2 className="mr-2 h-4 w-4" />
                )}
                {t("deleteSelected")}
              </AlertDialogAction>
            )}
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
