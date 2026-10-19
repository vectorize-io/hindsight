"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronDown, ChevronRight, History } from "lucide-react";
import { client, type DocumentEvent, type DocumentEventKind } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MemoryDetailModal } from "./memory-detail-modal";

// Same pill as the operations status badges; the colour says what kind of change it was.
const KIND_PILL: Record<DocumentEventKind, string> = {
  retain: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  append: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  reprocess: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  retag: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  fact_edited: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  fact_invalidated: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  fact_restored: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  fact_deleted: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  observations_cleared: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  consolidation: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
};

// The diff chips the observation history uses: red struck-through for removed, green for added.
const ID_CHIP = {
  created: "bg-green-500/15 text-green-700 dark:text-green-400 border-green-500/20",
  updated: "bg-muted text-muted-foreground border-border",
  removed: "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/20 line-through",
} as const;
type IdRole = keyof typeof ID_CHIP;
// Created or updated here, removed by a later change: faded and dashed, so it reads as gone.
const LATER_REMOVED = "opacity-50 border-dashed";

function Delta({ plus, changed, minus }: { plus: number; changed?: number; minus: number }) {
  if (!plus && !changed && !minus) return <span className="text-muted-foreground">—</span>;
  return (
    <span className="font-mono text-xs space-x-1.5">
      {plus > 0 && <span className="text-green-700 dark:text-green-400">+{plus}</span>}
      {!!changed && <span className="text-muted-foreground">~{changed}</span>}
      {minus > 0 && <span className="text-red-700 dark:text-red-400">−{minus}</span>}
    </span>
  );
}

const PAGE = 50;

export function DocumentHistory({ bankId, documentId }: { bankId: string; documentId: string }) {
  const t = useTranslations("documentHistory");
  const [events, setEvents] = useState<DocumentEvent[] | null>(null);
  const [total, setTotal] = useState(0);
  // Pages count operations, not events (see the API): the next page starts after the ones loaded.
  const [opsLoaded, setOpsLoaded] = useState(0);
  const [error, setError] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = async (offset: number) => {
    const data = await client.getDocumentHistory(documentId, bankId, offset, PAGE);
    setEvents((prev) => [...(offset ? (prev ?? []) : []), ...data.items]);
    setTotal(data.total);
    setOpsLoaded(offset + PAGE);
  };

  useEffect(() => {
    load(0).catch(() => setError(true));
  }, [bankId, documentId]);

  const loadOlder = async () => {
    setLoadingMore(true);
    try {
      await load(opsLoaded);
    } catch {
      setError(true);
    } finally {
      setLoadingMore(false);
    }
  };

  // The change that removed each observation, so a chip for one created or updated earlier can
  // say what deleted it instead of offering to open an observation that no longer exists.
  const removedBy = useMemo(
    () =>
      new Map(
        (events ?? []).flatMap((e) => e.observations_removed.ids.map((id) => [id, e] as const))
      ),
    [events]
  );

  if (error)
    return <div className="py-10 text-center text-sm text-muted-foreground">{t("loadError")}</div>;
  if (!events) {
    return (
      <div className="flex justify-center py-20">
        <Spinner size="xl" variant="jump" />
      </div>
    );
  }
  if (events.length === 0) {
    return (
      <div className="py-20 text-center">
        <History className="mx-auto mb-3 h-10 w-10 text-muted-foreground/50" />
        <div className="text-sm text-muted-foreground">{t("empty")}</div>
      </div>
    );
  }

  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const chip = (id: string, role: IdRole) => {
    const laterRemoval = role !== "removed" ? removedBy.get(id) : undefined;
    const gone = role === "removed" || !!laterRemoval;
    const status =
      role === "removed"
        ? t("chipRemoved")
        : laterRemoval
          ? t("chipRemovedLater", {
              kind: t(`kind.${laterRemoval.kind}`),
              when: new Date(laterRemoval.created_at).toLocaleString(),
            })
          : t("chipOpen");
    return (
      <button
        key={`${role}-${id}`}
        type="button"
        title={`${id}\n${status}`}
        disabled={gone}
        onClick={() => setOpenId(id)}
        onMouseEnter={() => setHoverId(id)}
        onMouseLeave={() => setHoverId(null)}
        className={`text-[10px] px-1.5 py-0.5 rounded-md border font-mono ${ID_CHIP[role]} ${
          hoverId === id ? "ring-1 ring-primary" : ""
        } ${laterRemoval ? LATER_REMOVED : ""} ${gone ? "cursor-default" : "hover:underline"}`}
      >
        {id.slice(0, 8)}
      </button>
    );
  };

  const idRow = (label: string, sample: DocumentEvent["observations_removed"], role: IdRole) =>
    sample.count > 0 && (
      <div className="flex flex-wrap items-center gap-1">
        <span className="w-20 text-xs text-muted-foreground">{label}</span>
        {sample.ids.map((id) => chip(id, role))}
        {sample.count > sample.ids.length && (
          <span className="text-xs text-muted-foreground">
            {t("moreIds", { count: sample.count - sample.ids.length })}
          </span>
        )}
      </div>
    );

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">{t("explainer")}</p>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className={`h-3 w-5 rounded-sm border ${ID_CHIP.created}`} />
          {t("legendExists")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className={`h-3 w-5 rounded-sm border ${ID_CHIP.created} ${LATER_REMOVED}`} />
          {t("legendRemovedLater")}
        </span>
        <span className="flex items-center gap-1.5">
          <span className={`h-3 w-5 rounded-sm border ${ID_CHIP.removed}`} />
          {t("legendRemovedHere")}
        </span>
      </div>
      <div className="rounded-lg border border-border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-6" />
              <TableHead>{t("colWhen")}</TableHead>
              <TableHead>{t("colChange")}</TableHead>
              <TableHead>{t("colFacts")}</TableHead>
              <TableHead>{t("colObservations")}</TableHead>
              <TableHead className="w-[100px]">{t("colOperation")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {events.map((e, i) => {
              const hasIds =
                e.observations_created.count +
                  e.observations_updated.count +
                  e.observations_removed.count >
                0;
              const open = expanded.has(i);
              const touchesHovered =
                !!hoverId &&
                [
                  ...e.observations_created.ids,
                  ...e.observations_updated.ids,
                  ...e.observations_removed.ids,
                ].includes(hoverId);
              return (
                <Fragment key={`${e.operation_id}-${i}`}>
                  <TableRow
                    className={`${hasIds ? "cursor-pointer" : ""} hover:bg-muted/50 ${
                      touchesHovered ? "bg-primary/5" : ""
                    }`}
                    onClick={() => hasIds && toggle(i)}
                  >
                    <TableCell className="text-muted-foreground">
                      {hasIds &&
                        (open ? (
                          <ChevronDown className="h-3.5 w-3.5" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5" />
                        ))}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground whitespace-nowrap">
                      {new Date(e.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${KIND_PILL[e.kind]}`}
                        >
                          {t(`kind.${e.kind}`)}
                        </span>
                        {e.fact_id && (
                          <button
                            type="button"
                            title={e.fact_id}
                            onClick={(ev) => {
                              ev.stopPropagation();
                              setOpenId(e.fact_id);
                            }}
                            className="font-mono text-xs text-muted-foreground hover:text-foreground hover:underline"
                          >
                            {e.fact_id.slice(0, 8)}
                          </button>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Delta plus={e.facts_added} minus={e.facts_removed} />
                    </TableCell>
                    <TableCell>
                      <Delta
                        plus={e.observations_created.count}
                        changed={e.observations_updated.count}
                        minus={e.observations_removed.count}
                      />
                    </TableCell>
                    <TableCell
                      className="font-mono text-xs text-muted-foreground"
                      title={e.operation_id ?? undefined}
                    >
                      {e.operation_id?.slice(0, 8)}
                    </TableCell>
                  </TableRow>
                  {open && (
                    <TableRow className="bg-muted/20 hover:bg-muted/20">
                      <TableCell />
                      <TableCell colSpan={5} className="space-y-1.5 py-2">
                        {idRow(t("idsRemoved"), e.observations_removed, "removed")}
                        {idRow(t("idsCreated"), e.observations_created, "created")}
                        {idRow(t("idsUpdated"), e.observations_updated, "updated")}
                      </TableCell>
                    </TableRow>
                  )}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>
      {total > opsLoaded && (
        <div className="flex justify-center">
          <Button variant="outline" size="sm" disabled={loadingMore} onClick={loadOlder}>
            {t("loadOlder")}
          </Button>
        </div>
      )}
      <MemoryDetailModal memoryId={openId} onClose={() => setOpenId(null)} />
    </div>
  );
}
