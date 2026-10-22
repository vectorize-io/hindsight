"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Eye, EyeOff, MessagesSquare, Search, CheckCheck } from "lucide-react";
import { client, type SessionSummary } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { isActive, relativeTime } from "@/lib/session-timeline";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { Spinner } from "@/components/ui/spinner";
import { HarnessLogo } from "@/components/ui/harness-logo";
import { resolveHarnessLogo } from "@/lib/harness-logo";
import { documentHarness } from "@/lib/harness-logo";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { SessionDetail } from "./session-detail";

const PAGE_SIZE = 50;
// The list is cheap (one indexed query) and a session's counts move every few
// seconds while an agent works, so it refreshes on a short cadence while visible.
const LIST_REFRESH_MS = 5000;
const SEARCH_DEBOUNCE_MS = 250;

/**
 * The client streaming a session, read like a document's: the `harness` metadata the
 * coding-agents plugin stamps (and the session's own `source` as a fallback).
 */
export function sessionHarness(session: Pick<SessionSummary, "metadata">) {
  return documentHarness(session.metadata, null);
}

/** The state dot every session row and header shares: pulsing when live, a spinner while checking. */
export function SessionStatusDot({
  session,
  now,
  className,
}: {
  session: Pick<SessionSummary, "last_event_at" | "checking">;
  now: number;
  className?: string;
}) {
  const t = useTranslations("sessionsView");
  if (session.checking) {
    return (
      <span className={cn("relative inline-flex h-2.5 w-2.5", className)} title={t("checking")}>
        <span className="absolute inset-0 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
      </span>
    );
  }
  const live = isActive(session, now);
  return (
    <span
      className={cn("relative inline-flex h-2.5 w-2.5", className)}
      title={live ? t("statusLive") : t("statusIdle")}
    >
      {live && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-60" />
      )}
      <span
        className={cn(
          "relative inline-flex h-2.5 w-2.5 rounded-full",
          live ? "bg-emerald-500" : "bg-muted-foreground/30"
        )}
      />
    </span>
  );
}

/**
 * Sessions tab: the bank's live conversations, and — when `sessionId` is set — one
 * of them followed live. The open session lives in the URL (`?session=`) so a link
 * to it can be shared and survives a reload.
 */
export function SessionsView({
  sessionId,
  onOpenSession,
}: {
  sessionId: string | null;
  onOpenSession: (id: string | null) => void;
}) {
  if (sessionId) {
    return <SessionDetail sessionId={sessionId} onBack={() => onOpenSession(null)} />;
  }
  return <SessionsList onOpenSession={onOpenSession} />;
}

function SessionsList({ onOpenSession }: { onOpenSession: (id: string) => void }) {
  const t = useTranslations("sessionsView");
  const locale = useLocale();
  const { currentBank } = useBank();
  const [items, setItems] = useState<SessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [page, setPage] = useState(1);
  const [now, setNow] = useState(() => Date.now());
  // The newest request wins: a slow refresh must not overwrite a newer search.
  const requestRef = useRef(0);

  useEffect(() => {
    const id = window.setTimeout(() => {
      setDebouncedQuery(query.trim());
      setPage(1);
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(id);
  }, [query]);

  const load = useCallback(async () => {
    if (!currentBank) return;
    const req = ++requestRef.current;
    try {
      const res = await client.listSessions(currentBank, {
        q: debouncedQuery || undefined,
        limit: PAGE_SIZE,
        offset: (page - 1) * PAGE_SIZE,
      });
      if (req !== requestRef.current) return;
      setItems(res.items);
      setTotal(res.total);
      setError(null);
    } catch (e) {
      if (req !== requestRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (req === requestRef.current) {
        setLoaded(true);
        setNow(Date.now());
      }
    }
  }, [currentBank, debouncedQuery, page]);

  // Refresh in place (rows keyed by id, no spinner) so the table never flickers,
  // and only while the tab is visible.
  useEffect(() => {
    load();
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") load();
    }, LIST_REFRESH_MS);
    const onVisible = () => document.visibilityState === "visible" && load();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [load]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
          <p className="text-muted-foreground max-w-2xl">{t("description")}</p>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("searchPlaceholder")}
            aria-label={t("searchPlaceholder")}
            className="pl-8"
          />
        </div>
        {loaded && (
          <span className="text-xs text-muted-foreground">
            {t("sessionCount", { count: total })}
          </span>
        )}
      </div>

      {!loaded ? (
        <div className="flex justify-center py-16">
          <Spinner size="lg" />
        </div>
      ) : error && items.length === 0 ? (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border py-16 text-center">
          <MessagesSquare className="h-8 w-8 text-muted-foreground/60" />
          <div className="text-sm font-medium">
            {debouncedQuery ? t("emptySearch") : t("empty")}
          </div>
          {!debouncedQuery && (
            <p className="max-w-md px-4 text-xs text-muted-foreground">{t("emptyHint")}</p>
          )}
        </div>
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("colSession")}</TableHead>
                <TableHead className="w-[11rem]">{t("colActivity")}</TableHead>
                <TableHead className="w-[8rem]">{t("colNotes")}</TableHead>
                <TableHead className="w-[9rem]">{t("colLastActivity")}</TableHead>
                <TableHead className="w-[6rem]">{t("colLive")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((s) => (
                <TableRow
                  key={s.session_id}
                  className="cursor-pointer"
                  onClick={() => onOpenSession(s.session_id)}
                >
                  <TableCell>
                    <div className="flex min-w-0 items-center gap-2.5">
                      <SessionStatusDot session={s} now={now} className="shrink-0" />
                      <HarnessLogo harness={sessionHarness(s)} size={16} />
                      <div className="min-w-0">
                        <button
                          type="button"
                          className="block max-w-full truncate text-left font-medium hover:underline"
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenSession(s.session_id);
                          }}
                        >
                          {s.session_id}
                        </button>
                        <div className="truncate text-[11px] text-muted-foreground">
                          {resolveHarnessLogo(sessionHarness(s))?.label ?? ""}
                        </div>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                    {t("counts", {
                      events: s.event_count,
                      checks: s.check_count,
                    })}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    {s.note_count > 0 ? (
                      <span
                        className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
                        title={
                          s.acked_seq > 0
                            ? t("deliveredUpTo", { seq: s.acked_seq })
                            : t("noneDelivered")
                        }
                      >
                        {t("noteCount", { count: s.note_count })}
                        {s.acked_seq > 0 && <CheckCheck className="h-3 w-3" />}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                    {s.note_count > 0 && s.credit_count + s.dismissed_count > 0 && (
                      <span
                        className="ml-1.5 text-[11px] text-muted-foreground"
                        title={t("noteMentions", {
                          used: s.credit_count,
                          skipped: s.dismissed_count,
                        })}
                      >
                        {t("usedCount", { count: s.credit_count })}
                      </span>
                    )}
                  </TableCell>
                  <TableCell
                    className="text-xs text-muted-foreground whitespace-nowrap"
                    title={new Date(s.last_event_at ?? s.created_at).toLocaleString(locale)}
                  >
                    {relativeTime(s.last_event_at ?? s.created_at, locale, now)}
                  </TableCell>
                  <TableCell>
                    {s.live ? (
                      <span className="inline-flex items-center gap-1 text-xs text-foreground">
                        <Eye className="h-3.5 w-3.5 text-primary" />
                        {t("liveOn")}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                        <EyeOff className="h-3.5 w-3.5" />
                        {t("liveOff")}
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Pagination page={page} pageSize={PAGE_SIZE} total={total} onPageChange={setPage} />
        </>
      )}
    </div>
  );
}
