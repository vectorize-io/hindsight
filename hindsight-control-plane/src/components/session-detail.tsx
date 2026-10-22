"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useLocale, useTranslations } from "next-intl";
import {
  ArrowDown,
  ArrowLeft,
  Bot,
  ChevronRight,
  CircleSlash,
  Eye,
  EyeOff,
  FileDiff,
  FileText,
  Flag,
  Pause,
  Pencil,
  Play,
  Save,
  Search,
  SquareTerminal,
  User,
  Wrench,
  X,
  Activity,
  Check,
  Clock,
  Send,
  SlidersHorizontal,
  Braces,
  CheckCheck,
} from "lucide-react";
import { client, type SessionEvent, type SessionSummary } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import {
  buildTimeline,
  checkSpans,
  decisionPath,
  filterRows,
  formatClock,
  formatDuration,
  isActive,
  isDelivered,
  isQuietCheck,
  nextPollDelay,
  relativeTime,
  replayEvents,
  replayOffsets,
  toolTarget,
  type CheckPayload,
  type CheckSpan,
  type CheckRow,
  type DecisionRoute,
  type RowFilters,
  type TimelineRow,
  type ToolFamily,
} from "@/lib/session-timeline";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { HarnessLogo } from "@/components/ui/harness-logo";
import { resolveHarnessLogo } from "@/lib/harness-logo";
import { CompactMarkdown } from "@/components/compact-markdown";
import { DocumentChunkModal } from "@/components/document-chunk-modal";
import { MemoryDetailModal } from "@/components/memory-detail-modal";
import { cn } from "@/lib/utils";
import { withBasePath } from "@/lib/base-path";
import { SessionStatusDot, sessionHarness } from "./sessions-view";

const EVENTS_PAGE = 1000;
// Re-read the summary (acked_seq, saved_seq, counts) on every Nth poll even when no
// event arrived: a client acking a note or a save finishing changes it silently.
const SUMMARY_EVERY_N_POLLS = 5;
const ERROR_RETRY_MS = 5000;
const AT_BOTTOM_PX = 120;
const FLASH_MS = 1600;
const REPLAY_TICK_MS = 100;
const REPLAY_SPEEDS = [1, 2, 5] as const;
// Two lanes from md up: the client's conversation on the left, Hindsight on the right.
// Each row keeps its place in time, so a check sits level with the moment it judged.
// From lg up a middle column holds the time rail (TimeRail): every row leaves it empty (<Gap />).
const LANES =
  "md:grid md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:grid-cols-[minmax(0,3fr)_5rem_minmax(0,2fr)]";
const Gap = () => <div aria-hidden className="hidden lg:block" />;
const HINDSIGHT_LANE = "md:border-l md:border-primary/20 md:bg-primary/[0.03] md:pl-4 md:pr-2";

type Translate = ReturnType<typeof useTranslations<"sessionsView">>;

const TRIGGERS = new Set(["task", "plan", "command", "edit", "failure"]);
const OUTCOMES = new Set(["note", "silent", "repeat", "skipped", "error"]);

function triggerLabel(t: Translate, trigger: string | undefined) {
  return trigger && TRIGGERS.has(trigger)
    ? t(`trigger.${trigger as "task" | "plan" | "command" | "edit" | "failure"}`)
    : trigger || "";
}

function outcomeLabel(t: Translate, outcome: string | undefined) {
  return outcome && OUTCOMES.has(outcome)
    ? t(`outcome.${outcome as "note" | "silent" | "repeat" | "skipped" | "error"}`)
    : outcome || "";
}

const OUTCOME_STYLE: Record<string, string> = {
  note: "bg-primary/15 text-primary border-primary/30",
  silent: "bg-muted text-muted-foreground border-border",
  repeat: "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30",
  skipped: "bg-transparent text-muted-foreground border-dashed border-border",
  error: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30",
};

// Memory checks on the time rail: all in Hindsight's colour (only a failure is red), stronger
// the more the check had to say.
const RAIL_CHECK: Record<string, string> = {
  note: "bg-primary",
  repeat: "bg-primary/60",
  silent: "bg-primary/30",
  skipped: "bg-primary/15",
  error: "bg-red-500",
};

const ROUTE_STYLE: Record<DecisionRoute, string> = {
  silent: "text-muted-foreground",
  nothing: "text-muted-foreground",
  writer: "text-teal-700 dark:text-teal-400",
  llm: "text-violet-700 dark:text-violet-400",
  unknown: "text-muted-foreground",
};

/** The element that scrolls the page: the bank page's `main`, or the document. */
function scrollParent(el: HTMLElement | null): HTMLElement {
  let node = el?.parentElement ?? null;
  while (node) {
    const oy = getComputedStyle(node).overflowY;
    if ((oy === "auto" || oy === "scroll") && node.scrollHeight > node.clientHeight) return node;
    node = node.parentElement;
  }
  return (document.scrollingElement as HTMLElement) ?? document.documentElement;
}

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

export function SessionDetail({ sessionId, onBack }: { sessionId: string; onBack: () => void }) {
  const t = useTranslations("sessionsView");
  const { currentBank } = useBank();
  const now = useNow(5000);

  const [session, setSession] = useState<SessionSummary | null>(null);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [liveChecking, setLiveChecking] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [notFound, setNotFound] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);

  const [filters, setFilters] = useState<RowFilters>({
    tools: true,
    silent: true,
    hindsightOnly: false,
    query: "",
  });
  const [replay, setReplay] = useState<{ speed: number; playing: boolean; elapsed: number } | null>(
    null
  );
  const [panel, setPanel] = useState<"filters" | "activity" | "raw" | null>(null);
  const [flashKey, setFlashKey] = useState<string | null>(null);
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const [newCount, setNewCount] = useState(0);
  const [openDocId, setOpenDocId] = useState<string | null>(null);
  const [openMemoryId, setOpenMemoryId] = useState<string | null>(null);

  const rootRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);
  // False until the first paint of the loaded log: rows mounted after it animate in.
  const [settled, setSettled] = useState(false);

  // ---- Live follow -------------------------------------------------------------
  useEffect(() => {
    if (!currentBank) return;
    let cancelled = false;
    let timer: number | undefined;
    let inFlight = false;
    let polls = 0;
    let lastSeq = 0;
    let lastEventAt = 0;

    setEvents([]);
    setSession(null);
    setLoaded(false);
    setNotFound(false);

    const refreshSummary = async () => {
      try {
        const s = await client.getSession(currentBank, sessionId);
        if (!cancelled) setSession(s);
      } catch {
        // The events poll reports the error; a stale header is fine meanwhile.
      }
    };

    const poll = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      window.clearTimeout(timer);
      let gotNew = false;
      let checking = false;
      try {
        // Catch up page by page: a long session (or a tab that was hidden for a
        // while) can be thousands of events behind.
        for (;;) {
          const res = await client.getSessionEvents(currentBank, sessionId, lastSeq, EVENTS_PAGE);
          if (cancelled) return;
          checking = res.checking;
          if (res.items.length > 0) {
            gotNew = true;
            const batch = res.items;
            lastSeq = batch[batch.length - 1].seq;
            lastEventAt = Date.parse(batch[batch.length - 1].created_at);
            setEvents((prev) => [...prev, ...batch]);
          }
          if (res.items.length < EVENTS_PAGE) break;
        }
        setLiveChecking(checking);
        setPollError(null);
        if (polls === 0 || gotNew || polls % SUMMARY_EVERY_N_POLLS === 0) await refreshSummary();
        setLoaded(true);
      } catch (e) {
        inFlight = false;
        if (cancelled) return;
        if ((e as { status?: number }).status === 404) {
          setNotFound(true);
          setLoaded(true);
          return;
        }
        setPollError(e instanceof Error ? e.message : String(e));
        timer = window.setTimeout(poll, ERROR_RETRY_MS);
        return;
      }
      inFlight = false;
      polls++;
      const delay = nextPollDelay({
        visible: document.visibilityState === "visible",
        checking,
        gotNew,
        idleMs: lastEventAt ? Math.max(0, Date.now() - lastEventAt) : Infinity,
      });
      if (!cancelled) timer = window.setTimeout(poll, delay);
    };

    // Coming back to the tab polls at once instead of waiting out the hidden backoff.
    const onVisible = () => {
      if (document.visibilityState === "visible") poll();
    };
    document.addEventListener("visibilitychange", onVisible);
    poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [currentBank, sessionId]);

  useEffect(() => {
    if (loaded) setSettled(true);
  }, [loaded]);

  // ---- Replay ------------------------------------------------------------------
  const offsets = useMemo(() => replayOffsets(events), [events]);
  const replayEnd = offsets.length ? offsets[offsets.length - 1] : 0;
  useEffect(() => {
    if (!replay?.playing) return;
    const id = window.setInterval(() => {
      setReplay((r) => {
        if (!r) return r;
        const elapsed = r.elapsed + REPLAY_TICK_MS * r.speed;
        return elapsed >= replayEnd
          ? { ...r, elapsed: replayEnd, playing: false }
          : { ...r, elapsed };
      });
    }, REPLAY_TICK_MS);
    return () => window.clearInterval(id);
  }, [replay?.playing, replayEnd]);

  const spans = useMemo(() => checkSpans(events, offsets), [events, offsets]);
  const visibleEvents = useMemo(
    () => (replay ? replayEvents(events, offsets, spans, replay.elapsed) : events),
    [events, offsets, spans, replay]
  );
  // Dragging the time rail: a paused replay at that moment, its newest row in view.
  const seek = useCallback((elapsed: number) => {
    setReplay((r) => ({ speed: r?.speed ?? 1, playing: false, elapsed }));
    atBottomRef.current = true;
    requestAnimationFrame(() => scrollToBottomRef.current(false));
  }, []);
  const scrollToBottomRef = useRef<(smooth: boolean) => void>(() => {});

  const timeline = useMemo(() => buildTimeline(visibleEvents), [visibleEvents]);
  const ackedSeq = session?.acked_seq ?? 0;
  const rows = useMemo(() => filterRows(timeline.rows, filters), [timeline.rows, filters]);
  // In a replay a note counts as delivered once its handoff is on screen, not by where the session ended.
  const handedOff = useMemo(
    () =>
      replay
        ? new Set(timeline.rows.flatMap((r) => (r.kind === "handoff" ? [r.note.seq] : [])))
        : null,
    [replay, timeline.rows]
  );
  const noteKeys = useMemo(() => rows.filter((r) => r.kind === "note").map((r) => r.key), [rows]);

  // ---- Scrolling ---------------------------------------------------------------
  const scroller = useCallback(() => scrollParent(rootRef.current), []);

  const scrollToBottom = useCallback(
    (smooth: boolean) => {
      const el = scroller();
      el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
      setNewCount(0);
    },
    [scroller]
  );
  scrollToBottomRef.current = scrollToBottom;

  const flashRow = useCallback((key: string) => {
    const el = document.querySelector(`[data-row-key="${key}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
    setFlashKey(key);
    window.setTimeout(() => setFlashKey((k) => (k === key ? null : k)), FLASH_MS);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const el = scroller();
    const target: HTMLElement | Window =
      el === document.scrollingElement || el === document.documentElement ? window : el;
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < AT_BOTTOM_PX;
      atBottomRef.current = atBottom;
      if (atBottom) setNewCount(0);
    };
    target.addEventListener("scroll", onScroll, { passive: true });
    return () => target.removeEventListener("scroll", onScroll);
  }, [loaded, scroller]);

  // New entries: follow them when the reader is at the bottom, otherwise count
  // them on the "N new" pill instead of yanking the page away from what they read.
  const shownCount = visibleEvents.length;
  const prevShownRef = useRef<number | null>(null);
  useLayoutEffect(() => {
    if (!loaded) return;
    const prev = prevShownRef.current;
    prevShownRef.current = shownCount;
    if (prev === null) {
      // First paint: a live session opens at its newest entry, a finished one at its start.
      if (session && isActive(session, Date.now())) scrollToBottom(false);
      else atBottomRef.current = false;
      return;
    }
    if (shownCount <= prev) return;
    if (atBottomRef.current) scrollToBottom(true);
    else setNewCount((n) => n + (shownCount - prev));
  }, [shownCount, loaded, session, scrollToBottom]);

  // Keyboard: j / k step through Hindsight's notes.
  const noteCursorRef = useRef(-1);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
      if ((e.key !== "j" && e.key !== "k") || noteKeys.length === 0) return;
      const step = e.key === "j" ? 1 : -1;
      const next = Math.min(noteKeys.length - 1, Math.max(0, noteCursorRef.current + step));
      noteCursorRef.current = next;
      flashRow(noteKeys[next]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [noteKeys, flashRow]);

  if (!loaded) {
    return (
      <div ref={rootRef} className="flex justify-center py-24">
        <Spinner size="lg" />
      </div>
    );
  }

  if (notFound) {
    return (
      <div ref={rootRef} className="flex flex-col items-center gap-4 py-24 text-center">
        <CircleSlash className="h-8 w-8 text-muted-foreground/60" />
        <div className="text-sm font-medium">{t("notFound")}</div>
        <div className="font-mono text-xs text-muted-foreground break-all px-4">{sessionId}</div>
        <Button variant="outline" size="sm" onClick={onBack}>
          <ArrowLeft />
          {t("backToSessions")}
        </Button>
      </div>
    );
  }

  const checking = liveChecking || !!session?.checking;
  const live = session ? isActive(session, now) : false;
  const harness = session ? sessionHarness(session) : null;

  return (
    <SessionStart.Provider value={events.length ? Date.parse(events[0].created_at) : 0}>
      <div ref={rootRef} className="relative pb-24">
        <SessionHeader
          t={t}
          session={session}
          sessionId={sessionId}
          harness={harness}
          checking={checking}
          live={live}
          now={now}
          onBack={onBack}
          onOpenDocument={setOpenDocId}
        />

        {/* The control strip stays pinned while the conversation scrolls under it —
          from sm up only: on a phone it wraps to ~170px, a fifth of the screen. */}
        <div className="sm:sticky top-0 z-20 -mx-4 sm:-mx-6 mb-4 border-b border-border/60 bg-background/90 px-4 py-2 backdrop-blur sm:px-6">
          <Toolbar
            t={t}
            filters={filters}
            setFilters={setFilters}
            replay={replay}
            setReplay={setReplay}
            replayEnd={replayEnd}
            canReplay={events.length > 1}
            shown={rows.length}
            noteCount={noteKeys.length}
            panel={panel}
            setPanel={setPanel}
          />
          <div
            className={cn("mt-2 hidden text-[11px] font-semibold uppercase tracking-wider", LANES)}
          >
            {/* One header per column, in one row, each underlined in its side's colour. */}
            <div className="border-b-2 border-agent pb-1">
              <div className="flex items-center gap-2 text-agent">
                {resolveHarnessLogo(harness) ? (
                  <HarnessLogo harness={harness} size={14} />
                ) : (
                  <Bot className="h-3.5 w-3.5" />
                )}
                {resolveHarnessLogo(harness)?.label ?? t("laneConversation")}
              </div>
              <div className="text-[10px] font-normal normal-case tracking-normal text-muted-foreground">
                {t("colAgentSub")}
              </div>
            </div>
            <div className="hidden border-b-2 border-border pb-1 text-center lg:block">
              <div className="text-muted-foreground">{t("colTimeline")}</div>
              <div className="text-[10px] font-normal normal-case tracking-normal text-muted-foreground">
                {t("colTimelineSub")}
              </div>
            </div>
            <div className="border-b-2 border-primary pb-1 pl-4">
              <div className="flex items-center gap-2 text-primary">
                <HindsightMark size={14} />
                {t("laneHindsight")}
              </div>
              <div className="text-[10px] font-normal normal-case tracking-normal text-muted-foreground">
                {t("colMemorySub")}
              </div>
            </div>
          </div>
        </div>

        {panel === "activity" && (
          <ActivityLog
            t={t}
            session={session}
            events={visibleEvents}
            checking={checking && !replay}
            onJump={(seq) => {
              const key = timeline.rowKeyBySeq.get(seq);
              if (key) flashRow(key);
            }}
          />
        )}

        {panel === "raw" && (
          // The session's log as stored, oldest first: what the lanes are drawn from.
          <pre className="mb-3 max-h-[60vh] overflow-auto rounded-lg border border-border/60 bg-muted/40 p-3 font-mono text-[11px] leading-relaxed">
            {JSON.stringify(visibleEvents, null, 2)}
          </pre>
        )}

        {pollError && (
          <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
            {t("pollError", { message: pollError })}
          </div>
        )}

        {/* The rail floats over the lanes' empty middle column (see LANES). */}
        <div className="relative">
          <div className="min-w-0">
            {rows.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border py-12 text-center text-sm text-muted-foreground">
                {events.length === 0 ? t("noEvents") : t("noMatches")}
              </div>
            ) : (
              <ol aria-live="polite" aria-relevant="additions">
                {rows.map((row, i) => (
                  <li
                    key={row.key}
                    data-row-key={row.key}
                    onPointerEnter={() => setHoverKey(row.key)}
                    onPointerLeave={() => setHoverKey((k) => (k === row.key ? null : k))}
                    className={cn(
                      "scroll-mt-40 rounded-lg transition-[box-shadow,background-color] duration-300",
                      // Each side lights in its own colour, here and on its marks on the rail.
                      flashKey === row.key &&
                        (HINDSIGHT_ROWS.has(row.kind)
                          ? "bg-primary/5 ring-2 ring-primary/40"
                          : "bg-agent/5 ring-2 ring-agent/40"),
                      hoverKey === row.key &&
                        (HINDSIGHT_ROWS.has(row.kind)
                          ? "bg-primary/5 ring-1 ring-primary/30"
                          : "bg-agent/5 ring-1 ring-agent/30")
                    )}
                  >
                    <LaneRow
                      settled={settled}
                      row={row}
                      run={toolRun(rows, i)}
                      t={t}
                      source={harness}
                      ackedSeq={ackedSeq}
                      handedOff={handedOff}
                      timeline={timeline}
                      onJump={flashRow}
                      onHoverHost={setHoverKey}
                      onOpenDocument={setOpenDocId}
                      onOpenMemory={setOpenMemoryId}
                    />
                  </li>
                ))}
              </ol>
            )}

            {checking && !replay && (
              <div className={LANES}>
                <div className="hidden md:block" />
                <Gap />
                <div className={cn("pb-2", HINDSIGHT_LANE)}>
                  <CheckingIndicator t={t} />
                </div>
              </div>
            )}
          </div>
          <TimeRail
            t={t}
            events={events}
            offsets={offsets}
            spans={spans}
            end={replayEnd}
            at={replay ? replay.elapsed : null}
            onSeek={seek}
            rowKeyBySeq={timeline.rowKeyBySeq}
            hoverKey={hoverKey}
            onHover={setHoverKey}
          />
        </div>

        {newCount > 0 && (
          <button
            type="button"
            onClick={() => scrollToBottom(true)}
            className="fixed bottom-[max(1.5rem,env(safe-area-inset-bottom))] left-1/2 z-30 inline-flex -translate-x-1/2 items-center gap-2 rounded-full bg-primary-gradient px-4 py-2 text-sm font-medium text-white shadow-lg animate-soft-fade-in"
          >
            <ArrowDown className="h-4 w-4" />
            {t("newEntries", { count: newCount })}
          </button>
        )}

        <DocumentChunkModal type="document" id={openDocId} onClose={() => setOpenDocId(null)} />
        <MemoryDetailModal memoryId={openMemoryId} onClose={() => setOpenMemoryId(null)} />
      </div>
    </SessionStart.Provider>
  );
}

// ---- Header ----------------------------------------------------------------------

function SessionHeader({
  t,
  session,
  sessionId,
  harness,
  checking,
  live,
  now,
  onBack,
  onOpenDocument,
}: {
  t: Translate;
  session: SessionSummary | null;
  sessionId: string;
  harness: string | null;
  checking: boolean;
  live: boolean;
  now: number;
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}) {
  const locale = useLocale();
  const end = session
    ? live
      ? now
      : Date.parse(session.last_event_at ?? session.created_at)
    : now;
  const duration = session ? Math.max(0, end - Date.parse(session.created_at)) : 0;
  const savedPct =
    session && session.last_seq > 0
      ? Math.min(100, Math.round((session.saved_seq / session.last_seq) * 100))
      : 0;

  return (
    <div className="mb-4">
      <button
        type="button"
        onClick={onBack}
        className="mb-3 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        {t("backToSessions")}
      </button>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-2.5">
            <HarnessLogo harness={harness} size={22} />
            {/* The client's id for the session: a prompt is not a name. */}
            <h1 className="min-w-0 break-all font-mono text-lg font-semibold text-foreground sm:text-xl">
              {sessionId}
            </h1>
            {live ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                <span className="relative inline-flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-70" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                </span>
                {t("live")}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-border px-2.5 py-0.5 text-xs text-muted-foreground">
                {t("statusIdle")}
              </span>
            )}
            {checking && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                <SessionStatusDot session={{ last_event_at: null, checking }} now={now} />
                {t("checking")}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {session && session.saved_seq > 0 && (
            <Button variant="outline" size="sm" onClick={() => onOpenDocument(session.document_id)}>
              <FileText />
              {t("openDocument")}
            </Button>
          )}
        </div>
      </div>

      {session && (
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label={t("statDuration")}
            value={formatDuration(duration)}
            hint={t("startedAt", { when: relativeTime(session.created_at, locale, now) })}
          />
          <Stat
            label={t("statActivity")}
            value={t("eventCount", { count: session.event_count })}
            hint={t("checkCount", { count: session.check_count })}
          />
          <Stat
            label={t("statNotes")}
            value={String(session.note_count)}
            hint={
              session.note_count > 0
                ? t("noteMentions", {
                    used: session.credit_count,
                    skipped: session.dismissed_count,
                  })
                : session.live
                  ? t("liveOnLong")
                  : t("liveOffLong")
            }
            accent={session.note_count > 0}
          />
          <div className="min-w-0 rounded-xl border border-border/60 bg-card px-3 py-2">
            <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              {t("statSaved")}
            </div>
            <div className="mt-1 text-sm font-semibold tabular-nums">
              {session.retain === "off"
                ? t("retainOff")
                : `${session.saved_seq} / ${session.last_seq}`}
            </div>
            {session.retain !== "off" && (
              <div
                className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted"
                role="progressbar"
                aria-valuenow={savedPct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={t("statSaved")}
              >
                <div
                  className="h-full rounded-full bg-primary-gradient transition-[width] duration-500"
                  style={{ width: `${savedPct}%` }}
                />
              </div>
            )}
            <div className="mt-1 truncate text-[11px] text-muted-foreground">
              {session.retain === "turns"
                ? t("retainTurns")
                : session.retain === "idle"
                  ? t("retainIdle")
                  : t("retainOffHint")}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-border/60 bg-card px-3 py-2">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className={cn("mt-1 text-sm font-semibold tabular-nums", accent && "text-primary")}>
        {value}
      </div>
      {hint && <div className="mt-1 truncate text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

/** Hindsight's own mark (the favicon: the logo without the word), wherever Hindsight acts. */
function HindsightMark({ size, className }: { size: number; className?: string }) {
  return (
    <img
      src={withBasePath("/favicon.png")}
      alt=""
      aria-hidden
      width={size}
      height={Math.round((size * 139) / 186)}
      className={cn("shrink-0 object-contain", className)}
    />
  );
}

// ---- Activity -------------------------------------------------------------------------

/**
 * Everything Hindsight did in the session, server side, in order: each check and its
 * outcome, each note, when each note went back to the agent, and each save into
 * memory. Read from the session log, the same entries the lanes draw.
 */
function ActivityLog({
  t,
  session,
  events,
  checking,
  onJump,
}: {
  t: Translate;
  session: SessionSummary | null;
  events: SessionEvent[];
  checking: boolean;
  onJump: (seq: number) => void;
}) {
  const start = events.length ? Date.parse(events[0].created_at) : 0;
  const entries = events.filter((e) => e.source === "hindsight");
  const line = (e: SessionEvent): { icon: React.ReactNode; text: string } => {
    const p = e.payload;
    switch (e.kind) {
      case "check": {
        const c = p as CheckPayload;
        return {
          icon: <HindsightMark size={14} />,
          text: `${t("actCheck", { trigger: triggerLabel(t, c.trigger), outcome: outcomeLabel(t, c.outcome) })}${
            typeof c.ms === "number" ? ` · ${formatDuration(c.ms)}` : ""
          }`,
        };
      }
      case "note":
        return {
          icon: <HindsightMark size={14} />,
          text: `${t("actNote")}: ${String(p.text ?? "")}`,
        };
      case "handoff": {
        const n = Array.isArray(p.with_seqs) ? p.with_seqs.length : 0;
        return {
          icon: <Send className="h-3.5 w-3.5 text-primary" />,
          text: n
            ? t("actHandoff", { seq: Number(p.note_seq), count: n })
            : t("actHandoffPoll", { seq: Number(p.note_seq) }),
        };
      }
      case "save":
        return {
          icon: <Save className="h-3.5 w-3.5 text-primary" />,
          text: t("saved", {
            from: Number(p.from_seq ?? 0),
            to: Number(p.to_seq ?? 0),
            turns: Number(p.turns ?? 0),
          }),
        };
      default:
        return { icon: <HindsightMark size={14} />, text: e.kind };
    }
  };
  return (
    <div className="mb-4 rounded-xl border border-border/60 bg-card p-3 text-xs animate-soft-fade-in">
      {session && (
        <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/60 pb-2 text-muted-foreground">
          <span className="font-medium text-foreground">
            {session.retain === "turns"
              ? t("retainTurns")
              : session.retain === "idle"
                ? t("retainIdle")
                : t("retainOffHint")}
          </span>
          {session.retain !== "off" && (
            <span className="tabular-nums">
              {t("savedUpTo", { saved: session.saved_seq, total: session.last_seq })}
            </span>
          )}
          <span>{session.live ? t("liveOnLong") : t("liveOffLong")}</span>
          {checking && <span className="text-primary">{t("checkingLong")}</span>}
        </div>
      )}
      {entries.length === 0 ? (
        <div className="py-2 text-muted-foreground">{t("activityEmpty")}</div>
      ) : (
        <ol className="max-h-72 space-y-0.5 overflow-y-auto">
          {entries.map((e) => {
            const { icon, text } = line(e);
            return (
              <li key={e.seq}>
                <button
                  type="button"
                  onClick={() => onJump(e.seq)}
                  className="flex w-full min-w-0 items-center gap-2 rounded-md px-1.5 py-1 text-left hover:bg-muted/60"
                >
                  <span className="w-14 shrink-0 font-mono tabular-nums text-muted-foreground">
                    {formatClock(Date.parse(e.created_at) - start)}
                  </span>
                  {icon}
                  <span className="truncate">{text}</span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

// ---- Toolbar -------------------------------------------------------------------------

function ToggleChip({
  pressed,
  onClick,
  children,
}: {
  pressed: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs transition-colors",
        pressed
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border text-muted-foreground hover:bg-accent"
      )}
    >
      {children}
    </button>
  );
}

function Toolbar({
  t,
  filters,
  setFilters,
  replay,
  setReplay,
  replayEnd,
  canReplay,
  shown,
  noteCount,
  panel,
  setPanel,
}: {
  t: Translate;
  panel: "filters" | "activity" | "raw" | null;
  setPanel: (p: "filters" | "activity" | "raw" | null) => void;
  filters: RowFilters;
  setFilters: React.Dispatch<React.SetStateAction<RowFilters>>;
  replay: { speed: number; playing: boolean; elapsed: number } | null;
  setReplay: React.Dispatch<
    React.SetStateAction<{ speed: number; playing: boolean; elapsed: number } | null>
  >;
  replayEnd: number;
  canReplay: boolean;
  shown: number;
  noteCount: number;
}) {
  const set = (patch: Partial<RowFilters>) => setFilters((f) => ({ ...f, ...patch }));
  const filtering = !!filters.query || !filters.tools || !filters.silent || filters.hindsightOnly;
  const toggle = (p: "filters" | "activity" | "raw") => setPanel(panel === p ? null : p);
  const filterControls = (
    <div className="mt-2 flex flex-wrap items-center gap-2 animate-soft-fade-in">
      <div className="relative w-full sm:w-56">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={filters.query}
          onChange={(e) => set({ query: e.target.value })}
          placeholder={t("searchInSession")}
          aria-label={t("searchInSession")}
          className="h-8 pl-8 text-xs"
        />
      </div>
      <ToggleChip pressed={filters.tools} onClick={() => set({ tools: !filters.tools })}>
        <Wrench className="h-3.5 w-3.5" />
        {t("filterTools")}
      </ToggleChip>
      <ToggleChip pressed={filters.silent} onClick={() => set({ silent: !filters.silent })}>
        {filters.silent ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
        {t("filterSilent")}
      </ToggleChip>
      <ToggleChip
        pressed={filters.hindsightOnly}
        onClick={() => set({ hindsightOnly: !filters.hindsightOnly })}
      >
        <HindsightMark size={14} />
        {t("filterHindsightOnly")}
      </ToggleChip>
    </div>
  );
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <ToggleChip pressed={panel === "filters"} onClick={() => toggle("filters")}>
          <SlidersHorizontal className="h-3.5 w-3.5" />
          {t("filters")}
          {filtering && <span className="h-1.5 w-1.5 rounded-full bg-primary" />}
        </ToggleChip>
        <ToggleChip pressed={panel === "activity"} onClick={() => toggle("activity")}>
          <Activity className="h-3.5 w-3.5" />
          {t("activity")}
        </ToggleChip>
        <ToggleChip pressed={panel === "raw"} onClick={() => toggle("raw")}>
          <Braces className="h-3.5 w-3.5" />
          {t("rawLog")}
        </ToggleChip>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {noteCount > 0 && (
            <span className="hidden text-[11px] text-muted-foreground lg:inline">
              {t("keyboardHint")}
            </span>
          )}
          {replay ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
                aria-label={replay.playing ? t("replayPause") : t("replayPlay")}
                title={replay.playing ? t("replayPause") : t("replayPlay")}
                onClick={() =>
                  setReplay((r) =>
                    r
                      ? {
                          ...r,
                          playing: !r.playing,
                          elapsed: !r.playing && r.elapsed >= replayEnd ? 0 : r.elapsed,
                        }
                      : r
                  )
                }
              >
                {replay.playing ? <Pause /> : <Play />}
              </Button>
              {REPLAY_SPEEDS.map((s) => (
                <ToggleChip
                  key={s}
                  pressed={replay.speed === s}
                  onClick={() => setReplay((r) => (r ? { ...r, speed: s } : r))}
                >
                  {s}×
                </ToggleChip>
              ))}
              <span className="tabular-nums text-[11px] text-muted-foreground">
                {formatDuration(replay.elapsed)} / {formatDuration(replayEnd)}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                aria-label={t("replayExit")}
                title={t("replayExit")}
                onClick={() => setReplay(null)}
              >
                <X />
              </Button>
            </div>
          ) : (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              disabled={!canReplay}
              onClick={() => setReplay({ speed: 1, playing: true, elapsed: 0 })}
            >
              <Play />
              {t("replay")}
            </Button>
          )}
          <span className="text-[11px] tabular-nums text-muted-foreground">
            {t("shownCount", { count: shown })}
          </span>
        </div>
      </div>
      {panel === "filters" && filterControls}
    </div>
  );
}

// ---- Rows -----------------------------------------------------------------------------

interface RowProps {
  row: TimelineRow;
  t: Translate;
  source: string | null;
  ackedSeq: number;
  timeline: ReturnType<typeof buildTimeline>;
  onJump: (key: string) => void;
  onHoverHost: (key: string | null) => void;
  onOpenDocument: (id: string) => void;
  onOpenMemory: (id: string) => void;
  /** The row arrived after the first paint: it animates in (see LaneRow). */
  enter?: boolean;
  /** Replay: the notes handed to the agent so far (null when not replaying). */
  handedOff?: Set<number> | null;
}

const HINDSIGHT_ROWS = new Set<TimelineRow["kind"]>(["note", "save", "check", "handoff"]);

const isToolRow = (r: TimelineRow | undefined) => r?.kind === "tool" || r?.kind === "state";

/** Where a tool row sits in a run of them: a run renders as one block. */
function toolRun(rows: TimelineRow[], i: number) {
  return isToolRow(rows[i])
    ? { first: !isToolRow(rows[i - 1]), last: !isToolRow(rows[i + 1]) }
    : null;
}

function LaneRow(
  rowProps: RowProps & { run: { first: boolean; last: boolean } | null; settled: boolean }
) {
  // Read once, when the row mounts: rows there at first paint stay still, rows that
  // arrive later (live, or a replay stepping forward) slide in from their own side.
  const [enter] = useState(rowProps.settled);
  const props = { ...rowProps, enter };
  const { row, run } = props;
  if (row.kind === "handoff") return <Handoff {...props} row={row} />;
  // A turn end closes both sides' turn: it runs across the two lanes.
  if (row.kind === "turn_end") {
    return (
      <div className={cn("py-2", enter && "animate-soft-fade-in")}>
        <RowBody {...props} />
      </div>
    );
  }
  const hindsight = HINDSIGHT_ROWS.has(row.kind);
  const checks = row.checks.length > 0 && (
    <div className="space-y-1">
      {row.checks.map((c) => (
        <CheckMarker
          key={c.key}
          check={c}
          t={props.t}
          onHover={(on) => props.onHoverHost(on ? row.key : null)}
          onOpenMemory={props.onOpenMemory}
          settled={rowProps.settled}
        />
      ))}
    </div>
  );
  return (
    <div className={LANES}>
      {run ? (
        // Tool calls are the agent acting, not speaking: one framed, monospace block per run of
        // them under its message, so they never read as part of the conversation.
        <div
          className={cn(
            "flex min-w-0 flex-col pl-9 sm:pl-11",
            enter && "animate-lane-in-left",
            run.first && "pt-1.5",
            run.last && "pb-1.5"
          )}
        >
          {/* flex-1: a taller Hindsight cell beside it must not open a gap in the block. */}
          <div
            className={cn(
              "flex-1 border-x border-border/70 bg-muted/40 px-1",
              run.first && "rounded-t-lg border-t pt-1",
              run.last && "rounded-b-lg border-b pb-1"
            )}
          >
            {run.first && (
              <div className="flex items-center gap-1.5 px-2 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                <Wrench className="h-3 w-3" />
                {props.t("filterTools")}
              </div>
            )}
            <RowBody {...props} />
          </div>
        </div>
      ) : (
        <div
          className={cn(
            "min-w-0 py-1.5",
            hindsight && "hidden md:block",
            enter && !hindsight && "animate-lane-in-left"
          )}
        >
          {!hindsight && <RowBody {...props} />}
        </div>
      )}
      <Gap />
      <div
        className={cn(
          "min-w-0 py-1.5 pl-9 sm:pl-11",
          HINDSIGHT_LANE,
          enter && hindsight && "animate-lane-in-right",
          !hindsight && !checks && "hidden md:block"
        )}
      >
        {hindsight ? <RowBody {...props} /> : checks}
      </div>
    </div>
  );
}

function Avatar({
  kind,
  source,
}: {
  kind: "user" | "agent" | "hindsight";
  source?: string | null;
}) {
  // An unknown source has no logo: fall back to the generic agent mark.
  if (kind === "agent" && resolveHarnessLogo(source)) {
    return (
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-card">
        <HarnessLogo harness={source} size={16} />
      </span>
    );
  }
  return (
    <span
      className={cn(
        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
        kind === "user" && "bg-indigo-500 text-white",
        kind === "agent" && "border border-border bg-card text-muted-foreground",
        kind === "hindsight" && "border border-primary/30 bg-card"
      )}
    >
      {kind === "user" ? (
        <User className="h-4 w-4" />
      ) : kind === "agent" ? (
        <Bot className="h-4 w-4" />
      ) : (
        <HindsightMark size={16} />
      )}
    </span>
  );
}

/** When the session started: every card shows its time from here ("0:41"), on both sides. */
const SessionStart = createContext(0);

function Time({ iso }: { iso: string }) {
  const locale = useLocale();
  const start = useContext(SessionStart);
  const d = new Date(iso);
  return (
    <time
      dateTime={iso}
      title={d.toLocaleString(locale)}
      className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground"
    >
      {formatClock(d.getTime() - start)}
    </time>
  );
}

function RowBody(props: RowProps) {
  const { row, t } = props;
  switch (row.kind) {
    case "message":
      if (row.role === "user") {
        return (
          <div className="flex items-start justify-end gap-2.5">
            <div className="min-w-0 max-w-[88%] sm:max-w-[80%]">
              <div className="mb-1 flex items-center justify-end gap-2">
                <Time iso={row.event.created_at} />
                <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400">
                  {t("voiceUser")}
                </span>
              </div>
              <div className="rounded-2xl rounded-tr-sm bg-indigo-500/10 px-3.5 py-2.5 ring-1 ring-indigo-500/20">
                <ExpandableText text={row.text} t={t} markdown />
              </div>
            </div>
            <Avatar kind="user" />
          </div>
        );
      }
      if (row.role === "system") {
        return (
          <div className="px-9 text-center text-xs text-muted-foreground sm:px-11">
            <ExpandableText text={row.text} t={t} />
          </div>
        );
      }
      return (
        <div className="flex items-start gap-2.5">
          <Avatar kind="agent" source={props.source} />
          <div className="min-w-0 max-w-[88%] sm:max-w-[80%]">
            <div className="mb-1 flex items-center gap-2">
              <span className="text-xs font-semibold text-foreground/80">{t("voiceAgent")}</span>
              <Time iso={row.event.created_at} />
            </div>
            <div className="rounded-2xl rounded-tl-sm border border-border/60 bg-card px-3.5 py-2.5">
              <ExpandableText text={row.text} t={t} markdown />
            </div>
          </div>
        </div>
      );
    case "tool":
      return <ToolLine row={row} t={t} />;
    case "state":
      return <StateLine row={row} t={t} />;
    case "turn_end":
      return (
        <div className="flex items-center gap-2 pl-9 text-[11px] text-muted-foreground sm:pl-11">
          <Flag className="h-3 w-3" />
          {t("turnEnd")}
          <div className="h-px flex-1 bg-border/60" />
        </div>
      );
    case "note":
      return <NoteCard {...props} row={row} />;
    case "save":
      return (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          <Save className="h-3.5 w-3.5 text-primary" />
          <span>
            {t("saved", {
              from: row.save.from_seq ?? 0,
              to: row.save.to_seq ?? 0,
              turns: row.save.turns ?? 0,
            })}
          </span>
          {row.save.document_id && (
            <button
              type="button"
              className="font-mono text-primary hover:underline"
              onClick={() => props.onOpenDocument(row.save.document_id!)}
            >
              {row.save.document_id}
            </button>
          )}
          <Time iso={row.event.created_at} />
        </div>
      );
    case "check":
      // Its moment is not in the log: the marker stands alone at its own time.
      return (
        <CheckMarker
          check={row}
          t={t}
          onHover={() => undefined}
          onOpenMemory={props.onOpenMemory}
          // Its row already slides in.
          settled={false}
        />
      );
    case "handoff":
      return null; // LaneRow draws it across both lanes (Handoff)
    default:
      return (
        <div className="pl-9 font-mono text-[11px] text-muted-foreground sm:pl-11">
          {row.event.source}/{row.event.kind}
        </div>
      );
  }
}

const LONG_TEXT = 900;

function ExpandableText({ text, t, markdown }: { text: string; t: Translate; markdown?: boolean }) {
  const [open, setOpen] = useState(false);
  const long = text.length > LONG_TEXT;
  const shown = long && !open ? `${text.slice(0, LONG_TEXT).trimEnd()}…` : text;
  return (
    <div className="min-w-0 break-words">
      {markdown ? (
        <CompactMarkdown>{shown}</CompactMarkdown>
      ) : (
        <span className="whitespace-pre-wrap">{shown}</span>
      )}
      {long && (
        <button
          type="button"
          className="mt-1 text-xs font-medium text-primary hover:underline"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? t("showLess") : t("showMore")}
        </button>
      )}
    </div>
  );
}

const TOOL_ICON: Record<ToolFamily, typeof Pencil> = {
  edit: Pencil,
  command: SquareTerminal,
  read: Search,
  other: Wrench,
};

function ToolLine({ row, t }: { row: Extract<TimelineRow, { kind: "tool" }>; t: Translate }) {
  const [open, setOpen] = useState(false);
  const Icon = TOOL_ICON[row.family];
  const quiet = row.family === "read" && !row.isError;
  const input = (row.call?.payload.input ?? null) as Record<string, unknown> | null;
  const resultText = typeof row.result?.payload.text === "string" ? row.result.payload.text : "";
  const oldStr = typeof input?.old_string === "string" ? input.old_string : null;
  const newStr = typeof input?.new_string === "string" ? input.new_string : null;

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={cn(
          "group flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1 text-left font-mono transition-colors hover:bg-muted",
          quiet ? "text-muted-foreground" : "text-foreground",
          row.isError && "bg-red-500/5 text-red-700 dark:text-red-400"
        )}
      >
        <ChevronRight
          className={cn("h-3 w-3 shrink-0 opacity-50 transition-transform", open && "rotate-90")}
        />
        <Icon
          className={cn(
            "h-3.5 w-3.5 shrink-0",
            row.family === "edit" && !row.isError && "text-amber-600 dark:text-amber-400",
            row.family === "command" && !row.isError && "text-sky-600 dark:text-sky-400"
          )}
        />
        <span className={cn("shrink-0 text-xs", quiet ? "font-normal" : "font-semibold")}>
          {row.tool || t("toolResult")}
        </span>
        <span className="min-w-0 truncate font-mono text-xs opacity-80">{row.target}</span>
        {row.isError && (
          <span className="shrink-0 rounded-full bg-red-500/15 px-1.5 text-[10px] font-semibold uppercase">
            {t("error")}
          </span>
        )}
        {!row.result && row.call && (
          <span className="ml-auto shrink-0">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-muted-foreground/60" />
          </span>
        )}
      </button>
      {open && (
        <div className="mt-1 space-y-2 rounded-lg border border-border/60 bg-muted/30 p-2.5 animate-soft-fade-in">
          {oldStr !== null && newStr !== null ? (
            <DiffBlock diff={toDiff(oldStr, newStr)} />
          ) : input ? (
            <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5">
              {typeof input.command === "string" ? input.command : JSON.stringify(input, null, 2)}
            </pre>
          ) : null}
          {row.result && (
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {t("toolResult")}
              </div>
              <pre
                className={cn(
                  "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background p-2 font-mono text-[11px] leading-5",
                  row.isError && "text-red-700 dark:text-red-400"
                )}
              >
                {resultText || "—"}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function toDiff(oldStr: string, newStr: string): string {
  const minus = oldStr.split("\n").map((l) => `-${l}`);
  const plus = newStr.split("\n").map((l) => `+${l}`);
  return [...minus, ...plus].join("\n");
}

function DiffBlock({ diff }: { diff: string }) {
  return (
    <pre className="max-h-96 overflow-auto rounded-md bg-background font-mono text-[11px] leading-5">
      {diff.split("\n").map((line, i) => (
        <div
          key={i}
          className={cn(
            "whitespace-pre-wrap break-words px-2",
            line.startsWith("+") &&
              !line.startsWith("+++") &&
              "bg-emerald-500/10 text-emerald-800 dark:text-emerald-300",
            line.startsWith("-") &&
              !line.startsWith("---") &&
              "bg-red-500/10 text-red-800 dark:text-red-300",
            line.startsWith("@@") && "text-sky-700 dark:text-sky-400",
            (line.startsWith("diff ") || line.startsWith("index ")) && "text-muted-foreground"
          )}
        >
          {line || " "}
        </div>
      ))}
    </pre>
  );
}

function StateLine({ row, t }: { row: Extract<TimelineRow, { kind: "state" }>; t: Translate }) {
  const [open, setOpen] = useState(false);
  const lines = row.diff.split("\n");
  const added = lines.filter((l) => l.startsWith("+") && !l.startsWith("+++")).length;
  const removed = lines.filter((l) => l.startsWith("-") && !l.startsWith("---")).length;
  const files = lines.filter((l) => l.startsWith("diff --git")).length;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full min-w-0 items-center gap-2 rounded-lg px-2 py-1 text-left text-xs text-muted-foreground hover:bg-muted/60"
      >
        <ChevronRight
          className={cn("h-3 w-3 shrink-0 opacity-50 transition-transform", open && "rotate-90")}
        />
        <FileDiff className="h-3.5 w-3.5 shrink-0" />
        <span className="font-medium text-foreground/80" title={t("diffHint")}>
          {t("changeSoFar")}
        </span>
        <span>{t("filesChanged", { count: files })}</span>
        <span className="font-mono text-emerald-600 dark:text-emerald-400">+{added}</span>
        <span className="font-mono text-red-600 dark:text-red-400">−{removed}</span>
      </button>
      {open && (
        <div className="mt-1 animate-soft-fade-in">
          {row.diff ? (
            <DiffBlock diff={row.diff} />
          ) : (
            <div className="px-2 text-xs text-muted-foreground">{t("noChanges")}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ---- Hindsight ---------------------------------------------------------------------------

/** What a host row was, in a few words: `terminal git log --oneline`. */
function hostLabel(row: TimelineRow): string {
  if (row.kind === "tool") return `${row.tool} ${row.target}`.trim();
  if (row.kind === "message") return row.text.split("\n")[0];
  return row.kind.replace(/_/g, " ");
}

/**
 * A note reaching the agent: an arrow leaves Hindsight's lane, crosses the middle
 * and lands in the agent's lane as a message there, next to what it arrived with.
 * The segments draw right to left in turn, then the message slides in.
 */
function Handoff({
  row,
  t,
  source,
  onJump,
  enter,
}: RowProps & { row: Extract<TimelineRow, { kind: "handoff" }> }) {
  const agent = resolveHarnessLogo(source)?.label ?? t("theAgent");
  const line = (delay: number, cls?: string) => (
    <span
      className={cn(
        "h-[3px] origin-right rounded-full bg-gradient-to-l from-primary to-primary/70",
        enter && "animate-handoff-line",
        cls
      )}
      style={enter ? { animationDelay: `${delay}ms` } : undefined}
    />
  );
  return (
    // Above the time rail (z-10), so the arrow is seen crossing it.
    <div className={cn("relative z-20 py-2", LANES)}>
      <div className="flex min-w-0 items-center pl-9 sm:pl-11">
        <button
          type="button"
          onClick={() => onJump(row.note.key)}
          title={t("handoffJump")}
          className={cn(
            "ml-auto min-w-0 max-w-[90%] rounded-2xl rounded-tr-sm bg-primary-gradient p-px text-left shadow-lg shadow-primary/20 transition-transform hover:-translate-y-px",
            enter && "animate-lane-in-left"
          )}
          style={enter ? { animationDelay: "300ms" } : undefined}
        >
          <div className="rounded-2xl rounded-tr-sm bg-card px-3.5 py-2.5">
            <div className="mb-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11px]">
              <HindsightMark size={14} />
              <span className="font-semibold text-primary">{t("handoffTitle", { agent })}</span>
              {row.host && (
                <span className="truncate text-muted-foreground">
                  {t("handoffWith", { what: hostLabel(row.host) })}
                </span>
              )}
              <span className="ml-auto">
                <Time iso={row.event.created_at} />
              </span>
            </div>
            <p className="line-clamp-3 text-sm text-foreground/90">{row.note.note.text}</p>
          </div>
        </button>
        {/* The arrowhead points at the agent's message. */}
        <span className="ml-1 h-0 w-0 shrink-0 border-y-[6px] border-r-[9px] border-y-transparent border-r-primary" />
        {line(200, "w-4 shrink-0 md:w-6")}
      </div>
      {line(100, "hidden self-center lg:block")}
      <div className={cn("hidden items-center md:flex", HINDSIGHT_LANE, "md:pl-0")}>
        {line(0, "w-8 shrink-0")}
        <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
          <Send className="h-3 w-3" />
          {t("handoffSent", { agent })}
        </span>
      </div>
    </div>
  );
}

function routeLabel(t: Translate, route: DecisionRoute) {
  switch (route) {
    case "silent":
      return t("routeSilent");
    case "nothing":
      return t("routeNothing");
    case "writer":
      return t("routeWriter");
    case "llm":
      return t("routeLlm");
    default:
      return t("routeUnknown");
  }
}

/** `decision none=0.02 · kept 1/8 → writer words it → abstain answer 1.2s` */
function DecisionPathLine({ check, t }: { check: CheckPayload; t: Translate }) {
  const path = decisionPath(check.llm_calls);
  if (!path.pick && path.after.length === 0) return null;
  return (
    <span className="inline-flex min-w-0 flex-wrap items-center gap-x-1 gap-y-0.5 text-[11px]">
      {path.pick && (
        <>
          <span className="text-muted-foreground">
            {t("decision")}{" "}
            {path.pick.none !== null && (
              // The decision model's own wording, kept as-is.
              <span className="font-mono">{`none=${path.pick.none.toFixed(2)}`}</span>
            )}
            {path.pick.kept !== null && path.pick.offered !== null && (
              <span className="font-mono">
                {" "}
                · {path.pick.kept}/{path.pick.offered}
              </span>
            )}
          </span>
          <span className="text-muted-foreground/60">→</span>
          <span className={cn("font-medium", ROUTE_STYLE[path.pick.route])}>
            {routeLabel(t, path.pick.route)}
          </span>
        </>
      )}
      {path.after.map((c, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          <span className="text-muted-foreground/60">→</span>
          <span className="text-foreground/80">{c.scope.replace(/_/g, " ")}</span>
          {typeof c.duration_ms === "number" && (
            <span className="font-mono text-muted-foreground">{formatDuration(c.duration_ms)}</span>
          )}
        </span>
      ))}
    </span>
  );
}

function CheckMarker({
  check,
  t,
  onHover,
  onOpenMemory,
  settled,
}: {
  check: CheckRow;
  t: Translate;
  onHover: (on: boolean) => void;
  onOpenMemory: (id: string) => void;
  settled: boolean;
}) {
  const [open, setOpen] = useState(false);
  // A check that lands on a row already shown slides in on its own.
  const [enter] = useState(settled);
  const c = check.check;
  if (typeof c.running_ms === "number") return <RunningCheck check={c} t={t} enter={enter} />;
  const outcome = c.outcome ?? "silent";
  const quiet = isQuietCheck(c);
  return (
    <div
      className={cn(
        "border-l-2 pl-2",
        enter && "animate-soft-fade-in",
        outcome === "note"
          ? "border-primary"
          : outcome === "error"
            ? "border-red-500"
            : outcome === "repeat"
              ? "border-amber-500"
              : "border-border",
        quiet && "opacity-75 hover:opacity-100"
      )}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 rounded-md px-1.5 py-1 text-left hover:bg-muted/60"
      >
        <ChevronRight
          className={cn("h-3 w-3 shrink-0 opacity-50 transition-transform", open && "rotate-90")}
        />
        <HindsightMark size={14} className="shrink-0" />
        <span className="text-[11px] font-medium text-foreground/80">
          {t("checkedOn", { trigger: triggerLabel(t, c.trigger) })}
        </span>
        <span
          className={cn(
            "rounded-full border px-1.5 text-[10px] font-semibold uppercase tracking-wide",
            OUTCOME_STYLE[outcome] ?? OUTCOME_STYLE.silent
          )}
        >
          {outcomeLabel(t, outcome)}
        </span>
        {typeof c.ms === "number" && (
          <span className="inline-flex items-center gap-0.5 font-mono text-[11px] text-muted-foreground">
            <Clock className="h-3 w-3" />
            {formatDuration(c.ms)}
          </span>
        )}
        {outcome === "skipped" && c.reason ? (
          <span className="text-[11px] text-muted-foreground">{c.reason}</span>
        ) : (
          <DecisionPathLine check={c} t={t} />
        )}
      </button>
      {open && <CheckDetails check={c} t={t} onOpenMemory={onOpenMemory} />}
    </div>
  );
}

/**
 * A check caught mid-run in a replay: its steps in the order they ran (memory search,
 * then each model call), lit up to where the clock is. The search is what is left of
 * the total time once the model calls are taken out.
 */
function RunningCheck({
  check: c,
  t,
  enter,
}: {
  check: CheckPayload;
  t: Translate;
  enter: boolean;
}) {
  const calls = c.llm_calls ?? [];
  const callMs = calls.reduce((n, x) => n + (x.duration_ms ?? 0), 0);
  const steps = [
    {
      label: t("stepSearch", { count: c.search_queries?.length ?? 0 }),
      ms: Math.max(0, (c.ms ?? 0) - callMs),
    },
    ...calls.map((x) => ({ label: x.scope.replace(/_/g, " "), ms: x.duration_ms ?? 0 })),
  ];
  const at = c.running_ms ?? 0;
  let from = 0;
  return (
    <div className={cn("border-l-2 border-primary/60 pl-2", enter && "animate-lane-in-right")}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 px-1.5 py-1">
        <HindsightMark size={14} className="shrink-0 animate-pulse" />
        <span className="text-[11px] font-medium text-primary">
          {t("checkRunning", { trigger: triggerLabel(t, c.trigger) })}
        </span>
        <span className="font-mono text-[11px] text-muted-foreground">{formatDuration(at)}</span>
      </div>
      <ol className="ml-6 space-y-0.5 pb-1 text-[11px]">
        {steps.map((s, i) => {
          const start = from;
          from += s.ms;
          const state = at >= from ? "done" : at >= start ? "now" : "next";
          return (
            <li
              key={i}
              className={cn(
                "flex items-center gap-1.5",
                state === "next" && "text-muted-foreground/50",
                state === "now" && "font-medium text-primary"
              )}
            >
              {state === "done" ? (
                <Check className="h-3 w-3 text-emerald-500" />
              ) : state === "now" ? (
                <Spinner size="sm" className="h-3 w-3" />
              ) : (
                <span className="h-3 w-3 rounded-full border border-border" />
              )}
              {s.label}
              {state === "done" && (
                <span className="font-mono text-muted-foreground">{formatDuration(s.ms)}</span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/**
 * The session top to bottom on the replay clock, between the two lanes it explains:
 * the agent's steps on its left half in the agent's colour, Hindsight's checks on its
 * right half in Hindsight's (a bar from start to end, so its height is how long it
 * took). Hovering a mark lights its card and hovering a card lights its mark, with a
 * line across both halves at that moment. Click or drag to move the replay there.
 */
function TimeRail({
  t,
  events,
  offsets,
  spans,
  end,
  at,
  onSeek,
  rowKeyBySeq,
  hoverKey,
  onHover,
}: {
  t: Translate;
  events: SessionEvent[];
  offsets: number[];
  spans: CheckSpan[];
  end: number;
  at: number | null;
  onSeek: (elapsed: number) => void;
  rowKeyBySeq: Map<number, string>;
  hoverKey: string | null;
  onHover: (key: string | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [legend, setLegend] = useState(false);
  if (events.length < 2 || end <= 0) return null;
  const pct = (ms: number) => `${(Math.min(end, Math.max(0, ms)) / end) * 100}%`;
  const seekAt = (clientY: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    onSeek(Math.round(Math.min(1, Math.max(0, (clientY - box.top) / box.height)) * end));
  };
  const keyOf = (seq: number) => rowKeyBySeq.get(seq) ?? null;
  const hover = (seq: number) => ({
    onPointerEnter: () => onHover(keyOf(seq)),
    onPointerLeave: () => onHover(null),
  });
  const lit = (seq: number) => hoverKey !== null && keyOf(seq) === hoverKey;
  // Where the hovered card sits: its first event, on that event's side.
  const hovered = hoverKey === null ? -1 : events.findIndex((e) => keyOf(e.seq) === hoverKey);
  const thumb = at ?? end;
  return (
    // 3fr : 5rem : 2fr, so the middle column starts 3/5 of the way across what the rail leaves.
    <aside className="absolute inset-y-0 left-[calc((100%-5rem)*0.6)] z-10 hidden w-20 px-2 lg:block">
      <div className="sticky top-44 flex h-[calc(100dvh-16rem)] flex-col items-stretch gap-1 text-[10px] text-muted-foreground">
        <div
          ref={ref}
          role="slider"
          tabIndex={0}
          aria-label={t("timeRail")}
          aria-valuemin={0}
          aria-valuemax={end}
          aria-valuenow={thumb}
          aria-valuetext={formatClock(thumb)}
          className="relative flex-1 cursor-ns-resize touch-none select-none overflow-visible rounded-md"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            seekAt(e.clientY);
          }}
          onPointerMove={(e) => {
            if (e.currentTarget.hasPointerCapture(e.pointerId)) seekAt(e.clientY);
          }}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10_000 : 1_000;
            if (e.key === "ArrowDown") onSeek(Math.min(end, thumb + step));
            else if (e.key === "ArrowUp") onSeek(Math.max(0, thumb - step));
            else return;
            e.preventDefault();
          }}
        >
          {/* The two lanes: the agent's half, then Hindsight's. */}
          <div className="absolute inset-y-0 left-0 w-1/2 rounded-l-md bg-agent/[0.07]" />
          <div className="absolute inset-y-0 right-0 w-1/2 rounded-r-md bg-primary/[0.07]" />
          {events.map((ev, i) => {
            const p = ev.payload;
            const top = pct(offsets[i]);
            const on = lit(ev.seq);
            if (ev.source === "hindsight") {
              if (ev.kind === "note")
                return (
                  <span
                    key={ev.seq}
                    {...hover(ev.seq)}
                    title={`${formatClock(offsets[i])} · ${t("voiceHindsightNote")}: ${String(p.text ?? "")}`}
                    className={cn(
                      "absolute right-0.5 h-2 w-2 -translate-y-1/2 rounded-full bg-primary transition-transform",
                      on && "scale-150 ring-2 ring-primary/40"
                    )}
                    style={{ top }}
                  />
                );
              if (ev.kind === "handoff")
                return (
                  <span
                    key={ev.seq}
                    {...hover(ev.seq)}
                    title={`${formatClock(offsets[i])} · ${t("handoffSent", { agent: t("theAgent") })}`}
                    className={cn(
                      "absolute inset-x-0 h-0.5 -translate-y-1/2 rounded-full bg-primary/70",
                      on && "h-1 bg-primary"
                    )}
                    style={{ top }}
                  />
                );
              return null;
            }
            const user = ev.kind === "message" && p.role === "user";
            const error = p.is_error === true;
            return (
              <span
                key={ev.seq}
                {...hover(ev.seq)}
                title={`${formatClock(offsets[i])} · ${
                  ev.kind === "message"
                    ? String(p.text ?? "")
                        .split("\n")[0]
                        .slice(0, 120)
                    : `${String(p.tool ?? ev.kind)} ${toolTarget(p.input)}`.slice(0, 120)
                }`}
                className={cn(
                  "absolute left-0.5 -translate-y-1/2 rounded-full transition-transform",
                  error
                    ? "h-1 w-3 bg-red-500"
                    : user
                      ? "h-1 w-5 bg-agent"
                      : ev.kind === "message"
                        ? "h-0.5 w-4 bg-agent"
                        : "h-0.5 w-2.5 bg-agent/50",
                  on && "scale-x-150 ring-2 ring-agent/40"
                )}
                style={{ top }}
              />
            );
          })}
          {spans.map((s) => (
            <span
              key={s.seq}
              {...hover(s.seq)}
              title={`${formatClock(s.start)} · ${triggerLabel(t, s.trigger)} · ${outcomeLabel(t, s.outcome)} · ${formatDuration(s.end - s.start)}`}
              className={cn(
                "absolute right-3.5 w-2 rounded-sm",
                RAIL_CHECK[s.outcome ?? "silent"] ?? RAIL_CHECK.silent,
                lit(s.seq) && "ring-2 ring-primary/50"
              )}
              style={{ top: pct(s.start), height: `max(3px, ${pct(s.end - s.start)})` }}
            />
          ))}
          {hovered >= 0 && (
            <div
              className={cn(
                "pointer-events-none absolute -inset-x-1 h-px -translate-y-1/2",
                events[hovered].source === "hindsight" ? "bg-primary" : "bg-agent"
              )}
              style={{ top: pct(offsets[hovered]) }}
            />
          )}
          {at !== null && (
            <div
              className="pointer-events-none absolute -inset-x-1 -translate-y-1/2 border-t-2 border-dashed border-foreground/70"
              style={{ top: pct(at) }}
            >
              <span className="absolute -top-2 left-full ml-1 rounded bg-foreground px-1 font-mono text-[9px] text-background">
                {formatClock(at)}
              </span>
            </div>
          )}
        </div>
        <div className="flex justify-between font-mono">
          <span>{formatClock(0)}</span>
          <span>{formatClock(end)}</span>
        </div>
        <div className="flex justify-between gap-1 text-[9px] font-medium">
          <span className="text-agent">{t("laneAgent")}</span>
          <span className="text-primary">{t("laneMemory")}</span>
        </div>
        <div className="relative flex justify-center">
          <button
            type="button"
            onClick={() => setLegend((v) => !v)}
            aria-expanded={legend}
            aria-label={t("legendTitle")}
            title={t("legendTitle")}
            className="flex h-5 w-5 items-center justify-center rounded-full border border-border text-[11px] font-semibold hover:bg-muted"
          >
            ?
          </button>
          {legend && (
            <div className="absolute bottom-full right-full z-30 mb-1 w-64 rounded-lg border border-border bg-popover p-3 text-[11px] text-popover-foreground shadow-lg animate-soft-fade-in">
              <div className="mb-2 font-semibold">{t("legendTitle")}</div>
              <ul className="space-y-1.5">
                {(
                  [
                    [<span key="a" className="h-0.5 w-4 rounded-full bg-agent" />, t("legendTick")],
                    [
                      <span key="b" className="h-1 w-3 rounded-full bg-red-500" />,
                      t("legendError"),
                    ],
                    [<span key="c" className="h-4 w-2 rounded-sm bg-primary/40" />, t("legendBar")],
                    [<span key="d" className="h-2 w-2 rounded-full bg-primary" />, t("legendDot")],
                    [
                      <span key="e" className="h-0.5 w-4 rounded-full bg-primary/70" />,
                      t("legendLine"),
                    ],
                    [
                      <span
                        key="f"
                        className="w-4 border-t-2 border-dashed border-foreground/70"
                      />,
                      t("legendCursor"),
                    ],
                  ] as const
                ).map(([mark, text], n) => (
                  <li key={n} className="flex items-center gap-2">
                    <span className="flex w-5 shrink-0 justify-center">{mark}</span>
                    {text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

function DetailSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      {children}
    </div>
  );
}

function CheckDetails({
  check: c,
  t,
  onOpenMemory,
}: {
  check: CheckPayload;
  t: Translate;
  onOpenMemory: (id: string) => void;
}) {
  const path = decisionPath(c.llm_calls);
  return (
    <div className="mb-1 ml-1 mt-1 space-y-3 rounded-lg border border-border/60 bg-muted/30 p-3 text-xs animate-soft-fade-in">
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {c.question && (
          <DetailSection label={t("detailQuestion")}>
            <span>
              {c.question === "headsup"
                ? t("questionHeadsup")
                : c.question === "conflict"
                  ? t("questionConflict")
                  : c.question}
            </span>
          </DetailSection>
        )}
        {c.usage && (
          <DetailSection label={t("detailTokens")}>
            <span className="font-mono">
              {t("tokens", {
                input: c.usage.input_tokens ?? 0,
                output: c.usage.output_tokens ?? 0,
              })}
            </span>
          </DetailSection>
        )}
        {path.pick?.ms != null && (
          <DetailSection label={t("detailDecisionTime")}>
            <span className="font-mono">{formatDuration(path.pick.ms)}</span>
          </DetailSection>
        )}
      </div>
      {c.error && (
        <DetailSection label={t("error")}>
          <pre className="whitespace-pre-wrap break-words font-mono text-red-600 dark:text-red-400">
            {c.error}
          </pre>
        </DetailSection>
      )}
      {c.reason && <DetailSection label={t("detailReason")}>{c.reason}</DetailSection>}
      {c.moment && (
        <DetailSection label={t("detailMoment")}>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px]">
            {c.moment}
          </pre>
        </DetailSection>
      )}
      {(c.search_queries?.length ?? 0) > 0 && (
        <DetailSection label={t("detailSearchQueries")}>
          <ol className="list-decimal space-y-1 pl-4">
            {c.search_queries!.map((q, i) => (
              <li key={i} className="whitespace-pre-wrap break-words font-mono text-[11px]">
                {q}
              </li>
            ))}
          </ol>
        </DetailSection>
      )}
      {c.context && (
        <DetailSection label={t("detailContext")}>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background p-2 font-mono text-[11px] leading-5">
            {c.context}
          </pre>
        </DetailSection>
      )}
      <DetailSection label={t("detailAnswer")}>
        {c.answer ? (
          <div className="whitespace-pre-wrap break-words">{c.answer}</div>
        ) : (
          <span className="text-muted-foreground">
            {c.abstained ? t("abstained") : t("noAnswer")}
          </span>
        )}
      </DetailSection>
      {(c.memory_ids?.length ?? 0) > 0 && (
        <DetailSection label={t("detailMemories")}>
          <MemoryChips ids={c.memory_ids!} onOpen={onOpenMemory} />
        </DetailSection>
      )}
    </div>
  );
}

function MemoryChips({ ids, onOpen }: { ids: string[]; onOpen: (id: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1">
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => onOpen(id)}
          className="rounded-md border border-border bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground hover:border-primary/40 hover:text-primary"
          title={id}
        >
          {id.slice(0, 8)}
        </button>
      ))}
    </div>
  );
}

function NoteCard({
  row,
  t,
  ackedSeq,
  timeline,
  onJump,
  onOpenDocument,
  onOpenMemory,
  enter,
  handedOff,
}: RowProps & { row: Extract<TimelineRow, { kind: "note" }> }) {
  const locale = useLocale();
  const n = row.note;
  const delivered = handedOff ? handedOff.has(row.seq) : isDelivered(row.seq, ackedSeq);
  const check = n.check_seq !== undefined ? timeline.eventBySeq.get(n.check_seq) : undefined;
  const momentSeq = check ? (check.payload as CheckPayload).moment_seq : undefined;
  const momentKey = momentSeq !== undefined ? timeline.rowKeyBySeq.get(momentSeq) : undefined;
  const momentText = check ? (check.payload as CheckPayload).moment : undefined;
  const src = n.source;

  return (
    <div className="flex items-start gap-2.5">
      <Avatar kind="hindsight" />
      <div
        className={cn(
          "min-w-0 flex-1 rounded-2xl rounded-tl-sm bg-primary-gradient p-px shadow-md shadow-primary/10",
          enter && "animate-note-glow"
        )}
      >
        <div className="rounded-[15px] rounded-tl-[3px] bg-card px-4 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="text-xs font-bold text-primary">{t("voiceHindsight")}</span>
            <span className="text-[11px] text-muted-foreground">{t("toAgent")}</span>
            {n.trigger && (
              <span className="rounded-full bg-primary/10 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
                {triggerLabel(t, n.trigger)}
              </span>
            )}
            <span className="ml-auto inline-flex items-center gap-2">
              {delivered ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                  <Check className="h-3 w-3" />
                  {t("delivered")}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-400">
                  <Clock className="h-3 w-3" />
                  {t("pending")}
                </span>
              )}
              {timeline.noteMentions.get(row.seq) === "used" && (
                <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                  <CheckCheck className="h-3 w-3" />
                  {t("noteUsed")}
                </span>
              )}
              {timeline.noteMentions.get(row.seq) === "unneeded" && (
                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                  {t("noteNotNeeded")}
                </span>
              )}
              <Time iso={row.event.created_at} />
            </span>
          </div>

          <div className="whitespace-pre-wrap break-words text-[15px] leading-relaxed text-foreground">
            {n.text}
          </div>

          {src?.quote && (
            <blockquote className="mt-3 border-l-2 border-primary/40 pl-3 text-[13px] italic leading-relaxed text-foreground/75">
              “{src.quote}”
              <footer className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] not-italic text-muted-foreground">
                {src.document_id && (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 font-mono text-primary hover:underline"
                    onClick={() => onOpenDocument(src.document_id!)}
                  >
                    <FileText className="h-3 w-3" />
                    {src.document_id}
                  </button>
                )}
                {src.mentioned_at && (
                  <span>
                    {t("mentionedAt", {
                      when: new Date(src.mentioned_at).toLocaleDateString(locale, {
                        day: "numeric",
                        month: "short",
                        year: "numeric",
                      }),
                    })}
                  </span>
                )}
              </footer>
            </blockquote>
          )}

          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px] text-muted-foreground">
            {momentKey && (
              <button
                type="button"
                onClick={() => onJump(momentKey)}
                className="inline-flex min-w-0 max-w-full items-center gap-1 hover:text-primary"
              >
                <span className="shrink-0">{t("inReplyTo")}</span>
                <span className="truncate italic">{(momentText || "").split("\n")[0]}</span>
              </button>
            )}
            {(n.memory_ids?.length ?? 0) > 0 && (
              <MemoryChips ids={n.memory_ids!} onOpen={onOpenMemory} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** A typing indicator for the third voice: shown while a live-session check runs. */
function CheckingIndicator({ t }: { t: Translate }) {
  return (
    <div className="mt-3 flex items-center gap-2.5 animate-soft-fade-in">
      <Avatar kind="hindsight" />
      <div className="inline-flex items-center gap-2 rounded-2xl rounded-tl-sm border border-primary/20 bg-primary/5 px-3.5 py-2 text-xs text-primary">
        <span className="inline-flex gap-1">
          {[0, 150, 300].map((d) => (
            <span
              key={d}
              className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary"
              style={{ animationDelay: `${d}ms` }}
            />
          ))}
        </span>
        {t("checkingLong")}
      </div>
    </div>
  );
}
