"use client";

/**
 * Hindsight Wrapped 2026: a Spotify-style story of the year across every bank,
 * opened from a button in the top bar. The story is the personal recap (it
 * names banks and quotes memories); its last slide offers a share image drawn
 * on a canvas, with counts only: no bank name, no entities, no memory text.
 *
 * The year is summed up per bank by the server (`stats/summary`) and kept in
 * localStorage per account: refreshed once a day while the year is running,
 * kept for good once it is over.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Download, Share2, X } from "lucide-react";
import { client } from "@/lib/api";
import { useBank, type BankInfo } from "@/lib/bank-context";
import { withBasePath } from "@/lib/base-path";
import { resolveHarnessLogo } from "@/lib/harness-logo";
import {
  accountStamp,
  computeWrapped,
  readCache,
  writeCache,
  wrappedInSeason,
  yearRange,
  type BankYear,
  type WrappedStats,
} from "@/lib/wrapped";
import { useFeatures } from "@/lib/features-context";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Constellation,
  HourDial,
  MemoryMapCanvas,
  Orbit,
  yearStars,
  type Star,
} from "@/components/wrapped-sky";
import { buildMemoryMap, paintMemoryMap, type BankGraph } from "@/lib/memory-map";

const YEAR = 2026;
// One stats-summary call per active bank (the server sums its year up), this many at a time.
const CONCURRENCY = 6;
// The memory constellation reads this many recent memories from each of the top banks.
const MAP_BANKS = 3;
const MAP_MEMORIES_PER_BANK = 120;
const SLIDE_MS = 7000;

// Hindsight palette (globals.css): navy page, brand blue → teal.
const NAVY = "#080C17";
const INK_MUTED = "rgba(220,232,255,0.62)";
const BLUE = "#5BB0FF";
// logo.png is the mark (x 0–185) followed by the wordmark, 139px tall.
const LOGO_SRC = "/logo.png";
const MARK_W = 185;
const MARK_H = 139;

// One faint blue light in the corner; every slide sits on the same ground.
const BACKGROUND = `radial-gradient(1200px 700px at 100% 0%, rgba(0,116,217,0.22), transparent 60%), ${NAVY}`;

async function inPool<T, R>(items: T[], fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, items.length) }, worker));
  return out;
}

async function listAllBanks(): Promise<BankInfo[]> {
  const banks: BankInfo[] = [];
  for (let offset = 0; ; offset += 100) {
    const r = await client.listBanks({ limit: 100, offset });
    banks.push(...r.banks);
    if (r.banks.length < 100 || banks.length >= r.total) return banks;
  }
}

/** The year, and whether every bank answered (a partial year is shown, never cached). */
async function loadWrapped(): Promise<{ stats: WrappedStats; complete: boolean }> {
  const { start, end } = yearRange(YEAR);
  const startDate = start.toISOString();
  // A bank that never had a fact, or was last written before the year, has nothing to add.
  const banks = (await listAllBanks()).filter(
    (b) => b.fact_count > 0 && (!b.last_write_at || b.last_write_at >= startDate)
  );
  // One bank deleted meanwhile, or refused to this caller, must not blank the whole story.
  const years = await inPool(banks, (b) =>
    client
      .getStatsSummary(b.bank_id, startDate, end.toISOString())
      .then((summary): BankYear => ({ name: b.display_alias ?? b.name ?? b.bank_id, summary }))
      .catch(() => null)
  );
  const answered = years.filter((y): y is BankYear => y !== null);
  // Every bank failing is the server failing, not an empty year: show the error, not "nothing this year".
  if (years.length && !answered.length) throw new Error("No bank could be summed up");
  const stats = computeWrapped(YEAR, answered);
  // The constellation is a picture, not a count: without it the story still stands.
  const busiest = [...answered]
    .map((y) => ({
      id: y.summary.bank_id,
      n: y.summary.hours.reduce((a, h) => a + h.world + h.experience + h.observation, 0),
    }))
    .sort((a, b) => b.n - a.n)
    .slice(0, MAP_BANKS);
  // ponytail: the graph has no date filter, so this reads each bank's latest memories; in
  // January most are the new year's and the sky is sparser. A since/until on the graph if it matters.
  const graphs = await Promise.all(
    busiest.map((b) =>
      client
        .getGraph({ bank_id: b.id, limit: MAP_MEMORIES_PER_BANK })
        .then((g) => g as BankGraph)
        .catch(() => null)
    )
  );
  const map = buildMemoryMap(
    graphs.filter((g): g is BankGraph => g !== null),
    start,
    end
  );
  stats.map = map.nodes.length ? map : null;
  return { stats, complete: answered.length === years.length };
}

/** The Hindsight mark alone, cropped out of the full logo. */
function HindsightMark({ size }: { size: number }) {
  return (
    <span
      className="inline-block overflow-hidden shrink-0"
      style={{ width: (size * MARK_W) / MARK_H, height: size }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={withBasePath(LOGO_SRC)} alt="" style={{ height: size, maxWidth: "none" }} />
    </span>
  );
}

// The brand blue → teal, dark enough to read on the light top bar too.
const GRADIENT_TEXT_ON_LIGHT =
  "bg-gradient-to-r from-[#0074D9] to-[#009296] dark:from-[#5BB0FF] dark:to-[#2DD4BF] bg-clip-text text-transparent";

/**
 * The top-bar button that opens the story of the year across every bank,
 * next to the GitHub link. The year is computed in the background as soon as
 * it mounts, so it is usually ready by the time someone presses it; the story
 * shows a loader if it is not.
 */
export function WrappedButton() {
  const t = useTranslations("wrapped");
  const { bankInfos, banksLoading, bankSearch } = useBank();
  const { features } = useFeatures();
  // Off on the server (HINDSIGHT_CP_WRAPPED_ENABLED=false) or past January: no button, nothing read.
  const shown = (features?.wrapped ?? false) && wrappedInSeason(YEAR);
  // The account is read off the unfiltered bank list only: a search in the bank picker narrows
  // `bankInfos`, and must not look like another account (or no account at all).
  // ponytail: the stamp moves once the new list is in, so an account switched in place (without
  // a page load, which nothing in this app does) shows the old year until then; a stamp from the
  // bank context's unfiltered first page if that ever happens.
  const [error, setError] = useState(false);
  const [stamp, setStamp] = useState<string | null>(null);
  const fresh = !banksLoading && !bankSearch ? accountStamp(bankInfos) : stamp;
  if (fresh !== stamp) {
    setStamp(fresh);
    setError(false); // the last account's failure is not this one's
  }
  // `stamp` is null on the server render, so localStorage is only touched in the browser.
  const cached = useMemo(() => (stamp ? readCache(localStorage, YEAR, stamp) : null), [stamp]);
  // Kept with the account it was computed for, so switching account never shows the last one's year.
  const [computed, setComputed] = useState<{ stamp: string | null; stats: WrappedStats } | null>(
    null
  );
  const stats = cached ?? (computed?.stamp === stamp ? computed.stats : null);
  // Bumped when the story is opened after a failed load, so a brief outage is retried.
  const [attempt, setAttempt] = useState(0);
  const [open, setOpen] = useState(false);

  // Computed in the background, then kept: an empty or partial year is not cached, so it fills in later.
  useEffect(() => {
    if (!shown || !stamp || stats) return;
    let cancelled = false;
    loadWrapped()
      .then(({ stats: s, complete }) => {
        if (cancelled) return;
        if (complete && s.memories > 0) writeCache(localStorage, YEAR, stamp, s);
        setComputed({ stamp, stats: s });
      })
      .catch(() => !cancelled && setError(true));
    return () => {
      cancelled = true;
    };
  }, [shown, stamp, stats, attempt]);

  if (!shown || !stamp) return null;
  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          if (error) {
            setError(false);
            setAttempt((a) => a + 1);
          }
          setOpen(true);
        }}
        title={t("bannerTitle", { period: YEAR })}
        aria-label={t("bannerTitle", { period: YEAR })}
        className="group h-9 shrink-0 gap-1.5 whitespace-nowrap px-2.5 sm:px-3"
      >
        <span className="transition-transform group-hover:-rotate-6 group-hover:scale-110">
          <HindsightMark size={18} />
        </span>
        <span className={`hidden lg:inline font-semibold ${GRADIENT_TEXT_ON_LIGHT}`}>
          {t("pill", { year: YEAR })}
        </span>
      </Button>
      {open && <WrappedStory stats={stats} error={error} onClose={() => setOpen(false)} />}
    </>
  );
}

/** A slide; `art` is its constellation, drawn beside the words (above them on a phone). */
type Slide = { id: string; render: () => React.ReactNode; art?: () => React.ReactNode };

// Where the top names sit in their constellation (a 100×70 sky), and how they are joined.
const NAME_SPOTS = [
  [22, 18],
  [50, 34],
  [80, 16],
  [70, 56],
  [28, 54],
];
const NAME_LINKS: [number, number][] = [
  [0, 1],
  [1, 2],
  [1, 3],
  [1, 4],
  [3, 4],
];
const BANK_SPOTS = [
  [50, 30],
  [24, 70],
  [78, 72],
];

const short = (s: string, n = 18) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function Rise({
  delay = 0,
  children,
  className = "",
}: {
  delay?: number;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`wrapped-rise ${className}`} style={{ animationDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

// One type scale for every slide: a blue kicker, one hero (a number or a
// title), a line of body. Left-aligned, flat, white on navy.
const HERO =
  "font-semibold tracking-[-0.04em] leading-[0.9] tabular-nums text-white text-[clamp(3.5rem,15vw,9.5rem)]";
const TITLE = "text-3xl sm:text-5xl font-semibold tracking-tight text-white";
const BODY = "text-lg text-white/60 leading-relaxed max-w-xl";

function Frame({ kicker, children }: { kicker: string; children: React.ReactNode }) {
  return (
    <div className="w-full max-w-2xl text-left">
      <Rise className="mb-6 text-sm font-medium uppercase tracking-[0.18em] text-[#5BB0FF]">
        {kicker}
      </Rise>
      {children}
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="border-t border-white/10 pt-4">
      <div className="text-4xl font-semibold tracking-tight tabular-nums text-white">{value}</div>
      <div className="mt-1 text-sm text-white/50 first-letter:uppercase">{label}</div>
    </div>
  );
}

function Quote({ text }: { text: string }) {
  return (
    <blockquote className="border-l-2 border-[#5BB0FF] pl-6 text-2xl sm:text-3xl font-medium leading-snug tracking-tight text-white">
      {text}
    </blockquote>
  );
}

function WrappedStory({
  stats,
  error,
  onClose,
}: {
  stats: WrappedStats | null;
  error: boolean;
  onClose: () => void;
}) {
  const t = useTranslations("wrapped");
  const locale = useLocale();
  const [index, setIndex] = useState(0);
  const nf = useMemo(() => new Intl.NumberFormat(locale), [locale]);

  const slides = useMemo<Slide[]>(() => {
    if (!stats) return [];
    if (stats.memories === 0)
      return [
        {
          id: "empty",
          render: () => (
            <Frame kicker={t("brand")}>
              <Rise delay={150} className={TITLE}>
                {t("emptyTitle", { period: YEAR })}
              </Rise>
              <Rise delay={300} className={`${BODY} mt-4`}>
                {t("emptyBody")}
              </Rise>
            </Frame>
          ),
        },
      ];
    const peakMonth = stats.bars.indexOf(Math.max(...stats.bars));
    const year = yearStars(stats.bars);
    const monthLetter = new Intl.DateTimeFormat(locale, { month: "narrow" });
    const busiest = stats.busiestDay
      ? new Intl.DateTimeFormat(locale, { weekday: "long", month: "long", day: "numeric" }).format(
          new Date(`${stats.busiestDay.name}T12:00:00`)
        )
      : null;
    const harness = stats.topHarness ? resolveHarnessLogo(stats.topHarness.name) : null;
    const s: Slide[] = [
      {
        id: "intro",
        render: () => (
          <Frame kicker={t("introKicker")}>
            <Rise delay={150} className={HERO}>
              {YEAR}
            </Rise>
            <Rise delay={300} className={`${BODY} mt-6`}>
              {t("introBody")}
            </Rise>
            <Rise delay={450} className="mt-10 text-sm text-white/40">
              {t("introBanks", { count: stats.banks })}
            </Rise>
          </Frame>
        ),
      },
      {
        id: "memories",
        render: () => (
          <Frame kicker={t("memoriesLead")}>
            <Rise delay={150} className={HERO}>
              {nf.format(stats.memories)}
            </Rise>
            <Rise delay={300} className={`${TITLE} mt-4`}>
              {t("memoriesTitle", { count: stats.memories })}
            </Rise>
            <Rise delay={450} className={`${BODY} mt-6`}>
              {stats.documents === null
                ? t("memoriesBodyNoDocuments", {
                    facts: nf.format(stats.facts),
                    observations: nf.format(stats.observations),
                  })
                : t("memoriesBody", {
                    facts: nf.format(stats.facts),
                    observations: nf.format(stats.observations),
                    documents: nf.format(stats.documents),
                  })}
            </Rise>
          </Frame>
        ),
        art: stats.map ? () => <MemoryMapCanvas map={stats.map!} /> : undefined,
      },
      {
        id: "rhythm",
        render: () => (
          <Frame kicker={t("rhythmTitle")}>
            <div className="mb-10">
              <Constellation
                stars={year}
                links={year.slice(1).map((_, i): [number, number] => [i, i + 1])}
                h={60}
              />
              <div className="relative mt-2 h-4 text-xs text-white/40">
                {year.map((st, i) => (
                  <span
                    key={i}
                    className={`absolute -translate-x-1/2 ${i === peakMonth ? "text-[#5BB0FF]" : ""}`}
                    style={{ left: `${st.x}%` }}
                  >
                    {monthLetter.format(new Date(YEAR, i, 1))}
                  </span>
                ))}
              </div>
            </div>
            <Rise delay={450} className="grid grid-cols-2 gap-8">
              <Stat
                value={nf.format(stats.activeDays)}
                label={t("activeDays", { count: stats.activeDays })}
              />
              <Stat value={nf.format(stats.longestStreak)} label={t("longestStreak")} />
            </Rise>
            {busiest && (
              <Rise delay={600} className={`${BODY} mt-8`}>
                {t("busiestDay", { day: busiest, count: stats.busiestDay!.count })}
              </Rise>
            )}
          </Frame>
        ),
      },
    ];
    if (stats.banks > 1)
      s.push({
        id: "banks",
        render: () => (
          <Frame kicker={t("banksLead")}>
            <Rise delay={150} className={HERO}>
              {nf.format(stats.banks)}
            </Rise>
            <Rise delay={300} className={`${TITLE} mt-4 mb-10`}>
              {t("banksTitle", { count: stats.banks })}
            </Rise>
            <ol className="border-b border-white/10">
              {stats.topBanks.map((b, i) => (
                <Rise key={i} delay={450 + i * 120}>
                  <li className="flex items-baseline gap-6 border-t border-white/10 py-3">
                    <span className="w-6 font-mono text-sm tabular-nums text-white/40">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="flex-1 truncate font-mono text-base text-white">{b.name}</span>
                    <span className="text-sm tabular-nums text-white/50">
                      {t("memoriesCount", { count: b.count })}
                    </span>
                  </li>
                </Rise>
              ))}
            </ol>
          </Frame>
        ),
        art: () => {
          const max = stats.topBanks[0]?.count || 1;
          const stars: Star[] = stats.topBanks.map((b, i) => ({
            x: BANK_SPOTS[i][0],
            y: BANK_SPOTS[i][1],
            r: 1.5 + 5 * Math.sqrt(b.count / max),
            tone: i === 0 ? "blue" : i === 1 ? "teal" : "white",
            halo: i === 0,
            label: String(i + 1).padStart(2, "0"),
          }));
          const links: [number, number][] =
            stars.length === 3
              ? [
                  [0, 1],
                  [1, 2],
                  [2, 0],
                ]
              : stars.length === 2
                ? [[0, 1]]
                : [];
          return <Constellation stars={stars} links={links} labelSize={4} />;
        },
      });
    if (stats.topEntities.length)
      s.push({
        id: "names",
        render: () => (
          <Frame kicker={t("namesTitle")}>
            <Constellation
              h={70}
              labelSize={4.2}
              stars={stats.topEntities.map((e, i) => ({
                x: NAME_SPOTS[i][0],
                y: NAME_SPOTS[i][1],
                r: 1.2 + 2.8 * Math.sqrt(e.count / stats.topEntities[0].count),
                tone: i === 0 ? "blue" : "white",
                halo: i === 0,
                label: short(e.name),
                sub: nf.format(e.count),
              }))}
              links={NAME_LINKS.filter(([a, b]) => Math.max(a, b) < stats.topEntities.length)}
            />
          </Frame>
        ),
      });
    if (stats.persona)
      s.push({
        id: "persona",
        render: () => (
          <Frame kicker={t("personaLead")}>
            <Rise delay={150} className={HERO}>
              {String(stats.peakHour).padStart(2, "0")}:00
            </Rise>
            <Rise delay={300} className={`${TITLE} mt-4`}>
              {t(`persona.${stats.persona!}.title`)}
            </Rise>
            <Rise delay={450} className={`${BODY} mt-6`}>
              {t(`persona.${stats.persona!}.body`, {
                hour: String(stats.peakHour).padStart(2, "0"),
              })}
            </Rise>
          </Frame>
        ),
        art: () => <HourDial hourly={stats.hourly} peak={stats.peakHour} />,
      });
    if (stats.topHarness)
      s.push({
        id: "agent",
        render: () => (
          <Frame kicker={t("agentLead")}>
            <Rise delay={300} className={TITLE}>
              {harness?.label ?? stats.topHarness!.name}
            </Rise>
            <Rise delay={450} className={`${BODY} mt-4`}>
              {t("agentBody", { count: stats.topHarness!.count })}
            </Rise>
          </Frame>
        ),
        art: () => <Orbit src={harness ? withBasePath(harness.src) : null} />,
      });
    if (stats.strongestBelief && stats.strongestBelief.proofs > 1)
      s.push({
        id: "belief",
        render: () => (
          <Frame kicker={t("beliefLead")}>
            <Rise delay={150}>
              <Quote text={stats.strongestBelief!.text} />
            </Rise>
            <Rise delay={300} className={`${BODY} mt-6`}>
              {t("beliefBody", { count: stats.strongestBelief!.proofs })}
            </Rise>
          </Frame>
        ),
      });
    if (stats.firstMemory)
      s.push({
        id: "first",
        render: () => (
          <Frame kicker={t("firstLead")}>
            <Rise delay={150}>
              <Quote text={stats.firstMemory!.text} />
            </Rise>
            <Rise delay={300} className={`${BODY} mt-6`}>
              {new Intl.DateTimeFormat(locale, { dateStyle: "long" }).format(
                new Date(stats.firstMemory!.date)
              )}
            </Rise>
          </Frame>
        ),
      });
    s.push({ id: "share", render: () => <ShareSlide stats={stats} /> });
    return s;
  }, [stats, t, locale, nf]);

  const last = slides.length - 1;
  // Clamped at 0 too: while loading there are no slides, and a tap must not go to -1.
  const next = useCallback(() => setIndex((i) => Math.max(0, Math.min(i + 1, last))), [last]);
  const prev = useCallback(() => setIndex((i) => Math.max(i - 1, 0)), []);

  // Auto-advance like a story; the share slide waits for the viewer.
  useEffect(() => {
    if (!slides.length || index >= last) return;
    const id = setTimeout(next, SLIDE_MS);
    return () => clearTimeout(id);
  }, [index, last, next, slides.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight" || e.key === " ") next();
      else if (e.key === "ArrowLeft") prev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, onClose]);

  const slide = slides[index];
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t("dialogLabel")}
      style={{ background: BACKGROUND }}
      className="fixed inset-0 z-50 flex flex-col text-white"
    >
      <div className="relative flex gap-1 px-4 pt-4">
        {slides.map((sl, i) => (
          <div key={sl.id} className="h-0.5 flex-1 bg-white/15 overflow-hidden">
            <div
              key={`${sl.id}-${index}`}
              className="h-full bg-white"
              style={
                i === index && index < last
                  ? { animation: `wrapped-progress ${SLIDE_MS}ms linear forwards` }
                  : { width: i <= index ? "100%" : "0%" }
              }
            />
          </div>
        ))}
      </div>
      <div className="relative flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2.5">
          <HindsightMark size={22} />
          <span className="font-semibold">Hindsight</span>
          <span className="text-sm text-white/50">{t("pill", { year: YEAR })}</span>
        </div>
        <button
          onClick={onClose}
          aria-label={t("close")}
          className="rounded-md p-2 text-white/70 hover:bg-white/10 hover:text-white"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="relative flex-1 min-h-0">
        {!stats && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
            <Spinner size="lg" variant="jump" />
            <div className="text-white/60">{t("loading", { period: YEAR })}</div>
          </div>
        )}
        {error && !stats && (
          <div className="absolute inset-0 flex items-center justify-center text-white/60">
            {t("error")}
          </div>
        )}
        {slide && (
          <>
            {/* Tap zones: left third goes back, the rest forward (not on the share slide). */}
            {index < last && (
              <>
                <button
                  aria-label={t("previous")}
                  onClick={prev}
                  className="absolute inset-y-0 left-0 w-1/3 z-10"
                />
                <button
                  aria-label={t("next")}
                  onClick={next}
                  className="absolute inset-y-0 right-0 w-2/3 z-10"
                />
              </>
            )}
            <div
              key={slide.id}
              className="absolute inset-0 overflow-y-auto flex items-center justify-center px-6 sm:px-12 pb-10"
            >
              {slide.art ? (
                <div className="grid w-full max-w-6xl items-center gap-6 md:grid-cols-2 md:gap-16">
                  <div className="order-2 min-w-0 md:order-1">{slide.render()}</div>
                  <div className="order-1 mx-auto w-full max-w-[34vh] md:order-2 md:max-w-[520px]">
                    {slide.art()}
                  </div>
                </div>
              ) : (
                slide.render()
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ShareSlide({ stats }: { stats: WrappedStats }) {
  const t = useTranslations("wrapped");
  const locale = useLocale();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [hint, setHint] = useState<{ blocked: boolean; copied: boolean } | null>(null);
  // How to paste here: a key on a keyboard, tap and hold on a touch screen.
  const touch = typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches;
  const mac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    const nf = new Intl.NumberFormat(locale);
    // Only numbers worth showing off: most banks are written in bursts, and
    // "1 active day, 1 day streak" reads as a complaint. Those show the busiest day instead.
    const tiles: CardTile[] = [];
    if (stats.banks > 1)
      tiles.push([nf.format(stats.banks), t("banksTitle", { count: stats.banks })]);
    if (stats.activeDays >= 7)
      tiles.push([nf.format(stats.activeDays), t("activeDays", { count: stats.activeDays })]);
    if (stats.longestStreak >= 3) tiles.push([nf.format(stats.longestStreak), t("longestStreak")]);
    // ...unless that one day is the whole year, which the headline already says.
    if (stats.busiestDay && stats.busiestDay.count < stats.memories && tiles.length < 2)
      tiles.push([
        nf.format(stats.busiestDay.count),
        t("busiestOn", {
          count: stats.busiestDay.count,
          date: new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" }).format(
            new Date(`${stats.busiestDay.name}T12:00:00`)
          ),
        }),
      ]);
    if (stats.documents)
      tiles.push([nf.format(stats.documents), t("documents", { count: stats.documents })]);
    if (stats.observations)
      tiles.push([nf.format(stats.observations), t("observations", { count: stats.observations })]);
    drawShareCard(canvas, {
      stats,
      tiles: tiles.slice(0, 4),
      text: {
        count: nf.format(stats.memories),
        pill: t("pill", { year: YEAR }).toUpperCase(),
        tagline: t("cardTagline").toUpperCase(),
        memories: t("memoriesTitle", { count: stats.memories }),
        persona: stats.persona ? t(`persona.${stats.persona}.title`) : null,
        peak:
          stats.peakHour === null
            ? ""
            : t("peakAt", { hour: String(stats.peakHour).padStart(2, "0") }).toUpperCase(),
        agent: t("agentShort").toUpperCase(),
      },
    }).then(() => canvas.toBlob((b) => !cancelled && setBlob(b), "image/png"));
    return () => {
      cancelled = true;
    };
  }, [stats, locale, t]);

  const fileName = `hindsight-wrapped-${YEAR}.png`;
  const download = () => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  // A post link can't carry an image, so the card goes on the clipboard (or is downloaded where
  // that is refused) and the post opens with the text filled in, ready for a paste.
  const postTo = (url: string) => {
    if (!blob) return;
    let copied: Promise<boolean>;
    try {
      copied = navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]).then(
        () => true,
        () => false
      );
    } catch {
      copied = Promise.resolve(false); // no image clipboard in this browser
    }
    // Both start inside the click: the clipboard needs the page focused, the new tab needs the click.
    // Not "noopener": that makes window.open return null always, hiding a blocked tab.
    const tab = window.open(url, "_blank");
    if (tab) tab.opener = null;
    void copied.then((ok) => {
      if (!ok) download();
      setHint({ blocked: !tab, copied: ok });
    });
  };
  const text = t("postText", { year: YEAR, count: stats.memories });
  const postText = encodeURIComponent(text);
  // The system share sheet takes the image itself (phones, Safari, recent Chrome): the app picked
  // there (X, Instagram, LinkedIn, ...) opens with the card already attached.
  const file = useMemo(
    () => (blob ? new File([blob], fileName, { type: "image/png" }) : null),
    [blob, fileName]
  );
  const canShare = useMemo(
    () =>
      !!file && typeof navigator !== "undefined" && !!navigator.canShare?.({ files: [file], text }),
    [file, text]
  );
  const targets = [
    { label: t("postOnX"), Logo: XLogo, url: `https://x.com/intent/post?text=${postText}` },
    {
      label: t("postOnLinkedIn"),
      Logo: LinkedInLogo,
      url: `https://www.linkedin.com/feed/?shareActive=true&text=${postText}`,
    },
  ];

  const secondary =
    "inline-flex items-center justify-center gap-2 rounded-md border border-white/15 px-4 py-2.5 text-sm font-medium text-white hover:bg-white/5 disabled:opacity-50";
  return (
    <div className="flex w-full max-w-4xl flex-col items-center gap-8 sm:flex-row sm:items-center sm:gap-14">
      <Rise className="shrink-0">
        <canvas
          ref={canvasRef}
          width={1080}
          height={1350}
          className="h-[46vh] sm:h-[64vh] max-h-[640px] w-auto rounded-md border border-white/10"
        />
      </Rise>
      <div className="w-full max-w-xs text-left">
        <Rise
          delay={150}
          className="mb-4 text-sm font-medium uppercase tracking-[0.18em] text-[#5BB0FF]"
        >
          {t("pill", { year: YEAR })}
        </Rise>
        <Rise delay={250} className="text-3xl font-semibold tracking-tight text-white">
          {t("shareTitle")}
        </Rise>
        <Rise delay={350} className="mt-8 flex flex-col gap-2">
          {targets.map(({ label, Logo, url }, n) => (
            <button
              key={url}
              onClick={() => postTo(url)}
              disabled={!blob}
              className={
                n === 0
                  ? "inline-flex items-center justify-center gap-2 rounded-md bg-white px-4 py-2.5 text-sm font-medium text-[#080C17] hover:bg-white/90 disabled:opacity-50"
                  : secondary
              }
            >
              <Logo className="w-4 h-4" /> {label}
            </button>
          ))}
          {canShare && (
            <button
              onClick={() => navigator.share({ files: [file!], text }).catch(() => {})}
              className={secondary}
            >
              <Share2 className="w-4 h-4" /> {t("shareMore")}
            </button>
          )}
          <button onClick={download} disabled={!blob} className={secondary}>
            <Download className="w-4 h-4" /> {t("download")}
          </button>
          {hint && (
            <p role="status" className="mt-2 text-sm text-white/60">
              {[
                hint.blocked && t("postHintBlocked"),
                !hint.copied
                  ? t("postHintDownloaded")
                  : !hint.blocked &&
                    (touch
                      ? t("postHintCopiedTouch")
                      : t("postHintCopied", { keys: mac ? "⌘V" : "Ctrl+V" })),
              ]
                .filter((line): line is string => !!line)
                .map((line) => (
                  <span key={line} className="block">
                    {line}
                  </span>
                ))}
            </p>
          )}
        </Rise>
      </div>
    </div>
  );
}

/** The X logo (the lucide icon set has none). */
function XLogo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={className} fill="currentColor">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

function LinkedInLogo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={className} fill="#0A66C2">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

function ellipsize(ctx: CanvasRenderingContext2D, text: string, max: number) {
  if (ctx.measureText(text).width <= max) return text;
  let s = text;
  while (s.length > 1 && ctx.measureText(`${s}…`).width > max) s = s.slice(0, -1);
  return `${s}…`;
}

/** A number and what it counts. */
type CardTile = [string, string];

type CardText = Record<"count" | "pill" | "tagline" | "memories" | "peak" | "agent", string> & {
  persona: string | null;
};

/**
 * The share image, 1080×1350 (4:5, the portrait size feeds show uncropped):
 * the year's memory count, up to four tiles, the persona and the top agent,
 * centred between the header and the footer. Counts only: no bank name, no
 * entities, no memory text.
 */
async function drawShareCard(
  canvas: HTMLCanvasElement,
  o: { stats: WrappedStats; tiles: CardTile[]; text: CardText }
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const P = 90;
  const gap = 24;
  const font = getComputedStyle(document.body).fontFamily || "system-ui, sans-serif";
  const f = (weight: number, size: number) => `${weight} ${size}px ${font}`;
  const spacing = (px: number) => {
    if ("letterSpacing" in ctx) ctx.letterSpacing = `${px}px`;
  };
  const { stats, text } = o;
  let { tiles } = o;
  const harness = stats.topHarness ? resolveHarnessLogo(stats.topHarness.name) : null;
  const harnessLabel = stats.topHarness ? (harness?.label ?? stats.topHarness.name) : null;
  const [logo, harnessImg] = await Promise.all([
    loadImage(withBasePath(LOGO_SRC)),
    harness ? loadImage(withBasePath(harness.src)) : Promise.resolve(null),
  ]);

  const panel = (x: number, y: number, w: number, h: number) => {
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 12);
    ctx.fill();
    ctx.stroke();
  };
  const drawMark = (x: number, y: number, h: number, alpha = 1) => {
    if (!logo) return;
    ctx.globalAlpha = alpha;
    ctx.drawImage(logo, 0, 0, MARK_W, MARK_H, x, y, (h * MARK_W) / MARK_H, h);
    ctx.globalAlpha = 1;
  };
  const tile = (
    x: number,
    y: number,
    w: number,
    h: number,
    value: string,
    label: string,
    vSize: number,
    lSize = 30,
    pad = 32
  ) => {
    panel(x, y, w, h);
    ctx.fillStyle = "#fff";
    ctx.font = f(600, vSize);
    ctx.fillText(value, x + pad, y + vSize + 18);
    ctx.fillStyle = INK_MUTED;
    ctx.font = f(500, lSize);
    ctx.fillText(ellipsize(ctx, label, w - 2 * pad), x + pad, y + h - 30);
  };

  // Background: the story's navy with its one faint blue light in the corner.
  ctx.fillStyle = NAVY;
  ctx.fillRect(0, 0, W, H);
  const glow = ctx.createRadialGradient(W, 0, 0, W, 0, 1100);
  glow.addColorStop(0, "rgba(0,116,217,0.30)");
  glow.addColorStop(1, "rgba(8,12,23,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  // Header: mark + wordmark, and "WRAPPED 2026" on the right.
  ctx.textBaseline = "alphabetic";
  drawMark(P, 82, 56);
  ctx.fillStyle = "#fff";
  ctx.font = f(600, 44);
  spacing(0);
  ctx.fillText("Hindsight", P + 94, 126);
  ctx.fillStyle = INK_MUTED;
  ctx.font = f(500, 26);
  spacing(4);
  ctx.textAlign = "right";
  ctx.fillText(text.pill, W - P, 122);
  ctx.textAlign = "left";
  spacing(0);

  // Persona and top agent: side by side at the bottom, no boxes. A small tracked label over
  // each value, like the tagline over the number.
  const footnotes = [
    text.persona ? { label: text.peak, value: text.persona, img: null } : null,
    harnessLabel ? { label: text.agent, value: harnessLabel, img: harnessImg } : null,
  ].filter((n): n is { label: string; value: string; img: HTMLImageElement | null } => !!n);
  const FOOT_H = 104;
  const drawFootnotes = (y: number) => {
    const colW = (W - 2 * P - (footnotes.length - 1) * 56) / footnotes.length;
    // One size for both values, shrunk until the longer one fits its column.
    let size = 44;
    ctx.font = f(600, size);
    const widest = () =>
      Math.max(...footnotes.map((n) => ctx.measureText(n.value).width + (n.img ? 60 : 0)));
    while (size > 32 && widest() > colW) ctx.font = f(600, --size);
    footnotes.forEach((n, i) => {
      const x = P + i * (colW + 56);
      ctx.fillStyle = BLUE;
      ctx.font = f(500, 24);
      spacing(4);
      ctx.fillText(n.label, x, y + 24);
      spacing(0);
      let vx = x;
      if (n.img) {
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.roundRect(x, y + 50, 44, 44, 10);
        ctx.fill();
        ctx.drawImage(n.img, x + 6, y + 56, 32, 32);
        vx += 60;
      }
      ctx.fillStyle = "#fff";
      ctx.font = f(600, size);
      ctx.fillText(ellipsize(ctx, n.value, x + colW - vx), vx, y + 88);
    });
  };

  // Lay the body out once to measure it, then centre it between header and footer.
  ctx.font = f(600, 220);
  spacing(-10);
  const num = ctx.measureText(text.count);
  spacing(0);
  // The label sits under the number's lowest ink, so a comma never runs into it.
  const numBase = 70 + num.actualBoundingBoxAscent;
  const labelBase = numBase + Math.max(0, num.actualBoundingBoxDescent) + 66;
  const heroH = labelBase + 16;
  const th = 150;
  const footH = footnotes.length ? FOOT_H : 0;
  const top = 180;
  const room = H - 130 - top;
  const layout = (n: number) => {
    const rows = Math.ceil(n / 2);
    const tilesH = rows ? rows * th + (rows - 1) * gap : 0;
    return { tilesH, bodyH: heroH + (tilesH ? 56 + tilesH : 0) + (footH ? 64 + footH : 0) };
  };
  // The memory constellation sits above the number; tiles give way, two at a time, to fit it.
  const SKY_MIN = 300;
  if (stats.map)
    while (tiles.length && room - layout(tiles.length).bodyH - 40 < SKY_MIN)
      tiles = tiles.slice(0, Math.max(0, tiles.length - 2));
  const { tilesH, bodyH } = layout(tiles.length);
  const skyH = stats.map ? Math.min(480, room - bodyH - 40) : 0;
  let y = top + Math.max(0, (room - bodyH - (skyH >= SKY_MIN ? skyH + 40 : 0)) / 2);
  if (stats.map && skyH >= SKY_MIN) {
    paintMemoryMap(ctx, stats.map, { x: P, y, w: W - 2 * P, h: skyH });
    y += skyH + 40;
  }

  ctx.fillStyle = BLUE;
  ctx.font = f(500, 28);
  spacing(5);
  ctx.fillText(text.tagline, P, y + 30);
  spacing(-10);
  ctx.font = f(600, 220);
  ctx.fillStyle = "#fff";
  ctx.fillText(text.count, P - 10, y + numBase);
  spacing(0);
  ctx.fillStyle = "#fff";
  ctx.font = f(600, 52);
  ctx.fillText(text.memories, P, y + labelBase);
  y += heroH;

  if (tilesH) {
    y += 56;
    const tw = (W - 2 * P - gap) / 2;
    tiles.forEach(([v, l], i) => {
      // An odd last tile takes the whole row.
      const full = i === tiles.length - 1 && i % 2 === 0;
      const ty = y + Math.floor(i / 2) * (th + gap);
      tile(P + (i % 2) * (tw + gap), ty, full ? W - 2 * P : tw, th, v, l, 72);
    });
    y += tilesH;
  }
  if (footH) drawFootnotes(y + 64);

  ctx.fillStyle = INK_MUTED;
  ctx.font = f(600, 28);
  spacing(1);
  ctx.fillText("hindsight.vectorize.io", P, H - 56);
  spacing(0);
}
