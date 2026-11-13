"use client";

// A knowledge bank's home, built from the memory bank home's own pieces: the stat
// strip and distribution of its "Memory store" card, its constellation card, and its
// activity chart's style. Every card links through to the documents it counts.

import * as React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { ArrowRight, CircleAlert, FolderOpen, HardDrive, Layers, Network } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { client } from "@/lib/api";
import {
  ACTIVITY_PERIODS,
  fillTimeseries,
  MEDIA_TYPES,
  documentLabel,
  formatBytes,
  knowledgeBankRoute,
  type ActivityBucket,
  type ActivityPeriod,
  type KnowledgeBank,
  type KnowledgeDocument,
  type MediaType,
} from "@/lib/knowledge-banks";
import { formatAbsoluteDateTime, formatRelativeTime } from "@/lib/relative-time";
import {
  AXIS_TICK_STYLE,
  CHART_COLORS,
  ChartTooltip,
  Distribution,
  formatBucketLabel,
  formatBucketTooltip,
  formatCompact,
  InlineStat,
} from "@/components/bank-stats-view";
import { KIND_HEX, KindIcon, StatusMark } from "@/components/knowledge-passage";
import { KnowledgeConstellation } from "@/components/knowledge-constellation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";

const SERIES = [
  { key: "documents", color: CHART_COLORS.semantic },
  { key: "passages", color: CHART_COLORS.temporal },
] as const;
type SeriesKey = (typeof SERIES)[number]["key"];

const RECENT_LIST = 6;
/** The most recent documents the home's constellation draws (one detail read each). */
const CONSTELLATION_DOCS = 20;
const CONSTELLATION_HEIGHT = 464;
const ALL = "all";

/** The card header's "View all →" link, as on the memory bank home. */
function ViewAll({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="flex shrink-0 items-center gap-1 text-xs text-primary hover:underline"
    >
      {label} <ArrowRight className="h-3 w-3" />
    </Link>
  );
}

function ActivityChart({ bankId, version }: { bankId: string; version: string }) {
  const t = useTranslations("knowledgeBanks.home");
  const tDocs = useTranslations("knowledgeBanks.documents");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const [period, setPeriod] = React.useState<ActivityPeriod>("7d");
  const [kind, setKind] = React.useState<MediaType | typeof ALL>(ALL);
  const [series, setSeries] = React.useState<ActivityBucket[] | null>(null);

  // `version` changes when the bank's counts move, so a finished upload shows up.
  React.useEffect(() => {
    let live = true;
    client
      .getKnowledgeTimeseries(bankId, period, kind === ALL ? undefined : kind)
      .then((res) => live && setSeries(res.buckets))
      .catch(() => live && setSeries([]));
    return () => {
      live = false;
    };
  }, [bankId, period, kind, version]);

  const [enabled, setEnabled] = React.useState<Record<SeriesKey, boolean>>({
    documents: true,
    passages: true,
  });
  const { trunc, buckets } = fillTimeseries(series ?? [], period);
  const data = buckets.map((b) => ({
    ...b,
    label: formatBucketLabel(b.time, trunc),
    tooltipLabel: formatBucketTooltip(b.time, trunc),
  }));
  const totalDocs = data.reduce((s, b) => s + b.documents, 0);
  const seriesLabel: Record<SeriesKey, string> = {
    documents: t("documents"),
    passages: t("passages"),
  };

  return (
    <Card>
      <CardHeader className="pb-2 space-y-2">
        <div className="flex flex-row items-center justify-between">
          <CardTitle className="text-sm font-semibold">{t("activityTitle")}</CardTitle>
          <div className="flex items-center gap-0.5 rounded-md bg-muted/60 p-0.5">
            {ACTIVITY_PERIODS.map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors tabular-nums ${
                  period === p
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <Select value={kind} onValueChange={(v) => setKind(v as MediaType | typeof ALL)}>
              <SelectTrigger className="h-7 w-36 text-xs" aria-label={tDocs("kindFilter")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>{tDocs("allKinds")}</SelectItem>
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
            {SERIES.map(({ key, color }) => (
              <button
                key={key}
                onClick={() => setEnabled((prev) => ({ ...prev, [key]: !prev[key] }))}
                className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium transition-all ${
                  enabled[key]
                    ? "bg-muted text-foreground"
                    : "bg-transparent text-muted-foreground/60 hover:text-muted-foreground"
                }`}
              >
                <span
                  className="w-2 h-2 rounded-[2px] transition-opacity"
                  style={{ backgroundColor: color, opacity: enabled[key] ? 1 : 0.3 }}
                />
                {seriesLabel[key]}
              </button>
            ))}
          </div>
          <Link
            href={knowledgeBankRoute(
              bankId,
              `?view=documents${kind === ALL ? "" : `&kind=${kind}`}`
            )}
            className="text-xs text-muted-foreground tabular-nums hover:text-foreground hover:underline"
          >
            {t("addedInPeriod", { count: totalDocs })}
          </Link>
        </div>
      </CardHeader>
      <CardContent>
        {series === null ? (
          <div className="h-[180px] flex items-center justify-center">
            <Spinner size="sm" />
          </div>
        ) : totalDocs === 0 ? (
          <div className="h-[180px] flex items-center justify-center text-sm text-muted-foreground">
            {t("noneInPeriod")}
          </div>
        ) : (
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <defs>
                  {SERIES.map(({ key, color }) => (
                    <linearGradient key={key} id={`kb-grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={color} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid
                  vertical={false}
                  stroke={CHART_COLORS.border}
                  strokeOpacity={0.5}
                  strokeDasharray="2 4"
                />
                <XAxis
                  dataKey="label"
                  tick={AXIS_TICK_STYLE}
                  axisLine={false}
                  tickLine={false}
                  dy={4}
                  minTickGap={20}
                />
                <YAxis
                  tick={AXIS_TICK_STYLE}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                  allowDecimals={false}
                  tickFormatter={(v: number) => formatCompact(v)}
                />
                <Tooltip
                  content={<ChartTooltip />}
                  labelFormatter={(_v, payload) => payload?.[0]?.payload?.tooltipLabel || ""}
                  cursor={{ stroke: CHART_COLORS.mutedFg, strokeWidth: 1, strokeDasharray: "3 3" }}
                />
                {SERIES.map(({ key, color }) =>
                  enabled[key] ? (
                    <Area
                      key={key}
                      type="monotone"
                      dataKey={key}
                      name={seriesLabel[key]}
                      stroke={color}
                      strokeWidth={2}
                      fill={`url(#kb-grad-${key})`}
                      isAnimationActive={false}
                    />
                  ) : null
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** A stat of the strip, as a link to the documents it counts. */
function StatLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link href={href} className="block transition-colors hover:bg-muted/40">
      {children}
    </Link>
  );
}

/** The stat strip and kinds of a knowledge bank, each linking to the documents it counts:
 *  its Home's first card, and the head of its Settings' General tab. */
export function KnowledgeBankStats({ bank }: { bank: KnowledgeBank }) {
  const t = useTranslations("knowledgeBanks.home");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const route = (query: string) => knowledgeBankRoute(bank.bank_id, query);
  const documentsHref = route("?view=documents");
  const ingestNote = (bank.pending_count > 0 || bank.failed_count > 0) && (
    <p className="mt-0.5 text-[11px] text-muted-foreground">
      {bank.pending_count > 0 && t("pending", { count: bank.pending_count })}
      {bank.pending_count > 0 && bank.failed_count > 0 && " · "}
      {bank.failed_count > 0 && (
        <span className="font-medium text-red-600 dark:text-red-400">
          {t("failed", { count: bank.failed_count })}
        </span>
      )}
    </p>
  );

  return (
    <Card>
      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-3 md:divide-x divide-y md:divide-y-0 divide-border/60">
          <StatLink href={documentsHref}>
            <InlineStat
              icon={FolderOpen}
              label={t("documents")}
              value={bank.document_count}
              note={ingestNote}
            />
          </StatLink>
          <StatLink href={documentsHref}>
            <InlineStat icon={Layers} label={t("passages")} value={bank.passage_count} />
          </StatLink>
          <StatLink href={documentsHref}>
            <InlineStat
              icon={HardDrive}
              label={t("size")}
              value={bank.total_bytes}
              display={formatBytes(bank.total_bytes)}
            />
          </StatLink>
        </div>
        {bank.failed_count > 0 && (
          // Failures need a look: a link straight to them.
          <Link
            href={route("?view=documents&status=failed")}
            className="flex items-center gap-2 border-t border-border/60 bg-red-500/5 px-4 py-2 text-xs font-medium text-red-600 hover:bg-red-500/10 dark:text-red-400"
          >
            <CircleAlert className="h-3.5 w-3.5 shrink-0" />
            <span className="min-w-0 flex-1">
              {t("failedNotice", { count: bank.failed_count })}
            </span>
            <ArrowRight className="h-3 w-3 shrink-0" />
          </Link>
        )}
        <div className="border-t border-border/60 p-5">
          <Distribution
            title={t("kindsTitle")}
            items={MEDIA_TYPES.map((kind) => ({
              name: tKinds(kind),
              value: bank.documents_by_media_type[kind] ?? 0,
              color: KIND_HEX[kind],
              href: route(`?view=documents&kind=${kind}`),
            }))}
            emptyLabel={t("noDocuments")}
          />
        </div>
      </CardContent>
    </Card>
  );
}

export function KnowledgeBankHome({
  bank,
  onOpenDocument,
}: {
  bank: KnowledgeBank | null;
  onOpenDocument: (documentId: string, passage?: number) => void;
}) {
  const t = useTranslations("knowledgeBanks.home");
  const [docs, setDocs] = React.useState<KnowledgeDocument[] | null>(null);

  const bankId = bank?.bank_id;
  // Refetch when the bank's counts move (an upload landed or finished).
  const version = bank ? `${bank.document_count}:${bank.pending_count}:${bank.failed_count}` : "";
  React.useEffect(() => {
    if (!bankId) return;
    let live = true;
    // One read feeds both cards: the newest few are "Recently updated", all of them the constellation.
    client
      .listKnowledgeDocuments(bankId, { limit: CONSTELLATION_DOCS, offset: 0 }, { quiet: true })
      .then((res) => live && setDocs(res.documents))
      .catch(() => live && setDocs([]));
    return () => {
      live = false;
    };
  }, [bankId, version]);

  if (!bank) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }

  const route = (query: string) => knowledgeBankRoute(bank.bank_id, query);
  const documentsHref = route("?view=documents");
  return (
    <div>
      <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
      <p className="text-muted-foreground mb-6">
        {t("subtitle")} <span className="font-mono text-foreground">{bank.bank_id}</span>
      </p>

      <KnowledgeBankStats bank={bank} />

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Same card as the memory bank home's constellation. */}
        <div className="flex flex-col overflow-hidden rounded-[16px] border border-border bg-card lg:col-span-2">
          <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 py-3">
            <div className="flex min-w-0 items-center gap-2">
              <Network className="h-4 w-4 shrink-0 text-muted-foreground" />
              <h2 className="truncate text-sm font-semibold text-foreground">
                {t("constellationTitle")}
              </h2>
            </div>
            <ViewAll href={route("?view=documents&layout=constellation")} label={t("viewAll")} />
          </div>
          {docs === null ? (
            <div
              className="flex items-center justify-center"
              style={{ height: CONSTELLATION_HEIGHT }}
            >
              <Spinner size="lg" variant="jump" />
            </div>
          ) : (
            <KnowledgeConstellation
              bankId={bank.bank_id}
              docs={docs}
              height={CONSTELLATION_HEIGHT}
              onOpenDocument={onOpenDocument}
            />
          )}
        </div>

        <div className="flex min-h-0 flex-col overflow-hidden rounded-[16px] border border-border bg-card">
          <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 py-3">
            <h2 className="truncate text-sm font-semibold text-foreground">{t("recentTitle")}</h2>
            <ViewAll href={documentsHref} label={t("viewAll")} />
          </div>
          {docs === null ? (
            <div className="flex justify-center py-6">
              <Spinner size="sm" />
            </div>
          ) : docs.length === 0 ? (
            <div className="px-4 py-6 text-sm text-muted-foreground">{t("noDocuments")}</div>
          ) : (
            <ul className="py-2">
              {docs.slice(0, RECENT_LIST).map((doc) => (
                <li key={doc.document_id}>
                  <Link
                    href={route(`?view=home&doc=${encodeURIComponent(doc.document_id)}`)}
                    className="flex w-full items-center justify-between gap-2 px-4 py-2 text-left hover:bg-muted"
                  >
                    <span className="inline-flex min-w-0 items-center gap-2">
                      <KindIcon kind={doc.media_type} />
                      <span className="truncate text-sm" title={doc.document_id}>
                        {documentLabel(doc)}
                      </span>
                    </span>
                    {doc.status !== "ready" ? (
                      <StatusMark status={doc.status} error={doc.error} />
                    ) : (
                      doc.updated_at && (
                        <span
                          className="flex-shrink-0 text-[11px] text-muted-foreground"
                          title={formatAbsoluteDateTime(doc.updated_at)}
                        >
                          {formatRelativeTime(doc.updated_at)}
                        </span>
                      )
                    )}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="mt-4">
        <ActivityChart bankId={bank.bank_id} version={version} />
      </div>
    </div>
  );
}
