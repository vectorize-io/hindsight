"use client";

// A knowledge bank's home, built from the memory bank home's own pieces: the stat
// strip and distribution of its "Memory store" card, and its activity chart's style.

import * as React from "react";
import { useTranslations } from "next-intl";
import { FolderOpen, HardDrive, Layers } from "lucide-react";
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
  formatBytes,
  type ActivityBucket,
  type ActivityPeriod,
  type KnowledgeBank,
  type KnowledgeConfig,
  type KnowledgeDocument,
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
import { KindIcon, StatusChip } from "@/components/knowledge-passage";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";

// Kind hues, matching the kind icons.
const KIND_COLORS: Record<string, string> = {
  document: "#ef4444",
  image: "#8b5cf6",
  audio: "#f59e0b",
  video: "#3b82f6",
  text: "#10b981",
};

const SERIES = [
  { key: "documents", color: CHART_COLORS.semantic },
  { key: "passages", color: CHART_COLORS.temporal },
] as const;
type SeriesKey = (typeof SERIES)[number]["key"];

const RECENT_LIST = 6;

function ActivityChart({ bankId, version }: { bankId: string; version: string }) {
  const t = useTranslations("knowledgeBanks.home");
  const [period, setPeriod] = React.useState<ActivityPeriod>("7d");
  const [series, setSeries] = React.useState<ActivityBucket[] | null>(null);

  // `version` changes when the bank's counts move, so a finished upload shows up.
  React.useEffect(() => {
    let live = true;
    client
      .getKnowledgeTimeseries(bankId, period)
      .then((res) => live && setSeries(res.buckets))
      .catch(() => live && setSeries([]));
    return () => {
      live = false;
    };
  }, [bankId, period, version]);

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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
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
          <span className="text-xs text-muted-foreground tabular-nums">
            {t("addedInPeriod", { count: totalDocs })}
          </span>
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

export function KnowledgeBankHome({
  bank,
  config,
  onOpenDocument,
}: {
  bank: KnowledgeBank | null;
  config: KnowledgeConfig | null;
  onOpenDocument: (documentId: string) => void;
}) {
  const t = useTranslations("knowledgeBanks.home");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const [docs, setDocs] = React.useState<KnowledgeDocument[] | null>(null);

  const bankId = bank?.bank_id;
  // Refetch when the bank's counts move (an upload landed or finished).
  const version = bank ? `${bank.document_count}:${bank.pending_count}:${bank.failed_count}` : "";
  React.useEffect(() => {
    if (!bankId) return;
    let live = true;
    client
      .listKnowledgeDocuments(bankId, { limit: RECENT_LIST, offset: 0 }, { quiet: true })
      .then((res) => {
        if (!live) return;
        setDocs(res.documents);
      })
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

  const ingestNote = (bank.pending_count > 0 || bank.failed_count > 0) && (
    <p className="mt-0.5 text-[11px] text-muted-foreground">
      {[
        bank.pending_count > 0 && t("pending", { count: bank.pending_count }),
        bank.failed_count > 0 && t("failed", { count: bank.failed_count }),
      ]
        .filter(Boolean)
        .join(" · ")}
    </p>
  );

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
      <p className="text-muted-foreground mb-6">
        {t("subtitle")} <span className="font-mono text-foreground">{bank.bank_id}</span>
        {config && (
          <span className="ml-2 text-sm">
            · {t("embedder", { model: config.model, dimension: config.dimension })}
          </span>
        )}
      </p>

      <Card>
        <CardContent className="p-0">
          <div className="grid grid-cols-1 md:grid-cols-3 md:divide-x divide-y md:divide-y-0 divide-border/60">
            <InlineStat
              icon={FolderOpen}
              label={t("documents")}
              value={bank.document_count}
              note={ingestNote}
            />
            <InlineStat icon={Layers} label={t("passages")} value={bank.passage_count} />
            <InlineStat
              icon={HardDrive}
              label={t("size")}
              value={bank.total_bytes}
              display={formatBytes(bank.total_bytes)}
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 md:divide-x divide-y md:divide-y-0 divide-border/60 border-t border-border/60">
            <div className="p-5">
              <Distribution
                title={t("kindsTitle")}
                items={MEDIA_TYPES.map((kind) => ({
                  name: tKinds(kind),
                  value: bank.documents_by_media_type[kind] ?? 0,
                  color: KIND_COLORS[kind],
                }))}
                emptyLabel={t("noDocuments")}
              />
            </div>
            <div className="p-5 space-y-3">
              <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">
                {t("recentTitle")}
              </h4>
              {docs === null ? (
                <Spinner size="sm" />
              ) : docs.length === 0 ? (
                <div className="text-xs text-muted-foreground py-2">{t("noDocuments")}</div>
              ) : (
                <ul className="-mx-2">
                  {docs.map((doc) => (
                    <li key={doc.document_id}>
                      <button
                        onClick={() => onOpenDocument(doc.document_id)}
                        className="w-full flex items-center justify-between gap-2 px-2 py-1 rounded hover:bg-muted text-left"
                      >
                        <span className="inline-flex items-center gap-1.5 min-w-0">
                          <KindIcon kind={doc.media_type} className="h-3.5 w-3.5" />
                          <span className="text-xs font-mono truncate" title={doc.document_id}>
                            {doc.document_id}
                          </span>
                        </span>
                        {doc.status !== "ready" ? (
                          <StatusChip status={doc.status} />
                        ) : (
                          doc.updated_at && (
                            <span
                              className="text-[11px] text-muted-foreground flex-shrink-0"
                              title={formatAbsoluteDateTime(doc.updated_at)}
                            >
                              {formatRelativeTime(doc.updated_at)}
                            </span>
                          )
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="mt-4">
        <ActivityChart bankId={bank.bank_id} version={version} />
      </div>
    </div>
  );
}
