"use client";

// A traced knowledge search, laid out like the Recall view's Trace tab: the parallel candidate
// searches, then each step with its own partial list and scores, down to how the token budget was
// filled. Every step card opens to its list.

import * as React from "react";
import { useTranslations } from "next-intl";
import { ArrowDown, ChevronDown, ChevronRight } from "lucide-react";

import type {
  KnowledgeSearchTrace,
  KnowledgeSearchTraceHit,
  KnowledgeTraceBudgetStep,
} from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { KindIcon } from "@/components/knowledge-passage";
import { Card, CardContent } from "@/components/ui/card";

/** Rows a step shows before "view all", as the Recall trace does. */
const INITIAL_ROWS = 10;

const fmt = (n: number) => n.toFixed(3);

const ARM_STYLE: Record<string, { letter: string; bg: string; text: string }> = {
  vector: { letter: "V", bg: "bg-blue-500/10", text: "text-blue-500" },
  extra_vectors: { letter: "X", bg: "bg-cyan-500/10", text: "text-cyan-500" },
  keyword: { letter: "K", bg: "bg-amber-500/10", text: "text-amber-500" },
};

const ACTION_STYLE: Record<KnowledgeTraceBudgetStep["action"], string> = {
  window: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  alone: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  media: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  merged: "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  widened: "bg-violet-500/15 text-violet-700 dark:text-violet-400",
  skipped: "bg-muted text-muted-foreground",
};

function Divider({ label }: { label: string }) {
  return (
    <div className="mb-3 flex items-center gap-2 text-xs font-medium text-muted-foreground">
      <div className="h-px flex-1 bg-border" />
      <span>{label}</span>
      <div className="h-px flex-1 bg-border" />
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-2">
      <ArrowDown className="h-4 w-4 text-muted-foreground/50" />
    </div>
  );
}

/** A step: a header card that opens to its list, like the Recall trace's steps. */
function Step({
  icon,
  iconClass,
  title,
  description,
  count,
  rows,
  render,
  open,
  onToggle,
}: {
  icon: React.ReactNode;
  iconClass: string;
  title: string;
  description: React.ReactNode;
  count: React.ReactNode;
  rows?: unknown[];
  render?: (row: any, i: number) => React.ReactNode;
  open: boolean;
  onToggle: () => void;
}) {
  const t = useTranslations("knowledgeBanks.trace");
  const [showAll, setShowAll] = React.useState(false);
  const expandable = !!rows && rows.length > 0 && !!render;
  const shown = rows ? (showAll ? rows : rows.slice(0, INITIAL_ROWS)) : [];
  return (
    <div>
      <Card
        className={cn(
          "transition-colors",
          expandable && "cursor-pointer",
          open ? "border-primary" : expandable && "hover:border-primary/50"
        )}
        onClick={expandable ? onToggle : undefined}
      >
        <CardContent className="py-4">
          <div className="flex items-center gap-4">
            <div
              className={cn(
                "flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold",
                iconClass
              )}
            >
              {icon}
            </div>
            <div className="min-w-0 flex-1">
              <div className="font-semibold text-foreground">{title}</div>
              <div className="mt-0.5 text-sm text-muted-foreground">{description}</div>
            </div>
            <div className="text-2xl font-bold text-foreground">{count}</div>
            {expandable &&
              (open ? (
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-5 w-5 text-muted-foreground" />
              ))}
          </div>
        </CardContent>
      </Card>
      {open && expandable && (
        <div className="ml-6 mt-2 max-h-[420px] space-y-2 overflow-y-auto border-l-2 border-muted pl-4">
          {shown.map((row, i) => render!(row, i))}
          {rows!.length > INITIAL_ROWS && (
            <button
              type="button"
              className="w-full rounded py-2 text-xs text-primary hover:bg-muted/50"
              onClick={() => setShowAll((v) => !v)}
            >
              {showAll ? t("showLess") : t("viewAll", { count: rows!.length })}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** One passage row inside a step: its rank there, what it is, and that step's scores. */
function Row({
  n,
  hit,
  children,
}: {
  n: number;
  hit: KnowledgeSearchTraceHit | undefined;
  children: React.ReactNode;
}) {
  const t = useTranslations("knowledgeBanks.trace");
  return (
    <div className="rounded-lg bg-muted/30 p-3">
      <div className="flex items-start gap-3">
        <span className="w-6 shrink-0 font-mono text-xs text-muted-foreground">{n}</span>
        <div className="min-w-0 flex-1">
          {hit ? (
            <>
              <div className="flex min-w-0 items-center gap-1.5">
                <KindIcon kind={hit.media_type} />
                <span className="truncate text-xs font-medium">
                  {hit.filename || hit.document_id}
                </span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {hit.page != null ? t("page", { page: hit.page }) : `#${hit.index}`}
                </span>
              </div>
              {hit.snippet && (
                <p className="mt-0.5 line-clamp-2 text-sm text-foreground">{hit.snippet}</p>
              )}
            </>
          ) : null}
          <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted-foreground">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

export function KnowledgeSearchTraceView({ trace }: { trace: KnowledgeSearchTrace }) {
  const t = useTranslations("knowledgeBanks.trace");
  const [open, setOpen] = React.useState<Set<string>>(new Set());
  const toggle = (key: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const byKey = new Map(trace.hits.map((h) => [`${h.document_id}\u0000${h.index}`, h]));
  const byRank = new Map(trace.hits.map((h) => [h.rank, h]));
  const meaning = [...trace.hits].sort((a, b) => b.with_document - a.with_document);
  const terms = Object.entries(trace.keyword_terms).sort((a, b) => b[1] - a[1]);
  const returned = trace.hits.filter((h) => h.outcome === "returned").length;

  // No space-y here: the arrows between steps are the spacing, as in the Recall trace.
  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {Object.entries(trace.timings_ms).map(([phase, ms]) => (
          <span key={phase}>
            {phase} <span className="font-medium text-foreground">{Math.round(ms)}ms</span>
          </span>
        ))}
      </div>

      <div>
        <Divider label={t("parallelSearches")} />
        <div className="space-y-2">
          {trace.candidate_searches.map((arm) => {
            const style = ARM_STYLE[arm.name] ?? ARM_STYLE.vector;
            return (
              <Step
                key={arm.name}
                icon={style.letter}
                iconClass={cn(style.bg, style.text)}
                title={t(`arm_${arm.name}`)}
                description={t(`arm_${arm.name}_help`)}
                count={arm.results.length}
                rows={arm.results}
                open={open.has(arm.name)}
                onToggle={() => toggle(arm.name)}
                render={(r, i) => (
                  <Row key={i} n={i + 1} hit={byKey.get(`${r.document_id}\u0000${r.index}`)}>
                    <span>
                      {t("armScore")} {fmt(r.score)}
                    </span>
                  </Row>
                )}
              />
            );
          })}
        </div>
        <Arrow />
      </div>

      <Step
        icon="∪"
        iconClass="bg-purple-500/10 text-purple-500"
        title={t("meaningTitle")}
        description={t("meaningHelp", { count: trace.candidates })}
        count={trace.candidates}
        rows={meaning}
        open={open.has("meaning")}
        onToggle={() => toggle("meaning")}
        render={(h: KnowledgeSearchTraceHit, i) => (
          <Row key={h.rank} n={i + 1} hit={h}>
            <span>
              {t("similarity")} {fmt(h.similarity)}
            </span>
            {Object.entries(h.roles).map(([role, v]) => (
              <span key={role}>
                {role} {fmt(v)}
              </span>
            ))}
            <span>
              {t("passageScore")} {fmt(h.passage_score)}
            </span>
            <span className="font-medium text-foreground">
              {t("withDocument")} {fmt(h.with_document)}
            </span>
          </Row>
        )}
      />
      <Arrow />

      <Step
        icon="K"
        iconClass="bg-amber-500/10 text-amber-500"
        title={t("rankingTitle")}
        description={
          <>
            {terms.length === 0
              ? t("noKeywords")
              : t("rankingHelp", { weight: Math.round(trace.keyword_weight * 100) })}
            {terms.length > 0 && (
              <span className="ml-2 inline-flex flex-wrap gap-1 align-middle">
                {terms.map(([word, idf]) => (
                  <span
                    key={word}
                    className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
                    title={t("idfHelp")}
                  >
                    {word} {idf.toFixed(2)}
                  </span>
                ))}
              </span>
            )}
          </>
        }
        count={trace.hits.length}
        rows={trace.hits}
        open={open.has("ranking")}
        onToggle={() => toggle("ranking")}
        render={(h: KnowledgeSearchTraceHit) => (
          <Row key={h.rank} n={h.rank} hit={h}>
            <span>
              {t("meaning")} {fmt(h.with_document)}
            </span>
            <span>
              {t("keyword")} {h.keyword > 0 ? h.keyword.toFixed(2) : "0"}
            </span>
            {h.reference_page && <span className="text-amber-600">{t("referencePage")}</span>}
            <span className="font-medium text-foreground">
              {t("score")} {fmt(h.score)}
            </span>
          </Row>
        )}
      />
      <Arrow />

      <Step
        icon="⌄"
        iconClass="bg-slate-500/10 text-slate-500"
        title={t("depthTitle", { budget: trace.budget })}
        description={t("depthHelp", { depth: trace.depth, total: trace.hits.length })}
        count={trace.depth}
        open={false}
        onToggle={() => {}}
      />
      <Arrow />

      <Step
        icon="Σ"
        iconClass="bg-emerald-500/10 text-emerald-500"
        title={t("budgetTitle")}
        description={
          <span className="flex items-center gap-2">
            <span className="h-1.5 w-32 overflow-hidden rounded-full bg-muted">
              <span
                className="block h-full bg-emerald-500"
                style={{ width: `${Math.min(100, (trace.tokens_used / trace.max_tokens) * 100)}%` }}
              />
            </span>
            {t("budgetHelp", { used: trace.tokens_used, max: trace.max_tokens })}
          </span>
        }
        count={trace.budget_steps.length}
        rows={trace.budget_steps}
        open={open.has("budget")}
        onToggle={() => toggle("budget")}
        render={(s: KnowledgeTraceBudgetStep, i) => (
          <Row key={i} n={i + 1} hit={byRank.get(s.rank)}>
            <span
              className={cn("rounded px-1.5 py-0.5 font-medium", ACTION_STYLE[s.action])}
              title={t(`action_${s.action}_help`)}
            >
              {/* into_rank is a candidate rank: the label names the result that holds it. */}
              {t(`action_${s.action}`, {
                rank: (s.into_rank != null && byRank.get(s.into_rank)?.result_rank) || 0,
              })}
            </span>
            <span>{t("candidateRank", { rank: s.rank })}</span>
            <span>{t("passages", { first: s.first, last: s.last })}</span>
            <span>+{s.tokens}</span>
            <span className="font-medium text-foreground">
              {t("total", { total: s.total, max: trace.max_tokens })}
            </span>
          </Row>
        )}
      />
      <Arrow />

      <Card className="border-primary/40">
        <CardContent className="flex items-center gap-4 py-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
            ✓
          </div>
          <div className="flex-1 font-semibold">{t("resultsTitle")}</div>
          <div className="text-2xl font-bold">{returned}</div>
        </CardContent>
      </Card>
    </div>
  );
}
