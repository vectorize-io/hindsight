"use client";

// A traced knowledge search, as the Recall view's Trace tab shows a memory recall: what the
// search found, how each candidate's score was built, and what the token budget kept.

import * as React from "react";
import { useTranslations } from "next-intl";

import type { KnowledgeSearchTrace, KnowledgeSearchTraceHit } from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { KindIcon } from "@/components/knowledge-passage";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

/** Rows shown before "show all": a mid search ranks up to 200. */
const INITIAL_ROWS = 50;

const OUTCOME_STYLE: Record<KnowledgeSearchTraceHit["outcome"], string> = {
  returned: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  merged: "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  over_budget: "bg-muted text-muted-foreground",
};

const score = (n: number) => n.toFixed(3);

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold">{children}</div>
    </div>
  );
}

export function KnowledgeSearchTraceView({ trace }: { trace: KnowledgeSearchTrace }) {
  const t = useTranslations("knowledgeBanks.trace");
  const [showAll, setShowAll] = React.useState(false);
  const rows = showAll ? trace.hits : trace.hits.slice(0, INITIAL_ROWS);
  const terms = Object.entries(trace.keyword_terms).sort((a, b) => b[1] - a[1]);
  const counts = trace.hits.reduce<Record<string, number>>((acc, h) => {
    acc[h.outcome] = (acc[h.outcome] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("summaryTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label={t("candidates")}>{trace.candidates}</Stat>
            <Stat label={t("ranked")}>
              {trace.hits.length}{" "}
              <span className="font-normal text-muted-foreground">
                {t("outcomes", {
                  returned: counts.returned ?? 0,
                  merged: counts.merged ?? 0,
                  over: counts.over_budget ?? 0,
                })}
              </span>
            </Stat>
            <Stat label={t("tokens")}>
              {trace.tokens_used} / {trace.max_tokens}
            </Stat>
            <Stat label={t("budget")}>{trace.budget}</Stat>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">
              {t("keywords", {
                language: trace.language ?? t("noLanguage"),
                weight: Math.round(trace.keyword_weight * 100),
              })}
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {terms.length === 0 ? (
                <span className="text-sm text-muted-foreground">{t("noKeywords")}</span>
              ) : (
                terms.map(([word, idf]) => (
                  <span
                    key={word}
                    className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs"
                    title={t("idfHelp")}
                  >
                    {word} <span className="text-muted-foreground">{idf.toFixed(2)}</span>
                  </span>
                ))
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {Object.entries(trace.timings_ms).map(([phase, ms]) => (
              <span key={phase}>
                {phase} <span className="font-medium text-foreground">{Math.round(ms)}ms</span>
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("rankingTitle")}</CardTitle>
          <p className="text-xs text-muted-foreground">{t("rankingHelp")}</p>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">#</TableHead>
                <TableHead>{t("passage")}</TableHead>
                <TableHead>{t("outcome")}</TableHead>
                <TableHead className="text-right" title={t("scoreHelp")}>
                  {t("score")}
                </TableHead>
                <TableHead className="text-right" title={t("similarityHelp")}>
                  {t("similarity")}
                </TableHead>
                <TableHead className="text-right" title={t("passageScoreHelp")}>
                  {t("passageScore")}
                </TableHead>
                <TableHead className="text-right" title={t("withDocumentHelp")}>
                  {t("withDocument")}
                </TableHead>
                <TableHead className="text-right" title={t("keywordHelp")}>
                  {t("keyword")}
                </TableHead>
                <TableHead>{t("foundBy")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((h) => (
                <TableRow key={h.rank}>
                  <TableCell className="font-mono text-xs tabular-nums">{h.rank}</TableCell>
                  <TableCell className="max-w-[22rem]">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <KindIcon kind={h.media_type} />
                      <span className="truncate text-xs font-medium" title={h.document_id}>
                        {h.filename || h.document_id}
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {h.page != null ? t("page", { page: h.page }) : `#${h.index}`}
                      </span>
                    </div>
                    {h.snippet && (
                      <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                        {h.snippet}
                      </p>
                    )}
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                        OUTCOME_STYLE[h.outcome]
                      )}
                    >
                      {h.outcome === "returned"
                        ? t("returned", { rank: h.result_rank ?? 0 })
                        : h.outcome === "merged"
                          ? t("merged", { rank: h.merged_into_rank ?? 0 })
                          : t("overBudget")}
                    </span>
                    {h.reference_page && (
                      <span className="ml-1 text-[11px] text-amber-600">{t("referencePage")}</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs font-semibold tabular-nums">
                    {score(h.score)}
                  </TableCell>
                  <TableCell
                    className="text-right font-mono text-xs tabular-nums"
                    title={Object.entries(h.roles)
                      .map(([role, v]) => `${role} ${score(v)}`)
                      .join(" · ")}
                  >
                    {score(h.similarity)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs tabular-nums">
                    {score(h.passage_score)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs tabular-nums">
                    {score(h.with_document)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs tabular-nums">
                    {h.keyword > 0 ? h.keyword.toFixed(2) : "—"}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs text-muted-foreground">
                    {h.found_by.join(", ")}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {trace.hits.length > INITIAL_ROWS && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="mt-3 text-xs font-medium text-primary hover:underline"
            >
              {showAll ? t("showLess") : t("showAll", { count: trace.hits.length })}
            </button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
