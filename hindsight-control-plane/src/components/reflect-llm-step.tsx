"use client";

import { useTranslations } from "next-intl";
import { Brain, Check, Clock, X } from "lucide-react";
import type { LLMStep } from "@/lib/reflect-timeline";

const TONE_CLASS: Record<LLMStep["tone"], string> = {
  answer: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  error: "bg-red-500/15 text-red-600 dark:text-red-400",
  check: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  turn: "bg-primary/10 text-primary",
};

/** One LLM call of the reflect loop, as a timeline row (used by reflect and page diagnostics). */
export function ReflectLLMStep({
  step,
  durationMs,
  offsetMs,
}: {
  step: LLMStep;
  durationMs: number;
  offsetMs: number;
}) {
  const t = useTranslations("thinkView");
  return (
    <div className="flex items-start gap-3 pb-3">
      <div
        className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${TONE_CLASS[step.tone]}`}
      >
        {step.tone === "answer" ? (
          <Check className="w-3.5 h-3.5" strokeWidth={2.5} />
        ) : step.tone === "check" ? (
          <Brain className="w-3.5 h-3.5" />
        ) : step.tone === "error" ? (
          <X className="w-3.5 h-3.5" strokeWidth={2.5} />
        ) : (
          <span className="text-[10px] font-semibold">{step.turn ?? "·"}</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium text-sm">{t(step.title, step.values)}</span>
          <span className="text-xs text-muted-foreground flex items-center gap-1 shrink-0">
            <Clock className="w-3 h-3" />
            {durationMs}ms
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          {t(step.detail, step.values)}
          <span className="font-mono"> · +{offsetMs}ms</span>
        </span>
      </div>
    </div>
  );
}
