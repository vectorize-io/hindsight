"use client";

import { useTranslations } from "next-intl";
import { FileText } from "lucide-react";
import { formatAbsoluteDateTime } from "@/lib/relative-time";

export interface KnowledgeSearchHit {
  id: string;
  name: string;
  snippet: string;
  score: number;
  updated_at?: string | null;
}

interface Props {
  hit: KnowledgeSearchHit;
  /** 1-based position in the ranking. */
  rank: number;
  /** Folder path of the page, bank root first. */
  path: string;
  selected?: boolean;
  onClick: () => void;
}

/**
 * One knowledge-search hit. Shared by the sidebar's live search and the advanced
 * dialog so the two never drift into showing different things about the same
 * result — the sidebar used to omit the rank, score and timestamp.
 */
export function KnowledgeSearchResult({ hit, rank, path, selected, onClick }: Props) {
  const t = useTranslations("knowledgeBase");
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 border-l-2 transition-colors ${
        selected ? "bg-primary/10 border-primary" : "border-transparent hover:bg-muted"
      }`}
    >
      <span className="block truncate pl-7 text-[11px] text-muted-foreground/70">{path}</span>
      <span className="flex items-center gap-1.5">
        <span className="w-5 text-xs tabular-nums text-muted-foreground">{rank}</span>
        <FileText className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
        <span className="text-sm truncate">{hit.name}</span>
        <span
          className="ml-auto pl-2 text-xs tabular-nums text-muted-foreground"
          title={String(hit.score)}
        >
          {Number(hit.score.toPrecision(4))}
        </span>
      </span>
      {hit.snippet && (
        <span className="mt-1 block pl-7 text-xs text-muted-foreground/80 line-clamp-3">
          {hit.snippet}
        </span>
      )}
      {hit.updated_at && (
        <span className="mt-1 block pl-7 text-[11px] text-muted-foreground/70">
          {t("updatedLabel")} {formatAbsoluteDateTime(hit.updated_at)}
        </span>
      )}
    </button>
  );
}
