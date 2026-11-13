"use client";

// The pieces every knowledge-bank screen shows: a document's kind and status, one
// passage rendered by its modality (page image, image, audio window, video window,
// text), and a search hit.

import * as React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import {
  AudioLines,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Clapperboard,
  FileText,
  Image as ImageIcon,
  Type,
  type LucideIcon,
} from "lucide-react";
import { ToneChip, TagChip } from "@/components/ui/facet-chip";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  documentLabel,
  isPicturePassage,
  formatWindow,
  isInFlight,
  knowledgeBankRoute,
  proxiedMediaUrl,
  type MediaType,
  type DocumentStatus,
  type KnowledgePassage,
  type KnowledgeSearchResult,
} from "@/lib/knowledge-banks";

// Each kind gets its own shape AND colour, the same everywhere (lists, cards, filters,
// the home chart and constellation), so people learn them.
const KIND_ICON: Record<MediaType, LucideIcon> = {
  document: FileText,
  image: ImageIcon,
  audio: AudioLines,
  video: Clapperboard,
  text: Type,
};

/** Kind hues for charts and graphs; the icons use the same ones. */
export const KIND_HEX: Record<MediaType, string> = {
  document: "#ef4444",
  image: "#8b5cf6",
  audio: "#f59e0b",
  video: "#3b82f6",
  text: "#10b981",
};

export function KindIcon({ kind, className }: { kind: MediaType; className?: string }) {
  const t = useTranslations("knowledgeBanks.kinds");
  const Icon = KIND_ICON[kind] ?? FileText;
  return (
    <Icon
      className={cn("h-4 w-4 shrink-0", className)}
      style={{ color: KIND_HEX[kind] }}
      aria-label={t(kind)}
      role="img"
    />
  );
}

/**
 * A document's state, shown only when it needs attention: a spinner while it ingests,
 * a red "Failed" (its reason on hover) when it failed, and nothing once it is ready.
 */
export function StatusMark({ status, error }: { status: DocumentStatus; error?: string | null }) {
  const t = useTranslations("knowledgeBanks.statuses");
  if (status === "failed") {
    return (
      <ToneChip
        tone="red"
        label={t("failed")}
        title={error ?? undefined}
        icon={<CircleAlert className="h-3 w-3" />}
      />
    );
  }
  if (!isInFlight(status)) return null;
  return (
    <span className="inline-flex shrink-0" title={t(status)} data-testid="status-spinner">
      <Spinner size="xs" />
      <span className="sr-only">{t(status)}</span>
    </span>
  );
}

/** Where a passage sits in its document: "Page 3", "1:00 – 2:00", or "Passage 4". */
export function PassageLocation({ passage }: { passage: KnowledgePassage }) {
  const t = useTranslations("knowledgeBanks.passage");
  const window = formatWindow(passage);
  if (window) return <>{window}</>;
  if (passage.page != null) return <>{t("page", { page: passage.page })}</>;
  return <>{t("number", { number: passage.index + 1 })}</>;
}

/**
 * One passage, by modality. Audio and video media are the window's own clip (the
 * server cuts each window out of the file), so a player starts at the window's
 * start without seeking; the window's offset in the file is shown as its label.
 */
export function PassageMedia({
  passage,
  compact = false,
  alt,
}: {
  passage: KnowledgePassage;
  /** Card-sized: a fixed-height thumbnail and a clamped text snippet. */
  compact?: boolean;
  alt: string;
}) {
  const src = passage.passage_url ? proxiedMediaUrl(passage.passage_url) : null;
  if (src && isPicturePassage(passage)) {
    return (
      // A raw <img>: the media is served by the control-plane proxy, which
      // next/image's optimiser cannot fetch with the user's session.
      <img
        src={src}
        alt={alt}
        loading="lazy"
        className={cn(
          "w-full bg-white object-contain",
          // Card-sized it fills the card's top edge, which carries the frame.
          compact ? "h-32" : "max-h-[80vh] rounded-md border border-border"
        )}
      />
    );
  }
  if (src && passage.media_type === "audio") {
    return <audio controls preload="metadata" src={src} className="w-full" />;
  }
  if (src && passage.media_type === "video") {
    return (
      <video
        controls
        playsInline
        preload="metadata"
        src={src}
        className={cn(
          "rounded-md bg-black",
          compact ? "max-h-32 w-full" : "max-h-[60vh] w-auto max-w-full"
        )}
      />
    );
  }
  return (
    <p
      className={cn(
        "whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground",
        compact && "line-clamp-5 text-xs"
      )}
    >
      {passage.text}
    </p>
  );
}

/**
 * A text hit: the matched passage with the text the search's token budget gave it around
 * it, clamped to a few lines until opened.
 */
function TextHit({ result }: { result: KnowledgeSearchResult }) {
  const t = useTranslations("knowledgeBanks.search");
  const [expanded, setExpanded] = React.useState(false);
  const long = (result.text?.length ?? 0) > LONG_HIT_CHARS;
  return (
    <div className="border-b border-border bg-muted/20 p-2.5 text-xs">
      <p
        className={cn(
          "whitespace-pre-wrap break-words leading-relaxed text-foreground",
          expanded ? "max-h-72 overflow-y-auto pr-1" : "line-clamp-5"
        )}
      >
        {result.text}
      </p>
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
          aria-expanded={expanded}
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? t("showLess") : t("showMore")}
        </button>
      )}
    </div>
  );
}

/** About five clamped lines of a card: a longer hit gets a "show more" toggle. */
const LONG_HIT_CHARS = 280;

export function SearchResultCard({
  result,
  bankId,
  rank,
  score,
}: {
  result: KnowledgeSearchResult;
  bankId: string;
  rank?: number;
  /** The score it was ranked by, from the search's trace: comparable within one search only. */
  score?: number;
}) {
  const t = useTranslations("knowledgeBanks.search");
  const label = documentLabel(result);
  const href = knowledgeBankRoute(
    bankId,
    // The document dialog opens over the search view, which keeps its results.
    `?view=search&doc=${encodeURIComponent(result.document_id)}&passage=${result.index}`
  );
  const playable = result.media_type === "audio" || result.media_type === "video";
  return (
    <article className="group flex min-w-0 flex-col overflow-hidden rounded-lg border border-border bg-card text-sm transition-shadow hover:shadow-md">
      {/* The preview on top: a page or image links through to the passage; a player
          stays interactive, so it is not wrapped in the link. */}
      {playable ? (
        <div className="flex min-h-28 items-center bg-muted/30 p-1.5">
          <PassageMedia passage={result} compact alt={label} />
        </div>
      ) : result.media_type === "text" ? (
        <TextHit result={result} />
      ) : (
        <Link
          href={href}
          className="block border-b border-border bg-white"
          aria-label={t("openPassage")}
        >
          <PassageMedia passage={result} compact alt={label} />
        </Link>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-1.5 p-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <KindIcon kind={result.media_type} />
          <Link
            href={href}
            className="min-w-0 flex-1 truncate text-[13px] font-medium text-card-foreground hover:underline"
            title={label}
          >
            {label}
          </Link>
          {rank != null && (
            <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
              #{rank}
              {score != null && <span className="ml-1.5">{score.toFixed(3)}</span>}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="tabular-nums">
            <PassageLocation passage={result} />
          </span>
        </div>
        {result.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {result.tags.map((tag) => (
              <TagChip key={tag} tag={tag} size="xs" />
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

/** Search hits as a responsive grid of cards. */
export function SearchResultGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
      {children}
    </div>
  );
}
