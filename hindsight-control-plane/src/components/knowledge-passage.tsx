"use client";

// The pieces every knowledge-bank screen shows: a document's kind and status, one
// passage rendered by its modality (page image, image, audio window, video window,
// text), and a search hit. The search panel and the memory-bank recall view both
// render hits, so the card lives here once.

import * as React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import {
  ChevronDown,
  ChevronUp,
  FileAudio,
  FileImage,
  FileText,
  FileType,
  FileVideo,
  type LucideIcon,
} from "lucide-react";
import { ToneChip, TagChip, type ChipTone } from "@/components/ui/facet-chip";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  documentLabel,
  isPicturePassage,
  splitContext,
  formatWindow,
  knowledgeBankRoute,
  proxiedMediaUrl,
  type MediaType,
  type DocumentStatus,
  type KnowledgePassage,
  type KnowledgeSearchResult,
} from "@/lib/knowledge-banks";

const KIND_ICON: Record<MediaType, LucideIcon> = {
  document: FileText,
  image: FileImage,
  audio: FileAudio,
  video: FileVideo,
  text: FileType,
};

const KIND_COLOR: Record<MediaType, string> = {
  document: "text-red-500",
  image: "text-violet-500",
  audio: "text-amber-500",
  video: "text-blue-500",
  text: "text-emerald-500",
};

export function KindIcon({ kind, className }: { kind: MediaType; className?: string }) {
  const t = useTranslations("knowledgeBanks.kinds");
  const Icon = KIND_ICON[kind] ?? FileText;
  return (
    <Icon
      className={cn("h-4 w-4 shrink-0", KIND_COLOR[kind], className)}
      aria-label={t(kind)}
      role="img"
    />
  );
}

const STATUS_TONE: Record<DocumentStatus, ChipTone> = {
  pending: "amber",
  processing: "blue",
  ready: "emerald",
  failed: "red",
};

export function StatusChip({ status, title }: { status: DocumentStatus; title?: string }) {
  const t = useTranslations("knowledgeBanks.statuses");
  return (
    <ToneChip
      tone={STATUS_TONE[status]}
      label={t(status)}
      title={title}
      icon={status === "processing" ? <Spinner size="xs" /> : undefined}
    />
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

function ScoreBar({ score }: { score: number }) {
  const t = useTranslations("knowledgeBanks.search");
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div className="flex items-center gap-2" title={t("scoreHelp")}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-primary/70" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs tabular-nums text-muted-foreground">
        {score.toFixed(3)}
      </span>
    </div>
  );
}

/**
 * A text hit: the passage that matched, and on demand the context around it (its
 * neighbouring passages, merged by the server) with the hit highlighted inside it.
 */
function TextHit({ result }: { result: KnowledgeSearchResult }) {
  const t = useTranslations("knowledgeBanks.search");
  const [expanded, setExpanded] = React.useState(false);
  const parts = splitContext(result.context, result.text);
  const boxRef = React.useRef<HTMLParagraphElement>(null);
  const markRef = React.useRef<HTMLElement>(null);

  // Open the context on the hit, not on the passages before it. Only the box
  // scrolls: scrollIntoView would move the page too.
  React.useEffect(() => {
    const box = boxRef.current;
    const mark = markRef.current;
    if (expanded && box && mark) box.scrollTop = mark.offsetTop - 8; // box is the mark's offsetParent
  }, [expanded]);
  const hasMore = !!result.context && result.context !== result.text;
  return (
    <div className="border-b border-border bg-muted/20 p-2.5 text-xs">
      <p
        ref={boxRef}
        className={cn(
          "relative whitespace-pre-wrap break-words leading-relaxed text-foreground",
          expanded ? "max-h-72 overflow-y-auto pr-1" : "line-clamp-5"
        )}
      >
        {expanded && parts ? (
          <>
            <span className="text-muted-foreground">{parts.before}</span>
            <mark
              ref={markRef}
              className="rounded-sm bg-amber-400/40 text-foreground dark:bg-amber-400/30"
            >
              {parts.hit}
            </mark>
            <span className="text-muted-foreground">{parts.after}</span>
          </>
        ) : expanded ? (
          result.context
        ) : (
          result.text
        )}
      </p>
      {hasMore && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
          aria-expanded={expanded}
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? t("hideContext") : t("showContext")}
        </button>
      )}
    </div>
  );
}

export function SearchResultCard({
  result,
  bankId,
  rank,
  view = "documents",
}: {
  result: KnowledgeSearchResult;
  bankId: string;
  rank?: number;
  /** The bank page section the document dialog opens over (search keeps its results). */
  view?: "documents" | "search";
}) {
  const t = useTranslations("knowledgeBanks.search");
  const label = documentLabel(result);
  const href = knowledgeBankRoute(
    result.bank_id ?? bankId,
    `?view=${view}&doc=${encodeURIComponent(result.document_id)}&passage=${result.index}`
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
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="tabular-nums">
            <PassageLocation passage={result} />
            {result.bank_id && (
              <span className="ml-2 font-mono text-[11px] text-muted-foreground/70">
                {result.bank_id}
              </span>
            )}
          </span>
          <ScoreBar score={result.score} />
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
