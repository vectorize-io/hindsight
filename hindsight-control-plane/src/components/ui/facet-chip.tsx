import { Braces, Tag as TagIcon, Users, X } from "lucide-react";

/**
 * The one place tags, entities and metadata are turned into chips.
 *
 * Before this existed each view styled its own: tags were blue in the memory
 * dialog, amber in the documents table, purple in directives; entities reused
 * the same blue as tags, so the two were indistinguishable.
 *
 * Colour is NOT how the kinds are told apart. Successive revisions gave each
 * kind its own saturated fill and every one of them read as busy — at this
 * size, several filled chips per row is visual noise, and it duplicated
 * information the text already carries. Kinds are distinguished by form:
 *
 *   tag       #name        pill      (clickable where it filters)
 *   entity    name         pill
 *   metadata  key=value    squared badge
 *
 * plus the header legend on any column that mixes them. Every chip uses the
 * same quiet neutral treatment; the brand-cyan accent is reserved for the one
 * state worth calling out, an active filter, and is an outline rather than a
 * fill. Tokens live in globals.css (`--chip*`) so both themes resolve through
 * the `.dark` block.
 */

export type FacetKind = "tag" | "entity" | "metadata";

export type ChipSize = "xs" | "sm" | "md";

// Sizes are set for legibility first. The earlier 10/12px at py-0.5 was
// cramped enough that the text felt like a label on the chip rather than the
// content of it; a point more type and roughly double the vertical padding
// reads much better at a glance without making rows any taller (the row height
// is set by the two-line document cell, not by the chips).
const SIZE_TEXT: Record<ChipSize, string> = {
  xs: "text-[11px] gap-0.5",
  sm: "text-[13px] gap-1",
  md: "text-sm gap-1",
};
const SIZE_PAD: Record<ChipSize, string> = {
  xs: "px-2 py-[3px]",
  sm: "px-2.5 py-1",
  md: "px-3 py-1.5",
};
const SIZE_CLASS: Record<ChipSize, string> = {
  xs: `${SIZE_TEXT.xs} ${SIZE_PAD.xs}`,
  sm: `${SIZE_TEXT.sm} ${SIZE_PAD.sm}`,
  md: `${SIZE_TEXT.md} ${SIZE_PAD.md}`,
};

// Shape separates the kinds: pills for the name-like facets, a squared badge
// for `key=value` metadata. Combined with the `#` prefix on tags and the header
// legend, that carries the distinction on its own.
const KIND_SHAPE: Record<FacetKind, string> = {
  tag: "rounded-full",
  entity: "rounded-full",
  metadata: "rounded-md",
};

// Tonal variants of one another rather than separate colours. Fill AND border
// both shift per kind: at this size the border is as much of the perceived tone
// as the fill, and varying the fill alone came out below the just-noticeable
// threshold (ΔE < 2 — indistinguishable in practice). See globals.css.
const KIND_TONE: Record<FacetKind, string> = {
  tag: "bg-chip-tag border-chip-tag-border",
  entity: "bg-chip-entity border-chip-entity-border",
  metadata: "bg-chip-meta border-chip-meta-border",
};

// The default. Deliberately quiet: a filled saturated chip at this size, three
// to a row, is what made the tables noisy.
const CHIP_NEUTRAL = "text-chip-fg";

// The exception, spent only on a chip worth calling out (an active filter).
// Outline rather than fill — it reads as emphasis without adding another block
// of colour to the row.
const CHIP_ACCENT = "bg-transparent text-chip-accent border-chip-accent/30";

const CHIP_HOVER = "hover:border-chip-accent/40 hover:text-chip-accent";

const FACET_ICON: Record<FacetKind, typeof TagIcon> = {
  tag: TagIcon,
  entity: Users,
  metadata: Braces,
};

function chipClass(kind: FacetKind, size: ChipSize, active: boolean): string {
  // No max-width here: callers pass their own via className, and two competing
  // max-w utilities resolve by stylesheet order rather than by which was passed
  // last, so a base value would silently beat the caller's.
  return [
    "inline-flex items-center border font-medium leading-[1.2]",
    SIZE_CLASS[size],
    KIND_SHAPE[kind],
    // The accent state carries its own background and border, so the per-kind
    // tone only applies to the neutral one.
    active ? CHIP_ACCENT : `${KIND_TONE[kind]} ${CHIP_NEUTRAL}`,
  ].join(" ");
}

interface BaseChipProps {
  size?: ChipSize;
  className?: string;
  title?: string;
  /** Renders the chip as a button and fires this on click. */
  onClick?: () => void;
  /** Marks the chip as selected (e.g. an active filter). */
  active?: boolean;
  /** Adds a trailing ✕ that fires this. Not combinable with onClick. */
  onRemove?: () => void;
  removeLabel?: string;
  /** Truncate long values instead of wrapping. */
  truncate?: boolean;
}

function ChipShell({
  kind,
  size = "sm",
  active = false,
  className,
  title,
  onClick,
  onRemove,
  removeLabel,
  children,
}: BaseChipProps & { kind: FacetKind; children: React.ReactNode }) {
  const cls = `${chipClass(kind, size, active)} ${
    onClick ? `transition-colors cursor-pointer ${active ? "" : CHIP_HOVER}` : ""
  } ${className ?? ""}`;

  // Chips commonly sit inside a clickable table row, where the row opens a
  // detail view. Clicking the chip must not also trigger the row, so the stop
  // lives here rather than being re-implemented at every call site.
  const stop = (e: React.MouseEvent, fn: () => void) => {
    e.stopPropagation();
    fn();
  };

  if (onClick) {
    return (
      <button type="button" title={title} onClick={(e) => stop(e, onClick)} className={cls}>
        {children}
      </button>
    );
  }
  return (
    <span title={title} className={cls}>
      {children}
      {onRemove && (
        <button
          type="button"
          onClick={(e) => stop(e, onRemove)}
          aria-label={removeLabel}
          className="opacity-50 hover:opacity-100 transition-opacity ml-0.5 shrink-0"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </span>
  );
}

// Each chip renders its text as ONE child of the shell. The shell's `gap` is
// there to space the icon / remove-button, and would otherwise open a space
// inside the label itself — "# tag", "key= value".

/** A tag. Always rendered with the `#` marker so it reads as a tag anywhere. */
export function TagChip({ tag, truncate, ...rest }: BaseChipProps & { tag: string }) {
  return (
    <ChipShell kind="tag" {...rest}>
      <span className={truncate ? "truncate" : undefined}>
        <span className="opacity-70 select-none font-mono">#</span>
        {tag}
      </span>
    </ChipShell>
  );
}

/**
 * An identifier the system routes on, such as a bank alias.
 *
 * Shares the entity pill's form and tone rather than getting a kind of its own:
 * it is the same sort of thing visually — a bare name, not a `key=value` — and
 * the kinds here are a closed set distinguished by form, so a fourth tone would
 * be a distinction without a difference. Rendered monospace because, unlike an
 * entity, it is a literal string someone has to type correctly.
 */
export function IdChip({ id, truncate, ...rest }: BaseChipProps & { id: string }) {
  return (
    <ChipShell kind="entity" {...rest}>
      <span className={`font-mono ${truncate ? "truncate" : ""}`}>{id}</span>
    </ChipShell>
  );
}

/** A resolved entity (person, place, thing) linked to a memory. */
export function EntityChip({ entity, truncate, ...rest }: BaseChipProps & { entity: string }) {
  return (
    <ChipShell kind="entity" {...rest}>
      <span className={truncate ? "truncate" : undefined}>{entity}</span>
    </ChipShell>
  );
}

/** One `key=value` pair of free-form document metadata. */
export function MetadataChip({
  entryKey,
  value,
  truncate,
  ...rest
}: BaseChipProps & { entryKey: string; value: string }) {
  return (
    <ChipShell kind="metadata" {...rest}>
      <span className={truncate ? "truncate" : "break-all"}>
        <span className="opacity-70">{entryKey}=</span>
        {value}
      </span>
    </ChipShell>
  );
}

/**
 * Legend for a column that mixes kinds: one labelled specimen per kind, in the
 * exact treatment the rows use. Lets the rows stay bare chips instead of
 * restating the category on every one of them.
 */
export function FacetLegend({ items }: { items: { kind: FacetKind; label: string }[] }) {
  return (
    <div className="flex items-center gap-1.5 normal-case tracking-normal">
      {items.map(({ kind, label }) => {
        const Icon = FACET_ICON[kind];
        return (
          <span key={kind} className={chipClass(kind, "sm", false)}>
            <Icon className="w-3 h-3 shrink-0" />
            {label}
          </span>
        );
      })}
    </div>
  );
}

/**
 * Renders a list of tags. Returns null when empty.
 * Kept as a named export because several views already call it.
 */
export function TagList({
  tags,
  size = "sm",
  showLabel = false,
}: {
  tags: string[];
  size?: ChipSize;
  showLabel?: boolean;
}) {
  if (!tags || tags.length === 0) return null;

  return (
    <div>
      {showLabel && (
        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
          <TagIcon className="w-3 h-3" />
          Tags
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        {tags.map((tag, idx) => (
          <TagChip key={idx} tag={tag} size={size} />
        ))}
      </div>
    </div>
  );
}

/**
 * A coloured chip, for a closed set of categories where colour is the point —
 * telling them apart at a glance down a long list (the kinds of model call in
 * the LLM requests view: LLM, embedding, rerank, decision) — and for a call's
 * status. The facets above stay neutral; this is the exception, kept here so it
 * shares their sizes, squared shape and border instead of being styled per view.
 *
 * `detail` is an optional second segment split off by a hairline, for what
 * qualifies the label: a scope ("Embedding | document") or a count ("LLM | 5").
 */
export type ChipTone = "blue" | "violet" | "amber" | "emerald" | "red";

// A light tint with a matching border, the same treatment as the status pills
// elsewhere in the app; follows the chart palette (--chart-1..4).
const TONE_CLASS: Record<ChipTone, string> = {
  blue: "bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-500/20",
  violet: "bg-violet-500/10 text-violet-700 dark:text-violet-300 border-violet-500/20",
  amber: "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/25",
  emerald: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20",
  red: "bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/20",
};

export function ToneChip({
  tone,
  label,
  detail,
  icon,
  monoDetail = false,
  size = "xs",
  title,
  className,
}: {
  tone: ChipTone;
  label: React.ReactNode;
  detail?: React.ReactNode;
  icon?: React.ReactNode;
  /** Render the detail monospace (a literal identifier, like a scope). */
  monoDetail?: boolean;
  size?: ChipSize;
  title?: string;
  className?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex min-w-0 max-w-full items-stretch overflow-hidden whitespace-nowrap rounded-md border font-sans font-medium leading-[1.2] ${SIZE_TEXT[size]} ${TONE_CLASS[tone]} ${className ?? ""}`}
    >
      <span className={`inline-flex shrink-0 items-center gap-1 ${SIZE_PAD[size]}`}>
        {icon}
        {label}
      </span>
      {detail != null && detail !== "" && (
        <span
          className={`truncate border-l border-inherit bg-background/70 font-normal tabular-nums text-foreground/80 ${SIZE_PAD[size]} ${
            monoDetail ? "font-mono" : ""
          }`}
        >
          {detail}
        </span>
      )}
    </span>
  );
}
