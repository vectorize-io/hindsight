import { withBasePath } from "@/lib/base-path";
import { cn } from "@/lib/utils";
import { resolveHarnessLogo } from "@/lib/harness-logo";

/**
 * The logo of the coding agent that wrote a document (see lib/harness-logo.ts
 * for the metadata convention it reads).
 *
 * A mark, not a chip: it sits inline next to the thing it qualifies and carries
 * no border or fill of its own, so a table row costs one glyph rather than
 * another `key=value` badge competing with the facet chips.
 *
 * Renders nothing for an unknown harness. Callers keep showing the raw value as
 * an ordinary metadata chip in that case, so a new agent is never invisible —
 * it just isn't branded until its icon is added to the registry.
 */
export function HarnessLogo({
  harness,
  size = 16,
  className,
  titlePrefix,
}: {
  harness: string | null | undefined;
  /** Rendered box in px. The image is contained in it, so non-square marks keep their ratio. */
  size?: number;
  className?: string;
  /**
   * Labels the tooltip, e.g. "Harness" → "Harness: Claude Code". Composed here
   * rather than by the caller because only this component knows the resolved
   * name — a caller building the whole string has to handle the unresolved case
   * it never actually renders.
   */
  titlePrefix?: string;
}) {
  const logo = resolveHarnessLogo(harness);
  if (!logo) return null;

  return (
    // A plain <img>, not next/image: these are static local assets a few KB
    // each, and the loader plus layout wrapper buy nothing at 16px.
    <img
      src={withBasePath(logo.src)}
      alt={logo.label}
      title={titlePrefix ? `${titlePrefix}: ${logo.label}` : logo.label}
      width={size}
      height={size}
      // `object-contain` keeps wordmark-ish icons from being squashed into the
      // square box; `invert` is applied only to the marks flagged monochrome.
      className={cn(
        "shrink-0 rounded-[3px] object-contain",
        logo.invertOnDark && "dark:invert",
        className
      )}
      style={{ width: size, height: size }}
    />
  );
}
