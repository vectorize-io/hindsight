"use client";

import { type ReactNode } from "react";
import { ChevronRight, Info, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

/**
 * Building blocks for the control plane's option forms (mental-model triggers,
 * recall, reflect): uppercase section titles, label + compact-control rows, help
 * behind an info icon that opens a small popover on click, and no divider lines.
 * Keep forms reading as labels and controls; explanations are one click away.
 */

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      {children}
    </section>
  );
}

// Click (not hover) opens a small popover, so the help works on touch and can
// hold a sentence or two without a native tooltip's delay and truncation.
export function Hint({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={text}
          className="inline-flex rounded-full text-muted-foreground/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        className="w-72 p-3 text-xs leading-relaxed text-foreground"
      >
        {text}
      </PopoverContent>
    </Popover>
  );
}

// Label on the left, a compact control on the right; stacks when narrow.
export function Row({
  label,
  description,
  htmlFor,
  children,
}: {
  label: string;
  description?: string;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="flex min-w-0 items-center gap-1.5">
        <label htmlFor={htmlFor} className="text-sm font-medium text-foreground">
          {label}
        </label>
        <Hint text={description} />
      </div>
      <div className="shrink-0 sm:w-64">{children}</div>
    </div>
  );
}

export function OptionCards<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  // description is the one-liner on the card; hint the full explanation on hover
  // (a native title: a popover button cannot nest inside the card button).
  options: { value: T; icon: LucideIcon; label: string; description: string; hint: string }[];
}) {
  return (
    <div
      role="radiogroup"
      className={cn("grid gap-2", options.length === 3 ? "sm:grid-cols-3" : "sm:grid-cols-2")}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={cn(
              "flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40",
              active
                ? "border-primary bg-primary/5 ring-1 ring-primary/40"
                : "border-border hover:bg-muted/50"
            )}
          >
            <span className="flex items-center gap-2 text-sm font-medium text-foreground">
              <option.icon
                className={cn("h-4 w-4", active ? "text-primary" : "text-muted-foreground")}
              />
              {option.label}
            </span>
            <span className="text-xs leading-snug text-muted-foreground">{option.description}</span>
          </button>
        );
      })}
    </div>
  );
}

/** A small inline toggle group, e.g. Low / Mid / High. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
  ariaLabel?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className="inline-flex rounded-md bg-muted/50 p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              active
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/** Uppercase disclosure header, styled like a Section title, for collapsible option groups. */
export function DisclosureButton({
  open,
  onToggle,
  label,
  badge,
}: {
  open: boolean;
  onToggle: () => void;
  label: string;
  /** Shown when collapsed, e.g. how many options differ from the defaults. */
  badge?: number;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      className="group inline-flex items-center gap-1.5 text-left"
    >
      <ChevronRight
        className={cn(
          "h-3.5 w-3.5 text-muted-foreground transition-transform",
          open && "rotate-90"
        )}
      />
      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground group-hover:text-foreground">
        {label}
      </span>
      {!open && badge ? (
        <span className="rounded-full bg-primary/15 px-1.5 text-[10px] font-medium text-primary">
          {badge}
        </span>
      ) : null}
    </button>
  );
}
