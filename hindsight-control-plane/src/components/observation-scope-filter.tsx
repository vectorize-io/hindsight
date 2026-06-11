"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Layers, Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export interface ObservationScope {
  tags: string[];
  count: number;
}

interface ObservationScopeFilterProps {
  scopes: ObservationScope[];
  /** Selected scope: a tag set (possibly empty = global), or null for "all scopes". */
  value: string[] | null;
  onChange: (scope: string[] | null) => void;
}

/** Stable key for a scope's tag set (order-independent, matches the trigger value). */
function scopeKey(tags: string[]): string {
  return JSON.stringify(tags);
}

/** Render a scope's tag set as inline pills, or the "global" label when empty. */
function ScopeTags({ tags, globalLabel }: { tags: string[]; globalLabel: string }) {
  if (tags.length === 0) {
    return <span className="italic text-muted-foreground">{globalLabel}</span>;
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 font-medium leading-none"
        >
          <span className="opacity-50 select-none font-mono">#</span>
          {tag}
        </span>
      ))}
    </span>
  );
}

/**
 * Searchable single-select for observation scopes (exact tag sets). Mirrors the
 * tag filter's type-to-search UX via a Popover + Command combobox: the trigger
 * shows a compact, single-line summary of the selected scope; the dropdown lists
 * every distinct scope with counts and filters as you type. The empty tag set is
 * the global/untagged scope; "All scopes" clears the filter.
 */
export function ObservationScopeFilter({ scopes, value, onChange }: ObservationScopeFilterProps) {
  const t = useTranslations("dataView");
  const [open, setOpen] = useState(false);

  const selectedKey = value === null ? null : scopeKey(value);
  const selectedCount =
    value === null ? null : scopes.find((s) => scopeKey(s.tags) === selectedKey)?.count;

  const select = (scope: string[] | null) => {
    onChange(scope);
    setOpen(false);
  };

  // Compact, single-line trigger summary (the full pills live in the dropdown),
  // so a multi-tag or long-tag scope truncates instead of overflowing.
  const triggerLabel = () => {
    if (value === null) {
      return <span className="truncate text-muted-foreground">{t("scopeAll")}</span>;
    }
    if (value.length === 0) {
      return <span className="truncate italic text-muted-foreground">{t("scopeGlobal")}</span>;
    }
    return <span className="truncate text-primary">{value.map((tag) => `#${tag}`).join(" ")}</span>;
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label={t("scopeLabel")}
          className="h-9 w-64 justify-start gap-1.5 px-3 font-normal"
        >
          <Layers className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
            {triggerLabel()}
            {selectedCount != null && (
              <span className="shrink-0 text-xs text-muted-foreground">({selectedCount})</span>
            )}
          </span>
          <ChevronsUpDown className="ml-auto h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[min(22rem,var(--radix-popover-content-available-width))] p-0"
        align="start"
      >
        <Command
          filter={(value, search, keywords) => {
            // Substring match over the scope's tags (like the tag filter), not
            // cmdk's default fuzzy scoring which over-matches scattered letters
            // across long multi-tag scopes.
            const q = search.trim().toLowerCase();
            if (!q) return 1;
            const haystack = `${value} ${(keywords ?? []).join(" ")}`.toLowerCase();
            return haystack.includes(q) ? 1 : 0;
          }}
        >
          <CommandInput placeholder={t("scopeSearch")} />
          <CommandList className="max-h-[min(60vh,var(--radix-popover-content-available-height))]">
            <CommandEmpty>{t("scopeNoResults")}</CommandEmpty>
            <CommandGroup>
              <CommandItem
                value="__all__"
                keywords={["all", "scopes"]}
                onSelect={() => select(null)}
              >
                <Check
                  className={cn("h-4 w-4 shrink-0", value === null ? "opacity-100" : "opacity-0")}
                />
                {t("scopeAll")}
              </CommandItem>
              {scopes.map((scope) => {
                const key = scopeKey(scope.tags);
                const isSelected = selectedKey === key;
                return (
                  <CommandItem
                    key={key}
                    value={key}
                    keywords={scope.tags.length ? scope.tags : ["global"]}
                    onSelect={() => select(scope.tags)}
                  >
                    <Check
                      className={cn("h-4 w-4 shrink-0", isSelected ? "opacity-100" : "opacity-0")}
                    />
                    <span className="inline-flex flex-1 items-center gap-2 overflow-hidden">
                      <ScopeTags tags={scope.tags} globalLabel={t("scopeGlobal")} />
                    </span>
                    <span className="shrink-0 text-xs text-muted-foreground">({scope.count})</span>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
