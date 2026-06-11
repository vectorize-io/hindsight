"use client";

import { useTranslations } from "next-intl";
import { Layers } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";

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

const ALL_VALUE = "__all__";

/** Encode a scope's tag set into a stable Select value (JSON of the tags). */
function scopeValue(tags: string[]): string {
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
 * Dropdown that enumerates every distinct observation scope (exact tag set) in
 * the bank and lets the user filter observations down to one scope. The empty
 * tag set is the global/untagged scope; "All scopes" clears the filter.
 */
export function ObservationScopeFilter({ scopes, value, onChange }: ObservationScopeFilterProps) {
  const t = useTranslations("dataView");

  const selectValue = value === null ? ALL_VALUE : scopeValue(value);
  const selectedCount =
    value === null ? null : scopes.find((s) => scopeValue(s.tags) === selectValue)?.count;

  const handleChange = (next: string) => {
    if (next === ALL_VALUE) {
      onChange(null);
      return;
    }
    onChange(JSON.parse(next) as string[]);
  };

  // The trigger renders a compact, single-line summary (not the wrapping pills
  // used in the list) so a multi-tag or long-tag scope truncates with an
  // ellipsis instead of spilling out of the fixed-height control. We render it
  // ourselves rather than via SelectValue, which would otherwise clone the
  // selected item's wrapping pill layout into the trigger and overflow it.
  const renderTriggerLabel = () => {
    if (value === null) {
      return <span className="truncate text-muted-foreground">{t("scopeAll")}</span>;
    }
    if (value.length === 0) {
      return <span className="truncate italic text-muted-foreground">{t("scopeGlobal")}</span>;
    }
    return <span className="truncate text-primary">{value.map((tag) => `#${tag}`).join(" ")}</span>;
  };

  return (
    <Select value={selectValue} onValueChange={handleChange}>
      <SelectTrigger className="h-9 w-64" aria-label={t("scopeLabel")}>
        <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden">
          <Layers className="h-4 w-4 shrink-0 text-muted-foreground" />
          {renderTriggerLabel()}
          {selectedCount != null && (
            <span className="shrink-0 text-xs text-muted-foreground">({selectedCount})</span>
          )}
        </div>
      </SelectTrigger>
      <SelectContent className="max-h-[min(60vh,var(--radix-select-content-available-height))] max-w-[22rem] overflow-y-auto">
        <SelectItem value={ALL_VALUE}>{t("scopeAll")}</SelectItem>
        {scopes.map((scope) => (
          <SelectItem key={scopeValue(scope.tags)} value={scopeValue(scope.tags)}>
            <span className="inline-flex items-center gap-2">
              <ScopeTags tags={scope.tags} globalLabel={t("scopeGlobal")} />
              <span className="text-xs text-muted-foreground">({scope.count})</span>
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
