"use client";

import { useTranslations } from "next-intl";
import { Layers } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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

  const handleChange = (next: string) => {
    if (next === ALL_VALUE) {
      onChange(null);
      return;
    }
    onChange(JSON.parse(next) as string[]);
  };

  return (
    <Select value={selectValue} onValueChange={handleChange}>
      <SelectTrigger className="h-9 w-64" aria-label={t("scopeLabel")}>
        <Layers className="h-4 w-4 text-muted-foreground shrink-0" />
        <SelectValue placeholder={t("scopeLabel")} />
      </SelectTrigger>
      <SelectContent>
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
