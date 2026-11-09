"use client";

import { useTranslations } from "next-intl";
import type { PageTagFilter, TagsMatch } from "@/lib/api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TagFilterInput } from "./tag-filter-input";

interface Props {
  bankId: string;
  value: PageTagFilter;
  onChange: (filter: PageTagFilter) => void;
}

/**
 * Tag filter for the page tree and search: tags + tags_match, with recall's
 * semantics. Compound `tag_groups` are API-only.
 */
export function PageTagFilterPanel({ bankId, value, onChange }: Props) {
  const tm = useTranslations("thinkView");
  return (
    <div className="flex gap-2 items-start" data-testid="kb-tag-filter">
      <TagFilterInput
        className="flex-none"
        value={value.tags ?? []}
        onChange={(tags) => onChange({ ...value, tags })}
        bankId={bankId}
        showMatchToggleAt={Infinity}
      />
      <Select
        value={value.tags_match ?? "any"}
        onValueChange={(v) => onChange({ ...value, tags_match: v as TagsMatch })}
      >
        <SelectTrigger className="h-9 w-40 shrink-0" aria-label={tm("tagsMatchLabel")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="any">{tm("tagsMatchAny")}</SelectItem>
          <SelectItem value="all">{tm("tagsMatchAll")}</SelectItem>
          <SelectItem value="any_strict">{tm("tagsMatchAnyStrict")}</SelectItem>
          <SelectItem value="all_strict">{tm("tagsMatchAllStrict")}</SelectItem>
          <SelectItem value="exact">{tm("tagsMatchExact")}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
