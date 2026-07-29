"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Input } from "@/components/ui/input";
import { Tag } from "lucide-react";
import { client } from "@/lib/api";
import { TagChip } from "@/components/ui/facet-chip";

type FetchSuggestions = (q: string) => Promise<string[]>;

interface TagFilterInputProps {
  value: string[];
  onChange: (tags: string[]) => void;
  bankId?: string | null;
  fetchSuggestions?: FetchSuggestions;
  placeholder?: string;
  className?: string;
  matchMode?: "any" | "all";
  onMatchModeChange?: (mode: "any" | "all") => void;
  showMatchToggleAt?: number;
}

const DEFAULT_SHOW_MATCH_TOGGLE_AT = 2;

export function TagFilterInput({
  value,
  onChange,
  bankId,
  fetchSuggestions,
  placeholder,
  className,
  matchMode,
  onMatchModeChange,
  showMatchToggleAt = DEFAULT_SHOW_MATCH_TOGGLE_AT,
}: TagFilterInputProps) {
  const t = useTranslations("common");
  const resolvedPlaceholder = placeholder ?? t("filterByTagPlaceholder");
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Default suggestion source uses the memory-units `list_tags` endpoint when a
  // bankId is given. Memoize so the effect below doesn't refire on every render.
  const defaultFetcher = useMemo<FetchSuggestions | undefined>(() => {
    if (!bankId) return undefined;
    return async (q: string) => {
      const pattern = q ? `${q}*` : undefined;
      const res = await client.listTags(bankId, pattern, 20);
      return res.items.map((i) => i.tag);
    };
  }, [bankId]);

  // Caller-supplied fetchSuggestions is typically defined inline (new identity per
  // render), which would refire the debounce effect after each fetch and create an
  // infinite suggestion-fetch loop. Hold it via a ref so the effect's dep list
  // only tracks input/value — the latest closure is used at fire time.
  const fetcherRef = useRef<FetchSuggestions | undefined>(undefined);
  fetcherRef.current = fetchSuggestions ?? defaultFetcher;

  // Debounced fetch of suggestions when typing
  useEffect(() => {
    const fetcher = fetcherRef.current;
    if (!fetcher) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const results = await fetcher(input.trim());
        if (cancelled) return;
        const filtered = results.filter((tag) => !value.includes(tag));
        setSuggestions(filtered);
        setActiveIndex(filtered.length > 0 ? 0 : -1);
      } catch {
        if (!cancelled) setSuggestions([]);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [input, value]);

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const addTag = (tag: string) => {
    const trimmed = tag.trim();
    if (!trimmed || value.includes(trimmed)) {
      setInput("");
      return;
    }
    onChange([...value, trimmed]);
    setInput("");
    setActiveIndex(-1);
  };

  const removeTag = (tag: string) => {
    onChange(value.filter((existing) => existing !== tag));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" && suggestions.length > 0) {
      e.preventDefault();
      setOpen(true);
      setActiveIndex((i) => (i + 1) % suggestions.length);
      return;
    }
    if (e.key === "ArrowUp" && suggestions.length > 0) {
      e.preventDefault();
      setOpen(true);
      setActiveIndex((i) => (i <= 0 ? suggestions.length - 1 : i - 1));
      return;
    }
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      if (open && activeIndex >= 0 && suggestions[activeIndex]) {
        addTag(suggestions[activeIndex]);
      } else if (input.trim()) {
        addTag(input);
      }
      return;
    }
    if (e.key === "Escape") {
      setOpen(false);
      return;
    }
    if (e.key === "Backspace" && !input && value.length > 0) {
      removeTag(value[value.length - 1]);
    }
  };

  const showMatchToggle =
    matchMode != null && onMatchModeChange != null && value.length >= showMatchToggleAt;

  // Two rows, not one. The applied-filter chips used to sit inline with the
  // input and the match toggle, so past two or three tags they wrapped and
  // shoved the toggle (which was `ml-auto`) around, pushing the whole toolbar
  // out of alignment. Giving the chips their own row below the controls keeps
  // the layout identical at 0 tags and at 10.
  //
  // `flex-1 min-w-0` is baked in rather than left to each caller. Without it
  // this block's flex-basis is `auto`, i.e. the max-content width of its widest
  // child — which is now the chips row — so a single long tag made the block
  // demand hundreds of pixels and collapsed the sibling search input to nothing.
  // Callers can still override via className.
  return (
    <div className={`flex min-w-0 flex-1 flex-col gap-2 ${className ?? ""}`}>
      <div className="flex items-center gap-2">
        <div ref={containerRef} className="relative w-56">
          <Tag className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={handleKeyDown}
            placeholder={resolvedPlaceholder}
            className="pl-8 h-9"
          />
          {open && suggestions.length > 0 && (
            <div className="absolute z-20 mt-1 w-full bg-popover border border-border rounded-md shadow-md max-h-60 overflow-y-auto">
              {suggestions.map((tag, idx) => (
                <button
                  key={tag}
                  type="button"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    addTag(tag);
                  }}
                  onMouseEnter={() => setActiveIndex(idx)}
                  className={`w-full text-left px-2 py-1.5 flex items-center gap-2 ${
                    idx === activeIndex ? "bg-accent" : "hover:bg-muted"
                  }`}
                >
                  {/* Suggestions are tags too — show them as the chip they will
                    become once picked, rather than as plain text. */}
                  <TagChip tag={tag} truncate className="max-w-full" />
                </button>
              ))}
            </div>
          )}
        </div>

        {showMatchToggle && <MatchToggle matchMode={matchMode!} onChange={onMatchModeChange!} />}
      </div>

      {value.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {/* An applied filter is a selected tag, so it uses the same chip the
              rows do — in its `active` state, since that is exactly what it
              represents.

              Truncated: tags like `gitlog-head:<40-char sha>` are wider than
              the toolbar, and a chip that cannot shrink overflows the row no
              matter what the container allows. Full value stays in the title. */}
          {value.map((tag) => (
            <TagChip
              key={tag}
              tag={tag}
              active
              truncate
              className="max-w-[240px]"
              title={tag}
              onRemove={() => removeTag(tag)}
              removeLabel={t("removeTag", { tag })}
            />
          ))}
          <button
            type="button"
            onClick={() => onChange([])}
            className="text-xs text-muted-foreground hover:text-foreground underline"
          >
            {t("clear")}
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * any / all segmented control, shown once enough tags are applied to matter.
 *
 * Calls useTranslations itself rather than taking `t` as a prop: the
 * used-keys test resolves `t("…")` back to a catalog entry by following the
 * enclosing `useTranslations("ns")` binding, and a `t` arriving as a parameter
 * is unresolvable, so these keys would silently stop being checked.
 */
function MatchToggle({
  matchMode,
  onChange,
}: {
  matchMode: "any" | "all";
  onChange: (mode: "any" | "all") => void;
}) {
  const t = useTranslations("common");
  return (
    <div className="flex items-center gap-1 bg-muted rounded-md p-0.5 h-9 shrink-0">
      {(["any", "all"] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode)}
          className={`px-2 py-1 rounded text-xs font-medium ${
            matchMode === mode ? "bg-background text-foreground shadow-sm" : "text-muted-foreground"
          }`}
          title={t(mode === "any" ? "matchAnyTooltip" : "matchAllTooltip")}
        >
          {t(mode === "any" ? "matchAny" : "matchAll")}
        </button>
      ))}
    </div>
  );
}
