"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Input } from "@/components/ui/input";
import { Tag, X } from "lucide-react";
import { client } from "@/lib/api";

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

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className ?? ""}`}>
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
                className={`w-full text-left px-3 py-1.5 text-sm flex items-center gap-2 ${
                  idx === activeIndex
                    ? "bg-accent text-accent-foreground"
                    : "text-foreground hover:bg-muted"
                }`}
              >
                <Tag className="w-3 h-3 text-muted-foreground" />
                <span className="truncate">{tag}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {value.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {value.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-md bg-primary/10 text-primary border border-primary/20 font-medium leading-none"
            >
              <span className="opacity-50 select-none font-mono">#</span>
              {tag}
              <button
                type="button"
                onClick={() => removeTag(tag)}
                className="opacity-50 hover:opacity-100 transition-opacity ml-0.5"
                aria-label={t("removeTag", { tag })}
              >
                <X className="w-3 h-3" />
              </button>
            </span>
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

      {showMatchToggle && (
        <div className="flex items-center gap-1 bg-muted rounded-md p-0.5 h-9 ml-auto">
          <button
            type="button"
            onClick={() => onMatchModeChange!("any")}
            className={`px-2 py-1 rounded text-xs font-medium ${
              matchMode === "any"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground"
            }`}
            title={t("matchAnyTooltip")}
          >
            {t("matchAny")}
          </button>
          <button
            type="button"
            onClick={() => onMatchModeChange!("all")}
            className={`px-2 py-1 rounded text-xs font-medium ${
              matchMode === "all"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground"
            }`}
            title={t("matchAllTooltip")}
          >
            {t("matchAll")}
          </button>
        </div>
      )}
    </div>
  );
}
