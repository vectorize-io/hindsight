"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Check, Copy } from "lucide-react";

interface JsonViewerProps {
  /** The value to display. Objects/arrays are pretty-printed; strings render as-is. */
  value: unknown;
  /**
   * Classes for the <pre> box (background, max-height, scroll, text color).
   * Defaults to `bg-muted`. Layout/wrap classes are always applied.
   */
  className?: string;
}

function toDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  // Unescape newlines inside string values so multi-line content (e.g. prompts)
  // renders as real line breaks under `whitespace-pre-wrap` instead of literal "\n".
  return JSON.stringify(value, null, 2).replace(/\\n/g, "\n");
}

/**
 * Read-only viewer for JSON (or plain text) with word-wrapping and a copy button.
 * Used in detail dialogs to show request/response/input/output/metadata payloads.
 */
export function JsonViewer({ value, className = "bg-muted" }: JsonViewerProps) {
  const t = useTranslations("common");
  const [copied, setCopied] = useState(false);
  const text = toDisplayText(value);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context) — ignore.
    }
  };

  return (
    <div className="relative group">
      <button
        type="button"
        onClick={handleCopy}
        title={copied ? t("copied") : t("copy")}
        aria-label={copied ? t("copied") : t("copy")}
        className="absolute top-1.5 right-1.5 p-1.5 rounded-md border border-border bg-background/70 text-muted-foreground opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-foreground hover:bg-muted transition-all"
      >
        {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
      </button>
      <pre className={`p-3 pr-10 rounded-md text-xs whitespace-pre-wrap break-words ${className}`}>
        {text}
      </pre>
    </div>
  );
}
