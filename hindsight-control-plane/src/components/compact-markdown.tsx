"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

const TABLE_LINE_RE = /^\s*\|.*\|\s*$/;
const TABLE_SEPARATOR_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/;

/** Normalize LLM-emitted markdown so GFM tables render reliably.
 *
 *  Two transforms:
 *    1. Strip leading whitespace from lines that look like table rows.
 *       Otherwise remark-gfm treats an indented pipe-table as a code block.
 *    2. Ensure a blank line precedes and follows each table so the parser
 *       enters table mode instead of folding rows into the prior paragraph. */
function normalizeMarkdown(input: string): string {
  const lines = input.split("\n");
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const header = lines[i];
    const sep = lines[i + 1];
    if (header && sep && TABLE_LINE_RE.test(header) && TABLE_SEPARATOR_RE.test(sep)) {
      if (out.length > 0 && out[out.length - 1].trim() !== "") out.push("");
      while (i < lines.length && TABLE_LINE_RE.test(lines[i])) {
        out.push(lines[i].replace(/^\s+/, ""));
        i++;
      }
      if (i < lines.length && lines[i].trim() !== "") out.push("");
      continue;
    }
    out.push(header);
    i++;
  }
  return out.join("\n");
}

export function CompactMarkdown({ children, className }: { children: string; className?: string }) {
  return (
    <div
      className={cn(
        "text-[13px] leading-6 text-foreground/90 space-y-2 [&>:first-child]:mt-0",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => (
            <div
              className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mt-4 mb-1"
              {...props}
            />
          ),
          h2: (props) => (
            <div
              className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mt-4 mb-1"
              {...props}
            />
          ),
          h3: (props) => (
            <div
              className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mt-3 mb-1"
              {...props}
            />
          ),
          h4: (props) => (
            <div
              className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mt-3 mb-1"
              {...props}
            />
          ),
          p: (props) => <p className="my-1.5" {...props} />,
          ul: (props) => <ul className="list-disc pl-5 my-1.5 space-y-0.5" {...props} />,
          ol: (props) => <ol className="list-decimal pl-5 my-1.5 space-y-0.5" {...props} />,
          li: (props) => <li className="leading-6" {...props} />,
          strong: (props) => <strong className="font-semibold text-foreground" {...props} />,
          code: (props) => (
            <code className="text-[12px] font-mono bg-muted/70 px-1 py-0.5 rounded" {...props} />
          ),
          a: (props) => <a className="text-primary underline" {...props} />,
          table: (props) => (
            <div className="overflow-x-auto my-3 rounded-md border border-border">
              <table className="text-[12px] w-full border-collapse" {...props} />
            </div>
          ),
          thead: (props) => <thead className="bg-muted/60" {...props} />,
          th: (props) => (
            <th
              className="text-left font-semibold px-3 py-1.5 border-b border-border whitespace-nowrap"
              {...props}
            />
          ),
          td: (props) => (
            <td
              className="px-3 py-1.5 border-b border-border/40 align-top [tr:last-child_&]:border-b-0"
              {...props}
            />
          ),
          hr: (props) => <hr className="my-3 border-border/60" {...props} />,
          blockquote: (props) => (
            <blockquote
              className="border-l-2 border-border/60 pl-3 my-2 text-muted-foreground italic"
              {...props}
            />
          ),
        }}
      >
        {normalizeMarkdown(children)}
      </ReactMarkdown>
    </div>
  );
}
