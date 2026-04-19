"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

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
            <div className="overflow-x-auto my-2">
              <table className="text-[12px] border-collapse" {...props} />
            </div>
          ),
          th: (props) => (
            <th className="text-left font-semibold px-2 py-1 border-b border-border" {...props} />
          ),
          td: (props) => <td className="px-2 py-1 border-b border-border/50" {...props} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
