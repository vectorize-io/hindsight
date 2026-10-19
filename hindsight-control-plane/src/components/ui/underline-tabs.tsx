"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface UnderlineTab<T extends string> {
  id: T;
  label: React.ReactNode;
  /** Shows a muted "Off" pill next to the label (feature disabled on this server). */
  off?: boolean;
}

interface UnderlineTabsProps<T extends string> {
  tabs: UnderlineTab<T>[];
  value: T;
  onChange: (id: T) => void;
  className?: string;
}

/**
 * The page-level sub-tab row (Memories, Knowledge, Bank Configuration).
 *
 * On a narrow screen the row scrolls sideways inside itself instead of widening
 * the page, and the active tab is scrolled into view, so adding a tab never
 * breaks the mobile layout.
 */
export function UnderlineTabs<T extends string>({
  tabs,
  value,
  onChange,
  className,
}: UnderlineTabsProps<T>) {
  const activeRef = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    activeRef.current?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [value]);

  return (
    <div className={cn("mb-6 border-b border-border", className)}>
      <div role="tablist" className="flex gap-1 overflow-x-auto">
        {tabs.map((tab) => {
          const active = tab.id === value;
          return (
            <button
              key={tab.id}
              ref={active ? activeRef : undefined}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onChange(tab.id)}
              className={cn(
                "relative shrink-0 whitespace-nowrap px-3 py-3 text-sm font-semibold transition-all sm:px-6",
                active ? "text-primary" : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tab.label}
              {tab.off && (
                <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                  Off
                </span>
              )}
              {active && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-gradient" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
