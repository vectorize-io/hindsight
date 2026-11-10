"use client";

// Memory banks and knowledge banks are both banks, so they are chosen in one place:
// this switch sits at the top of the header's bank picker and on both overview pages.

import { useTranslations } from "next-intl";
import { Brain, Library } from "lucide-react";
import { cn } from "@/lib/utils";

export type BankKind = "memory" | "knowledge";

export function BankKindSwitch({
  value,
  onChange,
  size = "default",
}: {
  value: BankKind;
  onChange: (kind: BankKind) => void;
  size?: "default" | "sm";
}) {
  const t = useTranslations("knowledgeBanks.kindSwitch");
  const options = [
    { kind: "memory" as const, label: t("memory"), Icon: Brain },
    { kind: "knowledge" as const, label: t("knowledge"), Icon: Library },
  ];
  return (
    <div
      role="group"
      aria-label={t("label")}
      className="inline-flex shrink-0 rounded-md border border-border bg-muted/40 p-0.5"
    >
      {options.map(({ kind, label, Icon }) => (
        <button
          key={kind}
          type="button"
          onClick={() => onChange(kind)}
          aria-pressed={value === kind}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-[5px] font-medium transition-colors",
            size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
            value === kind
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Icon className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} />
          {label}
        </button>
      ))}
    </div>
  );
}
