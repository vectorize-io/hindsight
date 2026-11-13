"use client";

// Which knowledge banks a memory bank reads from (bank config `knowledge_banks`).
// Reflect on the memory bank also searches the linked banks and is told their
// documents override conflicting memories. Recall and mental models never read them.
//
// A searchable multi-select inside the Memory Bank section, saved with that section.
// It used to be its own card of checkbox rows with a separate Save button, which grew
// with every knowledge bank on the server.

import * as React from "react";
import { useTranslations } from "next-intl";
import { AlertTriangle, Check, ChevronsUpDown, Library } from "lucide-react";

import { client } from "@/lib/api";
import { type KnowledgeBank } from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export function LinkedKnowledgeBanksPicker({
  value,
  onChange,
}: {
  value: string[];
  onChange: (ids: string[]) => void;
}) {
  const t = useTranslations("knowledgeBanks.linked");
  const [open, setOpen] = React.useState(false);
  const [banks, setBanks] = React.useState<KnowledgeBank[] | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    client
      .listAllKnowledgeBanks()
      .then((list) => !cancelled && setBanks(list))
      .catch(() => !cancelled && setBanks([]));
    return () => {
      cancelled = true;
    };
  }, []);

  const known = new Set(banks?.map((b) => b.bank_id));
  // A linked bank that no longer exists stays listed, so it can be unlinked.
  const missing = banks ? value.filter((id) => !known.has(id)) : [];
  const toggle = (id: string) =>
    onChange(value.includes(id) ? value.filter((x) => x !== id) : [...value, id]);

  const item = (id: string, extra: React.ReactNode) => (
    <CommandItem key={id} value={id} onSelect={() => toggle(id)}>
      <Check className={cn("h-4 w-4 shrink-0", value.includes(id) ? "opacity-100" : "opacity-0")} />
      <span className="min-w-0 flex-1 truncate font-mono text-sm">{id}</span>
      {extra}
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label={t("title")}
          className="h-9 w-full justify-start gap-1.5 px-3 font-normal"
        >
          <Library className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="min-w-0 flex-1 truncate text-left">
            {value.length === 0 ? (
              <span className="text-muted-foreground">{t("noneLinked")}</span>
            ) : (
              value.join(", ")
            )}
          </span>
          <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[min(22rem,var(--radix-popover-content-available-width))] p-0"
        align="end"
      >
        <Command>
          <CommandInput placeholder={t("search")} />
          <CommandList className="max-h-[min(60vh,var(--radix-popover-content-available-height))]">
            <CommandEmpty>{banks === null ? "…" : t("none")}</CommandEmpty>
            <CommandGroup>
              {banks?.map((bank) =>
                item(
                  bank.bank_id,
                  <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                    {bank.document_count}
                  </span>
                )
              )}
              {missing.map((id) =>
                item(
                  id,
                  <span className="inline-flex shrink-0 items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    {t("missing")}
                  </span>
                )
              )}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
