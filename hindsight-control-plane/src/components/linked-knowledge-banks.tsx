"use client";

// Which knowledge banks a memory bank reads from (bank config `knowledge_banks`).
// Reflect on the memory bank also searches the linked banks and is told their
// documents override conflicting memories. Recall and mental models never read them.

import * as React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { AlertTriangle, Library } from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import { knowledgeBankRoute, type KnowledgeBank } from "@/lib/knowledge-banks";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";

function sameSet(a: string[], b: string[]) {
  return a.length === b.length && a.every((x) => b.includes(x));
}

export function LinkedKnowledgeBanks({ bankId }: { bankId: string }) {
  const t = useTranslations("knowledgeBanks.linked");
  const [banks, setBanks] = React.useState<KnowledgeBank[] | null>(null);
  const [saved, setSaved] = React.useState<string[]>([]);
  const [selected, setSelected] = React.useState<string[]>([]);
  const [saving, setSaving] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    Promise.all([client.getBankConfig(bankId), client.listAllKnowledgeBanks()])
      .then(([cfg, list]) => {
        if (cancelled) return;
        const linked: string[] = Array.isArray(cfg.config.knowledge_banks)
          ? cfg.config.knowledge_banks
          : [];
        setSaved(linked);
        setSelected(linked);
        setBanks(list);
      })
      .catch(() => !cancelled && setBanks([]));
    return () => {
      cancelled = true;
    };
  }, [bankId]);

  const save = async () => {
    setSaving(true);
    try {
      // null, not [], when nothing is linked: the override then goes away and the
      // bank inherits the tenant/server default again.
      const res = await client.updateBankConfig(bankId, {
        knowledge_banks: selected.length ? selected : null,
      });
      const linked: string[] = Array.isArray(res.config.knowledge_banks)
        ? res.config.knowledge_banks
        : [];
      setSaved(linked);
      setSelected(linked);
      toast.success(t("saved"));
    } catch {
      // toasted by the API client
    } finally {
      setSaving(false);
    }
  };

  const known = new Set(banks?.map((b) => b.bank_id));
  const missing = selected.filter((id) => banks && !known.has(id));
  const toggle = (id: string, on: boolean) =>
    setSelected((prev) => (on ? [...prev, id] : prev.filter((x) => x !== id)));

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold">{t("title")}</h2>
        <p className="text-sm text-muted-foreground">{t("description")}</p>
      </div>
      <Card className="border-border/40 bg-muted/20">
        <div className="divide-y divide-border/40">
          {banks === null ? (
            <div className="flex items-center justify-center py-8">
              <Spinner size="sm" />
            </div>
          ) : banks.length === 0 && missing.length === 0 ? (
            <div className="flex flex-col items-center px-6 py-8 text-center text-sm text-muted-foreground">
              <Library className="mb-2 h-6 w-6" />
              {t("none")}
              <Link
                href="/knowledge-banks"
                className="mt-2 font-medium text-primary hover:underline"
              >
                {t("createOne")}
              </Link>
            </div>
          ) : (
            <>
              {banks.map((bank) => (
                <label
                  key={bank.bank_id}
                  className="flex cursor-pointer items-center gap-3 px-6 py-3 hover:bg-muted/30"
                >
                  <Checkbox
                    checked={selected.includes(bank.bank_id)}
                    onCheckedChange={(on) => toggle(bank.bank_id, on === true)}
                  />
                  <div className="min-w-0 flex-1 truncate font-mono text-sm font-medium">
                    {bank.bank_id}
                  </div>
                  <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                    {t("counts", { documents: bank.document_count, passages: bank.passage_count })}
                  </span>
                  <Link
                    href={knowledgeBankRoute(bank.bank_id)}
                    className="shrink-0 text-xs text-muted-foreground hover:text-foreground hover:underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {t("open")}
                  </Link>
                </label>
              ))}
              {missing.map((id) => (
                <label key={id} className="flex cursor-pointer items-center gap-3 px-6 py-3">
                  <Checkbox checked onCheckedChange={(on) => toggle(id, on === true)} />
                  <span className="min-w-0 flex-1 truncate font-mono text-sm">{id}</span>
                  <span className="inline-flex shrink-0 items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    {t("missing")}
                  </span>
                </label>
              ))}
            </>
          )}
        </div>
        <div className="flex justify-end border-t border-border/40 px-6 py-4">
          <Button size="sm" disabled={sameSet(saved, selected) || saving} onClick={save}>
            {saving && <Spinner size="sm" className="mr-2" />}
            {t("save")}
          </Button>
        </div>
      </Card>
    </section>
  );
}
