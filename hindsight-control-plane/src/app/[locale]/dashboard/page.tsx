"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { BankSelector } from "@/components/bank-selector";
import { Link } from "@/i18n/navigation";
import { Button } from "@/components/ui/button";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";

export default function DashboardPage() {
  const t = useTranslations("dashboard");
  const ts = useTranslations("settings");
  const router = useRouter();
  const { currentBank } = useBank();
  // undefined = unknown/unavailable (endpoint disabled or multi-tenant), true/false otherwise.
  const [llmConfigured, setLlmConfigured] = useState<boolean | undefined>(undefined);

  // Redirect to bank page if a bank is selected
  useEffect(() => {
    if (currentBank) {
      router.push(bankRoute(currentBank, "?view=data"));
    }
  }, [currentBank, router]);

  // First-run prompt: if the instance LLM config API is available and not configured,
  // nudge the user to set it up. Silently ignored when the endpoint is disabled.
  useEffect(() => {
    let active = true;
    client
      .getServerLlmConfig()
      .then((cfg) => active && setLlmConfigured(cfg.is_configured))
      .catch(() => active && setLlmConfigured(undefined));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="flex-1 flex flex-col">
        <BankSelector />

        {llmConfigured === false && (
          <div className="flex items-center justify-between gap-4 border-b bg-amber-50 px-6 py-3 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            <span>{ts("firstRunBanner")}</span>
            <Button asChild size="sm" variant="outline">
              <Link href="/settings">{ts("firstRunAction")}</Link>
            </Button>
          </div>
        )}

        <div className="flex items-center justify-center h-[calc(100vh-80px)] bg-muted/20">
          <div className="text-center p-10 bg-card rounded-lg border-2 border-border shadow-lg max-w-md">
            <h3 className="text-2xl font-bold mb-3 text-card-foreground">{t("welcome")}</h3>
            <p className="text-muted-foreground">{t("selectBank")}</p>
            {llmConfigured !== undefined && (
              <div className="mt-5">
                <Button asChild variant="ghost" size="sm">
                  <Link href="/settings">{ts("title")}</Link>
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
