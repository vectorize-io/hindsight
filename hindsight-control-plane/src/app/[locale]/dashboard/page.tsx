"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { BankSelector } from "@/components/bank-selector";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";

export default function DashboardPage() {
  const t = useTranslations("dashboard");
  const router = useRouter();
  const { currentBank } = useBank();

  // Redirect to bank page if a bank is selected
  useEffect(() => {
    if (currentBank) {
      router.push(bankRoute(currentBank, "?view=data"));
    }
  }, [currentBank, router]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="flex-1 flex flex-col">
        <BankSelector />

        <div className="flex items-center justify-center h-[calc(100vh-80px)] bg-muted/20">
          <div className="text-center p-10 bg-card rounded-lg border-2 border-border shadow-lg max-w-md">
            <h3 className="text-2xl font-bold mb-3 text-card-foreground">{t("welcome")}</h3>
            <p className="text-muted-foreground">{t("selectBank")}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
