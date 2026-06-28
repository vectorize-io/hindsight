"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { BankSelector } from "@/components/bank-selector";
import { DeploymentView } from "@/components/deployment-view";

export default function DeploymentPage() {
  const t = useTranslations("deployment");

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <BankSelector />
      <main className="flex-1 overflow-y-auto">
        <div className="p-6">
          <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
          <p className="text-muted-foreground mb-6">{t("description")}</p>
          <DeploymentView />
        </div>
      </main>
    </div>
  );
}
