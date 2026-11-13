"use client";

// A knowledge bank's settings, laid out like the memory bank's Bank Configuration and
// built from its pieces: the title with an Actions menu, then the same underline tabs.
// General is the bank's stats, its ingest operations (the memory banks' operations list:
// knowledge ingests are async operations of the same bank id) and its identity;
// Configuration is this bank's own ingest settings.

import * as React from "react";
import { useTranslations } from "next-intl";
import { MoreVertical, RotateCcw, Trash2 } from "lucide-react";

import { client } from "@/lib/api";
import { type KnowledgeBank } from "@/lib/knowledge-banks";
import { formatAbsoluteDateTime } from "@/lib/relative-time";
import { ConfigSection, FieldRow } from "@/components/bank-config-view";
import { BankOperationsView } from "@/components/bank-operations-view";
import { KnowledgeBankStats } from "@/components/knowledge-bank-home";
import { KnowledgeBankConfigView } from "@/components/knowledge-bank-config-view";
import { DeleteKnowledgeBankDialog } from "@/components/knowledge-banks-list";
import { UnderlineTabs } from "@/components/ui/underline-tabs";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export type SettingsTab = "general" | "configuration";
export const SETTINGS_TABS: SettingsTab[] = ["general", "configuration"];

function Value({ children, mono = false }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <p
      className={`text-sm text-muted-foreground md:text-right break-words ${mono ? "font-mono" : ""}`}
    >
      {children}
    </p>
  );
}

export function KnowledgeBankSettings({
  bank,
  tab,
  onTabChange,
  onDeleted,
}: {
  bank: KnowledgeBank | null;
  tab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
  onDeleted: () => void;
}) {
  const t = useTranslations("knowledgeBanks.settings");
  const tCommon = useTranslations("common");
  const [deleteOpen, setDeleteOpen] = React.useState(false);
  const [resetOpen, setResetOpen] = React.useState(false);
  const [resetting, setResetting] = React.useState(false);
  // Bumped by a reset from the Actions menu, so the Configuration tab reloads what the server kept.
  const [configVersion, setConfigVersion] = React.useState(0);

  if (!bank) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }

  const resetConfig = async () => {
    setResetting(true);
    try {
      await client.resetKnowledgeBankConfig(bank.bank_id);
      setConfigVersion((v) => v + 1);
      setResetOpen(false);
    } catch {
      // The API client shows the error.
    } finally {
      setResetting(false);
    }
  };

  // Read-only section: no Save footer, nothing dirty, no error slot.
  const readOnly = { error: null, dirty: false, saving: false };

  return (
    <div>
      <div className="flex justify-between items-start gap-4 mb-6">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
          <p className="text-muted-foreground">{t("description")}</p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="shrink-0">
              {tCommon("actions")}
              <MoreVertical className="w-4 h-4 ml-2" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem
              onClick={() => setResetOpen(true)}
              className="text-amber-600 dark:text-amber-400 focus:text-amber-700 dark:focus:text-amber-300"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              {t("resetConfig")}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => setDeleteOpen(true)}
              className="text-red-600 dark:text-red-400 focus:text-red-700 dark:focus:text-red-300"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              {t("delete")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <UnderlineTabs
        value={tab}
        onChange={onTabChange}
        tabs={[
          { id: "general", label: t("tabGeneral") },
          { id: "configuration", label: t("tabConfiguration") },
        ]}
      />

      {tab === "general" && (
        <div>
          <p className="text-sm text-muted-foreground mb-4">{t("generalIntro")}</p>
          <div className="space-y-6">
            <KnowledgeBankStats bank={bank} />
            <BankOperationsView bankId={bank.bank_id} hideTypeFilter />
            <ConfigSection
              title={t("aboutTitle")}
              description={t("aboutDescription")}
              {...readOnly}
            >
              <FieldRow label={t("bankId")}>
                <Value mono>{bank.bank_id}</Value>
              </FieldRow>
              <FieldRow label={t("created")}>
                <Value>{bank.created_at ? formatAbsoluteDateTime(bank.created_at) : "—"}</Value>
              </FieldRow>
            </ConfigSection>
            {/* Which memory banks link this one is not listed: it would take one config
                read per memory bank. The link is made from the memory bank's side. */}
            <ConfigSection title={t("linkTitle")} description={t("linkDescription")} {...readOnly}>
              <p className="px-6 py-4 text-sm text-muted-foreground">{t("linkHowTo")}</p>
            </ConfigSection>
          </div>
        </div>
      )}

      {tab === "configuration" && (
        <KnowledgeBankConfigView key={configVersion} bankId={bank.bank_id} />
      )}

      <AlertDialog open={resetOpen} onOpenChange={setResetOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("resetConfigTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("resetConfigPrompt", { bankId: bank.bank_id })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={resetting}>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                // Stay open until the reset lands, so a failure is not hidden by the dialog closing.
                e.preventDefault();
                resetConfig();
              }}
              disabled={resetting}
            >
              {resetting ? (
                <Spinner size="sm" className="mr-2" />
              ) : (
                <RotateCcw className="w-4 h-4 mr-2" />
              )}
              {t("resetConfig")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <DeleteKnowledgeBankDialog
        bank={deleteOpen ? bank : null}
        onOpenChange={setDeleteOpen}
        onDeleted={onDeleted}
      />
    </div>
  );
}
