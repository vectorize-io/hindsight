"use client";

// A knowledge bank's settings, laid out like the memory bank configuration page and
// built from its pieces: the title with an Actions menu, then read-only
// ConfigSection/FieldRow sections.

import * as React from "react";
import { useTranslations } from "next-intl";
import { MoreVertical, Trash2 } from "lucide-react";

import {
  formatBytes,
  formatMs,
  type KnowledgeBank,
  type KnowledgeConfig,
} from "@/lib/knowledge-banks";
import { formatAbsoluteDateTime } from "@/lib/relative-time";
import { ConfigSection, FieldRow } from "@/components/bank-config-view";
import { DeleteKnowledgeBankDialog } from "@/components/knowledge-banks-list";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

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
  config,
  onDeleted,
}: {
  bank: KnowledgeBank | null;
  config: KnowledgeConfig | null;
  onDeleted: () => void;
}) {
  const t = useTranslations("knowledgeBanks.settings");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const tCommon = useTranslations("common");
  const [deleteOpen, setDeleteOpen] = React.useState(false);

  if (!bank) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }

  const seconds = (s: number) => t("seconds", { count: s });
  // Read-only sections: no Save footer, nothing dirty, no error slot.
  const readOnly = { error: null, dirty: false, saving: false };

  return (
    <div>
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-foreground">{t("title")}</h1>
          <p className="text-muted-foreground">{t("description")}</p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              {tCommon("actions")}
              <MoreVertical className="w-4 h-4 ml-2" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
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

      <div className="space-y-6">
        <ConfigSection
          title={t("generalTitle")}
          description={t("generalDescription")}
          {...readOnly}
        >
          <FieldRow label={t("bankId")}>
            <Value mono>{bank.bank_id}</Value>
          </FieldRow>
          <FieldRow label={t("created")}>
            <Value>{bank.created_at ? formatAbsoluteDateTime(bank.created_at) : "—"}</Value>
          </FieldRow>
        </ConfigSection>

        <ConfigSection
          title={t("embedderTitle")}
          description={t("embedderDescription")}
          {...readOnly}
        >
          {config ? (
            <>
              <FieldRow label={t("provider")}>
                <Value>{config.provider}</Value>
              </FieldRow>
              <FieldRow label={t("model")}>
                <Value mono>{config.model}</Value>
              </FieldRow>
              <FieldRow label={t("dimension")}>
                <Value>{config.dimension}</Value>
              </FieldRow>
              <FieldRow label={t("mediaTypes")}>
                <Value>{config.media_types.map((m) => tKinds(m)).join(", ")}</Value>
              </FieldRow>
              <FieldRow label={t("maxFile")}>
                <Value>{formatBytes(config.max_file_bytes)}</Value>
              </FieldRow>
              <FieldRow label={t("maxMedia")}>
                <Value>{formatMs(config.max_media_seconds * 1000)}</Value>
              </FieldRow>
              <FieldRow label={t("maxPdfPages")}>
                <Value>{config.max_pdf_pages}</Value>
              </FieldRow>
              <FieldRow label={t("audioWindow")}>
                <Value>{seconds(config.audio_window_seconds)}</Value>
              </FieldRow>
              <FieldRow label={t("videoWindow")}>
                <Value>{seconds(config.video_window_seconds)}</Value>
              </FieldRow>
              <FieldRow label={t("overlap")}>
                <Value>{seconds(config.media_overlap_seconds)}</Value>
              </FieldRow>
              <FieldRow label={t("pdfPixels")}>
                <Value>{config.pdf_page_pixels} px</Value>
              </FieldRow>
              <FieldRow label={t("textTokens")}>
                <Value>{config.text_passage_tokens}</Value>
              </FieldRow>
            </>
          ) : (
            <div className="flex justify-center py-6">
              <Spinner size="sm" />
            </div>
          )}
        </ConfigSection>

        {/* Which memory banks link this one is not listed: it would take one config
            read per memory bank. The link is made from the memory bank's side. */}
        <ConfigSection title={t("linkTitle")} description={t("linkDescription")} {...readOnly}>
          <p className="px-6 py-4 text-sm text-muted-foreground">{t("linkHowTo")}</p>
        </ConfigSection>
      </div>

      <DeleteKnowledgeBankDialog
        bank={deleteOpen ? bank : null}
        onOpenChange={setDeleteOpen}
        onDeleted={onDeleted}
      />
    </div>
  );
}
