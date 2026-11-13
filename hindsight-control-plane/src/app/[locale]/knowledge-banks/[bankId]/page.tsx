"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { FileText, Home, Library, Search, Settings } from "lucide-react";

import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import {
  DOCUMENT_STATUSES,
  knowledgeBankRoute,
  MEDIA_TYPES,
  type DocumentStatus,
  type KnowledgeBank,
  type MediaType,
} from "@/lib/knowledge-banks";
import { BankSelector } from "@/components/bank-selector";
import { Sidebar, type SidebarItem } from "@/components/sidebar";
import { KnowledgeBankHome } from "@/components/knowledge-bank-home";
import {
  KnowledgeBankSettings,
  SETTINGS_TABS,
  type SettingsTab,
} from "@/components/knowledge-bank-settings";
import {
  DOCUMENT_LAYOUTS,
  KnowledgeDocumentsView,
  type DocumentsLayout,
} from "@/components/knowledge-documents-view";
import { KnowledgeDocumentDialog } from "@/components/knowledge-document-dialog";
import { KnowledgeSearchPanel } from "@/components/knowledge-search-panel";
import { Button } from "@/components/ui/button";

type Section = "home" | "documents" | "search" | "settings";
const SECTIONS: Section[] = ["home", "documents", "search", "settings"];

/** `value` when it is one of `allowed`, else undefined: a bad link is ignored, not obeyed. */
function oneOf<T extends string>(value: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((a) => a === value);
}

export default function KnowledgeBankPage() {
  const params = useParams<{ bankId: string }>();
  const bankId = decodeURIComponent(params.bankId);
  const router = useRouter();
  const searchParams = useSearchParams();
  const t = useTranslations("knowledgeBanks.bank");
  const { setCurrentBank } = useBank();

  const rawView = searchParams.get("view") as Section | null;
  const view: Section = rawView && SECTIONS.includes(rawView) ? rawView : "home";
  const openDoc = searchParams.get("doc");
  const passageParam = searchParams.get("passage");
  const highlightPassage = passageParam != null ? Number(passageParam) : null;
  // Links from Home open the documents list filtered (?kind=, ?status=) or as a constellation (?layout=).
  const requestedKind = oneOf<MediaType>(searchParams.get("kind"), MEDIA_TYPES);
  const requestedStatus = oneOf<DocumentStatus>(searchParams.get("status"), DOCUMENT_STATUSES);
  const requestedLayout = oneOf<DocumentsLayout>(searchParams.get("layout"), DOCUMENT_LAYOUTS);
  // Settings' sub-tab, in the URL like a memory bank's `bankConfigTab`.
  const settingsTab =
    oneOf<SettingsTab>(searchParams.get("settingsTab"), SETTINGS_TABS) ?? "general";

  const [bank, setBank] = useState<KnowledgeBank | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    setCurrentBank(null);
  }, [setCurrentBank]);

  const loadBank = useCallback(async () => {
    try {
      setBank(await client.getKnowledgeBank(bankId));
    } catch (e) {
      if ((e as { status?: number }).status === 404) setMissing(true);
    }
  }, [bankId]);

  useEffect(() => {
    loadBank();
  }, [loadBank]);

  const go = (query: string) => router.push(knowledgeBankRoute(bankId, query));
  const goSection = (section: Section) => go(section === "home" ? "" : `?view=${section}`);
  // The document dialog opens over whichever section asked for it, and closing it
  // returns there, like the memory bank documents dialog.
  const openDocument = (id: string, passage?: number) =>
    go(
      `?view=${view}&doc=${encodeURIComponent(id)}${passage != null ? `&passage=${passage}` : ""}`
    );
  const closeDocument = () =>
    router.replace(knowledgeBankRoute(bankId, view === "home" ? "" : `?view=${view}`));

  const items: SidebarItem<Section>[] = [
    { id: "home", label: t("navHome"), icon: Home },
    { id: "documents", label: t("navDocuments"), icon: FileText },
    { id: "search", label: t("navSearch"), icon: Search },
    { id: "settings", label: t("navSettings"), icon: Settings },
  ];

  return (
    // Same frame as a memory bank page: header, rail, and `main` as the only scroller.
    <div className="h-dvh overflow-hidden bg-background flex flex-col">
      <div className="shrink-0">
        <BankSelector kind="knowledge" knowledgeBankId={bankId} />
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {!missing && (
          <Sidebar
            currentTab={view}
            onTabChange={goSection}
            items={items}
            hrefFor={(id) => knowledgeBankRoute(bankId, id === "home" ? "" : `?view=${id}`)}
            title={bankId}
            description={t("sidebarDescription")}
          />
        )}

        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="p-4 sm:p-6 pb-[max(1rem,env(safe-area-inset-bottom))]">
            <div className="max-w-[1024px] xl:max-w-[1280px] 2xl:max-w-[1440px] mx-auto w-full">
              {missing ? (
                <div className="flex flex-col items-center rounded-xl border border-dashed border-border bg-card px-6 py-16 text-center">
                  <Library className="mb-4 h-10 w-10 text-muted-foreground" />
                  <h1 className="text-lg font-semibold">{t("notFound")}</h1>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">{bankId}</p>
                  <Button className="mt-6" variant="outline" asChild>
                    <Link href="/knowledge-banks">{t("allBanks")}</Link>
                  </Button>
                </div>
              ) : (
                <>
                  {view === "home" && (
                    <KnowledgeBankHome bank={bank} onOpenDocument={openDocument} />
                  )}

                  {/* Documents and search stay mounted while hidden, so switching back
                      finds the list's filters and the search results where they were — and
                      the header's "Add Document" always has the upload dialog to open. */}
                  <div className={view === "documents" ? undefined : "hidden"}>
                    <h1 className="mb-2 text-3xl font-bold text-foreground">
                      {t("documentsTitle")}
                    </h1>
                    <p className="mb-6 text-muted-foreground">{t("documentsDescription")}</p>
                    <KnowledgeDocumentsView
                      bankId={bankId}
                      onOpenDocument={openDocument}
                      onChanged={loadBank}
                      requestedKind={requestedKind}
                      requestedStatus={requestedStatus}
                      requestedLayout={requestedLayout}
                    />
                  </div>
                  <div className={view === "search" ? undefined : "hidden"}>
                    <h1 className="mb-2 text-3xl font-bold text-foreground">{t("searchTitle")}</h1>
                    <p className="mb-6 text-muted-foreground">{t("searchDescription")}</p>
                    <KnowledgeSearchPanel bankId={bankId} />
                  </div>

                  {view === "settings" && (
                    <KnowledgeBankSettings
                      bank={bank}
                      tab={settingsTab}
                      onTabChange={(tab) => go(`?view=settings&settingsTab=${tab}`)}
                      onDeleted={() => router.push("/knowledge-banks")}
                    />
                  )}
                </>
              )}
            </div>
          </div>
        </main>
      </div>

      <KnowledgeDocumentDialog
        bankId={bankId}
        documentId={openDoc}
        highlightPassage={Number.isFinite(highlightPassage) ? highlightPassage : null}
        onClose={closeDocument}
        onDeleted={() => {
          loadBank();
          closeDocument();
        }}
      />
    </div>
  );
}
