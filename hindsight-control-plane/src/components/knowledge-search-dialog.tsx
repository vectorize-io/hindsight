"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { JsonViewer } from "@/components/ui/json-viewer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Search } from "lucide-react";
import { KnowledgeSearchResult } from "./knowledge-search-result";

type SearchResponse = Awaited<ReturnType<typeof client.searchKnowledgePages>>;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bankId: string;
  /** Seed query, taken from the sidebar's search box when the dialog opens. */
  initialQuery: string;
  /** Folder path per node id, from the loaded tree — search returns neither. */
  pathById: Map<string, string>;
  onOpenPage: (pageId: string) => void;
}

/**
 * The knowledge search the sidebar runs, with its knobs exposed and the server's
 * answer shown raw beside the rendered hits. The endpoint takes a query and a
 * limit and nothing else — there is no fact-type/tag surface here as there is in
 * recall, because knowledge search has none.
 */
export function KnowledgeSearchDialog({
  open,
  onOpenChange,
  bankId,
  initialQuery,
  pathById,
  onOpenPage,
}: Props) {
  const t = useTranslations("knowledgeBase");
  const [query, setQuery] = useState(initialQuery);
  const [limit, setLimit] = useState(20);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [searching, setSearching] = useState(false);

  const run = async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    const started = performance.now();
    try {
      const r = await client.searchKnowledgePages(bankId, q, limit);
      setResponse(r);
      setTookMs(Math.round(performance.now() - started));
    } catch {
      // toast handled by interceptor
      setResponse(null);
    } finally {
      setSearching(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Fixed height, flex column: the panel is the same size empty as it is
          full, so running a search doesn't jump the dialog under the cursor. */}
      <DialogContent className="sm:max-w-3xl flex flex-col h-[70vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>{t("advancedSearchTitle")}</DialogTitle>
          <DialogDescription>{t("advancedSearchDescription")}</DialogDescription>
        </DialogHeader>

        <div className="flex items-end gap-2 shrink-0">
          <div className="flex-1 space-y-1">
            <Label htmlFor="kb-adv-query">{t("advancedQuery")}</Label>
            <Input
              id="kb-adv-query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              placeholder={t("searchPlaceholder")}
            />
          </div>
          <div className="w-24 space-y-1">
            <Label htmlFor="kb-adv-limit">{t("advancedLimit")}</Label>
            <Input
              id="kb-adv-limit"
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            />
          </div>
          <Button onClick={run} disabled={searching || !query.trim()}>
            {searching ? <Spinner size="sm" /> : <Search className="w-4 h-4" />}
            {t("advancedRun")}
          </Button>
        </div>

        {!response ? (
          <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
            {t("advancedIdle")}
          </div>
        ) : (
          <Tabs defaultValue="results" className="flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between gap-2 shrink-0">
              <TabsList>
                <TabsTrigger value="results">{t("advancedResults")}</TabsTrigger>
                <TabsTrigger value="json">{t("advancedJson")}</TabsTrigger>
              </TabsList>
              <span className="text-xs text-muted-foreground">
                {t("advancedMeta", { total: response.total, ms: tookMs ?? 0 })}
              </span>
            </div>

            <TabsContent value="results" className="flex-1 min-h-0 overflow-y-auto">
              {response.results.length === 0 ? (
                <p className="py-8 text-sm text-muted-foreground text-center">{t("searchEmpty")}</p>
              ) : (
                <ul className="divide-y divide-border">
                  {response.results.map((r, i) => (
                    <li key={r.id}>
                      <KnowledgeSearchResult
                        hit={r}
                        rank={i + 1}
                        path={pathById.get(r.id) ?? bankId}
                        onClick={() => {
                          onOpenPage(r.id);
                          onOpenChange(false);
                        }}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </TabsContent>

            <TabsContent value="json" className="flex-1 min-h-0">
              <JsonViewer value={response} className="bg-muted h-full overflow-y-auto text-xs" />
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
}
