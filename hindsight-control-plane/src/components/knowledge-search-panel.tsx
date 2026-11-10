"use client";

// Semantic search over a knowledge bank's passages, by text, by an image, or both.
// The image can be pasted, dropped, or picked; it goes to the API as base64.

import * as React from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  ImagePlus,
  Search,
  SearchX,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { client } from "@/lib/api";
import {
  MEDIA_TYPES,
  fileToBase64,
  parseTags,
  groupByMediaType,
  type KnowledgeSearchRequest,
  type KnowledgeSearchResponse,
  type MediaType,
} from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { KindIcon, SearchResultCard, SearchResultGrid } from "@/components/knowledge-passage";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const NEIGHBOR_OPTIONS = [0, 1, 2];

interface QueryImage {
  base64: string;
  previewUrl: string;
  name: string;
}

export function KnowledgeSearchPanel({ bankId }: { bankId: string }) {
  const t = useTranslations("knowledgeBanks.search");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const tRecall = useTranslations("searchDebug");
  const [query, setQuery] = React.useState("");
  // "exact" runs grep: the words as written (codes, ids), on text only — no image, no filters.
  const [mode, setMode] = React.useState<"meaning" | "exact">("meaning");
  const [image, setImage] = React.useState<QueryImage | null>(null);
  const [topK, setTopK] = React.useState(10);
  const [neighbors, setNeighbors] = React.useState(1);
  const [showFilters, setShowFilters] = React.useState(false);
  const [tags, setTags] = React.useState("");
  const [tagsMatch, setTagsMatch] = React.useState<"any" | "all">("any");
  const [mediaTypes, setMediaTypes] = React.useState<MediaType[]>([]);
  const [searching, setSearching] = React.useState(false);
  const [response, setResponse] = React.useState<KnowledgeSearchResponse | null>(null);
  const [viewMode, setViewMode] = React.useState<"results" | "json">("results");
  const [error, setError] = React.useState<string | null>(null);
  const [dragOver, setDragOver] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);

  // Free the preview's object URL when the image is replaced or the panel unmounts.
  React.useEffect(() => {
    if (!image) return;
    return () => URL.revokeObjectURL(image.previewUrl);
  }, [image]);

  const setImageFile = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.warning(t("notAnImage"));
      return;
    }
    setImage({
      base64: await fileToBase64(file),
      previewUrl: URL.createObjectURL(file),
      name: file.name || t("pastedImage"),
    });
  };

  const canSearch =
    (mode === "exact" ? query.trim() !== "" : query.trim() !== "" || image !== null) && !searching;
  const activeFilters = parseTags(tags).length + mediaTypes.length;

  const search = async () => {
    if (!canSearch) return;
    const parsedTags = parseTags(tags);
    // Exact text is the same search call with `pattern` instead of a query: the words as
    // typed (codes, ids), no image.
    const exact = mode === "exact";
    const body: KnowledgeSearchRequest = {
      top_k: Math.min(200, Math.max(1, topK || 10)),
      neighbors,
      ...(query.trim() ? (exact ? { pattern: query.trim() } : { query: query.trim() }) : {}),
      ...(image && !exact ? { query_image: image.base64 } : {}),
      ...(parsedTags.length ? { tags: parsedTags, tags_match: tagsMatch } : {}),
      ...(mediaTypes.length ? { media_types: mediaTypes } : {}),
    };
    setSearching(true);
    setError(null);
    try {
      setResponse(await client.searchKnowledgeBank(bankId, body));
    } catch (e) {
      // Shown as a banner over the results, with the server's reason: a 503 says the
      // embedder is unreachable, which is not something a toast should hide.
      setError((e as Error).message);
      setResponse(null);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <div
        className={cn(
          "space-y-4 rounded-xl border border-border bg-card p-4 transition-colors",
          dragOver && "border-primary bg-primary/5"
        )}
        // An image pasted or dropped anywhere on the form becomes the query image.
        onPaste={(e) => {
          const file = Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"));
          if (file) {
            e.preventDefault();
            setImageFile(file);
          }
        }}
        onDragOver={(e) => {
          if (!e.dataTransfer.types.includes("Files")) return;
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          const file = e.dataTransfer.files[0];
          setDragOver(false);
          if (!file) return;
          e.preventDefault();
          setImageFile(file);
        }}
      >
        <form
          className="flex flex-col gap-3 sm:flex-row sm:items-start"
          onSubmit={(e) => {
            e.preventDefault();
            search();
          }}
        >
          <div className="relative min-w-0 flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("placeholder")}
              className="h-10 pl-9"
              aria-label={t("placeholder")}
              autoFocus
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <div className="flex h-10 gap-1 rounded-lg bg-muted p-1" title={t("modeHelp")}>
              {(["meaning", "exact"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={mode === m}
                  onClick={() => setMode(m)}
                  className={`whitespace-nowrap rounded-md px-3 text-sm font-medium transition-colors ${
                    mode === m
                      ? "bg-background shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {m === "meaning" ? t("modeMeaning") : t("modeExact")}
                </button>
              ))}
            </div>
            <Button
              type="button"
              variant="outline"
              className="h-10"
              disabled={mode === "exact"}
              onClick={() => fileRef.current?.click()}
              title={t("imageHint")}
            >
              <ImagePlus className="mr-1.5 h-4 w-4" />
              {t("addImage")}
            </Button>
            <div className="flex items-center gap-1.5">
              <Label htmlFor="kb-top-k" className="whitespace-nowrap text-xs text-muted-foreground">
                {t("topK")}
              </Label>
              <Input
                id="kb-top-k"
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="h-10 w-20"
              />
            </div>
            {/* Passages on each side of a text hit, returned as its context. */}
            <div className="flex items-center gap-1.5" title={t("contextHelp")}>
              <Label
                htmlFor="kb-neighbors"
                className="whitespace-nowrap text-xs text-muted-foreground"
              >
                {t("context")}
              </Label>
              <Select value={String(neighbors)} onValueChange={(v) => setNeighbors(Number(v))}>
                <SelectTrigger id="kb-neighbors" className="h-10 w-16">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {NEIGHBOR_OPTIONS.map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button type="submit" className="h-10" disabled={!canSearch}>
              {searching ? (
                <Spinner size="sm" className="mr-2" />
              ) : (
                <Search className="mr-1.5 h-4 w-4" />
              )}
              {t("submit")}
            </Button>
          </div>
        </form>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          data-testid="kb-search-image-input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) setImageFile(file);
            e.target.value = "";
          }}
        />

        {image ? (
          <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-2">
            <img src={image.previewUrl} alt="" className="h-16 w-16 rounded object-cover" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{image.name}</p>
              <p className="text-xs text-muted-foreground">{t("imageQuery")}</p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => setImage(null)}
              aria-label={t("removeImage")}
              title={t("removeImage")}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{t("imageHint")}</p>
        )}

        <div>
          <button
            type="button"
            onClick={() => setShowFilters((v) => !v)}
            className="inline-flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground"
            aria-expanded={showFilters}
          >
            {showFilters ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            {t("filters")}
            {activeFilters > 0 && (
              <span className="ml-1 rounded-full bg-primary/15 px-1.5 text-xs text-primary">
                {activeFilters}
              </span>
            )}
          </button>
          {showFilters && (
            <div className="mt-3 grid gap-4 md:grid-cols-2">
              <div className="min-w-0 space-y-1.5">
                <Label htmlFor="kb-search-tags">{t("tags")}</Label>
                <div className="flex gap-2">
                  <Input
                    id="kb-search-tags"
                    value={tags}
                    onChange={(e) => setTags(e.target.value)}
                    placeholder="v2, onboarding"
                    className="min-w-0 flex-1"
                  />
                  <Select value={tagsMatch} onValueChange={(v) => setTagsMatch(v as "any" | "all")}>
                    <SelectTrigger className="w-24" aria-label={t("tagsMatch")}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="any">{t("tagsAny")}</SelectItem>
                      <SelectItem value="all">{t("tagsAll")}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="min-w-0 space-y-1.5">
                <span className="text-sm font-medium leading-none">{t("mediaTypes")}</span>
                <div className="flex flex-wrap gap-x-4 gap-y-2 pt-2">
                  {MEDIA_TYPES.map((m) => (
                    <label key={m} className="inline-flex items-center gap-1.5 text-sm">
                      <Checkbox
                        checked={mediaTypes.includes(m)}
                        onCheckedChange={(checked) =>
                          setMediaTypes((prev) =>
                            checked ? [...prev, m] : prev.filter((x) => x !== m)
                          )
                        }
                      />
                      {tKinds(m)}
                    </label>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">{t("mediaTypesHelp")}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {searching && !response && (
        <div className="space-y-3" aria-hidden>
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="h-32 animate-pulse rounded-lg bg-muted/60" />
          ))}
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>{t("failed")}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {response && (
        <div className={cn("space-y-4 transition-opacity", searching && "opacity-50")}>
          {/* Same summary row and Results/JSON switch as the Recall view. */}
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">{tRecall("resultsLabel")}</span>
              <span className="font-semibold">{response.results.length}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">{tRecall("durationLabel")}</span>
              <span className="font-semibold">{(response.took_ms / 1000).toFixed(2)}s</span>
            </div>
            <div className="flex-1" />
            <div className="flex gap-1 bg-muted p-1 rounded-lg">
              {(["results", "json"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    viewMode === mode
                      ? "bg-background shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {mode === "results" ? tRecall("viewResults") : tRecall("viewJson")}
                </button>
              ))}
            </div>
          </div>
          {viewMode === "json" ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">{tRecall("rawResponse")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted p-4 rounded-lg overflow-auto max-h-[600px]">
                  <JsonView src={response} collapsed={2} theme="default" />
                </div>
              </CardContent>
            </Card>
          ) : response.results.length === 0 ? (
            <div className="flex flex-col items-center rounded-xl border border-dashed border-border bg-card px-6 py-12 text-center">
              <SearchX className="mb-3 h-8 w-8 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">{t("noResults")}</p>
            </div>
          ) : (
            <div className="space-y-6">
              {groupByMediaType(response.results).map(({ mediaType, hits }) => (
                <section key={mediaType}>
                  <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
                    <KindIcon kind={mediaType} />
                    {tKinds(mediaType)}
                    <span className="font-normal text-muted-foreground tabular-nums">
                      ({hits.length})
                    </span>
                  </h3>
                  <SearchResultGrid>
                    {hits.map(({ result, rank }) => (
                      <SearchResultCard
                        key={`${result.document_id}-${result.index}`}
                        result={result}
                        bankId={bankId}
                        rank={rank}
                        view="search"
                      />
                    ))}
                  </SearchResultGrid>
                </section>
              ))}
            </div>
          )}
        </div>
      )}

      {!response && !searching && !error && (
        <div className="flex flex-col items-center px-6 py-12 text-center text-sm text-muted-foreground">
          <Search className="mb-3 h-8 w-8" />
          {t("idle")}
        </div>
      )}
    </div>
  );
}
