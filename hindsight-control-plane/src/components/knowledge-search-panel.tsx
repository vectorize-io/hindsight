"use client";

// One search over a knowledge bank: text, an image and an MP4 clip in any combination make one
// query (the server embeds them together), ranked in one list across every media type.
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
  SEARCH_BUDGETS,
  fileToBase64,
  parseTags,
  type KnowledgeSearchRequest,
  type KnowledgeSearchResponse,
  type MediaType,
  type SearchBudget,
} from "@/lib/knowledge-banks";
import { cn } from "@/lib/utils";
import { SearchResultCard, SearchResultGrid } from "@/components/knowledge-passage";
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

const DEFAULT_MAX_TOKENS = 4096; // the API's default
const MAX_MAX_TOKENS = 1_000_000; // the API's limit

interface QueryMedia {
  base64: string;
  previewUrl: string;
  name: string;
}

/** What a dropped, pasted or chosen file can be in a query: an image or an MP4 clip. */
function queryKind(file: File): "image" | "video" | null {
  if (file.type.startsWith("image/")) return "image";
  if (file.type === "video/mp4") return "video";
  return null;
}

export function KnowledgeSearchPanel({ bankId }: { bankId: string }) {
  const t = useTranslations("knowledgeBanks.search");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const tRecall = useTranslations("searchDebug");
  const [query, setQuery] = React.useState("");
  const [image, setImage] = React.useState<QueryMedia | null>(null);
  const [video, setVideo] = React.useState<QueryMedia | null>(null);
  const [maxTokens, setMaxTokens] = React.useState(DEFAULT_MAX_TOKENS);
  const [budget, setBudget] = React.useState<SearchBudget>("mid");
  const [showOptions, setShowOptions] = React.useState(false);
  const [tags, setTags] = React.useState("");
  const [tagsMatch, setTagsMatch] = React.useState<"any" | "all">("any");
  const [mediaTypes, setMediaTypes] = React.useState<MediaType[]>([]);
  const [searching, setSearching] = React.useState(false);
  const [response, setResponse] = React.useState<KnowledgeSearchResponse | null>(null);
  // Timed here: the response carries no timing.
  const [tookMs, setTookMs] = React.useState(0);
  const [viewMode, setViewMode] = React.useState<"results" | "json">("results");
  const [error, setError] = React.useState<string | null>(null);
  const [dragOver, setDragOver] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);

  // Free a preview's object URL when it is replaced or the panel unmounts.
  React.useEffect(() => {
    if (!image) return;
    return () => URL.revokeObjectURL(image.previewUrl);
  }, [image]);
  React.useEffect(() => {
    if (!video) return;
    return () => URL.revokeObjectURL(video.previewUrl);
  }, [video]);

  const setMediaFile = async (file: File) => {
    const kind = queryKind(file);
    if (!kind) {
      toast.warning(t("notMedia"));
      return;
    }
    const media = {
      base64: await fileToBase64(file),
      previewUrl: URL.createObjectURL(file),
      name: file.name || t("pastedMedia"),
    };
    (kind === "image" ? setImage : setVideo)(media);
  };

  const hasQuery = query.trim() !== "" || image !== null || video !== null;
  const canSearch = hasQuery && !searching;
  const activeOptions = parseTags(tags).length + mediaTypes.length;

  const search = async () => {
    if (!canSearch) return;
    const parsedTags = parseTags(tags);
    const body: KnowledgeSearchRequest = {
      max_tokens: Math.min(MAX_MAX_TOKENS, Math.max(1, maxTokens || DEFAULT_MAX_TOKENS)),
      ...(budget !== "mid" ? { budget } : {}),
      ...(query.trim() ? { query: query.trim() } : {}),
      ...(image ? { query_image: image.base64 } : {}),
      ...(video ? { query_video: video.base64 } : {}),
      ...(parsedTags.length ? { tags: parsedTags, tags_match: tagsMatch } : {}),
      ...(mediaTypes.length ? { media_types: mediaTypes } : {}),
    };
    setSearching(true);
    setError(null);
    const started = performance.now();
    try {
      setResponse(await client.searchKnowledgeBank(bankId, body));
      setTookMs(performance.now() - started);
    } catch (e) {
      // Shown as a banner over the results, with the server's reason: a 503 says the
      // embedder is unreachable, which is not something a toast should hide.
      setError((e as Error).message);
      setResponse(null);
    } finally {
      setSearching(false);
    }
  };

  const attached: {
    kind: "image" | "video";
    media: QueryMedia;
    label: string;
    clear: () => void;
  }[] = [];
  if (image)
    attached.push({
      kind: "image",
      media: image,
      label: t("imageQuery"),
      clear: () => setImage(null),
    });
  if (video)
    attached.push({
      kind: "video",
      media: video,
      label: t("videoQuery"),
      clear: () => setVideo(null),
    });

  return (
    <div className="space-y-6">
      <div
        className={cn(
          "space-y-4 rounded-xl border border-border bg-card p-4 transition-colors",
          dragOver && "border-primary bg-primary/5"
        )}
        // An image or a clip pasted or dropped anywhere on the form joins the query.
        onPaste={(e) => {
          const file = Array.from(e.clipboardData.files).find((f) => queryKind(f) !== null);
          if (file) {
            e.preventDefault();
            setMediaFile(file);
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
          setMediaFile(file);
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
            <Button
              type="button"
              variant="outline"
              className="h-10"
              onClick={() => fileRef.current?.click()}
              title={t("mediaHint")}
            >
              <ImagePlus className="mr-1.5 h-4 w-4" />
              {t("addMedia")}
            </Button>
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
          accept="image/*,video/mp4"
          className="hidden"
          data-testid="kb-search-media-input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) setMediaFile(file);
            e.target.value = "";
          }}
        />

        {attached.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {attached.map(({ kind, media: m, label, clear }) => (
              <div
                key={kind}
                className="flex min-w-0 max-w-full flex-1 items-center gap-3 rounded-lg border border-border bg-muted/30 p-2 sm:max-w-sm"
              >
                {kind === "video" ? (
                  <video src={m.previewUrl} muted className="h-16 w-16 rounded object-cover" />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element -- a local blob preview
                  <img src={m.previewUrl} alt="" className="h-16 w-16 rounded object-cover" />
                )}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{m.name}</p>
                  <p className="text-xs text-muted-foreground">{label}</p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  onClick={clear}
                  aria-label={t("removeMedia")}
                  title={t("removeMedia")}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{t("mediaHint")}</p>
        )}

        <div>
          <button
            type="button"
            onClick={() => setShowOptions((v) => !v)}
            className="inline-flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground"
            aria-expanded={showOptions}
          >
            {showOptions ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            {t("options")}
            {activeOptions > 0 && (
              <span className="ml-1 rounded-full bg-primary/15 px-1.5 text-xs text-primary">
                {activeOptions}
              </span>
            )}
          </button>
          {showOptions && (
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
              <div className="min-w-0 space-y-1.5">
                <Label htmlFor="kb-max-tokens">{t("maxTokens")}</Label>
                <Input
                  id="kb-max-tokens"
                  type="number"
                  min={1}
                  max={MAX_MAX_TOKENS}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-32"
                />
                <p className="text-xs text-muted-foreground">{t("maxTokensHelp")}</p>
              </div>
              <div className="min-w-0 space-y-1.5">
                <span className="text-sm font-medium leading-none">{t("budget")}</span>
                <div className="flex h-10 w-fit gap-1 rounded-lg bg-muted p-1">
                  {SEARCH_BUDGETS.map((b) => (
                    <button
                      key={b}
                      type="button"
                      aria-pressed={budget === b}
                      onClick={() => setBudget(b)}
                      className={`rounded-md px-3 text-sm font-medium transition-colors ${
                        budget === b
                          ? "bg-background shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t(b === "low" ? "budgetLow" : b === "mid" ? "budgetMid" : "budgetHigh")}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">{t("budgetHelp")}</p>
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
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">{tRecall("resultsLabel")}</span>
              <span className="font-semibold">{response.results.length}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">{tRecall("durationLabel")}</span>
              <span className="font-semibold">{(tookMs / 1000).toFixed(2)}s</span>
            </div>
            <div className="flex-1" />
            <div className="flex gap-1 bg-muted p-1 rounded-lg">
              {(["results", "json"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
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
            // One ranked list, in the server's order: the ranking already spans media types.
            <SearchResultGrid>
              {response.results.map((result, i) => (
                <SearchResultCard
                  key={`${result.document_id}-${result.index}`}
                  result={result}
                  bankId={bankId}
                  rank={i + 1}
                />
              ))}
            </SearchResultGrid>
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
