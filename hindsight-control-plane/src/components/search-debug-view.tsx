"use client";

import { useState } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Search,
  Clock,
  Zap,
  ChevronRight,
  ChevronDown,
  Database,
  FileText,
  Users,
  ArrowDown,
  Tag,
  Calendar,
} from "lucide-react";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";
import { MemoryDetailPanel } from "./memory-detail-panel";

type FactType = "world" | "experience" | "observation";
type Budget = "low" | "mid" | "high";
type TagsMatch = "any" | "all" | "any_strict" | "all_strict";
type ViewMode = "results" | "trace" | "json";

export function SearchDebugView() {
  const { currentBank } = useBank();

  // Query state
  const [query, setQuery] = useState("");
  const [factTypes, setFactTypes] = useState<FactType[]>(["world"]);
  const [budget, setBudget] = useState<Budget>("mid");
  const [maxTokens, setMaxTokens] = useState(4096);
  const [queryDate, setQueryDate] = useState("");
  const [includeChunks, setIncludeChunks] = useState(false);
  const [includeEntities, setIncludeEntities] = useState(false);
  const [tags, setTags] = useState("");
  const [tagsMatch, setTagsMatch] = useState<TagsMatch>("any");

  // Results state
  const [results, setResults] = useState<any[] | null>(null);
  const [entities, setEntities] = useState<any[] | null>(null);
  const [chunks, setChunks] = useState<any[] | null>(null);
  const [observations, setObservations] = useState<any[] | null>(null);
  const [trace, setTrace] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("results");
  const [selectedMemory, setSelectedMemory] = useState<any | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());

  const toggleStep = (step: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(step)) {
        next.delete(step);
      } else {
        next.add(step);
      }
      return next;
    });
  };

  const toggleExpandResults = (key: string) => {
    setExpandedResults((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const INITIAL_RESULTS_COUNT = 5;

  // Helper to find full memory data from results when clicking trace items
  const selectMemoryFromTrace = (traceResult: any) => {
    const nodeId = traceResult.id || traceResult.node_id;
    // Try to find the full result with all metadata
    const fullResult = results?.find((r: any) => r.id === nodeId || r.node_id === nodeId);
    setSelectedMemory(fullResult || traceResult);
  };

  const runSearch = async () => {
    if (!currentBank) {
      alert("Please select a memory bank first");
      return;
    }

    if (!query) {
      return;
    }

    // Must select at least one type
    if (factTypes.length === 0) {
      alert("Please select at least one type (World, Experience, or Observations)");
      return;
    }

    setLoading(true);

    try {
      // Parse tags from comma-separated string
      const parsedTags = tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const requestBody: any = {
        bank_id: currentBank,
        query: query,
        types: factTypes,
        budget: budget,
        max_tokens: maxTokens,
        trace: true,
        include: {
          entities: includeEntities ? { max_tokens: 500 } : null,
          chunks: includeChunks ? { max_tokens: 8192 } : null,
        },
        ...(queryDate && { query_timestamp: queryDate }),
        ...(parsedTags.length > 0 && { tags: parsedTags, tags_match: tagsMatch }),
      };

      const data: any = await client.recall(requestBody);

      setResults(data.results || []);
      setEntities(data.entities || null);
      setChunks(data.chunks || null);
      setObservations(data.observations || null);
      setTrace(data.trace || null);
      setViewMode("results");
    } catch (error) {
      console.error("Error running search:", error);
      alert("Error running search: " + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const toggleFactType = (ft: FactType) => {
    setFactTypes((prev) => (prev.includes(ft) ? prev.filter((t) => t !== ft) : [...prev, ft]));
  };

  if (!currentBank) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-16">
          <Database className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">No Bank Selected</h3>
          <p className="text-muted-foreground">Select a memory bank to start recalling.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to recall?"
                className="pl-10 h-12 text-lg"
                onKeyDown={(e) => e.key === "Enter" && runSearch()}
              />
            </div>
            <Button onClick={runSearch} disabled={loading || !query} className="h-12 px-8">
              {loading ? "Searching..." : "Recall"}
            </Button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-6 mt-4 pt-4 border-t">
            {/* Fact Types */}
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-muted-foreground">Types:</span>
              <div className="flex gap-3">
                {(["world", "experience"] as FactType[]).map((ft) => (
                  <label key={ft} className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                      checked={factTypes.includes(ft)}
                      onCheckedChange={() => toggleFactType(ft)}
                    />
                    <span className="text-sm capitalize">{ft}</span>
                  </label>
                ))}
                <label className="flex items-center gap-2 cursor-pointer">
                  <Checkbox
                    checked={factTypes.includes("observation")}
                    onCheckedChange={() => toggleFactType("observation")}
                  />
                  <span className="text-sm">Observations</span>
                </label>
              </div>
            </div>

            <div className="h-6 w-px bg-border" />

            {/* Budget */}
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              <Select value={budget} onValueChange={(v) => setBudget(v as Budget)}>
                <SelectTrigger className="w-24 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="mid">Mid</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Max Tokens */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Tokens:</span>
              <Input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-24 h-8"
              />
            </div>

            {/* Query Date */}
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <Input
                type="datetime-local"
                value={queryDate}
                onChange={(e) => setQueryDate(e.target.value)}
                className="h-8"
                placeholder="Query date"
              />
            </div>

            <div className="h-6 w-px bg-border" />

            {/* Include options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={includeChunks}
                  onCheckedChange={(c) => setIncludeChunks(c as boolean)}
                />
                <FileText className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Chunks</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={includeEntities}
                  onCheckedChange={(c) => setIncludeEntities(c as boolean)}
                />
                <Users className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Entities</span>
              </label>
            </div>
          </div>

          {/* Tags Filter */}
          <div className="flex items-center gap-4 mt-4 pt-4 border-t">
            <Tag className="h-4 w-4 text-muted-foreground" />
            <div className="flex-1 max-w-md">
              <Input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="Filter by tags (comma-separated)"
                className="h-8"
              />
            </div>
            <Select value={tagsMatch} onValueChange={(v) => setTagsMatch(v as TagsMatch)}>
              <SelectTrigger className="w-40 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any (incl. untagged)</SelectItem>
                <SelectItem value="all">All (incl. untagged)</SelectItem>
                <SelectItem value="any_strict">Any (strict)</SelectItem>
                <SelectItem value="all_strict">All (strict)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4" />
            <p className="text-muted-foreground">Searching memories...</p>
          </CardContent>
        </Card>
      )}

      {!loading && results && (
        <div className="space-y-4">
          {/* Summary Stats */}
          {trace?.summary && (
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Results:</span>
                <span className="font-semibold">{results.length}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Duration:</span>
                <span className="font-semibold">
                  {trace.summary.total_duration_seconds?.toFixed(2)}s
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Nodes visited:</span>
                <span className="font-semibold">{trace.summary.total_nodes_visited}</span>
              </div>

              <div className="flex-1" />

              {/* View Mode Tabs */}
              <div className="flex gap-1 bg-muted p-1 rounded-lg">
                {(["results", "trace", "json"] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      viewMode === mode
                        ? "bg-background shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {mode === "results" ? "Results" : mode === "trace" ? "Trace" : "JSON"}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Results View */}
          {viewMode === "results" && (
            <div className="space-y-4">
              {/* Observations Section */}
              {observations && observations.length > 0 && (
                <Card className="border-orange-500/30 bg-orange-500/5">
                  <CardHeader className="py-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Database className="h-4 w-4 text-orange-500" />
                      <span>Observations</span>
                      <span className="text-xs text-muted-foreground">({observations.length})</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0 space-y-2">
                    {observations.map((obs: any, idx: number) => (
                      <div
                        key={obs.id || idx}
                        className="p-3 bg-background rounded-lg border border-orange-500/20"
                      >
                        <p className="text-sm text-foreground">{obs.text}</p>
                        <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                          <span className="px-2 py-0.5 rounded bg-orange-500/10 text-orange-600">
                            Observation
                          </span>
                          <span>Proof count: {obs.proof_count || 1}</span>
                          <span>Relevance: {(obs.relevance || 0).toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Memories Section */}
              <div className="space-y-3">
                {results.length === 0 && (!observations || observations.length === 0) ? (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Search className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">No memories found for this query.</p>
                    </CardContent>
                  </Card>
                ) : (
                  results.map((result: any, idx: number) => {
                    const visit = trace?.visits?.find((v: any) => v.node_id === result.id);
                    const score = visit ? visit.weights.final_weight : result.score || 0;

                    return (
                      <Card
                        key={idx}
                        className="cursor-pointer hover:border-primary/50 transition-colors"
                        onClick={() => setSelectedMemory(result)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-start gap-4">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                              <span className="text-sm font-semibold text-primary">{idx + 1}</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-foreground">{result.text}</p>
                              <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                <span className="px-2 py-0.5 rounded bg-muted capitalize">
                                  {result.type || "world"}
                                </span>
                                {result.context && (
                                  <span className="truncate max-w-xs">{result.context}</span>
                                )}
                                {result.occurred_start && (
                                  <span>
                                    {new Date(result.occurred_start).toLocaleDateString()}
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="flex-shrink-0 text-right">
                              <div className="text-sm font-semibold">{score.toFixed(3)}</div>
                              <div className="text-xs text-muted-foreground">score</div>
                            </div>
                            <ChevronRight className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })
                )}
              </div>
            </div>
          )}

          {/* Trace View */}
          {viewMode === "trace" && trace && (
            <div className="space-y-4">
              {/* Parallel Retrieval Methods - Grouped by Fact Type */}
              {trace.retrieval_results &&
                trace.retrieval_results.length > 0 &&
                (() => {
                  // Group retrieval results by fact type
                  const factTypeGroups: Record<string, any[]> = {};
                  trace.retrieval_results.forEach((method: any) => {
                    const ft = method.fact_type || "all";
                    if (!factTypeGroups[ft]) factTypeGroups[ft] = [];
                    factTypeGroups[ft].push(method);
                  });
                  const factTypes = Object.keys(factTypeGroups);

                  return (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-2">
                        <div className="flex-1 h-px bg-border" />
                        <span>PARALLEL RETRIEVAL</span>
                        <div className="flex-1 h-px bg-border" />
                      </div>

                      {/* Fact type lanes */}
                      <div className="space-y-2">
                        {factTypes.map((factType, ftIdx) => {
                          const methods = factTypeGroups[factType];
                          const laneKey = `lane-${factType}`;
                          const isLaneExpanded = expandedSteps.has(laneKey);
                          const totalResults = methods.reduce(
                            (sum: number, m: any) => sum + (m.results?.length || 0),
                            0
                          );
                          const totalDuration = Math.max(
                            ...methods.map((m: any) => m.duration_seconds || 0)
                          );

                          // Color coding for fact types
                          const ftColors: Record<
                            string,
                            { bg: string; text: string; border: string }
                          > = {
                            world: {
                              bg: "bg-blue-500/10",
                              text: "text-blue-500",
                              border: "border-blue-500/30",
                            },
                            experience: {
                              bg: "bg-green-500/10",
                              text: "text-green-500",
                              border: "border-green-500/30",
                            },
                            opinion: {
                              bg: "bg-purple-500/10",
                              text: "text-purple-500",
                              border: "border-purple-500/30",
                            },
                            all: {
                              bg: "bg-gray-500/10",
                              text: "text-gray-500",
                              border: "border-gray-500/30",
                            },
                          };
                          const colors = ftColors[factType] || ftColors.all;

                          return (
                            <Card
                              key={laneKey}
                              className={`transition-colors ${isLaneExpanded ? "border-primary" : colors.border}`}
                            >
                              <CardContent className="py-3 px-4">
                                {/* Lane Header */}
                                <div
                                  className="flex items-center gap-3 cursor-pointer"
                                  onClick={() => toggleStep(laneKey)}
                                >
                                  <div
                                    className={`w-8 h-8 rounded-lg ${colors.bg} flex items-center justify-center`}
                                  >
                                    <span className={`text-sm font-bold ${colors.text} capitalize`}>
                                      {factType.charAt(0).toUpperCase()}
                                    </span>
                                  </div>
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                      <span className="font-semibold text-foreground capitalize">
                                        {factType}
                                      </span>
                                      <span className="text-xs text-muted-foreground">
                                        {methods.length} methods
                                      </span>
                                    </div>
                                    {/* Method summary pills */}
                                    <div className="flex gap-1.5 mt-1">
                                      {methods.map((m: any, mIdx: number) => (
                                        <span
                                          key={mIdx}
                                          className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground capitalize"
                                        >
                                          {m.method_name}: {m.results?.length || 0}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-2xl font-bold text-foreground">
                                      {totalResults}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground">
                                      {totalDuration.toFixed(2)}s
                                    </div>
                                  </div>
                                  {isLaneExpanded ? (
                                    <ChevronDown className="h-5 w-5 text-muted-foreground" />
                                  ) : (
                                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                                  )}
                                </div>

                                {/* Expanded: Show methods grid */}
                                {isLaneExpanded && (
                                  <div className="mt-4 pt-4 border-t border-border">
                                    <div
                                      className={`grid gap-3 ${
                                        methods.length === 1
                                          ? "grid-cols-1"
                                          : methods.length === 2
                                            ? "grid-cols-2"
                                            : methods.length === 3
                                              ? "grid-cols-3"
                                              : "grid-cols-4"
                                      }`}
                                    >
                                      {methods.map((method: any, mIdx: number) => {
                                        const methodKey = `${laneKey}-method-${mIdx}`;
                                        const isMethodExpanded = expandedSteps.has(methodKey);
                                        const methodResults = method.results || [];

                                        return (
                                          <div key={methodKey} className="flex flex-col">
                                            <div
                                              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                                                isMethodExpanded
                                                  ? "bg-primary/10 border border-primary"
                                                  : "bg-muted/50 hover:bg-muted"
                                              }`}
                                              onClick={(e) => {
                                                e.stopPropagation();
                                                toggleStep(methodKey);
                                              }}
                                            >
                                              <div className="flex items-center justify-between mb-1">
                                                <div className="flex items-center gap-2">
                                                  <span className="font-medium text-sm text-foreground capitalize">
                                                    {method.method_name}
                                                  </span>
                                                  {/* Show temporal range inline */}
                                                  {method.method_name === "temporal" &&
                                                    method.metadata?.constraint && (
                                                      <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                                                        <Calendar className="h-3 w-3" />
                                                        {method.metadata.constraint.start
                                                          ? new Date(
                                                              method.metadata.constraint.start
                                                            ).toLocaleDateString()
                                                          : "any"}
                                                        {" → "}
                                                        {method.metadata.constraint.end
                                                          ? new Date(
                                                              method.metadata.constraint.end
                                                            ).toLocaleDateString()
                                                          : "any"}
                                                      </span>
                                                    )}
                                                </div>
                                                {isMethodExpanded ? (
                                                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                                                ) : (
                                                  <ChevronRight className="h-3 w-3 text-muted-foreground" />
                                                )}
                                              </div>
                                              <div className="flex items-end justify-between">
                                                <div className="text-2xl font-bold text-foreground">
                                                  {methodResults.length}
                                                </div>
                                                <div className="text-[10px] text-muted-foreground">
                                                  {method.duration_seconds?.toFixed(2)}s
                                                </div>
                                              </div>
                                            </div>

                                            {/* Method Results */}
                                            {isMethodExpanded &&
                                              methodResults.length > 0 &&
                                              (() => {
                                                const resultsKey = `results-${methodKey}`;
                                                const showAll = expandedResults.has(resultsKey);
                                                const displayResults = showAll
                                                  ? methodResults
                                                  : methodResults.slice(0, INITIAL_RESULTS_COUNT);
                                                const hasMore =
                                                  methodResults.length > INITIAL_RESULTS_COUNT;

                                                return (
                                                  <div className="mt-2 space-y-1.5 max-h-[300px] overflow-y-auto">
                                                    {displayResults.map((r: any, rIdx: number) => (
                                                      <div
                                                        key={rIdx}
                                                        className="p-2 bg-background rounded cursor-pointer hover:bg-muted/50 transition-colors border border-border"
                                                        onClick={(e) => {
                                                          e.stopPropagation();
                                                          selectMemoryFromTrace(r);
                                                        }}
                                                      >
                                                        <div className="flex items-start gap-2">
                                                          <span className="text-[10px] font-mono text-muted-foreground mt-0.5">
                                                            {rIdx + 1}
                                                          </span>
                                                          <div className="flex-1 min-w-0">
                                                            <p className="text-xs text-foreground line-clamp-2">
                                                              {r.text}
                                                            </p>
                                                            <div className="flex items-center gap-2 mt-1">
                                                              <span className="text-[10px] text-muted-foreground">
                                                                {(
                                                                  r.score ||
                                                                  r.similarity ||
                                                                  0
                                                                ).toFixed(4)}
                                                              </span>
                                                            </div>
                                                          </div>
                                                        </div>
                                                      </div>
                                                    ))}
                                                    {hasMore && (
                                                      <button
                                                        className="w-full text-[10px] text-primary hover:text-primary/80 py-1.5 hover:bg-muted/50 rounded transition-colors"
                                                        onClick={(e) => {
                                                          e.stopPropagation();
                                                          toggleExpandResults(resultsKey);
                                                        }}
                                                      >
                                                        {showAll
                                                          ? `Show less`
                                                          : `View all ${methodResults.length} results`}
                                                      </button>
                                                    )}
                                                  </div>
                                                );
                                              })()}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          );
                        })}
                      </div>

                      {/* Parallel indicator - vertical lines showing all run together */}
                      <div className="flex justify-center py-2">
                        <div className="flex items-center gap-2">
                          {factTypes.map((ft, i) => {
                            const ftColors: Record<string, string> = {
                              world: "bg-blue-500",
                              experience: "bg-green-500",
                              opinion: "bg-purple-500",
                              all: "bg-gray-500",
                            };
                            return (
                              <div key={i} className="flex flex-col items-center">
                                <div
                                  className={`w-1 h-4 ${ftColors[ft] || ftColors.all} rounded-full opacity-50`}
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      <div className="flex justify-center">
                        <ArrowDown className="h-5 w-5 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Step 2: RRF Merge */}
              {trace.rrf_merged &&
                (() => {
                  const stepKey = "rrf-merge";
                  const isExpanded = expandedSteps.has(stepKey);

                  return (
                    <div>
                      <Card
                        className={`cursor-pointer transition-colors ${isExpanded ? "border-primary" : "hover:border-primary/50"}`}
                        onClick={() => toggleStep(stepKey)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                              <span className="text-sm font-bold text-purple-500">∪</span>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-foreground">RRF Fusion</span>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                                  merge
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground mt-0.5">
                                Reciprocal Rank Fusion of all retrieval results
                              </div>
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                              {trace.rrf_merged.length}
                            </div>
                            {isExpanded ? (
                              <ChevronDown className="h-5 w-5 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-5 w-5 text-muted-foreground" />
                            )}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Expanded Results */}
                      {isExpanded &&
                        trace.rrf_merged.length > 0 &&
                        (() => {
                          const resultsKey = "results-rrf";
                          const showAll = expandedResults.has(resultsKey);
                          const displayResults = showAll
                            ? trace.rrf_merged
                            : trace.rrf_merged.slice(0, INITIAL_RESULTS_COUNT);
                          const hasMore = trace.rrf_merged.length > INITIAL_RESULTS_COUNT;

                          return (
                            <div className="ml-6 mt-2 space-y-2 border-l-2 border-muted pl-4 max-h-[400px] overflow-y-auto">
                              {displayResults.map((r: any, rIdx: number) => (
                                <div
                                  key={rIdx}
                                  className="p-3 bg-muted/30 rounded-lg cursor-pointer hover:bg-muted/50 transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    selectMemoryFromTrace(r);
                                  }}
                                >
                                  <div className="flex items-start gap-3">
                                    <span className="text-xs font-mono text-muted-foreground">
                                      {rIdx + 1}
                                    </span>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm text-foreground line-clamp-2">
                                        {r.text}
                                      </p>
                                      <div className="text-xs text-muted-foreground mt-1">
                                        RRF Score: {(r.rrf_score || r.score || 0).toFixed(4)}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                              {hasMore && (
                                <button
                                  className="w-full text-xs text-primary hover:text-primary/80 py-2 hover:bg-muted/50 rounded transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandResults(resultsKey);
                                  }}
                                >
                                  {showAll
                                    ? `Show less`
                                    : `View all ${trace.rrf_merged.length} results`}
                                </button>
                              )}
                            </div>
                          );
                        })()}

                      {/* Arrow */}
                      <div className="flex justify-center py-2">
                        <ArrowDown className="h-4 w-4 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Step 3: Combined Scoring */}
              {trace.reranked &&
                (() => {
                  const stepKey = "reranking";
                  const isExpanded = expandedSteps.has(stepKey);

                  return (
                    <div>
                      <Card
                        className={`cursor-pointer transition-colors ${isExpanded ? "border-primary" : "hover:border-primary/50"}`}
                        onClick={() => toggleStep(stepKey)}
                      >
                        <CardContent className="py-4">
                          <div className="flex items-center gap-4">
                            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-amber-500/10 flex items-center justify-center">
                              <span className="text-sm font-bold text-amber-500">⚡</span>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-foreground">
                                  Combined Scoring
                                </span>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                                  rerank
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground mt-0.5">
                                <span className="font-mono text-xs">
                                  0.6×cross_encoder + 0.2×rrf + 0.1×temporal + 0.1×recency
                                </span>
                              </div>
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                              {trace.reranked.length}
                            </div>
                            {isExpanded ? (
                              <ChevronDown className="h-5 w-5 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-5 w-5 text-muted-foreground" />
                            )}
                          </div>
                        </CardContent>
                      </Card>

                      {/* Expanded Results */}
                      {isExpanded &&
                        trace.reranked.length > 0 &&
                        (() => {
                          const resultsKey = "results-rerank";
                          const showAll = expandedResults.has(resultsKey);
                          const displayResults = showAll
                            ? trace.reranked
                            : trace.reranked.slice(0, INITIAL_RESULTS_COUNT);
                          const hasMore = trace.reranked.length > INITIAL_RESULTS_COUNT;

                          return (
                            <div className="ml-6 mt-2 space-y-2 border-l-2 border-muted pl-4 max-h-[400px] overflow-y-auto">
                              {displayResults.map((r: any, rIdx: number) => {
                                const sc = r.score_components || {};
                                return (
                                  <div
                                    key={rIdx}
                                    className="p-3 bg-muted/30 rounded-lg cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      selectMemoryFromTrace(r);
                                    }}
                                  >
                                    <div className="flex items-start gap-3">
                                      <span className="text-xs font-mono text-muted-foreground">
                                        {rIdx + 1}
                                      </span>
                                      <div className="flex-1 min-w-0">
                                        <p className="text-sm text-foreground line-clamp-2">
                                          {r.text}
                                        </p>
                                        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px] text-muted-foreground font-mono">
                                          <span className="font-semibold text-foreground">
                                            = {(r.rerank_score || r.score || 0).toFixed(4)}
                                          </span>
                                          {sc.cross_encoder_score_normalized !== undefined && (
                                            <span title="Cross-encoder (60%)">
                                              CE: {sc.cross_encoder_score_normalized.toFixed(3)}
                                            </span>
                                          )}
                                          {sc.rrf_normalized !== undefined && (
                                            <span
                                              title={`RRF normalized (20%) - raw: ${sc.rrf_score?.toFixed(4) || "N/A"}`}
                                            >
                                              RRF: {sc.rrf_normalized.toFixed(3)}
                                            </span>
                                          )}
                                          {sc.temporal !== undefined && (
                                            <span title="Temporal proximity (10%)">
                                              Tmp: {sc.temporal.toFixed(3)}
                                            </span>
                                          )}
                                          {sc.recency !== undefined && (
                                            <span title="Recency (10%)">
                                              Rec: {sc.recency.toFixed(3)}
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                              {hasMore && (
                                <button
                                  className="w-full text-xs text-primary hover:text-primary/80 py-2 hover:bg-muted/50 rounded transition-colors"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandResults(resultsKey);
                                  }}
                                >
                                  {showAll
                                    ? `Show less`
                                    : `View all ${trace.reranked.length} results`}
                                </button>
                              )}
                            </div>
                          );
                        })()}

                      {/* Arrow */}
                      <div className="flex justify-center py-2">
                        <ArrowDown className="h-4 w-4 text-muted-foreground/50" />
                      </div>
                    </div>
                  );
                })()}

              {/* Final: Results */}
              <Card className="border-primary bg-primary/5">
                <CardContent className="py-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                      <span className="text-sm font-bold text-primary">✓</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-foreground">Final Results</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                          output
                        </span>
                      </div>
                      <div className="text-sm text-muted-foreground mt-0.5">
                        Top results after all processing steps
                      </div>
                    </div>
                    <div className="text-2xl font-bold text-primary">{results?.length || 0}</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* JSON View */}
          {viewMode === "json" && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Raw Response</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted p-4 rounded-lg overflow-auto max-h-[600px]">
                  <JsonView
                    src={{
                      results,
                      ...(entities && { entities }),
                      ...(chunks && { chunks }),
                      ...(observations && { observations }),
                      trace,
                    }}
                    collapsed={2}
                    theme="default"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Empty State */}
      {!loading && !results && (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Search className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Ready to Recall</h3>
            <p className="text-muted-foreground text-center max-w-md">
              Enter a query above to search through your memories. Use filters to narrow down by
              fact type, budget, and more.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Memory Detail Panel */}
      {selectedMemory && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l shadow-2xl z-50 overflow-y-auto">
          <MemoryDetailPanel
            memory={selectedMemory}
            onClose={() => setSelectedMemory(null)}
            inPanel
            bankId={currentBank || undefined}
          />
        </div>
      )}
    </div>
  );
}
