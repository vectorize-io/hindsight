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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Sparkles,
  Info,
  Tag,
  Clock,
  Database,
  Brain,
  MessageSquare,
  Shield,
  X,
  Check,
  Play,
} from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";
import { MemoryDetailModal } from "./memory-detail-modal";
import { MentalModelDetailModal } from "./mental-model-detail-modal";

type TagsMatch = "any" | "all" | "any_strict" | "all_strict";
type ViewMode = "answer" | "trace" | "json";
type BasedOnTab = "directives" | "mental_models" | "observations" | "world" | "experience";

export function ThinkView() {
  const { currentBank } = useBank();
  const [query, setQuery] = useState("");
  const [budget, setBudget] = useState<"low" | "mid" | "high">("mid");
  const [maxTokens, setMaxTokens] = useState<number>(4096);
  const [includeFacts, setIncludeFacts] = useState(true);
  const [includeToolCalls, setIncludeToolCalls] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("answer");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [tags, setTags] = useState("");
  const [tagsMatch, setTagsMatch] = useState<TagsMatch>("any");
  const [feedback, setFeedback] = useState("");
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [selectedDirective, setSelectedDirective] = useState<any | null>(null);
  const [fullDirective, setFullDirective] = useState<any | null>(null);
  const [loadingDirective, setLoadingDirective] = useState(false);
  const [selectedObservation, setSelectedObservation] = useState<any | null>(null);
  const [fullObservation, setFullObservation] = useState<any | null>(null);
  const [loadingObservation, setLoadingObservation] = useState(false);
  const [selectedMentalModelId, setSelectedMentalModelId] = useState<string | null>(null);
  const [activeBasedOnTab, setActiveBasedOnTab] = useState<BasedOnTab>("world");

  const FEEDBACK_DIRECTIVE_NAME = "General Feedback";

  // Load full directive data when one is selected
  const handleSelectDirective = async (directive: any) => {
    setSelectedDirective(directive);
    setFullDirective(null);
    if (!currentBank || !directive?.id) return;

    setLoadingDirective(true);
    try {
      const directives = await client.listDirectives(currentBank);
      const fullDir = directives.items?.find((d: any) => d.id === directive.id);
      setFullDirective(fullDir || directive);
    } catch (error) {
      console.error("Failed to load directive:", error);
      setFullDirective(directive); // Fall back to partial data
    } finally {
      setLoadingDirective(false);
    }
  };

  // Load full observation data when one is selected
  const handleSelectObservation = async (observation: any) => {
    setSelectedObservation(observation);
    setFullObservation(null);
    if (!currentBank || !observation?.id) return;

    setLoadingObservation(true);
    try {
      const observations = await client.listObservations(currentBank);
      const fullObs = observations.items?.find((o: any) => o.id === observation.id);
      setFullObservation(fullObs || observation);
    } catch (error) {
      console.error("Failed to load observation:", error);
      setFullObservation(observation); // Fall back to partial data
    } finally {
      setLoadingObservation(false);
    }
  };

  const submitFeedback = async () => {
    if (!currentBank || !feedback.trim()) return;

    setFeedbackSubmitting(true);
    try {
      // Find existing "General Feedback" directive
      const directives = await client.listDirectives(currentBank);
      const existingDirective = directives.items?.find((d) => d.name === FEEDBACK_DIRECTIVE_NAME);

      if (existingDirective) {
        // Append to existing directive content
        const newContent = existingDirective.content
          ? `${existingDirective.content}\n${feedback.trim()}`
          : feedback.trim();
        await client.updateDirective(currentBank, existingDirective.id, {
          content: newContent,
        });
      } else {
        // Create new directive
        await client.createDirective(currentBank, {
          name: FEEDBACK_DIRECTIVE_NAME,
          content: feedback.trim(),
        });
      }

      setFeedback("");
      setFeedbackSubmitted(true);
      setTimeout(() => setFeedbackSubmitted(false), 3000);
    } catch (error) {
      console.error("Error submitting feedback:", error);
      alert("Error submitting feedback: " + (error as Error).message);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  const runReflect = async () => {
    if (!currentBank || !query) return;

    setLoading(true);
    setViewMode("answer");
    try {
      // Parse tags from comma-separated string
      const parsedTags = tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const data: any = await client.reflect({
        bank_id: currentBank,
        query,
        budget,
        max_tokens: maxTokens,
        include_facts: includeFacts,
        include_tool_calls: includeToolCalls,
        ...(parsedTags.length > 0 && { tags: parsedTags, tags_match: tagsMatch }),
      });
      setResult(data);
    } catch (error) {
      console.error("Error running reflect:", error);
      alert("Error running reflect: " + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  if (!currentBank) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-16">
          <Database className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">No Bank Selected</h3>
          <p className="text-muted-foreground">Select a memory bank to start reflecting.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Query Input */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Sparkles className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to reflect on?"
                className="pl-10 h-12 text-lg"
                onKeyDown={(e) => e.key === "Enter" && runReflect()}
              />
            </div>
            <Button onClick={runReflect} disabled={loading || !query} className="h-12 px-8">
              {loading ? "Reflecting..." : "Reflect"}
            </Button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-6 mt-4 pt-4 border-t">
            {/* Budget */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-muted-foreground">Budget:</span>
              <Select value={budget} onValueChange={(value: any) => setBudget(value)}>
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
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 4096)}
                className="w-24 h-8"
              />
            </div>

            <div className="h-6 w-px bg-border" />

            {/* Include options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={includeFacts}
                  onCheckedChange={(c) => setIncludeFacts(c as boolean)}
                />
                <span className="text-sm">Include Source</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={includeToolCalls}
                  onCheckedChange={(c) => setIncludeToolCalls(c as boolean)}
                />
                <span className="text-sm">Include Tools</span>
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

      {/* Loading State */}
      {loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4" />
            <p className="text-muted-foreground">Reflecting on memories...</p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {!loading && result && (
        <div className="space-y-4">
          {/* Summary Stats & Tabs */}
          <div className="flex items-center gap-6 text-sm">
            {result.usage && (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Input tokens:</span>
                  <span className="font-semibold">
                    {result.usage.input_tokens?.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Output tokens:</span>
                  <span className="font-semibold">
                    {result.usage.output_tokens?.toLocaleString()}
                  </span>
                </div>
              </>
            )}
            {result.trace?.tool_calls && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Tool calls:</span>
                <span className="font-semibold">{result.trace.tool_calls.length}</span>
                <span className="text-muted-foreground">
                  (
                  {result.trace.tool_calls.reduce(
                    (sum: number, tc: any) => sum + tc.duration_ms,
                    0
                  )}
                  ms)
                </span>
              </div>
            )}
            {result.trace?.llm_calls && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">LLM calls:</span>
                <span className="font-semibold">{result.trace.llm_calls.length}</span>
                <span className="text-muted-foreground">
                  (
                  {result.trace.llm_calls.reduce((sum: number, lc: any) => sum + lc.duration_ms, 0)}
                  ms)
                </span>
              </div>
            )}

            <div className="flex-1" />

            {/* View Mode Tabs */}
            <div className="flex gap-1 bg-muted p-1 rounded-lg">
              {(["answer", "trace", "json"] as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    viewMode === mode
                      ? "bg-background shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {mode === "answer" ? "Answer" : mode === "trace" ? "Trace" : "JSON"}
                </button>
              ))}
            </div>
          </div>

          {/* Answer View */}
          {viewMode === "answer" && (
            <div className="space-y-6">
              {/* Main Answer */}
              <Card>
                <CardHeader>
                  <CardTitle>Answer</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-base leading-relaxed whitespace-pre-wrap">{result.text}</div>
                </CardContent>
              </Card>

              {/* Directive */}
              <Card className="border-blue-200 dark:border-blue-800">
                <CardHeader className="py-4">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <MessageSquare className="w-4 h-4" />
                    Add Directive
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Hard rules injected into prompts that the agent must follow
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  {feedbackSubmitted ? (
                    <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                      <span className="text-lg">&#10003;</span>
                      <span className="text-sm font-medium">
                        Directive saved to {FEEDBACK_DIRECTIVE_NAME}
                      </span>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <Textarea
                        value={feedback}
                        onChange={(e) => setFeedback(e.target.value)}
                        placeholder="e.g., Always respond in formal English..."
                        className="flex-1 min-h-[60px] resize-none"
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                            submitFeedback();
                          }
                        }}
                      />
                      <Button
                        onClick={submitFeedback}
                        disabled={feedbackSubmitting || !feedback.trim()}
                        className="self-end"
                      >
                        {feedbackSubmitting ? "Saving..." : "Save"}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Trace View - Split Layout */}
          {viewMode === "trace" && (
            <div className="space-y-4">
              {/* Observations Created */}
              {result.observations_created && result.observations_created.length > 0 && (
                <Card className="border-emerald-200 dark:border-emerald-800">
                  <CardHeader className="bg-emerald-50 dark:bg-emerald-950 py-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Brain className="w-4 h-4 text-emerald-600" />
                      Observations Created ({result.observations_created.length})
                    </CardTitle>
                    <CardDescription className="text-xs">
                      New observations learned during this reflection
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-4">
                    <div className="space-y-2">
                      {result.observations_created.map((obs: any, i: number) => (
                        <div
                          key={i}
                          className="p-3 bg-emerald-50 dark:bg-emerald-950/50 rounded-lg border border-emerald-200 dark:border-emerald-800"
                        >
                          <div className="font-medium text-sm text-emerald-900 dark:text-emerald-100">
                            {obs.name}
                          </div>
                          <div className="text-xs text-emerald-700 dark:text-emerald-300 mt-1">
                            {obs.description}
                          </div>
                          <div className="text-[10px] text-muted-foreground mt-2 font-mono">
                            ID: {obs.id}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Left: Execution Trace (LLM + Tool Calls) */}
                <Card className="h-fit">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Execution Trace</CardTitle>
                    <CardDescription className="text-xs">
                      {result.iterations || 0} iteration
                      {(result.iterations || 0) !== 1 ? "s" : ""} •{" "}
                      {(result.trace?.llm_calls?.reduce(
                        (sum: number, lc: any) => sum + lc.duration_ms,
                        0
                      ) || 0) +
                        (result.trace?.tool_calls?.reduce(
                          (sum: number, tc: any) => sum + tc.duration_ms,
                          0
                        ) || 0)}
                      ms total
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!includeToolCalls ? (
                      <div className="flex items-start gap-3 p-3 bg-muted border border-border rounded-lg">
                        <Info className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-sm text-foreground">Not included</p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            Enable "Include Tool Calls" to see trace.
                          </p>
                        </div>
                      </div>
                    ) : (result.trace?.llm_calls && result.trace.llm_calls.length > 0) ||
                      (result.trace?.tool_calls && result.trace.tool_calls.length > 0) ? (
                      <div className="max-h-[500px] overflow-y-auto pr-2">
                        {/* Build timeline: LLM -> Tools -> LLM -> Tools */}
                        {(() => {
                          const llmCalls = result.trace?.llm_calls || [];
                          const toolCalls = result.trace?.tool_calls || [];

                          // Build interleaved timeline
                          const timeline: Array<{
                            type: "llm" | "tools";
                            llm?: any;
                            tools?: any[];
                            iteration: number;
                            isFinal?: boolean;
                          }> = [];

                          llmCalls.forEach((lc: any, idx: number) => {
                            // Add tools for this iteration (using iteration field from tool trace)
                            const iterTools = toolCalls.filter(
                              (tc: any) => tc.iteration === idx + 1
                            );
                            // Determine if this is the final LLM call:
                            // - scope includes "final", OR
                            // - it's the last LLM call AND no tools were called after it
                            const isLastLLMCall = idx === llmCalls.length - 1;
                            const isFinal =
                              lc.scope.includes("final") ||
                              (isLastLLMCall && iterTools.length === 0);
                            const iterNum = isFinal ? llmCalls.length : idx + 1;

                            // Add LLM call
                            timeline.push({
                              type: "llm",
                              llm: lc,
                              iteration: iterNum,
                              isFinal,
                            });

                            if (iterTools.length > 0) {
                              timeline.push({
                                type: "tools",
                                tools: iterTools,
                                iteration: idx + 1,
                              });
                            }
                          });

                          return timeline.map((item, idx) => (
                            <div key={idx} className="relative">
                              {/* Timeline connector */}
                              {idx < timeline.length - 1 && (
                                <div className="absolute left-3 top-6 bottom-0 w-0.5 bg-border" />
                              )}

                              {item.type === "llm" ? (
                                // LLM Call
                                <div className="flex items-start gap-3 pb-3">
                                  <div
                                    className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                                      item.isFinal
                                        ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                        : "bg-primary/10 text-primary"
                                    }`}
                                  >
                                    {item.isFinal ? (
                                      <Check className="w-3.5 h-3.5" strokeWidth={2.5} />
                                    ) : (
                                      <span className="text-[10px] font-semibold">
                                        {item.iteration}
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                      <span className="font-medium text-sm">
                                        {item.isFinal ? "Response generated" : "Agent decided"}
                                      </span>
                                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        {item.llm.duration_ms}ms
                                      </span>
                                    </div>
                                    <span className="text-xs text-muted-foreground">
                                      {item.isFinal ? "Final answer" : "Called tools below"}
                                    </span>
                                  </div>
                                </div>
                              ) : (
                                // Tool Calls
                                <div className="flex items-start gap-3 pb-3">
                                  <div className="w-6 h-6 rounded-full flex items-center justify-center bg-blue-500/15 text-blue-600 dark:text-blue-400 flex-shrink-0">
                                    <Play className="w-3 h-3" fill="currentColor" />
                                  </div>
                                  <div className="flex-1 min-w-0 space-y-2">
                                    <div className="text-xs text-muted-foreground">
                                      Executing {item.tools?.length} tool
                                      {item.tools?.length !== 1 ? "s" : ""}
                                    </div>
                                    {item.tools?.map((tc: any, tcIdx: number) => (
                                      <div
                                        key={tcIdx}
                                        className="border border-border rounded-lg overflow-hidden"
                                      >
                                        <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50">
                                          <span className="font-medium text-sm text-foreground">
                                            {tc.tool}
                                          </span>
                                          <span className="text-xs text-muted-foreground flex items-center gap-1">
                                            <Clock className="w-3 h-3" />
                                            {tc.duration_ms}ms
                                          </span>
                                        </div>
                                        <div className="p-2 space-y-2">
                                          <div>
                                            <p className="text-[10px] font-semibold text-muted-foreground mb-1">
                                              Input:
                                            </p>
                                            <div className="bg-muted p-1.5 rounded text-xs overflow-auto max-h-32">
                                              <JsonView
                                                src={tc.input}
                                                collapsed={1}
                                                theme="default"
                                              />
                                            </div>
                                          </div>
                                          {tc.output && (
                                            <div>
                                              <p className="text-[10px] font-semibold text-muted-foreground mb-1">
                                                Output:
                                              </p>
                                              <div className="bg-muted p-1.5 rounded text-xs overflow-auto max-h-32">
                                                <JsonView
                                                  src={tc.output}
                                                  collapsed={1}
                                                  theme="default"
                                                />
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ));
                        })()}
                      </div>
                    ) : (
                      <div className="flex items-start gap-3 p-3 bg-muted border border-border rounded-lg">
                        <Info className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-sm text-foreground">No operations</p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            No LLM or tool calls were made during this reflection.
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Right: Based On Facts */}
                <Card className="h-fit">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Based On</CardTitle>
                    <CardDescription className="text-xs">
                      {(result.based_on?.memories?.length || 0) +
                        (result.based_on?.observations?.filter(
                          (o: any) => o.subtype !== "directive"
                        )?.length || 0) +
                        (result.based_on?.directives?.length || 0)}{" "}
                      items used
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!includeFacts ? (
                      <div className="flex items-start gap-3 p-3 bg-muted border border-border rounded-lg">
                        <Info className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-sm text-foreground">Not included</p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            Enable "Include Source" to see memories.
                          </p>
                        </div>
                      </div>
                    ) : (result.based_on?.memories && result.based_on.memories.length > 0) ||
                      (result.based_on?.mental_models &&
                        result.based_on.mental_models.length > 0) ||
                      (result.based_on?.directives && result.based_on.directives.length > 0) ||
                      (result.based_on?.observations && result.based_on.observations.length > 0) ? (
                      (() => {
                        const memories = result.based_on?.memories || [];
                        const worldFacts = memories.filter((f: any) => f.type === "world");
                        const experienceFacts = memories.filter(
                          (f: any) => f.type === "experience"
                        );
                        // Mental models are in based_on.mental_models
                        const mentalModelFacts = result.based_on?.mental_models || [];
                        const observations = (result.based_on?.observations || []).filter(
                          (o: any) => o.subtype !== "directive"
                        );
                        // Directives are in based_on.directives
                        const directives = result.based_on?.directives || [];

                        // Build tabs array with all categories
                        const tabs: { id: BasedOnTab; label: string; count: number }[] = [
                          { id: "directives", label: "Directives", count: directives.length },
                          {
                            id: "mental_models",
                            label: "Mental Models",
                            count: mentalModelFacts.length,
                          },
                          { id: "observations", label: "Observations", count: observations.length },
                          { id: "world", label: "World", count: worldFacts.length },
                          { id: "experience", label: "Experience", count: experienceFacts.length },
                        ];

                        const currentTab = activeBasedOnTab;

                        const getCurrentFacts = () => {
                          switch (currentTab) {
                            case "directives":
                              return directives;
                            case "mental_models":
                              return mentalModelFacts;
                            case "observations":
                              return observations;
                            case "world":
                              return worldFacts;
                            case "experience":
                              return experienceFacts;
                            default:
                              return [];
                          }
                        };

                        const currentFacts = getCurrentFacts();

                        return (
                          <div>
                            {/* Tabs */}
                            <div className="flex items-center gap-1 bg-muted rounded-lg p-1 mb-4">
                              {tabs.map((tab) => (
                                <button
                                  key={tab.id}
                                  onClick={() => setActiveBasedOnTab(tab.id)}
                                  className={`flex-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                                    currentTab === tab.id
                                      ? "bg-background text-foreground shadow-sm"
                                      : "text-muted-foreground hover:text-foreground"
                                  }`}
                                >
                                  {tab.label} ({tab.count})
                                </button>
                              ))}
                            </div>

                            {/* Tab Content */}
                            {currentFacts.length > 0 ? (
                              <div className="max-h-[400px] overflow-y-auto pr-2 space-y-3">
                                {currentFacts.map((item: any, i: number) => (
                                  <div
                                    key={item.id || i}
                                    className={`p-4 bg-muted/50 rounded-lg border border-border/50 ${
                                      currentTab !== "directives"
                                        ? "cursor-pointer hover:bg-muted/80 transition-colors"
                                        : ""
                                    }`}
                                    onClick={() => {
                                      if (currentTab === "directives") return; // Not clickable
                                      if (currentTab === "observations")
                                        handleSelectObservation(item);
                                      else if (currentTab === "mental_models")
                                        setSelectedMentalModelId(item.id);
                                      else setSelectedMemoryId(item.id);
                                    }}
                                  >
                                    {currentTab === "directives" ? (
                                      <>
                                        <div className="font-medium text-sm">{item.name}</div>
                                        {item.content && (
                                          <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                                            {item.content}
                                          </p>
                                        )}
                                      </>
                                    ) : currentTab === "observations" ? (
                                      <div className="font-medium text-sm">{item.name}</div>
                                    ) : currentTab === "mental_models" ? (
                                      (() => {
                                        const colonIdx = item.text?.indexOf(": ") ?? -1;
                                        const name =
                                          colonIdx > 0 ? item.text.slice(0, colonIdx) : item.id;
                                        return (
                                          <>
                                            <div className="font-medium text-sm">{name}</div>
                                            <code className="text-xs font-mono text-muted-foreground">
                                              {item.id}
                                            </code>
                                          </>
                                        );
                                      })()
                                    ) : (
                                      <>
                                        <p className="text-sm text-foreground leading-relaxed">
                                          {item.text}
                                        </p>
                                        {item.context && (
                                          <div className="text-xs text-muted-foreground mt-2">
                                            {item.context}
                                          </div>
                                        )}
                                      </>
                                    )}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-sm text-muted-foreground text-center py-4">
                                No {currentTab} items
                              </p>
                            )}
                          </div>
                        );
                      })()
                    ) : (
                      <div className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg">
                        <Info className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-sm text-amber-900 dark:text-amber-100">
                            No facts found
                          </p>
                          <p className="text-xs text-amber-700 dark:text-amber-300 mt-0.5">
                            No memories were used to generate this answer.
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
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
                  <JsonView src={result} collapsed={2} theme="default" />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Empty State */}
      {!loading && !result && (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Sparkles className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Ready to Reflect</h3>
            <p className="text-muted-foreground text-center max-w-md">
              Enter a question above to query the memory bank and generate a disposition-aware
              response.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Memory Detail Modal */}
      <MemoryDetailModal memoryId={selectedMemoryId} onClose={() => setSelectedMemoryId(null)} />

      {/* Directive Detail Panel */}
      {selectedDirective && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l shadow-2xl z-50 overflow-y-auto">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                <h2 className="text-lg font-semibold">Directive</h2>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setSelectedDirective(null);
                  setFullDirective(null);
                }}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            {loadingDirective ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground">Name</h3>
                  <p className="mt-1 font-medium">
                    {fullDirective?.name || selectedDirective.name}
                  </p>
                </div>
                {fullDirective?.description && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground">Description</h3>
                    <p className="mt-1 text-sm">{fullDirective.description}</p>
                  </div>
                )}
                {fullDirective?.tags && fullDirective.tags.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-1">Tags</h3>
                    <div className="flex flex-wrap gap-1">
                      {fullDirective.tags.map((tag: string) => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground flex items-center gap-1"
                        >
                          <Tag className="w-2.5 h-2.5" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {/* Show content from directive */}
                {(fullDirective?.content || selectedDirective.content) && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Content</h3>
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {fullDirective?.content || selectedDirective.content}
                      </div>
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t">
                  <h3 className="text-sm font-medium text-muted-foreground">ID</h3>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {selectedDirective.id}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Observation Detail Panel */}
      {selectedObservation && (
        <div className="fixed right-0 top-0 h-screen w-[420px] bg-card border-l shadow-2xl z-50 overflow-y-auto">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                <h2 className="text-lg font-semibold">Observation</h2>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setSelectedObservation(null);
                  setFullObservation(null);
                }}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            {loadingObservation ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground">Text</h3>
                  <p className="mt-1 font-medium">
                    {fullObservation?.text || selectedObservation.text}
                  </p>
                </div>
                {fullObservation?.tags && fullObservation.tags.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-1">Tags</h3>
                    <div className="flex flex-wrap gap-1">
                      {fullObservation.tags.map((tag: string) => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground flex items-center gap-1"
                        >
                          <Tag className="w-2.5 h-2.5" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {fullObservation?.source_memories && fullObservation.source_memories.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">
                      Source Memories ({fullObservation.source_memories.length})
                    </h3>
                    <div className="space-y-2">
                      {fullObservation.source_memories.map((mem: any, i: number) => (
                        <div key={i} className="p-3 bg-muted rounded-lg">
                          <div className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {mem.text || (typeof mem === "string" ? mem : "")}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t">
                  <h3 className="text-sm font-medium text-muted-foreground">ID</h3>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {selectedObservation.id}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mental Model Detail Modal */}
      <MentalModelDetailModal
        mentalModelId={selectedMentalModelId}
        onClose={() => setSelectedMentalModelId(null)}
      />
    </div>
  );
}
