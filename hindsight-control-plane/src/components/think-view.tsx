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
import { Sparkles, Info, Tag, Clock, Database, Brain, MessageSquare, Shield } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";

type TagsMatch = "any" | "all" | "any_strict" | "all_strict";
type ViewMode = "answer" | "trace" | "json";

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

  const FEEDBACK_DIRECTIVE_NAME = "General Feedback";

  const submitFeedback = async () => {
    if (!currentBank || !feedback.trim()) return;

    setFeedbackSubmitting(true);
    try {
      // Find existing "General Feedback" directive
      const directives = await client.listDirectives(currentBank);
      const existingDirective = directives.items?.find((d) => d.name === FEEDBACK_DIRECTIVE_NAME);

      if (existingDirective) {
        // Append to existing directive description
        const newDescription = existingDirective.description
          ? `${existingDirective.description}\n${feedback.trim()}`
          : feedback.trim();
        await client.updateMentalModel(currentBank, existingDirective.id, {
          description: newDescription,
        });
      } else {
        // Create new directive with observation
        await client.createMentalModel(currentBank, {
          name: FEEDBACK_DIRECTIVE_NAME,
          description: "User feedback for improving responses",
          subtype: "directive",
          observations: [{ title: "Feedback", content: feedback.trim() }],
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

              {/* New Opinions Formed */}
              {result.new_opinions && result.new_opinions.length > 0 && (
                <Card className="border-green-200 dark:border-green-800">
                  <CardHeader className="bg-green-50 dark:bg-green-950">
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5" />
                      New Opinions Formed
                    </CardTitle>
                    <CardDescription>New beliefs generated from this interaction</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="space-y-3">
                      {result.new_opinions.map((opinion: any, i: number) => (
                        <div key={i} className="p-3 bg-muted rounded-lg border border-border">
                          <div className="font-semibold text-foreground">{opinion.text}</div>
                          <div className="text-sm text-muted-foreground mt-1">
                            Confidence: {opinion.confidence?.toFixed(2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Feedback */}
              <Card className="border-blue-200 dark:border-blue-800">
                <CardHeader className="py-4">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <MessageSquare className="w-4 h-4" />
                    Provide Feedback
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Your feedback will be saved as a directive to improve future responses
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  {feedbackSubmitted ? (
                    <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                      <span className="text-lg">&#10003;</span>
                      <span className="text-sm font-medium">
                        Feedback saved to {FEEDBACK_DIRECTIVE_NAME}
                      </span>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <Textarea
                        value={feedback}
                        onChange={(e) => setFeedback(e.target.value)}
                        placeholder="Enter your feedback here..."
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
              {/* Mental Models Created */}
              {result.mental_models_created && result.mental_models_created.length > 0 && (
                <Card className="border-emerald-200 dark:border-emerald-800">
                  <CardHeader className="bg-emerald-50 dark:bg-emerald-950 py-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <Brain className="w-4 h-4 text-emerald-600" />
                      Mental Models Created ({result.mental_models_created.length})
                    </CardTitle>
                    <CardDescription className="text-xs">
                      New mental models learned during this reflection
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-4">
                    <div className="space-y-2">
                      {result.mental_models_created.map((model: any, i: number) => (
                        <div
                          key={i}
                          className="p-3 bg-emerald-50 dark:bg-emerald-950/50 rounded-lg border border-emerald-200 dark:border-emerald-800"
                        >
                          <div className="font-medium text-sm text-emerald-900 dark:text-emerald-100">
                            {model.name}
                          </div>
                          <div className="text-xs text-emerald-700 dark:text-emerald-300 mt-1">
                            {model.description}
                          </div>
                          <div className="text-[10px] text-muted-foreground mt-2 font-mono">
                            ID: {model.id}
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
                      <div className="max-h-[500px] overflow-y-auto">
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
                                    className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 ${
                                      item.isFinal
                                        ? "bg-emerald-100 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300"
                                        : "bg-violet-100 dark:bg-violet-900 text-violet-700 dark:text-violet-300"
                                    }`}
                                  >
                                    {item.isFinal ? "✓" : item.iteration}
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
                                  <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 flex-shrink-0">
                                    ⚡
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
                        (result.based_on?.mental_models?.filter(
                          (m: any) => m.subtype !== "directive"
                        )?.length || 0) +
                        (result.trace?.mental_models?.filter((m: any) => m.subtype === "directive")
                          ?.length || 0)}{" "}
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
                        result.based_on.mental_models.length > 0) ? (
                      <div className="space-y-4 max-h-[500px] overflow-y-auto">
                        {(() => {
                          const memories = result.based_on?.memories || [];
                          const worldFacts = memories.filter((f: any) => f.type === "world");
                          const experienceFacts = memories.filter(
                            (f: any) => f.type === "experience"
                          );
                          const opinionFacts = memories.filter((f: any) => f.type === "opinion");
                          const mentalModels = (result.based_on?.mental_models || []).filter(
                            (m: any) => m.subtype !== "directive"
                          );
                          const directives =
                            result.trace?.mental_models?.filter(
                              (m: any) => m.subtype === "directive"
                            ) || [];

                          return (
                            <>
                              {/* Directives */}
                              {directives.length > 0 && (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                                    <Shield className="w-3 h-3" />
                                    Directives ({directives.length})
                                  </div>
                                  <div className="space-y-1.5">
                                    {directives.map((directive: any, i: number) => (
                                      <div key={i} className="p-2 bg-muted rounded text-xs">
                                        <div className="font-medium">{directive.name}</div>
                                        {directive.observations &&
                                          directive.observations.length > 0 && (
                                            <ul className="mt-1 space-y-0.5">
                                              {directive.observations.map(
                                                (obs: string, j: number) => (
                                                  <li
                                                    key={j}
                                                    className="text-[10px] text-muted-foreground flex items-start gap-1"
                                                  >
                                                    <span>•</span>
                                                    <span>{obs}</span>
                                                  </li>
                                                )
                                              )}
                                            </ul>
                                          )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Mental Models */}
                              {mentalModels.length > 0 && (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 text-xs font-semibold text-orange-600 dark:text-orange-400">
                                    <div className="w-2 h-2 rounded-full bg-orange-500" />
                                    Mental Models ({mentalModels.length})
                                  </div>
                                  <div className="space-y-1.5">
                                    {mentalModels.map((model: any, i: number) => (
                                      <div key={i} className="p-2 bg-muted rounded text-xs">
                                        <div className="font-medium">{model.name}</div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* World Facts */}
                              {worldFacts.length > 0 && (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 text-xs font-semibold text-blue-600 dark:text-blue-400">
                                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                                    World ({worldFacts.length})
                                  </div>
                                  <div className="space-y-1.5">
                                    {worldFacts.map((fact: any, i: number) => (
                                      <div key={i} className="p-2 bg-muted rounded text-xs">
                                        {fact.text}
                                        {fact.context && (
                                          <div className="text-[10px] text-muted-foreground mt-1">
                                            {fact.context}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Experience Facts */}
                              {experienceFacts.length > 0 && (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 text-xs font-semibold text-green-600 dark:text-green-400">
                                    <div className="w-2 h-2 rounded-full bg-green-500" />
                                    Experience ({experienceFacts.length})
                                  </div>
                                  <div className="space-y-1.5">
                                    {experienceFacts.map((fact: any, i: number) => (
                                      <div key={i} className="p-2 bg-muted rounded text-xs">
                                        {fact.text}
                                        {fact.context && (
                                          <div className="text-[10px] text-muted-foreground mt-1">
                                            {fact.context}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Opinion Facts */}
                              {opinionFacts.length > 0 && (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 text-xs font-semibold text-purple-600 dark:text-purple-400">
                                    <div className="w-2 h-2 rounded-full bg-purple-500" />
                                    Opinions ({opinionFacts.length})
                                  </div>
                                  <div className="space-y-1.5">
                                    {opinionFacts.map((fact: any, i: number) => (
                                      <div key={i} className="p-2 bg-muted rounded text-xs">
                                        {fact.text}
                                        {fact.context && (
                                          <div className="text-[10px] text-muted-foreground mt-1">
                                            {fact.context}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>
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
    </div>
  );
}
