"use client";

import { useState } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles, Info, Tag } from "lucide-react";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";

type TagsMatch = "any" | "all" | "any_strict" | "all_strict";

export function ThinkView() {
  const { currentBank } = useBank();
  const [query, setQuery] = useState("");
  const [context, setContext] = useState("");
  const [budget, setBudget] = useState<"low" | "mid" | "high">("mid");
  const [includeFacts, setIncludeFacts] = useState(true);
  const [showRawJson, setShowRawJson] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [tags, setTags] = useState("");
  const [tagsMatch, setTagsMatch] = useState<TagsMatch>("any");

  const runReflect = async () => {
    if (!currentBank || !query) return;

    setLoading(true);
    setShowRawJson(false);
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
        context: context || undefined,
        include_facts: includeFacts,
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

  return (
    <div className="max-w-6xl">
      <Card>
        <CardContent className="p-5 space-y-4">
          <div className="flex gap-4 items-end flex-wrap">
            <div className="flex-1 min-w-[300px]">
              <label className="font-bold block mb-2 text-card-foreground">Question:</label>
              <Input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your question..."
                onKeyDown={(e) => e.key === "Enter" && runReflect()}
              />
            </div>
            <div>
              <label className="font-bold block mb-2 text-card-foreground">Budget:</label>
              <Select value={budget} onValueChange={(value: any) => setBudget(value)}>
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="mid">Mid</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="include-facts"
                checked={includeFacts}
                onCheckedChange={(checked) => setIncludeFacts(checked as boolean)}
              />
              <label htmlFor="include-facts" className="text-sm cursor-pointer">
                Include Facts
              </label>
            </div>
            <Button onClick={runReflect} disabled={loading || !query}>
              <Sparkles className="w-4 h-4 mr-2" />
              Reflect
            </Button>
          </div>
          <div>
            <label className="font-bold block mb-2 text-card-foreground">Context (optional):</label>
            <Textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="Additional context for the LLM (not used in search)..."
              rows={3}
            />
          </div>
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

      {loading && (
        <Card className="mt-6">
          <CardContent className="text-center py-10">
            <Sparkles className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
            <div className="text-lg text-muted-foreground">Reflecting...</div>
          </CardContent>
        </Card>
      )}

      {result && !loading && (
        <div className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Answer</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-muted rounded-lg border-l-4 border-primary text-base leading-relaxed whitespace-pre-wrap">
                {result.text}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Details</CardTitle>
                  <CardDescription>View facts and raw response</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={!showRawJson ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowRawJson(false)}
                  >
                    Based On
                  </Button>
                  <Button
                    variant={showRawJson ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowRawJson(true)}
                  >
                    Raw JSON
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {!showRawJson ? (
                includeFacts && result.based_on && result.based_on.length > 0 ? (
                  (() => {
                    // Group facts by type
                    const worldFacts = result.based_on.filter((f: any) => f.type === "world");
                    const experienceFacts = result.based_on.filter(
                      (f: any) => f.type === "experience"
                    );
                    const opinionFacts = result.based_on.filter((f: any) => f.type === "opinion");

                    return (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card>
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base">World Facts</CardTitle>
                            <CardDescription className="text-xs">General Knowledge</CardDescription>
                          </CardHeader>
                          <CardContent>
                            {worldFacts.length > 0 ? (
                              <ul className="text-sm space-y-2">
                                {worldFacts.map((fact: any, i: number) => (
                                  <li key={i} className="p-2 bg-muted rounded">
                                    {fact.text}
                                    {fact.context && (
                                      <div className="text-xs text-muted-foreground mt-1">
                                        {fact.context}
                                      </div>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-muted-foreground text-sm">None</p>
                            )}
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base">Experience</CardTitle>
                            <CardDescription className="text-xs">
                              Conversations & Events
                            </CardDescription>
                          </CardHeader>
                          <CardContent>
                            {experienceFacts.length > 0 ? (
                              <ul className="text-sm space-y-2">
                                {experienceFacts.map((fact: any, i: number) => (
                                  <li key={i} className="p-2 bg-muted rounded">
                                    {fact.text}
                                    {fact.context && (
                                      <div className="text-xs text-muted-foreground mt-1">
                                        {fact.context}
                                      </div>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-muted-foreground text-sm">None</p>
                            )}
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base">Opinions</CardTitle>
                            <CardDescription className="text-xs">
                              Beliefs & Preferences
                            </CardDescription>
                          </CardHeader>
                          <CardContent>
                            {opinionFacts.length > 0 ? (
                              <ul className="text-sm space-y-2">
                                {opinionFacts.map((fact: any, i: number) => (
                                  <li key={i} className="p-2 bg-muted rounded">
                                    {fact.text}
                                    {fact.context && (
                                      <div className="text-xs text-muted-foreground mt-1">
                                        {fact.context}
                                      </div>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-muted-foreground text-sm">None</p>
                            )}
                          </CardContent>
                        </Card>
                      </div>
                    );
                  })()
                ) : includeFacts ? (
                  <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg">
                    <Info className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-semibold text-amber-900 dark:text-amber-100">
                        No facts found
                      </p>
                      <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                        No memories were found or used to generate this answer.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg">
                    <Info className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-semibold text-amber-900 dark:text-amber-100">
                        Facts not included
                      </p>
                      <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                        Enable "Include Facts" above to see which memories were used to generate
                        this answer.
                      </p>
                    </div>
                  </div>
                )
              ) : (
                <div className="bg-muted p-4 rounded border border-border overflow-auto max-h-[600px]">
                  <JsonView src={result} collapsed={1} theme="default" />
                </div>
              )}
            </CardContent>
          </Card>

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
        </div>
      )}
    </div>
  );
}
