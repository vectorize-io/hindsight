"use client";

import { useState, useEffect, useRef } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  BarChart3,
  Search,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Beaker,
  Play,
  Plus,
  FlaskConical,
  Brain,
  TrendingUp,
  Zap,
  RefreshCw,
} from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface StatCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
}

function StatCard({ label, value, icon }: StatCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  );
}

interface TestSuite {
  id: string;
  name: string;
  tests: number;
  passRate: number;
  lastRun: string;
  status: "passing" | "failing" | "never_run";
  models: string[];
}

interface EvalRun {
  id: string;
  suite: string;
  model: string;
  score: number;
  duration: string;
  status: "completed" | "running" | "failed";
  timestamp: string;
}

interface ModelScore {
  model: string;
  accuracy: number;
  latency: string;
  cost: string;
  status: "active" | "inactive";
}

export default function EvaluationPage() {
  const t = useTranslations("operator.evaluation");

  const [activeTab, setActiveTab] = useState<"suites" | "runs" | "comparison" | "playground">(
    "suites"
  );
  const [searchQuery, setSearchQuery] = useState("");

  // Playground state
  const [pgModels, setPgModels] = useState<string[]>([]);
  const [pgModel, setPgModel] = useState("auto");
  const [pgSystemPrompt, setPgSystemPrompt] = useState("");
  const [pgUserMessage, setPgUserMessage] = useState("");
  const [pgTemperature, setPgTemperature] = useState(0.7);
  const [pgMaxTokens, setPgMaxTokens] = useState(512);
  const [pgRunning, setPgRunning] = useState(false);
  const [pgResponse, setPgResponse] = useState("");
  const [pgLatency, setPgLatency] = useState<number | null>(null);
  const pgAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch("/api/chat/models")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.data) setPgModels(["auto", ...data.data.map((m: any) => m.id)]);
      })
      .catch(() => {});
  }, []);

  const runPlayground = async () => {
    if (!pgUserMessage.trim() || pgRunning) return;
    pgAbort.current?.abort();
    pgAbort.current = new AbortController();
    setPgRunning(true);
    setPgResponse("");
    setPgLatency(null);
    const t0 = Date.now();
    try {
      const messages = [
        ...(pgSystemPrompt.trim() ? [{ role: "system", content: pgSystemPrompt.trim() }] : []),
        { role: "user", content: pgUserMessage.trim() },
      ];
      const res = await fetch("/api/chat/proxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages,
          model: pgModel,
          stream: true,
          temperature: pgTemperature,
          max_tokens: pgMaxTokens,
        }),
        signal: pgAbort.current.signal,
      });
      if (!res.ok) {
        setPgResponse(`Error: ${res.status}`);
        return;
      }
      const reader = res.body?.getReader();
      if (!reader) {
        return;
      }
      const decoder = new TextDecoder();
      let text = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          try {
            const parsed = JSON.parse(raw);
            const delta = parsed.choices?.[0]?.delta?.content ?? "";
            text += delta;
            setPgResponse(text);
          } catch {
            /* ignore */
          }
        }
      }
      setPgLatency(Date.now() - t0);
    } catch (e: any) {
      if (e?.name !== "AbortError") setPgResponse(`Error: ${e?.message}`);
    } finally {
      setPgRunning(false);
    }
  };

  const testSuites: TestSuite[] = [
    {
      id: "suite_1",
      name: "Reasoning Benchmark",
      tests: 45,
      passRate: 92,
      lastRun: "2h ago",
      status: "passing",
      models: ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"],
    },
    {
      id: "suite_2",
      name: "Fact Extraction Accuracy",
      tests: 120,
      passRate: 87,
      lastRun: "5h ago",
      status: "passing",
      models: ["gpt-4o", "llama-3.3-70b"],
    },
    {
      id: "suite_3",
      name: "Tool Calling Precision",
      tests: 67,
      passRate: 78,
      lastRun: "1d ago",
      status: "passing",
      models: ["claude-sonnet-4", "gpt-4o"],
    },
    {
      id: "suite_4",
      name: "Context Adherence",
      tests: 34,
      passRate: 94,
      lastRun: "3h ago",
      status: "passing",
      models: ["gpt-4o", "claude-sonnet-4", "gemini-2.0-flash"],
    },
    {
      id: "suite_5",
      name: "Multi-turn Coherence",
      tests: 89,
      passRate: 63,
      lastRun: "1d ago",
      status: "failing",
      models: ["llama-3.3-70b", "mistral-large"],
    },
    {
      id: "suite_6",
      name: "Safety & Guardrails",
      tests: 200,
      passRate: 99,
      lastRun: "6h ago",
      status: "passing",
      models: ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"],
    },
  ];

  const evalRuns: EvalRun[] = [
    {
      id: "eval_001",
      suite: "Reasoning Benchmark",
      model: "gpt-4o",
      score: 94,
      duration: "3m 12s",
      status: "completed",
      timestamp: "2h ago",
    },
    {
      id: "eval_002",
      suite: "Reasoning Benchmark",
      model: "claude-sonnet-4",
      score: 96,
      duration: "4m 05s",
      status: "completed",
      timestamp: "2h ago",
    },
    {
      id: "eval_003",
      suite: "Fact Extraction",
      model: "llama-3.3-70b",
      score: 82,
      duration: "8m 30s",
      status: "completed",
      timestamp: "5h ago",
    },
    {
      id: "eval_004",
      suite: "Tool Calling",
      model: "claude-sonnet-4",
      score: 91,
      duration: "2m 45s",
      status: "completed",
      timestamp: "1d ago",
    },
    {
      id: "eval_005",
      suite: "Multi-turn Coherence",
      model: "llama-3.3-70b",
      score: 63,
      duration: "12m 00s",
      status: "completed",
      timestamp: "1d ago",
    },
    {
      id: "eval_006",
      suite: "Safety & Guardrails",
      model: "gpt-4o",
      score: 99,
      duration: "5m 20s",
      status: "completed",
      timestamp: "6h ago",
    },
    {
      id: "eval_007",
      suite: "Context Adherence",
      model: "gemini-2.0-flash",
      score: 91,
      duration: "3m 00s",
      status: "running",
      timestamp: "Now",
    },
  ];

  const modelScores: ModelScore[] = [
    { model: "gpt-4o", accuracy: 94, latency: "1.2s", cost: "$0.01/req", status: "active" },
    {
      model: "claude-sonnet-4",
      accuracy: 96,
      latency: "1.8s",
      cost: "$0.015/req",
      status: "active",
    },
    {
      model: "gemini-2.5-pro",
      accuracy: 91,
      latency: "0.9s",
      cost: "$0.005/req",
      status: "active",
    },
    { model: "llama-3.3-70b", accuracy: 78, latency: "2.4s", cost: "$0.003/req", status: "active" },
    {
      model: "mistral-large",
      accuracy: 72,
      latency: "1.5s",
      cost: "$0.004/req",
      status: "inactive",
    },
  ];

  const filteredSuites = testSuites.filter((s) =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline">
              <Play className="w-3.5 h-3.5 mr-1.5" /> {t("runTest")}
            </Button>
            <Button size="sm">
              <Plus className="w-3.5 h-3.5 mr-1.5" /> {t("createSuite")}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={t("statTotalTests")} value="555" icon={<Beaker className="h-4 w-4" />} />
          <StatCard
            label={t("statPassRate")}
            value="87.2%"
            icon={<CheckCircle2 className="h-4 w-4" />}
          />
          <StatCard
            label={t("statAvgScore")}
            value="84.6%"
            icon={<TrendingUp className="h-4 w-4" />}
          />
          <StatCard label={t("statModelsTested")} value="5" icon={<Brain className="h-4 w-4" />} />
        </div>

        <div className="flex items-center gap-4 border-b pb-2">
          <button
            onClick={() => setActiveTab("suites")}
            className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "suites" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          >
            {t("testSuites")}
          </button>
          <button
            onClick={() => setActiveTab("runs")}
            className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "runs" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          >
            {t("runHistory")}
          </button>
          <button
            onClick={() => setActiveTab("comparison")}
            className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "comparison" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          >
            {t("modelComparison")}
          </button>
          <button
            onClick={() => setActiveTab("playground")}
            className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "playground" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}
          >
            Playground
          </button>
          <div className="ml-auto relative">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search..."
              className="pl-8 h-9 w-48 text-xs"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        {activeTab === "suites" &&
          (filteredSuites.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                {t("noSuites")}
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredSuites.map((suite) => (
                <Card key={suite.id} className="hover:bg-accent/30 transition-colors">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">{suite.name}</CardTitle>
                      <Badge
                        variant={suite.status === "passing" ? "default" : "destructive"}
                        className="text-[10px]"
                      >
                        {suite.status === "passing" ? `${suite.passRate}%` : `${suite.passRate}%`}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{suite.tests} tests</span>
                      <span>{suite.lastRun}</span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {suite.models.map((m) => (
                        <Badge key={m} variant="outline" className="text-[10px]">
                          {m}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ))}

        {activeTab === "runs" &&
          (evalRuns.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                {t("noRuns")}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">{t("runHistory")}</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="divide-y">
                  {evalRuns.map((run) => (
                    <div
                      key={run.id}
                      className="flex items-center justify-between px-6 py-3 text-sm"
                    >
                      <div className="flex items-center gap-3">
                        {run.status === "running" ? (
                          <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                        ) : run.status === "completed" ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-500" />
                        )}
                        <div>
                          <span className="font-medium">{run.suite}</span>
                          <span className="text-muted-foreground ml-2">({run.model})</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span>Score: {run.score}%</span>
                        <span>{run.duration}</span>
                        <span>{run.timestamp}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}

        {activeTab === "comparison" && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t("modelComparison")}</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground text-xs">
                    <th className="text-left font-medium px-6 py-3">Model</th>
                    <th className="text-left font-medium px-6 py-3">Accuracy</th>
                    <th className="text-left font-medium px-6 py-3">Latency</th>
                    <th className="text-left font-medium px-6 py-3">Cost</th>
                    <th className="text-left font-medium px-6 py-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {modelScores.map((ms) => (
                    <tr key={ms.model} className="hover:bg-accent/30">
                      <td className="px-6 py-3 font-medium">{ms.model}</td>
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-2 bg-secondary rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${ms.accuracy}%` }}
                            />
                          </div>
                          <span className="text-xs">{ms.accuracy}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-3 text-muted-foreground">{ms.latency}</td>
                      <td className="px-6 py-3 text-muted-foreground">{ms.cost}</td>
                      <td className="px-6 py-3">
                        <Badge
                          variant={ms.status === "active" ? "default" : "secondary"}
                          className="text-[10px]"
                        >
                          {ms.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        )}

        {/* Playground Tab */}
        {activeTab === "playground" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Panel */}
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="h-4 w-4 text-primary" /> Model Playground
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Model Selector */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Model
                    </label>
                    <select
                      className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm"
                      value={pgModel}
                      onChange={(e) => setPgModel(e.target.value)}
                    >
                      {pgModels.length > 0 ? (
                        pgModels.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))
                      ) : (
                        <option value="auto">auto</option>
                      )}
                    </select>
                  </div>
                  {/* System Prompt */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      System Prompt
                    </label>
                    <Textarea
                      placeholder="Optional system prompt..."
                      value={pgSystemPrompt}
                      onChange={(e) => setPgSystemPrompt(e.target.value)}
                      rows={3}
                      className="text-sm resize-none"
                    />
                  </div>
                  {/* User Message */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      User Message
                    </label>
                    <Textarea
                      placeholder="Enter your test message..."
                      value={pgUserMessage}
                      onChange={(e) => setPgUserMessage(e.target.value)}
                      rows={4}
                      className="text-sm resize-none"
                    />
                  </div>
                  {/* Parameters */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Temperature ({pgTemperature.toFixed(1)})
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={pgTemperature}
                        onChange={(e) => setPgTemperature(parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Max Tokens ({pgMaxTokens})
                      </label>
                      <input
                        type="range"
                        min="64"
                        max="4096"
                        step="64"
                        value={pgMaxTokens}
                        onChange={(e) => setPgMaxTokens(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      className="flex-1"
                      onClick={runPlayground}
                      disabled={pgRunning || !pgUserMessage.trim()}
                    >
                      {pgRunning ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin mr-1.5" /> Running...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-1.5" /> Run Test
                        </>
                      )}
                    </Button>
                    {pgRunning && (
                      <Button variant="outline" onClick={() => pgAbort.current?.abort()}>
                        Stop
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Output Panel */}
            <Card className="flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">Response</CardTitle>
                  <div className="flex items-center gap-3">
                    {pgLatency !== null && (
                      <span className="text-[11px] text-muted-foreground font-mono">
                        {pgLatency}ms
                      </span>
                    )}
                    {pgResponse && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-[11px]"
                        onClick={() => {
                          setPgResponse("");
                          setPgLatency(null);
                        }}
                      >
                        <RefreshCw className="h-3 w-3 mr-1" /> Clear
                      </Button>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1">
                {pgRunning && !pgResponse ? (
                  <div className="flex items-center justify-center h-32 text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin mr-2" /> Waiting for response...
                  </div>
                ) : pgResponse ? (
                  <div className="relative">
                    <pre className="text-sm text-foreground whitespace-pre-wrap font-sans leading-relaxed break-words">
                      {pgResponse}
                      {pgRunning && (
                        <span className="inline-block w-1 h-4 bg-primary ml-0.5 animate-pulse" />
                      )}
                    </pre>
                    {pgLatency && (
                      <div className="mt-4 pt-3 border-t text-[11px] text-muted-foreground flex items-center gap-4">
                        <span>
                          Model: <span className="font-mono">{pgModel}</span>
                        </span>
                        <span>Latency: {pgLatency}ms</span>
                        <span>Temp: {pgTemperature.toFixed(1)}</span>
                        <span>MaxTok: {pgMaxTokens}</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-48 text-muted-foreground text-center gap-2">
                    <FlaskConical className="h-8 w-8 opacity-30" />
                    <p className="text-sm">Enter a message and click Run Test to start</p>
                    <p className="text-xs">Results stream in real-time via SSE</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </OperatorShell>
  );
}
