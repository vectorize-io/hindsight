"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Settings2,
  Zap,
  Monitor,
  Loader2,
  Save,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
} from "lucide-react";

interface BankConfig {
  retain_mission?: string;
  retain_extraction_mode?: string;
  retain_chunk_size?: number;
  retain_chunk_batch_size?: number;
  retain_structured_chunk_size?: number;
  disposition_skepticism?: number;
  disposition_literalism?: number;
  disposition_empathy?: number;
  enable_observations?: boolean;
  [key: string]: unknown;
}

export default function SettingsPage() {
  const t = useTranslations("operator");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stackConfig, setStackConfig] = useState<any>(null);
  const [bankConfig, setBankConfig] = useState<BankConfig>({});

  // Editable fields
  const [retainMission, setRetainMission] = useState("");
  const [extractionMode, setExtractionMode] = useState("concise");
  const [chunkSize, setChunkSize] = useState(3000);
  const [structuredChunkSize, setStructuredChunkSize] = useState(12000);
  const [chunkBatchSize, setChunkBatchSize] = useState(10);
  const [enableObservations, setEnableObservations] = useState(true);
  const [skepticism, setSkepticism] = useState(3);
  const [literalism, setLiteralism] = useState(3);
  const [empathy, setEmpathy] = useState(3);

  useEffect(() => {
    const load = async () => {
      try {
        // Read-only stack config
        const sysRes = await fetch("/api/system/config");
        if (sysRes.ok) setStackConfig(await sysRes.json());

        // Editable bank config
        const bankRes = await fetch("/api/banks/default/config");
        if (bankRes.ok) {
          const data: BankConfig = await bankRes.json();
          setBankConfig(data);
          setRetainMission(data.retain_mission || "");
          setExtractionMode(data.retain_extraction_mode || "concise");
          setChunkSize(data.retain_chunk_size ?? 3000);
          setStructuredChunkSize(data.retain_structured_chunk_size ?? 12000);
          setChunkBatchSize(data.retain_chunk_batch_size ?? 10);
          setEnableObservations(data.enable_observations ?? true);
          setSkepticism(data.disposition_skepticism ?? 3);
          setLiteralism(data.disposition_literalism ?? 3);
          setEmpathy(data.disposition_empathy ?? 3);
        }
      } catch {
        /* ignore */
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {};
      if (bankConfig.retain_mission !== undefined) payload.retain_mission = retainMission;
      if (bankConfig.retain_extraction_mode !== undefined)
        payload.retain_extraction_mode = extractionMode;
      if (bankConfig.retain_chunk_size !== undefined) payload.retain_chunk_size = chunkSize;
      if (bankConfig.retain_structured_chunk_size !== undefined)
        payload.retain_structured_chunk_size = structuredChunkSize;
      if (bankConfig.retain_chunk_batch_size !== undefined)
        payload.retain_chunk_batch_size = chunkBatchSize;
      if (bankConfig.enable_observations !== undefined)
        payload.enable_observations = enableObservations;
      if (bankConfig.disposition_skepticism !== undefined)
        payload.disposition_skepticism = skepticism;
      if (bankConfig.disposition_literalism !== undefined)
        payload.disposition_literalism = literalism;
      if (bankConfig.disposition_empathy !== undefined) payload.disposition_empathy = empathy;

      const res = await fetch("/api/banks/default/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Update failed" }));
        throw new Error(err.error || err.detail || "Update failed");
      }

      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e: any) {
      setError(e.message || "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <OperatorShell>
        <div className="p-6">
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading settings...
          </div>
        </div>
      </OperatorShell>
    );
  }

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Settings2 className="h-6 w-6 text-primary" />
            {t("panels.settings")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.settings")}</p>
        </div>

        <Tabs defaultValue="general" className="space-y-4">
          <TabsList>
            <TabsTrigger value="general" className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" /> Extraction
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <Zap className="h-4 w-4" /> Processing
            </TabsTrigger>
            <TabsTrigger value="stack" className="flex items-center gap-2">
              <Monitor className="h-4 w-4" /> Stack Config
            </TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Extraction Settings</CardTitle>
                <CardDescription>Controls how memories are extracted and organized</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Extraction Mode</label>
                  <select
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    value={extractionMode}
                    onChange={(e) => setExtractionMode(e.target.value)}
                  >
                    <option value="concise">Concise (selective, fast)</option>
                    <option value="verbose">Verbose (richer facts)</option>
                    <option value="verbatim">Verbatim (store as-is)</option>
                    <option value="chunks">Chunks (zero LLM cost)</option>
                  </select>
                  <p className="text-[10px] text-muted-foreground">
                    Controls the level of detail in extracted memory facts
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Retain Mission (steering)</label>
                  <textarea
                    className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-y"
                    value={retainMission}
                    onChange={(e) => setRetainMission(e.target.value)}
                    placeholder="Focus on technical decisions, architecture choices..."
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Natural language instruction that steers what gets extracted during retain
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Enable Observations</label>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setEnableObservations(!enableObservations)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                        enableObservations ? "bg-primary" : "bg-input"
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          enableObservations ? "translate-x-6" : "translate-x-1"
                        }`}
                      />
                    </button>
                    <span className="text-xs text-muted-foreground">
                      {enableObservations
                        ? "Auto-synthesis of observations after retain"
                        : "Disabled"}
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Skepticism ({skepticism}/5)</label>
                    <Input
                      type="range"
                      min={1}
                      max={5}
                      value={skepticism}
                      onChange={(e) => setSkepticism(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">Critical evaluation level</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Literalism ({literalism}/5)</label>
                    <Input
                      type="range"
                      min={1}
                      max={5}
                      value={literalism}
                      onChange={(e) => setLiteralism(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Literal vs abstract interpretation
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Empathy ({empathy}/5)</label>
                    <Input
                      type="range"
                      min={1}
                      max={5}
                      value={empathy}
                      onChange={(e) => setEmpathy(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Emotional context consideration
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Processing Settings</CardTitle>
                <CardDescription>
                  Chunk sizes, batching, and memory processing tuning
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chunk Size ({chunkSize} chars)</label>
                    <Input
                      type="range"
                      min={500}
                      max={8000}
                      step={100}
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Target max characters per content chunk
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Structured Chunk Size ({structuredChunkSize} chars)
                    </label>
                    <Input
                      type="range"
                      min={2000}
                      max={24000}
                      step={500}
                      value={structuredChunkSize}
                      onChange={(e) => setStructuredChunkSize(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Max characters for JSONL/conversation turns
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Batch Size ({chunkBatchSize})</label>
                    <Input
                      type="range"
                      min={1}
                      max={50}
                      value={chunkBatchSize}
                      onChange={(e) => setChunkBatchSize(Number(e.target.value))}
                    />
                    <p className="text-[10px] text-muted-foreground">
                      Number of chunks to process in parallel
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="stack" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Live Stack Configuration</CardTitle>
                <CardDescription>
                  Read-only view of the active Hindsight stack config
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {stackConfig ? (
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                        LLM
                      </h4>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Provider</div>
                          <div className="text-sm font-mono font-medium">
                            {stackConfig.llm?.provider || "—"}
                          </div>
                        </div>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Model</div>
                          <div className="text-sm font-mono font-medium truncate">
                            {stackConfig.llm?.model || "—"}
                          </div>
                        </div>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Prompt Cache</div>
                          <div className="text-sm font-medium">
                            {stackConfig.llm?.prompt_cache_enabled ? "Enabled" : "Disabled"}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                        Embeddings
                      </h4>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Provider</div>
                          <div className="text-sm font-mono font-medium">
                            {stackConfig.embeddings?.provider || "—"}
                          </div>
                        </div>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Model</div>
                          <div className="text-sm font-mono font-medium truncate">
                            {stackConfig.embeddings?.model || "—"}
                          </div>
                        </div>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Dimensions</div>
                          <div className="text-sm font-medium">
                            {stackConfig.embeddings?.dimension ?? "—"}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                          Reranker
                        </h4>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Provider</div>
                          <div className="text-sm font-mono font-medium">
                            {stackConfig.reranker?.provider || "—"}
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                          Database
                        </h4>
                        <div className="p-2 rounded border bg-muted/30">
                          <div className="text-[10px] text-muted-foreground">Type</div>
                          <div className="text-sm font-mono font-medium">
                            {stackConfig.database?.type || "—"}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">Config not available</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="flex items-center gap-3">
          {error && (
            <div className="flex items-center gap-2 text-sm text-red-600">
              <AlertCircle className="h-4 w-4" /> {error}
            </div>
          )}
          <Button onClick={handleSave} disabled={saving} className="flex items-center gap-2">
            {saving ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Saving...
              </>
            ) : saved ? (
              <>
                <CheckCircle2 className="h-4 w-4" /> Saved
              </>
            ) : (
              <>
                <Save className="h-4 w-4" /> Save Settings
              </>
            )}
          </Button>
        </div>
      </div>
    </OperatorShell>
  );
}
