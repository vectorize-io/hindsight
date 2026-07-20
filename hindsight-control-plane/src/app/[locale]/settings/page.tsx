"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { BankSelector } from "@/components/bank-selector";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { client, type ServerLlmConfig } from "@/lib/api";

// User-facing subset of supported providers. The backend validates the full set.
const PROVIDERS = [
  "openai",
  "anthropic",
  "gemini",
  "groq",
  "deepseek",
  "openrouter",
  "minimax",
  "ollama",
  "lmstudio",
  "none",
] as const;

// Providers that don't need an API key (mirrors the backend's keyless set).
const KEYLESS = new Set(["ollama", "lmstudio", "llamacpp", "none", "mock", "vertexai"]);

type Health = Awaited<ReturnType<typeof client.testServerLlm>>["operations"];

export default function SettingsPage() {
  const t = useTranslations("settings");

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<ServerLlmConfig | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");

  const load = useCallback(async () => {
    try {
      const cfg = await client.getServerLlmConfig();
      setConfig(cfg);
      setProvider(cfg.provider || "openai");
      setModel(cfg.model || "");
      setBaseUrl(cfg.base_url || "");
      setApiKey("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const keyless = KEYLESS.has(provider);

  const save = async () => {
    setSaving(true);
    setError(null);
    setHealth(null);
    try {
      const body: {
        provider: string;
        model?: string | null;
        api_key?: string | null;
        base_url?: string | null;
      } = {
        provider,
        model: model.trim() || null,
        base_url: baseUrl.trim() || null,
      };
      // Only send the key when the user typed one; empty means "leave unchanged".
      if (apiKey.trim()) body.api_key = apiKey.trim();
      const updated = await client.updateServerLlmConfig(body);
      setConfig(updated);
      setApiKey("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const clear = async () => {
    setSaving(true);
    setError(null);
    setHealth(null);
    try {
      const updated = await client.resetServerLlmConfig();
      setConfig(updated);
      setProvider(updated.provider || "openai");
      setModel(updated.model || "");
      setBaseUrl(updated.base_url || "");
      setApiKey("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const testConnection = async () => {
    setTesting(true);
    setError(null);
    try {
      const res = await client.testServerLlm();
      setHealth(res.operations);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <BankSelector />
      <div className="flex-1 p-6 flex justify-center">
        <div className="w-full max-w-2xl space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t("title")}</h1>
            <p className="text-muted-foreground mt-1">{t("description")}</p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {t("llmTitle")}
                {config &&
                  (config.is_configured ? (
                    <span className="inline-flex items-center gap-1 text-sm font-normal text-green-600">
                      <CheckCircle2 className="h-4 w-4" /> {t("statusConfigured")}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-sm font-normal text-muted-foreground">
                      <XCircle className="h-4 w-4" /> {t("statusNotConfigured")}
                    </span>
                  ))}
              </CardTitle>
              <CardDescription>{t("llmDescription")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label htmlFor="provider">{t("providerLabel")}</Label>
                <Select value={provider} onValueChange={setProvider} disabled={loading}>
                  <SelectTrigger id="provider" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PROVIDERS.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="model">{t("modelLabel")}</Label>
                <Input
                  id="model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder={t("modelPlaceholder")}
                  disabled={loading}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="apiKey">{t("apiKeyLabel")}</Label>
                <Input
                  id="apiKey"
                  type="password"
                  autoComplete="off"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={
                    config?.api_key_is_set
                      ? t("apiKeyConfiguredPlaceholder")
                      : t("apiKeyPlaceholder")
                  }
                  disabled={loading || keyless}
                />
                <p className="text-xs text-muted-foreground">
                  {keyless ? t("apiKeyNotNeeded") : t("apiKeyDescription")}
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="baseUrl">{t("baseUrlLabel")}</Label>
                <Input
                  id="baseUrl"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder={t("baseUrlPlaceholder")}
                  disabled={loading}
                />
              </div>

              {health && (
                <div className="space-y-1 rounded-md border p-3">
                  {health.map((op) => (
                    <div key={op.operation} className="flex items-center gap-2 text-sm">
                      {op.ok ? (
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      ) : (
                        <XCircle className="h-4 w-4 text-destructive" />
                      )}
                      <span className="capitalize">{op.operation}</span>
                      <span className="text-muted-foreground">— {op.status}</span>
                      {op.latency_ms != null && (
                        <span className="text-muted-foreground">
                          ({Math.round(op.latency_ms)}ms)
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="flex flex-wrap gap-2 pt-1">
                <Button onClick={save} disabled={loading || saving}>
                  {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {t("save")}
                </Button>
                <Button variant="outline" onClick={testConnection} disabled={loading || testing}>
                  {testing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {t("testConnection")}
                </Button>
                {config?.is_configured && (
                  <Button variant="ghost" onClick={clear} disabled={loading || saving}>
                    {t("clear")}
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
