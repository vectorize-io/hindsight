"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  KeyRound,
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Server,
  Wrench,
  Eye,
  Shield,
  Clock,
} from "lucide-react";

interface SystemConfig {
  llm: {
    provider: string;
    model: string;
    prompt_cache_enabled: boolean;
  };
  embeddings: {
    provider: string;
    model: string;
    dimension: number;
  };
  reranker: {
    provider: string;
  };
  database: {
    type: string;
  };
  storage: {
    type: string;
    s3_bucket: string | null;
    s3_region: string | null;
  };
}

interface ProviderHealth {
  name: string;
  type: string;
  model: string;
  status: "healthy" | "degraded" | "unknown";
  lastChecked: string | null;
  capabilities: string[];
}

export default function ConfigPage() {
  const t = useTranslations("operator");
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [providers, setProviders] = useState<ProviderHealth[]>([]);
  const [checkingProvider, setCheckingProvider] = useState<string | null>(null);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/system/config");
      if (res.ok) {
        const data = await res.json();
        setConfig(data);

        // Build provider cards from config
        const providerList: ProviderHealth[] = [];
        if (data.llm) {
          providerList.push({
            name: data.llm.provider || "unknown",
            type: "LLM",
            model: data.llm.model || "—",
            status: "unknown",
            lastChecked: null,
            capabilities: ["chat", "reasoning"],
          });
        }
        if (data.embeddings) {
          providerList.push({
            name: data.embeddings.provider || "unknown",
            type: "Embeddings",
            model: data.embeddings.model || "—",
            status: "unknown",
            lastChecked: null,
            capabilities: ["embeddings"],
          });
        }
        if (data.reranker) {
          providerList.push({
            name: data.reranker.provider || "unknown",
            type: "Reranker",
            model: "—",
            status: "unknown",
            lastChecked: null,
            capabilities: ["reranking"],
          });
        }
        // Add database as a provider
        if (data.database) {
          providerList.push({
            name: data.database.type || "unknown",
            type: "Database",
            model: "—",
            status: "unknown",
            lastChecked: null,
            capabilities: ["storage", "vector-search"],
          });
        }
        setProviders(providerList);
      }
    } catch (e) {
      console.error("Failed to load config:", e);
    } finally {
      setLoading(false);
    }
  };

  const checkProviderHealth = async (providerName: string, type: string) => {
    setCheckingProvider(providerName);
    try {
      // Try hitting relevant health endpoints based on provider type
      let healthy = false;
      if (type === "LLM") {
        const res = await fetch("http://localhost:8888/health");
        healthy = res.ok;
      } else if (type === "Embeddings") {
        const res = await fetch("http://localhost:11434/api/tags", {
          signal: AbortSignal.timeout(3000),
        });
        healthy = res.ok;
      } else if (type === "Database") {
        const res = await fetch("http://localhost:8888/health");
        if (res.ok) {
          const data = await res.json();
          healthy = data.database === "connected";
        }
      } else if (type === "Reranker") {
        // Reranker health via API health
        const res = await fetch("http://localhost:8888/health");
        healthy = res.ok;
      }

      setProviders((prev) =>
        prev.map((p) =>
          p.name === providerName
            ? {
                ...p,
                status: healthy ? ("healthy" as const) : ("degraded" as const),
                lastChecked: new Date().toISOString(),
              }
            : p
        )
      );
    } catch {
      setProviders((prev) =>
        prev.map((p) =>
          p.name === providerName
            ? {
                ...p,
                status: "degraded" as const,
                lastChecked: new Date().toISOString(),
              }
            : p
        )
      );
    } finally {
      setCheckingProvider(null);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const statusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case "degraded":
        return <AlertCircle className="w-4 h-4 text-amber-500" />;
      default:
        return <HelpCircle className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400 border-green-200 dark:border-green-800";
      case "degraded":
        return "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400 border-amber-200 dark:border-amber-800";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  };

  const providerIcon = (type: string) => {
    switch (type) {
      case "LLM":
        return <Cpu className="h-5 w-5 text-primary" />;
      case "Embeddings":
        return <Database className="h-5 w-5 text-primary" />;
      case "Reranker":
        return <Wrench className="h-5 w-5 text-primary" />;
      case "Database":
        return <Server className="h-5 w-5 text-primary" />;
      default:
        return <HelpCircle className="h-5 w-5 text-muted-foreground" />;
    }
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Page header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <KeyRound className="h-6 w-6 text-primary" />
            {t("panels.config")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.config")}</p>
        </div>

        <Tabs defaultValue="models" className="space-y-4">
          <TabsList>
            <TabsTrigger value="models" className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              {t("models.tabModels")}
            </TabsTrigger>
            <TabsTrigger value="environment" className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              {t("models.tabEnvironment")}
            </TabsTrigger>
            <TabsTrigger value="secrets" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              {t("models.tabSecrets")}
            </TabsTrigger>
          </TabsList>

          {/* Model Manager Tab */}
          <TabsContent value="models" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Server className="h-5 w-5 text-primary" />
                  {t("models.providerRegistry")}
                  <span className="text-sm font-normal text-muted-foreground">
                    · {providers.length} {t("models.registered")}
                  </span>
                </CardTitle>
                <CardDescription>
                  AI providers and services configured in the system
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center py-8 text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin mr-2" /> {t("models.loading")}
                  </div>
                ) : providers.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Server className="h-10 w-10 mx-auto mb-2 opacity-30" />
                    <p>{t("models.noProviders")}</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {providers.map((provider) => (
                      <div
                        key={`${provider.type}-${provider.name}`}
                        className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                      >
                        <div className="flex items-start gap-3">
                          {providerIcon(provider.type)}
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">{provider.name}</span>
                              <Badge variant="outline" className="text-[10px] h-5">
                                {provider.type}
                              </Badge>
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                              Model: <span className="font-mono">{provider.model}</span>
                            </div>
                            <div className="flex flex-wrap gap-1 mt-1.5">
                              {provider.capabilities.map((cap) => (
                                <Badge
                                  key={cap}
                                  variant="secondary"
                                  className="text-[10px] h-5 px-1.5"
                                >
                                  {cap}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 flex-shrink-0">
                          <div className="text-right">
                            <div className="flex items-center gap-1 justify-end">
                              {statusIcon(provider.status)}
                              <Badge
                                variant="outline"
                                className={`text-[10px] h-5 ${statusColor(provider.status)}`}
                              >
                                {provider.status}
                              </Badge>
                            </div>
                            {provider.lastChecked && (
                              <div className="text-[10px] text-muted-foreground mt-1 flex items-center gap-1">
                                <Clock className="w-2.5 h-2.5" />
                                {new Date(provider.lastChecked).toLocaleTimeString()}
                              </div>
                            )}
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => checkProviderHealth(provider.name, provider.type)}
                            disabled={checkingProvider === provider.name}
                            className="text-xs h-8"
                          >
                            {checkingProvider === provider.name ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <>
                                <RefreshCw className="w-3 h-3 mr-1" />
                                {t("models.recheck")}
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Environment Tab */}
          <TabsContent value="environment" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Eye className="h-5 w-5 text-primary" />
                  {t("models.environmentTitle")}
                </CardTitle>
                <CardDescription>{t("models.environmentDesc")}</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center py-8 text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin mr-2" /> {t("models.loading")}
                  </div>
                ) : !config ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Eye className="h-10 w-10 mx-auto mb-2 opacity-30" />
                    <p>{t("models.noData")}</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* LLM Config */}
                    <Card className="ring-1 ring-border/40">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Cpu className="h-4 w-4 text-primary" />
                          LLM Configuration
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("models.provider")}</span>
                          <span className="font-mono text-xs">{config.llm.provider}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("models.model")}</span>
                          <span className="font-mono text-xs">{config.llm.model}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Prompt Cache</span>
                          <Badge
                            variant="outline"
                            className={`text-[10px] h-5 ${config.llm.prompt_cache_enabled ? "border-green-200 text-green-700" : "text-muted-foreground"}`}
                          >
                            {config.llm.prompt_cache_enabled ? "Enabled" : "Disabled"}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Embeddings Config */}
                    <Card className="ring-1 ring-border/40">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Database className="h-4 w-4 text-primary" />
                          Embeddings Configuration
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("models.provider")}</span>
                          <span className="font-mono text-xs">{config.embeddings.provider}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("models.model")}</span>
                          <span className="font-mono text-xs">{config.embeddings.model}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Dimension</span>
                          <span className="font-mono text-xs">{config.embeddings.dimension}</span>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Reranker Config */}
                    <Card className="ring-1 ring-border/40">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Wrench className="h-4 w-4 text-primary" />
                          Reranker Configuration
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">{t("models.provider")}</span>
                          <span className="font-mono text-xs">{config.reranker.provider}</span>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Storage Config */}
                    <Card className="ring-1 ring-border/40">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <HardDrive className="h-4 w-4 text-primary" />
                          Storage Configuration
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Type</span>
                          <span className="font-mono text-xs">{config.storage.type}</span>
                        </div>
                        {config.database.type && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Database</span>
                            <span className="font-mono text-xs">{config.database.type}</span>
                          </div>
                        )}
                        {config.storage.s3_bucket && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">S3 Bucket</span>
                            <span className="font-mono text-xs">{config.storage.s3_bucket}</span>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Secrets Tab */}
          <TabsContent value="secrets">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  {t("models.tabSecrets")}
                </CardTitle>
                <CardDescription>{t("models.secretsDesc")}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-center py-16 text-muted-foreground">
                  <div className="text-center space-y-2">
                    <Shield className="h-10 w-10 mx-auto opacity-20" />
                    <p>{t("models.secretsComingSoon")}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </OperatorShell>
  );
}
