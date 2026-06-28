"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Server,
  Database,
  Cpu,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Zap,
  Tag,
  Globe,
  AlertCircle,
  HardDrive,
  Cog,
} from "lucide-react";
import { client } from "@/lib/api";
import { DATAPLANE_URL } from "@/lib/hindsight-client";

interface VersionInfo {
  api_version: string;
  features: {
    observations: boolean;
    mcp: boolean;
    worker: boolean;
    bank_config_api: boolean;
    file_upload_api: boolean;
    access_key_auth: boolean;
    [key: string]: boolean;
  };
}

interface HealthInfo {
  status: string;
  service: string;
  dataplane?: {
    status: string;
    url: string;
    error?: string;
  };
}

function FeatureBadge({ enabled, label }: { enabled: boolean; label: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <span className="text-sm text-foreground">{label}</span>
      {enabled ? (
        <Badge
          variant="outline"
          className="gap-1 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40"
        >
          <CheckCircle2 className="w-3 h-3" />
          On
        </Badge>
      ) : (
        <Badge variant="outline" className="gap-1 text-muted-foreground border-border bg-muted/40">
          <XCircle className="w-3 h-3" />
          Off
        </Badge>
      )}
    </div>
  );
}

function StatusIndicator({ connected }: { connected: boolean }) {
  return (
    <span className="relative flex items-center gap-2">
      <span
        className={`inline-flex h-2.5 w-2.5 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`}
      />
      {connected ? (
        <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
          Connected
        </span>
      ) : (
        <span className="text-sm font-medium text-red-600 dark:text-red-400">Disconnected</span>
      )}
    </span>
  );
}

export function DeploymentView() {
  const t = useTranslations("deployment");

  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverConfig, setServerConfig] = useState<Record<string, unknown> | null>(null);

  const load = useCallback(async (showRefresh = false) => {
    if (showRefresh) setIsRefreshing(true);
    else setIsLoading(true);
    setError(null);

    try {
      const [version, health, config] = await Promise.all([
        client.getVersion(),
        fetch("/api/health").then((r) => r.json() as Promise<HealthInfo>),
        fetch("/api/system/config").then((r) => r.json().catch(() => null)) as Promise<Record<string, unknown> | null>,
      ]);
      setVersionInfo(version);
      setHealthInfo(health);
      setServerConfig(config);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load deployment info");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const featureLabels: Record<string, string> = {
    observations: t("features.observations"),
    mcp: t("features.mcp"),
    worker: t("features.worker"),
    bank_config_api: t("features.bankConfigApi"),
    file_upload_api: t("features.fileUploadApi"),
    access_key_auth: t("features.accessKeyAuth"),
  };

  const dataplaneConnected = healthInfo?.dataplane?.status === "connected";

  return (
    <div className="space-y-6">
      {/* Refresh button */}
      <div className="flex justify-end">
        <Button
          variant="outline"
          size="sm"
          onClick={() => load(true)}
          disabled={isRefreshing || isLoading}
          className="gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? "animate-spin" : ""}`} />
          {t("refresh")}
        </Button>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Control Plane */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Server className="w-4 h-4 text-muted-foreground" />
              {t("controlPlane")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("service")}</span>
              {isLoading ? (
                <div className="h-4 w-32 bg-muted rounded animate-pulse" />
              ) : (
                <span className="text-sm font-mono text-foreground">
                  {healthInfo?.service ?? "hindsight-control-plane"}
                </span>
              )}
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("status")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <Badge
                  variant="outline"
                  className="gap-1 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40"
                >
                  <CheckCircle2 className="w-3 h-3" />
                  {healthInfo?.status ?? "ok"}
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Dataplane */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Database className="w-4 h-4 text-muted-foreground" />
              {t("dataplane")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("connection")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <StatusIndicator connected={dataplaneConnected} />
              )}
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("url")}</span>
              {isLoading ? (
                <div className="h-4 w-40 bg-muted rounded animate-pulse" />
              ) : (
                <span
                  className="text-sm font-mono text-foreground truncate max-w-[200px]"
                  title={DATAPLANE_URL}
                >
                  {DATAPLANE_URL}
                </span>
              )}
            </div>
            {!isLoading && healthInfo?.dataplane?.error && (
              <div className="flex items-start gap-2 p-3 rounded-md bg-destructive/10 border border-destructive/20 text-destructive text-xs">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <span className="break-all">{healthInfo.dataplane.error}</span>
              </div>
            )}
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("apiVersion")}</span>
              {isLoading ? (
                <div className="h-4 w-20 bg-muted rounded animate-pulse" />
              ) : (
                <span className="text-sm font-mono text-foreground">
                  {versionInfo?.api_version ?? "—"}
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Storage Configuration */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <HardDrive className="w-4 h-4 text-muted-foreground" />
              {t("storageConfig")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("storageType")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <Badge variant="outline" className="text-xs font-mono">
                  {serverConfig?.storage ? (serverConfig.storage as Record<string, unknown>).type as string : "native"}
                </Badge>
              )}
            </div>
            {!isLoading && (serverConfig?.storage as Record<string, unknown>)?.type !== "native" ? (
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-sm text-muted-foreground">{t("storageBucket")}</span>
                <span className="text-sm font-mono text-foreground truncate max-w-[200px]">
                  {(serverConfig?.storage as Record<string, unknown>)?.s3_bucket as string || "—"}
                </span>
              </div>
            ) : !isLoading ? (
              <p className="text-xs text-muted-foreground">{t("nativeStorageHint")}</p>
            ) : null}
          </CardContent>
        </Card>

        {/* LLM Configuration */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Cog className="w-4 h-4 text-muted-foreground" />
              {t("llmConfig")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("provider")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <span className="text-sm font-mono text-foreground">
                  {serverConfig?.llm ? (serverConfig.llm as Record<string, unknown>).provider as string : "—"}
                </span>
              )}
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("model")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <span className="text-sm font-mono text-foreground">
                  {serverConfig?.llm ? (serverConfig.llm as Record<string, unknown>).model as string : "—"}
                </span>
              )}
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-sm text-muted-foreground">{t("promptCache")}</span>
              {isLoading ? (
                <div className="h-5 w-12 bg-muted rounded animate-pulse" />
              ) : serverConfig?.llm ? (
                <Badge
                  variant="outline"
                  className={
                    (serverConfig.llm as Record<string, unknown>).prompt_cache_enabled
                      ? "gap-1 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40"
                      : "gap-1 text-muted-foreground border-border bg-muted/40"
                  }
                >
                  {(serverConfig.llm as Record<string, unknown>).prompt_cache_enabled ? (
                    <><CheckCircle2 className="w-3 h-3" /> On</>
                  ) : (
                    <><XCircle className="w-3 h-3" /> Off</>
                  )}
                </Badge>
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-muted-foreground">{t("embeddingsModel")}</span>
              {isLoading ? (
                <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              ) : (
                <span className="text-sm font-mono text-foreground truncate max-w-[200px]">
                  {serverConfig?.embeddings ? (serverConfig.embeddings as Record<string, unknown>).model as string : "—"}
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Features */}
        <Card className="md:col-span-2">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Zap className="w-4 h-4 text-muted-foreground" />
              {t("features.title")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-2 border-b border-border last:border-0"
                  >
                    <div className="h-4 w-36 bg-muted rounded animate-pulse" />
                    <div className="h-5 w-12 bg-muted rounded animate-pulse" />
                  </div>
                ))}
              </div>
            ) : versionInfo ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8">
                {Object.entries(versionInfo.features).map(([key, enabled]) => (
                  <FeatureBadge key={key} enabled={enabled} label={featureLabels[key] ?? key} />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{t("noData")}</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
