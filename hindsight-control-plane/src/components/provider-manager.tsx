"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Cpu,
  Bot,
  Wifi,
  WifiOff,
  Server,
  Activity,
  Layers,
} from "lucide-react";

interface Provider {
  id: string;
  provider_id: string;
  display_name: string;
  base_url: string;
  provider_type: string;
  api_style: string;
  auth_type: string;
  enabled: boolean;
  supports_chat: boolean;
  supports_completion: boolean;
  supports_embeddings: boolean;
  supports_tools: boolean;
  supports_streaming: boolean;
  health_status: string;
  last_health_check: string | null;
}

interface AIModel {
  id: string;
  model_id: string;
  display_name: string;
  provider_id: string;
  model_type: string;
  capabilities: string[];
  context_length: number;
  is_default: boolean;
}

type Tab = "providers" | "models";

export function ProviderManager() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [models, setModels] = useState<AIModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("providers");
  const [error, setError] = useState<string | null>(null);

  const fetchProviders = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/central/ai/providers");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setProviders(data.providers || []);
    } catch (e: any) {
      setError(e.message);
      setProviders([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const res = await fetch("/api/central/ai/models");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setModels(data.models || []);
    } catch {
      setModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  const handleHealthCheck = async (providerId: string) => {
    setHealthLoading(providerId);
    try {
      const res = await fetch(`/api/central/ai/providers/${providerId}/health`, {
        method: "POST",
      });
      await fetchProviders(); // Refresh all providers to get updated health status
    } catch {}
    setHealthLoading(null);
  };

  const handleRefreshModels = async () => {
    setModelsLoading(true);
    try {
      await fetch(`/api/central/ai/models/refresh`, { method: "POST" });
      await fetchModels();
    } catch {}
    setModelsLoading(false);
  };

  useEffect(() => {
    fetchProviders();
    fetchModels();
    const interval = setInterval(fetchProviders, 30000);
    return () => clearInterval(interval);
  }, [fetchProviders, fetchModels]);

  const capabilityBadges = (p: Provider) => (
    <div className="flex flex-wrap gap-1 mt-1">
      {p.supports_chat && (
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          chat
        </Badge>
      )}
      {p.supports_completion && (
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          completion
        </Badge>
      )}
      {p.supports_embeddings && (
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          embeddings
        </Badge>
      )}
      {p.supports_tools && (
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          tools
        </Badge>
      )}
      {p.supports_streaming && (
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          streaming
        </Badge>
      )}
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Model & Provider Manager</h2>
          <p className="text-sm text-muted-foreground">
            Central API provider registry —{" "}
            {loading ? "loading..." : `${providers.length} providers, ${models.length} models`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchProviders();
              fetchModels();
            }}
            disabled={loading}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b pb-px">
        <button
          onClick={() => setActiveTab("providers")}
          className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "providers"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          <Server className="w-3 h-3 inline mr-1" />
          Providers
        </button>
        <button
          onClick={() => setActiveTab("models")}
          className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "models"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          <Layers className="w-3 h-3 inline mr-1" />
          Models
          {models.length > 0 && (
            <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-primary/10 text-[10px]">
              {models.length}
            </span>
          )}
        </button>
      </div>

      {/* Providers tab */}
      {activeTab === "providers" && (
        <>
          {loading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading providers...
            </div>
          ) : error ? (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">
                <AlertCircle className="h-8 w-8 mx-auto mb-2 text-amber-500" />
                <p className="text-sm font-medium">Central API unreachable</p>
                <p className="text-xs mt-1">{error}</p>
                <Button variant="outline" size="sm" className="mt-3" onClick={fetchProviders}>
                  Retry
                </Button>
              </CardContent>
            </Card>
          ) : providers.length === 0 ? (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">
                <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No providers registered</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {providers.map((p) => (
                <Card key={p.id}>
                  <CardHeader className="pb-2 pt-3 px-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-primary shrink-0" />
                        <div>
                          <CardTitle className="text-sm">{p.display_name}</CardTitle>
                          <CardDescription className="text-[10px] font-mono">
                            {p.provider_id} · {p.api_style}
                          </CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {healthLoading === p.provider_id ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                        ) : p.health_status === "healthy" ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                        ) : p.health_status === "down" ? (
                          <WifiOff className="h-3.5 w-3.5 text-red-500" />
                        ) : (
                          <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                        )}
                        <Badge
                          variant="outline"
                          className={`text-[9px] h-4 px-1 ${
                            p.enabled
                              ? "border-green-300 text-green-700 bg-green-50 dark:border-green-800 dark:text-green-400 dark:bg-green-950/30"
                              : "border-gray-300 text-gray-500"
                          }`}
                        >
                          {p.enabled ? "enabled" : "disabled"}
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="px-3 pb-3 space-y-2">
                    <div className="text-[11px] space-y-0.5">
                      <div className="flex items-center gap-1.5 text-muted-foreground">
                        <Wifi className="w-3 h-3" />
                        <span className="font-mono truncate" title={p.base_url}>
                          {p.base_url}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 text-muted-foreground">
                        <Activity className="w-3 h-3" />
                        <span>
                          Health:{" "}
                          <span
                            className={
                              p.health_status === "healthy"
                                ? "text-green-600"
                                : p.health_status === "down"
                                  ? "text-red-500"
                                  : "text-amber-500"
                            }
                          >
                            {p.health_status}
                          </span>
                          {p.last_health_check && (
                            <span className="ml-1">
                              · {new Date(p.last_health_check).toLocaleTimeString()}
                            </span>
                          )}
                        </span>
                      </div>
                    </div>
                    {capabilityBadges(p)}
                    <div className="flex gap-1.5 pt-1">
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-6 text-[10px]"
                        onClick={() => handleHealthCheck(p.provider_id)}
                        disabled={healthLoading === p.provider_id}
                      >
                        {healthLoading === p.provider_id ? (
                          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                        ) : (
                          <Activity className="h-3 w-3 mr-1" />
                        )}
                        Check Health
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </>
      )}

      {/* Models tab */}
      {activeTab === "models" && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Model Inventory</CardTitle>
              <Button
                size="sm"
                variant="outline"
                onClick={handleRefreshModels}
                disabled={modelsLoading}
                className="h-7 text-xs"
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${modelsLoading ? "animate-spin" : ""}`} />
                Refresh Models
              </Button>
            </div>
            <CardDescription>
              {models.length === 0
                ? "No models loaded — click Refresh to sync from providers"
                : `${models.length} models across ${providers.length} providers`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {modelsLoading ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin mr-2" /> Syncing models...
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Layers className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Model inventory is empty</p>
                <p className="text-xs mt-1">Click "Refresh Models" to sync from all providers</p>
              </div>
            ) : (
              <div className="border rounded-lg overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="text-left py-2 px-3 font-medium text-muted-foreground">
                        Model
                      </th>
                      <th className="text-left py-2 px-3 font-medium text-muted-foreground">
                        Provider
                      </th>
                      <th className="text-left py-2 px-3 font-medium text-muted-foreground hidden sm:table-cell">
                        Type
                      </th>
                      <th className="text-right py-2 px-3 font-medium text-muted-foreground hidden md:table-cell">
                        Context
                      </th>
                      <th className="text-center py-2 px-3 font-medium text-muted-foreground">
                        Default
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.id} className="border-b last:border-0 hover:bg-muted/30">
                        <td className="py-2 px-3 font-mono text-xs">
                          {m.display_name || m.model_id}
                        </td>
                        <td className="py-2 px-3">{m.provider_id}</td>
                        <td className="py-2 px-3 hidden sm:table-cell">
                          <Badge variant="outline" className="text-[9px]">
                            {m.model_type}
                          </Badge>
                        </td>
                        <td className="py-2 px-3 text-right font-mono hidden md:table-cell">
                          {m.context_length?.toLocaleString() || "—"}
                        </td>
                        <td className="py-2 px-3 text-center">
                          {m.is_default ? (
                            <CheckCircle2 className="w-3 h-3 text-green-500 inline" />
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
