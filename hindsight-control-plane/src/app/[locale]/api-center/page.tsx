"use client";

import { useState, useEffect } from "react";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  FileJson,
  RefreshCw,
  Loader2,
  Server,
  Search,
  Activity,
  CheckCircle2,
  Terminal,
  BarChart3,
  AlertCircle,
  BookOpen,
  Hash,
} from "lucide-react";

interface Endpoint {
  method: string;
  path: string;
  summary: string;
  tags: string[];
  deprecated?: boolean;
}

interface OpenApiSpec {
  openapi: string;
  info: {
    title: string;
    version: string;
    description?: string;
  };
  paths: Record<string, Record<string, any>>;
}

const METHOD_COLORS: Record<string, string> = {
  get: "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400 border-blue-200",
  post: "bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400 border-green-200",
  put: "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400 border-amber-200",
  patch:
    "bg-purple-100 text-purple-700 dark:bg-purple-950/30 dark:text-purple-400 border-purple-200",
  delete: "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400 border-red-200",
};

const TAG_COLORS: Record<string, string> = {
  audit: "bg-sky-100 text-sky-700",
  banks: "bg-indigo-100 text-indigo-700",
  directives: "bg-emerald-100 text-emerald-700",
  documents: "bg-teal-100 text-teal-700",
  entities: "bg-violet-100 text-violet-700",
  files: "bg-orange-100 text-orange-700",
  "llm traces": "bg-pink-100 text-pink-700",
  memory: "bg-blue-100 text-blue-700",
  "mental models": "bg-amber-100 text-amber-700",
  monitoring: "bg-gray-100 text-gray-700",
  operations: "bg-rose-100 text-rose-700",
  webhooks: "bg-cyan-100 text-cyan-700",
  "bank templates": "bg-lime-100 text-lime-700",
  "document transfer": "bg-yellow-100 text-yellow-700",
};

function getTagColor(tag: string): string {
  const lower = tag.toLowerCase();
  return TAG_COLORS[lower] || "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300";
}

function getMethodColor(method: string): string {
  return METHOD_COLORS[method.toLowerCase()] || METHOD_COLORS.get;
}

export default function ApiCenterPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [spec, setSpec] = useState<OpenApiSpec | null>(null);
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [health, setHealth] = useState<{ status: string; database: string } | null>(null);
  const [fetchedSize, setFetchedSize] = useState(0);

  const fetchSpec = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/system/openapi", { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
      const data: OpenApiSpec = await res.json();
      setSpec(data);

      // Extract endpoints from paths
      const eps: Endpoint[] = [];
      const tagSet = new Set<string>();
      for (const [path, methods] of Object.entries(data.paths)) {
        for (const [method, detail] of Object.entries(methods)) {
          if (["get", "post", "put", "patch", "delete"].includes(method)) {
            const epTags = (detail as any).tags || ["uncategorized"];
            epTags.forEach((t: string) => tagSet.add(t));
            eps.push({
              method: method.toUpperCase(),
              path,
              summary: (detail as any).summary || (detail as any).description || "",
              tags: epTags,
              deprecated: (detail as any).deprecated || false,
            });
          }
        }
      }
      eps.sort((a, b) => a.path.localeCompare(b.path));
      setEndpoints(eps);
      setTags([...tagSet].sort());
      setFetchedSize(new Blob([JSON.stringify(data)]).size);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setEndpoints([]);
    }
    setLoading(false);
  };

  const fetchHealth = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_DATAPLANE_URL || "http://localhost:8888";
      const res = await fetch("/api/system/services", { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        const apiService = (data.services || []).find((s: any) => s.name === "Hindsight API");
        if (apiService) {
          setHealth({
            status: apiService.health,
            database: data.services?.find((s: any) => s.name === "PostgreSQL")?.health || "unknown",
          });
        }
      }
    } catch {}
  };

  useEffect(() => {
    fetchSpec();
    fetchHealth();
  }, []);

  const filteredEndpoints = endpoints.filter((ep) => {
    if (tagFilter && !ep.tags.some((t) => t.toLowerCase() === tagFilter.toLowerCase()))
      return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        ep.path.toLowerCase().includes(q) ||
        ep.summary.toLowerCase().includes(q) ||
        ep.method.toLowerCase().includes(q) ||
        ep.tags.some((t) => t.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const totalMethods = {
    get: endpoints.filter((e) => e.method === "GET").length,
    post: endpoints.filter((e) => e.method === "POST").length,
    put: endpoints.filter((e) => e.method === "PUT" || e.method === "PATCH").length,
    delete: endpoints.filter((e) => e.method === "DELETE").length,
  };

  const isHealthy = health?.status === "healthy";

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <FileJson className="h-6 w-6 text-primary" />
              API Center
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Live OpenAPI endpoint browser — powered by the running Hindsight API
              {spec ? (
                <span className="ml-2 text-xs text-muted-foreground">
                  · {spec.info.title} v{spec.info.version} · {spec.openapi}
                </span>
              ) : null}
            </p>
          </div>
          <button
            onClick={() => {
              fetchSpec();
              fetchHealth();
            }}
            disabled={loading}
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-md border text-xs font-medium hover:bg-accent transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Card>
            <CardHeader className="pb-1 pt-3 px-3">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Server className="w-3.5 h-3.5" /> Total Endpoints
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3">
              <div className="text-xl font-bold">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : endpoints.length}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-3">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> API Health
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3">
              <div className="flex items-center gap-1.5 text-sm">
                {health ? (
                  <>
                    {isHealthy ? (
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-amber-500" />
                    )}
                    <span className="font-medium">{isHealthy ? "Healthy" : "Degraded"}</span>
                  </>
                ) : (
                  <span className="text-muted-foreground">Checking...</span>
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-3">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Hash className="w-3.5 h-3.5" /> API Version
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3">
              <div className="text-xl font-bold font-mono text-xs">
                {spec ? (
                  spec.info.version
                ) : loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "—"
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-3">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5" /> Tag Groups
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3">
              <div className="text-xl font-bold">{tags.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1 pt-3 px-3">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" /> Spec Size
              </CardTitle>
            </CardHeader>
            <CardContent className="px-3 pb-3">
              <div className="text-xl font-bold">
                {fetchedSize > 0 ? (
                  `${(fetchedSize / 1024).toFixed(0)} KB`
                ) : loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "—"
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Method breakdown */}
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline" className="text-[10px] font-mono bg-blue-50 text-blue-700">
            GET {totalMethods.get}
          </Badge>
          <Badge variant="outline" className="text-[10px] font-mono bg-green-50 text-green-700">
            POST {totalMethods.post}
          </Badge>
          <Badge variant="outline" className="text-[10px] font-mono bg-amber-50 text-amber-700">
            PUT/PATCH {totalMethods.put}
          </Badge>
          <Badge variant="outline" className="text-[10px] font-mono bg-red-50 text-red-700">
            DELETE {totalMethods.delete}
          </Badge>
        </div>

        {/* Endpoint Browser */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Terminal className="h-4 w-4 text-blue-500" />
              Endpoint Browser
            </CardTitle>
            <CardDescription>
              {spec
                ? `${filteredEndpoints.length} of ${endpoints.length} endpoints · ${tags.length} groups`
                : "Loading specification from /openapi.json..."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Search */}
            <div className="flex items-center gap-2 mb-4 flex-wrap">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  placeholder="Search endpoints, methods, tags..."
                  className="pl-8 h-9 text-xs"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              {/* Tag filter pills */}
              <div className="flex gap-1 flex-wrap">
                <button
                  onClick={() => setTagFilter(null)}
                  className={`px-2 py-1 rounded text-[10px] font-medium border transition-colors ${
                    tagFilter === null
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-transparent text-muted-foreground border-border hover:bg-accent"
                  }`}
                >
                  All
                </button>
                {tags.slice(0, 12).map((tag) => (
                  <button
                    key={tag}
                    onClick={() => setTagFilter(tag === tagFilter ? null : tag)}
                    className={`px-2 py-1 rounded text-[10px] font-medium border transition-colors ${
                      tagFilter === tag
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-transparent text-muted-foreground border-border hover:bg-accent"
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Error state */}
            {error && (
              <div className="flex flex-col items-center gap-3 py-8 text-muted-foreground">
                <AlertCircle className="h-6 w-6 text-red-400" />
                <p className="text-sm">Failed to load OpenAPI spec: {error}</p>
                <button
                  onClick={fetchSpec}
                  className="text-xs text-primary underline hover:no-underline"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Loading state */}
            {loading && endpoints.length === 0 && !error && (
              <div className="flex items-center justify-center py-12 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading OpenAPI specification...
              </div>
            )}

            {/* Endpoint table */}
            {!loading && endpoints.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left pb-2 font-medium w-20">Method</th>
                      <th className="text-left pb-2 font-medium">Path</th>
                      <th className="text-left pb-2 font-medium hidden md:table-cell">Summary</th>
                      <th className="text-left pb-2 font-medium hidden lg:table-cell w-32">Tags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredEndpoints.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="text-center py-8 text-muted-foreground">
                          No endpoints match your filters
                        </td>
                      </tr>
                    ) : (
                      filteredEndpoints.map((ep, i) => (
                        <tr
                          key={`${ep.method}-${ep.path}-${i}`}
                          className={`border-b last:border-0 hover:bg-accent/30 transition-colors ${
                            ep.deprecated ? "opacity-50" : ""
                          }`}
                        >
                          <td className="py-2 pr-3">
                            <Badge
                              variant="outline"
                              className={`text-[10px] font-mono w-14 justify-center border ${getMethodColor(ep.method)}`}
                            >
                              {ep.method}
                            </Badge>
                          </td>
                          <td className="py-2 pr-3">
                            <code className="font-mono text-[11px] break-all">{ep.path}</code>
                            {ep.deprecated && (
                              <Badge variant="destructive" className="text-[9px] h-4 ml-1">
                                DEPRECATED
                              </Badge>
                            )}
                          </td>
                          <td className="py-2 pr-3 text-muted-foreground hidden md:table-cell max-w-xs truncate">
                            {ep.summary || "—"}
                          </td>
                          <td className="py-2 hidden lg:table-cell">
                            <div className="flex gap-1 flex-wrap">
                              {ep.tags.map((tag) => (
                                <Badge
                                  key={tag}
                                  variant="outline"
                                  className={`text-[9px] h-4 px-1.5 ${getTagColor(tag)}`}
                                >
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}
