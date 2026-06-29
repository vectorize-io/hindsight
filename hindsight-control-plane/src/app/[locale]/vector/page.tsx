"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Search, Database, Layers, HardDrive, Hash, CheckCircle2, AlertCircle } from "lucide-react";

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

interface Collection {
  id: string;
  name: string;
  vectors: number;
  dimensions: number;
  indexType: string;
  status: "ready" | "indexing" | "error";
  size: string;
  lastUpdated: string;
}

interface SampleVector {
  id: string;
  collection: string;
  preview: string;
  score: number;
  timestamp: string;
}

export default function VectorPage() {
  const t = useTranslations("operator.vector");
  const [searchQuery, setSearchQuery] = useState("");

  const collections: Collection[] = [
    {
      id: "col_1",
      name: "memories_default",
      vectors: 125430,
      dimensions: 768,
      indexType: "HNSW",
      status: "ready",
      size: "412 MB",
      lastUpdated: "1m ago",
    },
    {
      id: "col_2",
      name: "memories_opencode",
      vectors: 34210,
      dimensions: 768,
      indexType: "HNSW",
      status: "ready",
      size: "128 MB",
      lastUpdated: "5m ago",
    },
    {
      id: "col_3",
      name: "documents_ingest",
      vectors: 8905,
      dimensions: 768,
      indexType: "Flat",
      status: "indexing",
      size: "34 MB",
      lastUpdated: "Now",
    },
    {
      id: "col_4",
      name: "entity_embeddings",
      vectors: 45678,
      dimensions: 384,
      indexType: "HNSW",
      status: "ready",
      size: "89 MB",
      lastUpdated: "1h ago",
    },
    {
      id: "col_5",
      name: "mental_models",
      vectors: 1200,
      dimensions: 768,
      indexType: "HNSW",
      status: "ready",
      size: "4.2 MB",
      lastUpdated: "2d ago",
    },
    {
      id: "col_6",
      name: "observations_archive",
      vectors: 567890,
      dimensions: 768,
      indexType: "IVFFlat",
      status: "error",
      size: "2.1 GB",
      lastUpdated: "3d ago",
    },
  ];

  const sampleVectors: SampleVector[] = [
    {
      id: "vec_001",
      collection: "memories_default",
      preview: "The user prefers declarative configuration over imperative...",
      score: 0.94,
      timestamp: "1m ago",
    },
    {
      id: "vec_002",
      collection: "memories_default",
      preview: "Split Ollama configuration with embeddings on port 11434...",
      score: 0.89,
      timestamp: "5m ago",
    },
    {
      id: "vec_003",
      collection: "entity_embeddings",
      preview: "Entity: CollabMind — Type: Organization — Related: Hindsight...",
      score: 0.87,
      timestamp: "15m ago",
    },
    {
      id: "vec_004",
      collection: "documents_ingest",
      preview: "CHAPTER 3: Memory Architecture — The biomimetic approach...",
      score: 0.82,
      timestamp: "1h ago",
    },
    {
      id: "vec_005",
      collection: "memories_default",
      preview: "Docker Desktop 4.79.0 has an Electron crash bug affecting...",
      score: 0.78,
      timestamp: "2h ago",
    },
  ];

  const filteredCollections = collections.filter((c) =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            label={t("statVectors")}
            value="783,313"
            icon={<Database className="h-4 w-4" />}
          />
          <StatCard label={t("statDimensions")} value="768" icon={<Hash className="h-4 w-4" />} />
          <StatCard label={t("statCollections")} value="6" icon={<Layers className="h-4 w-4" />} />
          <StatCard
            label={t("statIndexSize")}
            value="2.77 GB"
            icon={<HardDrive className="h-4 w-4" />}
          />
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder={t("searchVectors")}
                className="pl-8 h-9 text-xs"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          {filteredCollections.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                {t("noCollections")}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">{t("collections")}</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-muted-foreground text-xs">
                      <th className="text-left font-medium px-6 py-3">Name</th>
                      <th className="text-left font-medium px-6 py-3">Vectors</th>
                      <th className="text-left font-medium px-6 py-3">Dimensions</th>
                      <th className="text-left font-medium px-6 py-3">Index</th>
                      <th className="text-left font-medium px-6 py-3">Size</th>
                      <th className="text-left font-medium px-6 py-3">Status</th>
                      <th className="text-left font-medium px-6 py-3">Updated</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {filteredCollections.map((col) => (
                      <tr key={col.id} className="hover:bg-accent/30">
                        <td className="px-6 py-3 font-medium text-xs">{col.name}</td>
                        <td className="px-6 py-3 text-xs">{col.vectors.toLocaleString()}</td>
                        <td className="px-6 py-3 text-xs">{col.dimensions}</td>
                        <td className="px-6 py-3 text-xs font-mono">{col.indexType}</td>
                        <td className="px-6 py-3 text-xs">{col.size}</td>
                        <td className="px-6 py-3">
                          <Badge
                            variant={
                              col.status === "ready"
                                ? "default"
                                : col.status === "indexing"
                                  ? "secondary"
                                  : "destructive"
                            }
                            className="text-[10px]"
                          >
                            {col.status === "ready" && (
                              <CheckCircle2 className="w-2.5 h-2.5 mr-1" />
                            )}
                            {col.status === "indexing" && (
                              <div className="w-2.5 h-2.5 mr-1 rounded-full bg-blue-500 animate-pulse" />
                            )}
                            {col.status === "error" && <AlertCircle className="w-2.5 h-2.5 mr-1" />}
                            {col.status}
                          </Badge>
                        </td>
                        <td className="px-6 py-3 text-xs text-muted-foreground">
                          {col.lastUpdated}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">{t("sampleVectors")}</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y">
              {sampleVectors.map((v) => (
                <div key={v.id} className="px-6 py-3 text-sm flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px] shrink-0">
                        {v.collection}
                      </Badge>
                      <span className="text-xs truncate">{v.preview}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 shrink-0 ml-4">
                    <span className="text-xs text-muted-foreground">
                      Score: {(v.score * 100).toFixed(0)}%
                    </span>
                    <span className="text-xs text-muted-foreground">{v.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}
