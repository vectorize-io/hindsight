"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Search,
  Plus,
  Gavel,
  CheckCircle2,
  XCircle,
  Clock,
  Tag,
  ArrowUpDown,
  BookOpen,
  AlertTriangle,
} from "lucide-react";

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

interface Directive {
  id: string;
  name: string;
  content: string;
  priority: number;
  tags: string[];
  status: "active" | "inactive";
  lastModified: string;
}

export default function DirectivesPage() {
  const t = useTranslations("operator.directives");
  const [searchQuery, setSearchQuery] = useState("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");

  const directives: Directive[] = [
    { id: "dir_1", name: "Safety First", content: "Never execute destructive operations without explicit user confirmation. Always verify intent before file deletion, data destruction, or system modifications.", priority: 10, tags: ["safety", "core"], status: "active", lastModified: "1d ago" },
    { id: "dir_2", name: "Memory Persistence", content: "Store all session outcomes, decisions, and learnings to both Memlord and Hindsight memory systems before session end.", priority: 9, tags: ["memory", "workflow"], status: "active", lastModified: "3d ago" },
    { id: "dir_3", name: "Code Quality", content: "Prefer small, reversible changes. Add tests for behavior changes. Follow existing code conventions. Run lint/typecheck before committing.", priority: 8, tags: ["code", "quality"], status: "active", lastModified: "1w ago" },
    { id: "dir_4", name: "Configuration Protection", content: "Never modify .env programmatically. Never change port assignments without explicit approval. Never delete Docker volumes.", priority: 10, tags: ["infrastructure", "core"], status: "active", lastModified: "2d ago" },
    { id: "dir_5", name: "Evidence-Based Reasoning", content: "Label uncertain reasoning as VERIFIED, HYPOTHESIS, or UNKNOWN. Prioritize evidence over assumptions. Read the existing codebase before changing code.", priority: 7, tags: ["reasoning", "methodology"], status: "active", lastModified: "5d ago" },
    { id: "dir_6", name: "Split Architecture", content: "Ollama uses two isolated lanes: Embeddings (port 11434) and LLM (port 11435). Never mix these configurations or change port assignments.", priority: 9, tags: ["infrastructure", "ollama"], status: "active", lastModified: "1w ago" },
    { id: "dir_7", name: "MCP Dual Memory", content: "Two independent MCP servers must both receive critical data: Memlord (port 8005) for cross-session persistence, Hindsight (port 8888) for agent operational context.", priority: 8, tags: ["mcp", "memory"], status: "active", lastModified: "1w ago" },
    { id: "dir_8", name: "Session Recovery", content: "At session start: verify MCP health, check recent memories, review AGENTS.md changes, verify critical services. At session end: store all learnings.", priority: 6, tags: ["workflow", "startup"], status: "active", lastModified: "2w ago" },
    { id: "dir_9", name: "Legacy Pattern", content: "Use older API pattern for certain legacy endpoints. Do not use for new development.", priority: 3, tags: ["legacy", "deprecated"], status: "inactive", lastModified: "1mo ago" },
    { id: "dir_10", name: "Experimental Feature Flag", content: "When testing experimental features, always gate behind a feature flag and never affect stable code paths.", priority: 5, tags: ["experimental", "process"], status: "inactive", lastModified: "3w ago" },
  ];

  let filtered = directives;
  if (searchQuery) filtered = filtered.filter(d => d.name.toLowerCase().includes(searchQuery.toLowerCase()) || d.content.toLowerCase().includes(searchQuery.toLowerCase()));
  if (priorityFilter) filtered = filtered.filter(d => d.priority.toString() === priorityFilter);
  if (statusFilter) filtered = filtered.filter(d => d.status === statusFilter);

  const priorityColor = (p: number) => {
    if (p >= 9) return "destructive";
    if (p >= 7) return "default";
    if (p >= 5) return "secondary";
    return "outline";
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
          <Button size="sm">
            <Plus className="w-3.5 h-3.5 mr-1.5" /> {t("createDirective")}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={t("statActive")} value="8" icon={<CheckCircle2 className="h-4 w-4" />} />
          <StatCard label={t("statInactive")} value="2" icon={<XCircle className="h-4 w-4" />} />
          <StatCard label={t("statPriorityTypes")} value="4" icon={<ArrowUpDown className="h-4 w-4" />} />
          <StatCard label={t("statLastModified")} value="1d ago" icon={<Clock className="h-4 w-4" />} />
        </div>

        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input placeholder={t("searchDirectives")} className="pl-8 h-9 text-xs" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
          </div>
          <select className="h-9 text-xs border rounded-md px-2 bg-background text-muted-foreground" value={priorityFilter} onChange={e => setPriorityFilter(e.target.value)}>
            <option value="">All priorities</option>
            <option value="10">Critical (10)</option>
            <option value="9">Urgent (9)</option>
            <option value="8">High (8)</option>
            <option value="7">Medium (7)</option>
            <option value="6">Standard (6)</option>
            <option value="5">Low (5)</option>
            <option value="3">Minimal (3)</option>
          </select>
          <select className="h-9 text-xs border rounded-md px-2 bg-background text-muted-foreground" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="">All statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>

        {filtered.length === 0 ? (
          <Card><CardContent className="py-12 text-center text-muted-foreground">{t("noDirectives")}</CardContent></Card>
        ) : (
          <div className="space-y-3">
            {filtered.map(d => (
              <Card key={d.id} className="hover:bg-accent/30 transition-colors">
                <CardHeader className="pb-2 pt-3 px-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Gavel className="w-4 h-4 text-primary" />
                      <CardTitle className="text-sm font-medium">{d.name}</CardTitle>
                      <Badge variant={priorityColor(d.priority)} className="text-[10px]">P{d.priority}</Badge>
                    </div>
                    <Badge variant={d.status === "active" ? "default" : "secondary"} className="text-[10px]">{d.status}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="px-4 pb-3">
                  <p className="text-xs text-muted-foreground mb-2">{d.content}</p>
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                    {d.tags.map(tag => (
                      <Badge key={tag} variant="outline" className="text-[10px] flex items-center gap-1">
                        <Tag className="w-2 h-2" />{tag}
                      </Badge>
                    ))}
                    <span className="ml-auto">{d.lastModified}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </OperatorShell>
  );
}
