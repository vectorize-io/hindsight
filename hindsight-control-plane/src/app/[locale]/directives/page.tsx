"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Plus,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowUpDown,
  Loader2,
  RefreshCw,
  Gavel,
  Tag,
  AlertTriangle,
  Brain,
} from "lucide-react";

interface Bank {
  bank_id: string;
  name: string | null;
}

interface Directive {
  id: string;
  name: string;
  content: string;
  priority: number;
  tags: string[];
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export default function DirectivesPage() {
  const t = useTranslations("operator.directives");

  const [banks, setBanks] = useState<Bank[]>([]);
  const [selectedBankId, setSelectedBankId] = useState<string>("");
  const [directives, setDirectives] = useState<Directive[]>([]);
  const [loading, setLoading] = useState(false);
  const [banksLoading, setBanksLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"" | "active" | "inactive">("");

  // Create dialog
  const [creating, setCreating] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [newName, setNewName] = useState("");
  const [newContent, setNewContent] = useState("");
  const [newPriority, setNewPriority] = useState("5");
  const [newTags, setNewTags] = useState("");

  useEffect(() => {
    const loadBanks = async () => {
      setBanksLoading(true);
      try {
        const res = await fetch("/api/banks");
        if (res.ok) {
          const data = await res.json();
          const b: Bank[] = data.banks || [];
          setBanks(b);
          if (b.length > 0) setSelectedBankId(b[0].bank_id);
        }
      } catch {
        /* ignore */
      }
      setBanksLoading(false);
    };
    loadBanks();
  }, []);

  const loadDirectives = useCallback(async () => {
    if (!selectedBankId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/banks/${selectedBankId}/directives`);
      if (res.ok) {
        const data = await res.json();
        setDirectives(data.directives || data.items || []);
      }
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, [selectedBankId]);

  useEffect(() => {
    loadDirectives();
  }, [loadDirectives]);

  const handleCreate = async () => {
    if (!selectedBankId || !newName.trim() || !newContent.trim()) return;
    setCreateLoading(true);
    try {
      const res = await fetch(`/api/banks/${selectedBankId}/directives`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newName.trim(),
          content: newContent.trim(),
          priority: parseInt(newPriority),
          tags: newTags
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          is_active: true,
        }),
      });
      if (res.ok) {
        setCreating(false);
        setNewName("");
        setNewContent("");
        setNewPriority("5");
        setNewTags("");
        loadDirectives();
      }
    } catch {
      /* ignore */
    }
    setCreateLoading(false);
  };

  const priorityVariant = (p: number): "destructive" | "default" | "secondary" | "outline" => {
    if (p >= 9) return "destructive";
    if (p >= 7) return "default";
    if (p >= 5) return "secondary";
    return "outline";
  };

  const filtered = directives.filter((d) => {
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      if (!d.name.toLowerCase().includes(q) && !d.content.toLowerCase().includes(q)) return false;
    }
    if (statusFilter === "active" && !d.is_active) return false;
    if (statusFilter === "inactive" && d.is_active) return false;
    return true;
  });

  const activeCount = directives.filter((d) => d.is_active).length;
  const inactiveCount = directives.filter((d) => !d.is_active).length;
  const highPriorityCount = directives.filter((d) => d.priority >= 7).length;
  const latestDate =
    directives.length > 0
      ? new Date(
          Math.max(...directives.map((d) => new Date(d.updated_at || d.created_at || 0).getTime()))
        ).toLocaleDateString()
      : "—";

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
              <Gavel className="h-6 w-6 text-primary" />
              {t("title")}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Persistent rules governing agent behaviour
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadDirectives} disabled={loading}>
              <RefreshCw className={`h-3.5 w-3.5 mr-1 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
            <Button size="sm" onClick={() => setCreating(true)} disabled={!selectedBankId}>
              <Plus className="w-3.5 h-3.5 mr-1.5" /> {t("createDirective")}
            </Button>
          </div>
        </div>

        {/* Bank Picker */}
        {banksLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading banks...
          </div>
        ) : banks.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No banks configured. Create a bank first via the Memory Console.
            </CardContent>
          </Card>
        ) : (
          <div className="flex items-center gap-3">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Bank:</span>
            <Select value={selectedBankId} onValueChange={setSelectedBankId}>
              <SelectTrigger className="w-48 h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {banks.map((b) => (
                  <SelectItem key={b.bank_id} value={b.bank_id} className="text-xs">
                    {b.name || b.bank_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Stat Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {t("statActive")}
              </CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : activeCount}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {t("statInactive")}
              </CardTitle>
              <XCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-muted-foreground">
                {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : inactiveCount}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                P7+ Priority
              </CardTitle>
              <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : highPriorityCount}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {t("statLastModified")}
              </CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-sm font-bold">{loading ? "..." : latestDate}</div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder={t("searchDirectives")}
              className="pl-8 h-9 text-xs"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as any)}>
            <SelectTrigger className="w-32 h-9 text-xs">
              <SelectValue placeholder="All status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="" className="text-xs">
                All status
              </SelectItem>
              <SelectItem value="active" className="text-xs">
                Active only
              </SelectItem>
              <SelectItem value="inactive" className="text-xs">
                Inactive only
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Directives List */}
        {loading ? (
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading directives...
          </div>
        ) : filtered.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center space-y-3">
              <Gavel className="h-10 w-10 text-muted-foreground mx-auto" />
              <p className="text-sm text-muted-foreground">
                {directives.length === 0
                  ? "No directives yet. Create your first directive to govern agent behaviour."
                  : "No directives match the current filter."}
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {filtered
              .sort((a, b) => b.priority - a.priority)
              .map((d) => (
                <Card key={d.id} className={d.is_active ? "" : "opacity-60"}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-3 min-w-0">
                        <div className="mt-0.5">
                          {d.is_active ? (
                            <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                          ) : (
                            <XCircle className="h-4 w-4 text-muted-foreground shrink-0" />
                          )}
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-1">
                            <span className="text-sm font-semibold truncate">{d.name}</span>
                            <Badge
                              variant={priorityVariant(d.priority)}
                              className="text-[10px] h-5"
                            >
                              P{d.priority}
                            </Badge>
                            {!d.is_active && (
                              <Badge
                                variant="outline"
                                className="text-[10px] h-5 text-muted-foreground"
                              >
                                inactive
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {d.content}
                          </p>
                          {d.tags && d.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              <Tag className="h-3 w-3 text-muted-foreground mt-0.5" />
                              {d.tags.map((tag) => (
                                <Badge key={tag} variant="outline" className="text-[9px] h-4 px-1">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="text-[10px] text-muted-foreground shrink-0 text-right">
                        {d.updated_at
                          ? new Date(d.updated_at).toLocaleDateString()
                          : d.created_at
                            ? new Date(d.created_at).toLocaleDateString()
                            : ""}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
          </div>
        )}
      </div>

      {/* Create Directive Dialog */}
      <Dialog open={creating} onOpenChange={setCreating}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Gavel className="h-5 w-5" /> New Directive
            </DialogTitle>
            <DialogDescription>
              Create a persistent governance rule for bank{" "}
              <span className="font-mono text-xs">{selectedBankId}</span>
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="e.g. Safety First"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Content</label>
              <Textarea
                placeholder="Describe the rule the agent must follow..."
                value={newContent}
                onChange={(e) => setNewContent(e.target.value)}
                rows={4}
                className="text-sm"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Priority (0–10)</label>
                <Input
                  type="number"
                  min="0"
                  max="10"
                  value={newPriority}
                  onChange={(e) => setNewPriority(e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Tags (comma-separated)</label>
                <Input
                  placeholder="safety, core, memory"
                  value={newTags}
                  onChange={(e) => setNewTags(e.target.value)}
                />
              </div>
            </div>
            {parseInt(newPriority) >= 9 && (
              <div className="flex items-center gap-2 text-xs text-amber-600 bg-amber-50 dark:bg-amber-950/20 p-2 rounded border border-amber-200 dark:border-amber-800">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                High-priority directives (P9+) are applied first in all agent reasoning sessions.
              </div>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setCreating(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreate}
                disabled={createLoading || !newName.trim() || !newContent.trim()}
              >
                {createLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Create Directive
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </OperatorShell>
  );
}
