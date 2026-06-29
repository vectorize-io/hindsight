"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ScrollText,
  CheckCircle2,
  Shield,
  Clock,
  AlertTriangle,
  FileCheck,
  PenLine,
  Scale,
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

interface Article {
  id: string;
  number: number;
  title: string;
  content: string;
  category: string;
  status: "ratified" | "pending" | "amended";
  lastReviewed: string;
}

interface Amendment {
  id: string;
  article: string;
  description: string;
  date: string;
  author: string;
}

export default function ConstitutionPage() {
  const t = useTranslations("operator.constitution");
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);

  const articles: Article[] = [
    { id: "art_1", number: 1, title: "Operator Sovereignty", content: "The operator retains ultimate authority over all system decisions. No agent may execute destructive operations without explicit operator consent. This article cannot be amended.", category: "Governance", status: "ratified", lastReviewed: "2026-06-01" },
    { id: "art_2", number: 2, title: "Memory Integrity", content: "All session outcomes must be persisted to both Memlord and Hindsight memory systems. Knowledge is the collective asset — no session ends without knowledge capture.", category: "Memory", status: "ratified", lastReviewed: "2026-06-15" },
    { id: "art_3", number: 3, title: "Defense in Depth", content: "Every change must be reviewable, reversible, and verifiable. Prefer small, atomic commits. Maintain rollback paths. No single point of failure in agent operations.", category: "Security", status: "ratified", lastReviewed: "2026-06-10" },
    { id: "art_4", number: 4, title: "Evidence Over Authority", content: "All reasoning must be grounded in evidence. Label uncertainty explicitly (VERIFIED, HYPOTHESIS, UNKNOWN). Read before writing. Understand before changing.", category: "Methodology", status: "ratified", lastReviewed: "2026-06-20" },
    { id: "art_5", number: 5, title: "Infrastructure Immutability", content: "Core infrastructure configuration (ports, service endpoints, database URLs, credentials) must never be modified programmatically. Changes require explicit operator action.", category: "Infrastructure", status: "ratified", lastReviewed: "2026-06-25" },
    { id: "art_6", number: 6, title: "Dual Memory Architecture", content: "Two independent MCP memory systems must always be operational: Memlord (port 8005) for cross-session persistence, Hindsight (port 8888/mcp) for agent operational context.", category: "Architecture", status: "ratified", lastReviewed: "2026-06-22" },
    { id: "art_7", number: 7, title: "Session Continuity", content: "Every session begins with system verification and ends with knowledge storage. No session creates unintentional technical debt or untracked configuration drift.", category: "Operations", status: "pending", lastReviewed: "2026-06-28" },
    { id: "art_8", number: 8, title: "Multi-Agent Coordination", content: "When multiple agents operate concurrently, they must use shared state coordination, avoid duplicate work, and synchronize through the operator console.", category: "Architecture", status: "amended", lastReviewed: "2026-06-27" },
  ];

  const amendments: Amendment[] = [
    { id: "amend_1", article: "Article 8", description: "Added concurrent agent coordination rules and shared state requirements", date: "2026-06-27", author: "Oliver" },
    { id: "amend_2", article: "Article 6", description: "Updated MCP port configuration to reflect dual architecture", date: "2026-06-25", author: "Oliver" },
    { id: "amend_3", article: "Article 4", description: "Added uncertainty classification requirement for agent reasoning", date: "2026-06-20", author: "Oliver" },
  ];

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
          <Button size="sm" variant="outline">
            <PenLine className="w-3.5 h-3.5 mr-1.5" /> {t("sign")}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={t("statArticles")} value="8" icon={<ScrollText className="h-4 w-4" />} />
          <StatCard label={t("statLastAmended")} value="3d ago" icon={<Clock className="h-4 w-4" />} />
          <StatCard label={t("statComplianceScore")} value="96%" icon={<CheckCircle2 className="h-4 w-4" />} />
          <StatCard label={t("statReviewCycle")} value="7 days" icon={<Scale className="h-4 w-4" />} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-3">
            {articles.length === 0 ? (
              <Card><CardContent className="py-12 text-center text-muted-foreground">{t("noArticles")}</CardContent></Card>
            ) : (
              articles.map(article => (
                <Card
                  key={article.id}
                  className={`hover:bg-accent/30 transition-colors cursor-pointer ${selectedArticle === article.id ? "ring-1 ring-primary" : ""}`}
                  onClick={() => setSelectedArticle(selectedArticle === article.id ? null : article.id)}
                >
                  <CardHeader className="pb-2 pt-3 px-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Scale className="w-4 h-4 text-primary" />
                        <CardTitle className="text-sm font-medium">
                          {t("articleNumber")} {article.number}: {article.title}
                        </CardTitle>
                      </div>
                      <Badge variant={article.status === "ratified" ? "default" : article.status === "pending" ? "secondary" : "outline"} className="text-[10px]">
                        {article.status === "ratified" && <Shield className="w-2.5 h-2.5 mr-1" />}
                        {article.status === "pending" && <Clock className="w-2.5 h-2.5 mr-1" />}
                        {article.status === "amended" && <PenLine className="w-2.5 h-2.5 mr-1" />}
                        {article.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="px-4 pb-3">
                    <p className="text-xs text-muted-foreground">{article.content}</p>
                    <div className="flex items-center gap-2 mt-2 text-[10px] text-muted-foreground">
                      <Badge variant="outline" className="text-[10px]">{article.category}</Badge>
                      <span className="flex items-center gap-1"><Clock className="w-2.5 h-2.5" />Reviewed {article.lastReviewed}</span>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><FileCheck className="w-4 h-4" />{t("complianceCheck")}</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                {articles.filter(a => a.status === "ratified").map(a => (
                  <div key={a.id} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Article {a.number}</span>
                    <span className="text-green-500 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" />Compliant</span>
                  </div>
                ))}
                {articles.filter(a => a.status === "pending").map(a => (
                  <div key={a.id} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Article {a.number}</span>
                    <span className="text-amber-500 flex items-center gap-1"><AlertTriangle className="w-3 h-3" />Pending</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><PenLine className="w-4 h-4" />{t("recentAmendments")}</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                {amendments.map(a => (
                  <div key={a.id} className="text-xs border-b pb-2 last:border-0 last:pb-0">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{a.article}</span>
                      <span className="text-muted-foreground">{a.date}</span>
                    </div>
                    <p className="text-muted-foreground mt-0.5">{a.description}</p>
                    <span className="text-muted-foreground">by {a.author}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </OperatorShell>
  );
}
