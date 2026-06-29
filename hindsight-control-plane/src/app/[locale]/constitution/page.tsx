"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ScrollText,
  Scale,
  CheckCircle2,
  AlertCircle,
  Clock,
  Shield,
  FileText,
  BookOpen,
  Gavel,
  Loader2,
  ExternalLink,
  Pen,
} from "lucide-react";
import Link from "next/link";

interface Article {
  number: number;
  title: string;
  category: string;
  status: "active" | "pending" | "amended";
  lastReviewed: string;
  content: string;
}

interface Amendment {
  id: string;
  article: number;
  date: string;
  summary: string;
  author: string;
}

export default function ConstitutionPage() {
  const t = useTranslations("operator");
  const params = useParams();
  const locale = (params?.locale as string) || "en";
  const [loading, setLoading] = useState(true);
  const [banks, setBanks] = useState<{ bank_id: string; name: string; mission: string }[]>([]);
  const [complianceScore, setComplianceScore] = useState(87);

  const [articles] = useState<Article[]>([
    {
      number: 1,
      title: "Data Sovereignty",
      category: "Governance",
      status: "active",
      lastReviewed: "2026-06-15",
      content: "All user data remains under user control. No data is shared with third parties without explicit consent.",
    },
    {
      number: 2,
      title: "Transparency",
      category: "Ethics",
      status: "active",
      lastReviewed: "2026-06-10",
      content: "All AI decisions must be explainable and auditable. Users have the right to know how decisions affecting them are made.",
    },
    {
      number: 3,
      title: "Privacy by Default",
      category: "Privacy",
      status: "active",
      lastReviewed: "2026-05-28",
      content: "Minimum data collection by default. Privacy settings default to maximum protection.",
    },
    {
      number: 4,
      title: "Human Oversight",
      category: "Governance",
      status: "active",
      lastReviewed: "2026-06-01",
      content: "Critical decisions require human approval. AI operates as an assistant, not an autonomous decision-maker.",
    },
    {
      number: 5,
      title: "Fairness & Bias Prevention",
      category: "Ethics",
      status: "amended",
      lastReviewed: "2026-06-20",
      content: "Systems must be regularly audited for bias. Training data must be representative and inclusive.",
    },
    {
      number: 6,
      title: "Security & Integrity",
      category: "Security",
      status: "active",
      lastReviewed: "2026-06-18",
      content: "All systems must maintain industry-standard security practices. Regular penetration testing and security audits are mandatory.",
    },
    {
      number: 7,
      title: "Accountability",
      category: "Governance",
      status: "active",
      lastReviewed: "2026-06-12",
      content: "Clear lines of responsibility for all system actions. Incident response procedures must be documented and tested.",
    },
    {
      number: 8,
      title: "Continuous Improvement",
      category: "Operations",
      status: "pending",
      lastReviewed: "2026-06-22",
      content: "Regular review and updates to all policies. Feedback loops for continuous system improvement.",
    },
  ]);

  const [amendments] = useState<Amendment[]>([
    { id: "a1", article: 5, date: "2026-06-20", summary: "Added AI fairness audit requirement", author: "Oliver" },
    { id: "a2", article: 3, date: "2026-05-28", summary: "Updated data retention policy", author: "Oliver" },
    { id: "a3", article: 1, date: "2026-05-15", summary: "Initial ratification", author: "Oliver" },
  ]);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("/api/banks");
        if (res.ok) {
          const data = await res.json();
          setBanks(data.banks || []);
        }
      } catch {
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <OperatorShell>
        <div className="p-6">
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading constitution...
          </div>
        </div>
      </OperatorShell>
    );
  }

  const statusBadge = (status: string) => {
    switch (status) {
      case "active":
        return <Badge className="bg-green-100 text-green-700 dark:bg-green-950/30 dark:text-green-400 border-green-200 dark:border-green-800">{status}</Badge>;
      case "pending":
        return <Badge variant="outline" className="text-amber-600 border-amber-300">{status}</Badge>;
      case "amended":
        return <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400 border-blue-200 dark:border-blue-800">{status}</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const activeArticles = articles.filter((a) => a.status === "active").length;

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <ScrollText className="h-6 w-6 text-primary" />
            {t("panels.constitution")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.constitution")}</p>
        </div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground font-medium">Active Articles</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold">{activeArticles}</span>
              <span className="text-xs text-muted-foreground ml-2">/ {articles.length}</span>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground font-medium">Last Amended</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold">3d ago</span>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground font-medium">Compliance Score</CardTitle>
            </CardHeader>
            <CardContent className="flex items-center gap-2">
              <span className="text-2xl font-bold">{complianceScore}%</span>
              <Badge className="bg-green-100 text-green-700 text-[10px]">Good</Badge>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground font-medium">Review Cycle</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="text-2xl font-bold">30d</span>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Articles */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-primary" />
                  Articles
                </CardTitle>
                <CardDescription>Prime directives governing system behavior</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {articles.map((article) => (
                  <div key={article.number} className="p-4 rounded-lg border bg-card">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-primary bg-primary/10 px-2 py-0.5 rounded">
                          Art. {article.number}
                        </span>
                        <span className="text-sm font-semibold">{article.title}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {statusBadge(article.status)}
                        <Badge variant="outline" className="text-[10px]">{article.category}</Badge>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{article.content}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-muted-foreground">
                        Last reviewed: {article.lastReviewed}
                      </span>
                      {article.status === "pending" && (
                        <Button variant="outline" size="sm" className="h-7 text-[11px]">
                          <Pen className="h-3 w-3 mr-1" /> Sign & Ratify
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Compliance Check */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Shield className="h-4 w-4 text-primary" />
                  Compliance Check
                </CardTitle>
              </CardHeader>
              <CardContent className="text-center py-6">
                <div className="text-4xl font-bold text-primary mb-1">{complianceScore}%</div>
                <p className="text-xs text-muted-foreground mb-4">
                  Overall compliance score across {banks.length} banks
                </p>
                <div className="space-y-2 text-left">
                  {banks.slice(0, 4).map((bank) => (
                    <div key={bank.bank_id} className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground truncate max-w-[150px]">{bank.name || bank.bank_id}</span>
                      <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recent Amendments */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Gavel className="h-4 w-4 text-primary" />
                  Recent Amendments
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {amendments.map((amendment) => (
                  <div key={amendment.id} className="border-l-2 border-primary/30 pl-3 py-1">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-medium">Art. {amendment.article}</span>
                      <span className="text-muted-foreground">{amendment.date}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">{amendment.summary}</p>
                    <span className="text-[10px] text-muted-foreground">by {amendment.author}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Bank Links */}
            {banks.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <FileText className="h-4 w-4 text-primary" />
                    Bank Constitutions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {banks.map((bank) => (
                    <Link
                      key={bank.bank_id}
                      href={`/${locale}/banks/${bank.bank_id}?view=bankConfig&bankConfigTab=general`}
                      className="flex items-center justify-between p-2 rounded hover:bg-accent/50 transition-colors text-xs"
                    >
                      <span className="truncate">{bank.name || bank.bank_id}</span>
                      <ExternalLink className="h-3 w-3 text-muted-foreground shrink-0" />
                    </Link>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </OperatorShell>
  );
}
