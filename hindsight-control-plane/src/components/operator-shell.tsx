"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import { OperatorSidebar } from "./operator-sidebar";
import { Shield, Activity, Users, Brain, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { Badge } from "./ui/badge";

export function OperatorShell({ children }: { children: React.ReactNode }) {
  const t = useTranslations("operator");
  const params = useParams();
  const locale = (params?.locale as string) || "en";
  const [systemStatus, setSystemStatus] = useState<"healthy" | "degraded" | "checking">("checking");
  const [agentCount, setAgentCount] = useState<number>(0);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("http://localhost:8888/health", {
          signal: AbortSignal.timeout(3000),
        });
        setSystemStatus(res.ok ? "healthy" : "degraded");
      } catch {
        setSystemStatus("degraded");
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <OperatorSidebar locale={locale} />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-10 border-b border-border bg-card flex items-center justify-between px-4 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-primary tracking-wider uppercase hidden sm:inline">
              CollabMind
            </span>
            <span className="text-[10px] text-muted-foreground hidden md:inline">
              Operator-Owned Intelligence
            </span>
            <Badge
              variant="outline"
              className={`text-[10px] h-5 px-1.5 ${
                systemStatus === "healthy"
                  ? "text-green-600 border-green-300 bg-green-50 dark:bg-green-950/30 dark:border-green-800 dark:text-green-400"
                  : "text-amber-600 border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-800 dark:text-amber-400"
              }`}
            >
              {systemStatus === "checking" ? (
                <Loader2 className="w-2.5 h-2.5 mr-1 animate-spin" />
              ) : systemStatus === "healthy" ? (
                <CheckCircle2 className="w-2.5 h-2.5 mr-1" />
              ) : (
                <AlertCircle className="w-2.5 h-2.5 mr-1" />
              )}
              Operator Mode
            </Badge>
          </div>

          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            <span className="hidden md:flex items-center gap-1">
              {systemStatus === "healthy" ? (
                <CheckCircle2 className="w-3 h-3 text-green-500" />
              ) : (
                <AlertCircle className="w-3 h-3 text-amber-500" />
              )}
              {systemStatus === "healthy" ? "All systems nominal" : "Systems degraded"}
            </span>
            <span className="hidden lg:flex items-center gap-1">
              <Users className="w-3 h-3" />
              {agentCount} agent{agentCount !== 1 ? "s" : ""} online
            </span>
            <span className="flex items-center gap-1 font-medium">
              <Brain className="w-3 h-3 text-primary" />
              Oliver
            </span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
