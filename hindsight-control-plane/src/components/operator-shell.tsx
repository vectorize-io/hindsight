"use client";

import { useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import { OperatorSidebar } from "./operator-sidebar";
import { Shield, Activity, Users, Brain } from "lucide-react";
import { Badge } from "./ui/badge";

/**
 * OperatorShell — root layout wrapper for the CollabMind Operator Console.
 * Provides the persistent 13-panel sidebar and top status bar.
 */
export function OperatorShell({ children }: { children: React.ReactNode }) {
  const t = useTranslations("operator");
  const params = useParams();
  const locale = (params?.locale as string) || "en";

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Persistent sidebar */}
      <OperatorSidebar locale={locale} />

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top status bar */}
        <header className="h-10 border-b border-border bg-card flex items-center justify-between px-4 shrink-0">
          {/* Left: brand + mission */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-primary tracking-wider uppercase hidden sm:inline">
              CollabMind
            </span>
            <span className="text-[10px] text-muted-foreground hidden md:inline">
              Operator-Owned Intelligence
            </span>
            <Badge variant="outline" className="text-[10px] h-5 px-1.5 text-green-600 border-green-300 bg-green-50 dark:bg-green-950/30 dark:border-green-800 dark:text-green-400">
              <Shield className="w-2.5 h-2.5 mr-1" />
              Operator Mode
            </Badge>
          </div>

          {/* Right: status indicators */}
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            <span className="hidden md:flex items-center gap-1">
              <Activity className="w-3 h-3 text-green-500" />
              All systems nominal
            </span>
            <span className="hidden lg:flex items-center gap-1">
              <Users className="w-3 h-3" />
              0 agents online
            </span>
            <span className="flex items-center gap-1 font-medium">
              <Brain className="w-3 h-3 text-primary" />
              Oliver
            </span>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
