"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { OperatorPanel } from "@/lib/operator-nav-types";
import {
  LayoutDashboard,
  MessageSquare,
  History,
  Bot,
  Brain,
  Database,
  FlaskConical,
  ScrollText,
  Network,
  Wrench,
  KeyRound,
  Shield,
  Settings,
  BookOpen,
  GitCompare,
  FileJson,
  Cpu,
  Mic,
  Plug,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const PANELS: { id: OperatorPanel; href: string; icon: React.ElementType }[] = [
  { id: "cockpit", href: "/cockpit", icon: LayoutDashboard },
  { id: "chat", href: "/chat", icon: MessageSquare },
  { id: "runs", href: "/runs", icon: History },
  { id: "agents", href: "/agents", icon: Bot },
  { id: "memories", href: "/memories", icon: Brain },
  { id: "vector", href: "/vector", icon: Database },
  { id: "evaluation", href: "/evaluation", icon: FlaskConical },
  { id: "directives", href: "/directives", icon: ScrollText },
  { id: "backplane", href: "/backplane", icon: Network },
  { id: "tools", href: "/tools", icon: Wrench },
  { id: "config", href: "/config", icon: KeyRound },
  { id: "audit", href: "/audit", icon: Shield },
  { id: "settings", href: "/settings", icon: Settings },
  { id: "constitution", href: "/constitution", icon: BookOpen },
  { id: "router", href: "/router", icon: GitCompare },
  { id: "api-center", href: "/api-center", icon: FileJson },
  { id: "providers", href: "/providers", icon: Cpu },
  { id: "voice", href: "/voice", icon: Mic },
  { id: "connectors", href: "/connectors", icon: Plug },
];

interface OperatorSidebarProps {
  locale?: string;
}

export function OperatorSidebar({ locale = "en" }: OperatorSidebarProps) {
  const t = useTranslations("operator");
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(true);

  const currentPanel = pathname.split("/").filter(Boolean).slice(1).join("/");
  const activePanel = PANELS.find(
    (p) => currentPanel === p.href.slice(1) || currentPanel.startsWith(p.href.slice(1) + "/")
  )?.id;

  return (
    <aside
      className={cn(
        "bg-card border-r border-border flex flex-col transition-all duration-300 shrink-0",
        isCollapsed ? "w-16" : "w-56"
      )}
    >
      {/* Brand header */}
      <div className={cn("p-3 border-b border-border", isCollapsed && "px-0 text-center")}>
        {isCollapsed ? (
          <div className="h-10 w-10 mx-auto flex items-center justify-center">
            <Brain className="h-6 w-6 text-primary" />
          </div>
        ) : (
          <div className="px-3 py-1">
            <p className="text-xs font-semibold text-primary tracking-wider uppercase">
              CollabMind
            </p>
            <p className="text-[10px] text-muted-foreground">Operator Console</p>
          </div>
        )}
      </div>

      {/* Navigation items */}
      <nav className="flex-1 p-2 space-y-0.5 overflow-y-auto">
        {PANELS.map((panel) => {
          const Icon = panel.icon;
          const labelKey = panel.id;
          const isActive = activePanel === panel.id;
          const href = `/${locale}${panel.href}`;

          return (
            <Link
              key={panel.id}
              href={href}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                isActive
                  ? "bg-primary-gradient text-white shadow-sm"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                isCollapsed && "justify-center px-0"
              )}
              title={isCollapsed ? t(`panels.${labelKey}`) : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {!isCollapsed && <span className="truncate">{t(`panels.${labelKey}`)}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="p-2 border-t border-border">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={cn(
            "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors",
            isCollapsed && "justify-center px-0"
          )}
          title={isCollapsed ? t("expandSidebar") : t("collapseSidebar")}
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>{t("collapseSidebar")}</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
