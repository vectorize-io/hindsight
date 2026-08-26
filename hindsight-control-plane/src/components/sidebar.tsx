"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";
import {
  Home,
  Search,
  Sparkles,
  Database,
  FileText,
  Users,
  Network,
  ChevronLeft,
  ChevronRight,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { client } from "@/lib/api";

type NavItem =
  "home" | "recall" | "reflect" | "data" | "documents" | "entities" | "knowledge" | "profile";

interface SidebarProps {
  currentTab: NavItem;
  onTabChange: (tab: NavItem) => void;
}

export function Sidebar({ currentTab, onTabChange }: SidebarProps) {
  const t = useTranslations("bank.sidebar");
  const tBank = useTranslations("bank");
  const { currentBank } = useBank();
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [apiVersion, setApiVersion] = useState<string | null>(null);

  useEffect(() => {
    client
      .getVersion()
      .then((v) => setApiVersion(v.api_version))
      .catch(() => setApiVersion(null));
  }, []);

  if (!currentBank) {
    return null;
  }

  const navItems = [
    { id: "home" as NavItem, label: t("home"), icon: Home },
    { id: "data" as NavItem, label: t("memories"), icon: Database },
    { id: "knowledge" as NavItem, label: t("knowledge"), icon: Network },
    { id: "recall" as NavItem, label: t("recall"), icon: Search },
    { id: "reflect" as NavItem, label: t("reflect"), icon: Sparkles },
    { id: "documents" as NavItem, label: t("documents"), icon: FileText },
    { id: "entities" as NavItem, label: t("entities"), icon: Users },
    { id: "profile" as NavItem, label: tBank("bankConfiguration"), icon: Settings },
  ];

  // Toggle when the user clicks the aside chrome itself — nav links and the
  // collapse button stop propagation so clicking an item navigates without
  // also flipping the collapsed state.
  const toggleCollapsed = () => setIsCollapsed((v) => !v);

  return (
    <aside
      onClick={toggleCollapsed}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        // Only when the aside itself has focus. Without this guard, pressing
        // Enter on a focused nav link bubbles up here and collapses the rail
        // on every keyboard navigation.
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleCollapsed();
        }
      }}
      aria-label={isCollapsed ? t("expandSidebar") : t("collapseSidebar")}
      aria-expanded={!isCollapsed}
      className={cn(
        "bg-card border-r border-border flex flex-col h-full transition-all duration-300 cursor-pointer select-none",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <nav className="flex-1 p-3 pt-4">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            const href = bankRoute(currentBank, `?view=${item.id}`);

            return (
              <li key={item.id}>
                <Link
                  href={href}
                  onClick={(e) => {
                    // Don't bubble — clicking an item navigates, it doesn't
                    // toggle the sidebar.
                    e.stopPropagation();
                    // For left-click, prevent default and use the callback
                    // This allows the parent to handle navigation without full page reload
                    if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
                      e.preventDefault();
                      onTabChange(item.id);
                      // Give the header logo a playful spin on navigation.
                      window.dispatchEvent(new CustomEvent("hindsight:logo-spin"));
                    }
                    // Middle-click or Ctrl/Cmd+click will naturally open in new tab
                  }}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all cursor-pointer",
                    isActive
                      ? "bg-primary-gradient text-white shadow-sm"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                    isCollapsed && "justify-center px-0"
                  )}
                  title={isCollapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!isCollapsed && <span>{item.label}</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Collapse/Expand button at bottom — kept as an explicit affordance
          alongside the ambient click-to-toggle on the aside chrome. */}
      <div className="p-3 border-t border-border">
        {apiVersion && (
          <div
            className={cn(
              "mb-2 text-xs text-muted-foreground/60 text-center select-none",
              isCollapsed ? "px-0" : "px-1"
            )}
            title={`Hindsight API v${apiVersion}`}
          >
            {isCollapsed ? `v${apiVersion}` : `Hindsight v${apiVersion}`}
          </div>
        )}
        <button
          onClick={(e) => {
            // Don't double-toggle when the aside's onClick also fires.
            e.stopPropagation();
            toggleCollapsed();
          }}
          className={cn(
            "w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors cursor-pointer",
            isCollapsed && "justify-center px-0"
          )}
          title={isCollapsed ? t("expandSidebar") : t("collapseSidebar")}
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span>{t("collapse")}</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
