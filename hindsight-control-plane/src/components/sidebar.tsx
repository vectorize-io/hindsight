"use client";

import { useEffect, useState } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
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
  Menu,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { client } from "@/lib/api";

type NavItem =
  | "home"
  | "recall"
  | "reflect"
  | "data"
  | "documents"
  | "entities"
  | "knowledge"
  | "profile";

interface SidebarProps {
  currentTab: NavItem;
  onTabChange: (tab: NavItem) => void;
}

// The header's menu button (phones only) opens the drawer through this event, so
// the two can live in separate components without shared state.
const OPEN_NAV_EVENT = "hindsight:open-nav";

/** Opens the bank navigation drawer. Rendered in the header, hidden from md up. */
export function SidebarMenuButton() {
  const t = useTranslations("bank.sidebar");
  return (
    <button
      type="button"
      onClick={() => window.dispatchEvent(new Event(OPEN_NAV_EVENT))}
      className="md:hidden inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
      aria-label={t("openNavigation")}
      title={t("openNavigation")}
    >
      <Menu className="h-5 w-5" />
    </button>
  );
}

export function Sidebar({ currentTab, onTabChange }: SidebarProps) {
  const t = useTranslations("bank.sidebar");
  const tBank = useTranslations("bank");
  const tCommon = useTranslations("common");
  const { currentBank, currentBankName } = useBank();
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [apiVersion, setApiVersion] = useState<string | null>(null);

  useEffect(() => {
    client
      .getVersion()
      .then((v) => setApiVersion(v.api_version))
      .catch(() => setApiVersion(null));
  }, []);

  useEffect(() => {
    const open = () => setDrawerOpen(true);
    window.addEventListener(OPEN_NAV_EVENT, open);
    return () => window.removeEventListener(OPEN_NAV_EVENT, open);
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

  const renderNav = (collapsed: boolean, id?: string) => (
    <nav id={id} className="flex-1 p-3 pt-4 overflow-y-auto">
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
                    setDrawerOpen(false);
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
                  collapsed && "justify-center px-0"
                )}
                title={collapsed ? item.label : undefined}
                aria-label={item.label}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );

  // Mouse shortcut: clicking the aside chrome toggles it. Nav links and the
  // collapse button stop propagation so they don't also flip it. The aside
  // carries no button role — it wraps the nav, and the bottom button is the
  // one control assistive tech and the keyboard use to toggle.
  const toggleCollapsed = () => setIsCollapsed((v) => !v);
  const toggleLabel = isCollapsed ? t("expandSidebar") : t("collapseSidebar");

  return (
    <>
      {/* Desktop and tablet: a rail beside the content. Phones get the drawer
          below instead, so the content keeps the full screen width. */}
      <aside
        onClick={toggleCollapsed}
        className={cn(
          "hidden md:flex bg-card border-r border-border flex-col h-full transition-all duration-300 cursor-pointer select-none",
          isCollapsed ? "w-16" : "w-64"
        )}
      >
        {renderNav(isCollapsed, "bank-sidebar-nav")}

        {/* Collapse/Expand button at bottom — the accessible toggle. */}
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
            aria-expanded={!isCollapsed}
            aria-controls="bank-sidebar-nav"
            aria-label={toggleLabel}
            title={toggleLabel}
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

      {/* Phones: the same nav in a drawer, opened by SidebarMenuButton. */}
      <DialogPrimitive.Root open={drawerOpen} onOpenChange={setDrawerOpen}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 md:hidden animate-soft-fade-in" />
          <DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 flex w-72 max-w-[85vw] flex-col bg-card border-r border-border shadow-xl pb-[env(safe-area-inset-bottom)] md:hidden animate-drawer-in">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <DialogPrimitive.Title className="truncate text-sm font-semibold">
                {currentBankName || currentBank}
              </DialogPrimitive.Title>
              <DialogPrimitive.Close
                className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-label={tCommon("close")}
              >
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>
            <DialogPrimitive.Description className="sr-only">
              {t("memoryBank")}
            </DialogPrimitive.Description>
            {renderNav(false)}
            {apiVersion && (
              <div className="border-t border-border p-3 text-center text-xs text-muted-foreground/60">
                {`Hindsight v${apiVersion}`}
              </div>
            )}
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    </>
  );
}
