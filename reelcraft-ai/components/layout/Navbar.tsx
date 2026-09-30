"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Film, Brain, MessageSquare, FolderOpen, Sparkles, Plus } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Home", icon: null },
  { href: "/analyze", label: "Analyze", icon: Film },
  { href: "/create", label: "Create", icon: Plus },
  { href: "/projects", label: "Projects", icon: FolderOpen },
  { href: "/memory", label: "Memory", icon: Brain },
  { href: "/chat", label: "Chat", icon: MessageSquare },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-white/[0.06]"
      style={{ background: "rgba(10,10,15,0.85)", backdropFilter: "blur(20px)" }}>
      <div className="max-w-7xl mx-auto h-full px-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-display font-bold text-lg tracking-tight">
            <span className="gradient-text">ReelCraft</span>
            <span className="text-white"> AI</span>
          </span>
        </Link>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || (href !== "/" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150",
                  active
                    ? "text-white bg-white/10"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                )}
              >
                {Icon && <Icon className="w-3.5 h-3.5" />}
                {label}
              </Link>
            );
          })}
        </nav>

        {/* CTA */}
        <Link href="/analyze" className="btn-primary text-xs px-4 py-2">
          <Film className="w-3.5 h-3.5" />
          Analyze Reel
        </Link>
      </div>
    </header>
  );
}
