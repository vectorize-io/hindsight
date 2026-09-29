"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface SectionCardProps {
  number: number;
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string;
  badgeColor?: string;
}

export function SectionCard({
  number,
  title,
  subtitle,
  icon,
  children,
  defaultOpen = false,
  badge,
  badgeColor = "bg-brand-500/15 text-brand-300 border-brand-500/30",
}: SectionCardProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={cn("glass rounded-2xl overflow-hidden transition-all duration-200", open && "ring-1 ring-brand-500/20")}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-4 p-5 text-left hover:bg-white/[0.02] transition-colors"
      >
        {/* Number */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-bold"
          style={{ background: open ? "linear-gradient(135deg, #c026d3, #ea580c)" : "rgba(255,255,255,0.06)" }}>
          {open ? icon ?? number : number}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-white text-sm">{title}</span>
            {badge && (
              <span className={cn("badge text-xs", badgeColor)}>{badge}</span>
            )}
          </div>
          {subtitle && !open && (
            <p className="text-xs text-gray-500 mt-0.5 truncate">{subtitle}</p>
          )}
        </div>

        <ChevronDown className={cn("w-4 h-4 text-gray-500 flex-shrink-0 transition-transform duration-200", open && "rotate-180")} />
      </button>

      {open && (
        <div className="px-5 pb-5 border-t border-white/[0.06] pt-5">
          {children}
        </div>
      )}
    </div>
  );
}
