"use client";

import { useState } from "react";
import { HelpCircle, X } from "lucide-react";

interface WhyButtonProps {
  explanation: string;
  label?: string;
}

export function WhyButton({ explanation, label = "Why?" }: WhyButtonProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="inline-block relative">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium
                   bg-brand-500/10 text-brand-400 border border-brand-500/20
                   hover:bg-brand-500/20 transition-colors"
      >
        <HelpCircle className="w-3 h-3" />
        {label}
      </button>

      {open && (
        <div className="absolute z-20 bottom-full mb-2 left-0 w-72 glass rounded-xl p-4 shadow-2xl">
          <div className="flex items-start justify-between gap-2 mb-2">
            <span className="text-xs font-semibold text-brand-300">Why this technique?</span>
            <button onClick={() => setOpen(false)} className="text-gray-500 hover:text-gray-300">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{explanation}</p>
        </div>
      )}
    </div>
  );
}
