"use client";

import { Brain, X } from "lucide-react";
import { useState } from "react";

interface MemoryBannerProps {
  memories: string[];
  onDismiss?: () => void;
}

export function MemoryBanner({ memories, onDismiss }: MemoryBannerProps) {
  const [expanded, setExpanded] = useState(false);
  const [visible, setVisible] = useState(true);

  if (!visible || memories.length === 0) return null;

  const handleDismiss = () => {
    setVisible(false);
    onDismiss?.();
  };

  return (
    <div className="rounded-xl border border-brand-500/30 bg-brand-500/5 p-4 mb-6 relative">
      <button
        onClick={handleDismiss}
        className="absolute top-3 right-3 text-gray-500 hover:text-gray-300 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Brain className="w-4 h-4 text-brand-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-brand-300 mb-1">
            🧠 Personalizing this guide using your creative memory
          </p>
          <p className="text-xs text-gray-400 mb-3">
            ReelCraft AI recalled {memories.length} preference{memories.length !== 1 ? "s" : ""} from your previous sessions to adapt this guide.
          </p>

          {expanded ? (
            <div className="space-y-1.5">
              {memories.map((m, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-gray-300">
                  <span className="text-brand-400 mt-0.5">•</span>
                  <span>{m}</span>
                </div>
              ))}
              <button onClick={() => setExpanded(false)} className="text-xs text-brand-400 hover:text-brand-300 mt-1">
                Show less
              </button>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {memories.slice(0, 3).map((m, i) => (
                <span key={i} className="memory-pill truncate max-w-xs">{m.slice(0, 60)}{m.length > 60 ? "…" : ""}</span>
              ))}
              {memories.length > 3 && (
                <button onClick={() => setExpanded(true)} className="memory-pill hover:bg-brand-400/15 transition-colors">
                  +{memories.length - 3} more
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
