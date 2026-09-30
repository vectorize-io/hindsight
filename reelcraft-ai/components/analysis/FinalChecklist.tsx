"use client";

import { useState } from "react";
import { CheckSquare, Square } from "lucide-react";

export function FinalChecklist({ items }: { items: string[] }) {
  const [checked, setChecked] = useState<Set<number>>(new Set());

  const toggle = (i: number) => {
    setChecked((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  };

  const progress = Math.round((checked.size / items.length) * 100);

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${progress}%`,
              background: "linear-gradient(90deg, #c026d3, #ea580c)"
            }}
          />
        </div>
        <span className="text-sm font-semibold text-white w-12 text-right">
          {checked.size}/{items.length}
        </span>
      </div>

      {progress === 100 && (
        <div className="glass rounded-xl p-4 text-center border border-green-500/30">
          <p className="text-green-400 font-semibold">🎉 Reel is ready to publish!</p>
        </div>
      )}

      {/* Items */}
      <div className="space-y-2">
        {items.map((item, i) => {
          const isChecked = checked.has(i);
          const label = item.replace(/^[☐✓□]\s*/, "");
          return (
            <button
              key={i}
              onClick={() => toggle(i)}
              className={`w-full flex items-start gap-3 p-3 rounded-xl text-left transition-all duration-150
                ${isChecked ? "bg-green-500/5 border border-green-500/20" : "glass-hover"}`}
            >
              {isChecked
                ? <CheckSquare className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                : <Square className="w-4 h-4 text-gray-500 flex-shrink-0 mt-0.5" />}
              <span className={`text-sm ${isChecked ? "line-through text-gray-500" : "text-gray-300"}`}>
                {label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
