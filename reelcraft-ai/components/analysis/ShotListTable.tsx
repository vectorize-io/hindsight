"use client";

import type { ShotListItem } from "@/types";
import { Camera, Clock, Zap } from "lucide-react";

export function ShotListTable({ shots }: { shots: ShotListItem[] }) {
  if (!shots.length) return <p className="text-gray-500 text-sm">No shot list generated.</p>;

  return (
    <div className="space-y-3">
      {shots.map((shot) => (
        <div key={shot.shotNumber} className="glass rounded-xl p-4">
          <div className="flex items-start gap-4">
            {/* Shot number */}
            <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-bold text-brand-300"
              style={{ background: "linear-gradient(135deg, rgba(192,38,211,0.2), rgba(234,88,12,0.2))" }}>
              {shot.shotNumber}
            </div>

            <div className="flex-1 min-w-0">
              {/* Header row */}
              <div className="flex flex-wrap items-center gap-3 mb-2 text-xs text-gray-400">
                <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{shot.timestamp}</span>
                <span className="text-gray-600">·</span>
                <span>{shot.duration}</span>
                <span className="text-gray-600">·</span>
                <span className="flex items-center gap-1"><Camera className="w-3 h-3" />{shot.shotType}</span>
                {shot.cameraAngle && <><span className="text-gray-600">·</span><span>{shot.cameraAngle}</span></>}
                {shot.cameraMovement && shot.cameraMovement !== "Static" && (
                  <><span className="text-gray-600">·</span><span className="flex items-center gap-1"><Zap className="w-3 h-3" />{shot.cameraMovement}</span></>
                )}
              </div>

              {/* Subject + action */}
              <p className="text-sm font-semibold text-white mb-1">
                {shot.subject} — {shot.action}
              </p>

              {/* How to shoot — highlighted */}
              <div className="rounded-lg p-3 mt-2" style={{ background: "rgba(192,38,211,0.06)", borderLeft: "3px solid rgba(192,38,211,0.4)" }}>
                <p className="text-xs font-semibold text-brand-300 mb-1">📹 How to shoot this:</p>
                <p className="text-sm text-gray-300 leading-relaxed">{shot.howToShoot}</p>
              </div>

              {/* Details */}
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-gray-500">
                {shot.lighting && <span>💡 {shot.lighting}</span>}
                {shot.audio && shot.audio !== "None" && <span>🎵 {shot.audio}</span>}
                {shot.text && shot.text !== "None" && <span>📝 Text: "{shot.text}"</span>}
                {shot.transition && <span>→ {shot.transition}</span>}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
