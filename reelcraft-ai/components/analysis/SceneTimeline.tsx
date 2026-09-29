"use client";

import { useState } from "react";
import type { SceneAnalysis, VideoMetadata } from "@/types";
import { formatDuration } from "@/lib/utils";
import { Clock, Camera, Zap } from "lucide-react";

interface SceneTimelineProps {
  scenes: SceneAnalysis[];
  metadata: VideoMetadata;
}

export function SceneTimeline({ scenes, metadata }: SceneTimelineProps) {
  const [activeScene, setActiveScene] = useState<number | null>(null);

  if (!scenes.length) return null;

  return (
    <div className="space-y-4">
      {/* Timeline bar */}
      <div className="relative h-10 rounded-full overflow-hidden bg-white/5 flex">
        {scenes.map((scene) => {
          const left = (scene.startTime / metadata.duration) * 100;
          const width = (scene.duration / metadata.duration) * 100;
          const isActive = activeScene === scene.sceneNumber;

          const colors = [
            "from-brand-600 to-brand-500",
            "from-accent-600 to-accent-500",
            "from-blue-600 to-blue-500",
            "from-emerald-600 to-emerald-500",
            "from-yellow-600 to-yellow-500",
            "from-pink-600 to-pink-500",
            "from-purple-600 to-purple-500",
            "from-teal-600 to-teal-500",
          ];
          const color = colors[(scene.sceneNumber - 1) % colors.length];

          return (
            <button
              key={scene.sceneNumber}
              className={`absolute top-0 h-full bg-gradient-to-r ${color} transition-all duration-200
                         ${isActive ? "opacity-100 scale-y-100" : "opacity-60 hover:opacity-80"}
                         border-r border-black/20`}
              style={{ left: `${left}%`, width: `${Math.max(width, 2)}%` }}
              onClick={() => setActiveScene(isActive ? null : scene.sceneNumber)}
              title={`Scene ${scene.sceneNumber}: ${scene.purpose}`}
            />
          );
        })}
      </div>

      {/* Timestamp markers */}
      <div className="flex justify-between text-xs text-gray-600 px-0.5">
        <span>0:00</span>
        <span>{formatDuration(metadata.duration / 4)}</span>
        <span>{formatDuration(metadata.duration / 2)}</span>
        <span>{formatDuration((metadata.duration * 3) / 4)}</span>
        <span>{formatDuration(metadata.duration)}</span>
      </div>

      {/* Scene cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
        {scenes.map((scene) => (
          <button
            key={scene.sceneNumber}
            onClick={() => setActiveScene(activeScene === scene.sceneNumber ? null : scene.sceneNumber)}
            className={`text-left p-4 rounded-xl border transition-all duration-200
              ${activeScene === scene.sceneNumber
                ? "border-brand-500/50 bg-brand-500/10"
                : "border-white/[0.06] bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
              }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-bold text-brand-400">Scene {scene.sceneNumber}</span>
              <span className="text-xs text-gray-500">
                {scene.startTime.toFixed(1)}s – {scene.endTime.toFixed(1)}s
              </span>
            </div>
            <p className="text-sm font-medium text-white line-clamp-1">{scene.purpose}</p>
            <p className="text-xs text-gray-500 mt-1 line-clamp-1">{scene.cameraAngle} • {scene.framing}</p>

            {activeScene === scene.sceneNumber && (
              <div className="mt-3 pt-3 border-t border-white/[0.06] space-y-2 text-xs text-gray-300">
                <Row icon={<Clock className="w-3 h-3" />} label="Duration" value={`${scene.duration.toFixed(1)}s`} />
                <Row icon={<Camera className="w-3 h-3" />} label="Camera" value={`${scene.cameraAngle} · ${scene.cameraMovement}`} />
                <Row icon={<Zap className="w-3 h-3" />} label="Transition" value={scene.transition} />
                {scene.text && <Row icon={<span>T</span>} label="Text" value={scene.text} />}
                {scene.lighting && <Row icon={<span>💡</span>} label="Lighting" value={scene.lighting} />}
                {scene.editingTechnique && <Row icon={<span>✂️</span>} label="Editing" value={scene.editingTechnique} />}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function Row({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  if (!value || value === "None" || value === "N/A") return null;
  return (
    <div className="flex items-start gap-2">
      <span className="text-gray-600 flex-shrink-0 mt-0.5">{icon}</span>
      <span className="text-gray-500 flex-shrink-0 w-16">{label}:</span>
      <span className="text-gray-300">{value}</span>
    </div>
  );
}
