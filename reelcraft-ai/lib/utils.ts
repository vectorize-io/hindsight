import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return formatDate(iso);
}

export function truncate(str: string, maxLen: number): string {
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

export function generateId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export function skillLevelColor(level: string): string {
  switch (level.toLowerCase()) {
    case "beginner": return "text-green-400 bg-green-400/10 border-green-400/30";
    case "intermediate": return "text-yellow-400 bg-yellow-400/10 border-yellow-400/30";
    case "advanced": return "text-red-400 bg-red-400/10 border-red-400/30";
    default: return "text-gray-400 bg-gray-400/10 border-gray-400/30";
  }
}

export function categoryColor(category: string): string {
  const map: Record<string, string> = {
    profile: "text-brand-400 bg-brand-400/10 border-brand-400/30",
    creative: "text-accent-400 bg-accent-400/10 border-accent-400/30",
    workflow: "text-blue-400 bg-blue-400/10 border-blue-400/30",
    project: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
  };
  return map[category] ?? "text-gray-400 bg-gray-400/10 border-gray-400/30";
}

export function categoryIcon(category: string): string {
  const map: Record<string, string> = {
    profile: "👤",
    creative: "🎨",
    workflow: "⚙️",
    project: "🎬",
  };
  return map[category] ?? "🧠";
}

export const DEMO_USER_ID = "demo-user-hackathon";
export const DEFAULT_USER_ID = process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";
