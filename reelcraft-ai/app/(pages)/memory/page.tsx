"use client";

import { useEffect, useState } from "react";
import { Brain, Plus, Trash2, RefreshCw, AlertCircle, CheckCircle } from "lucide-react";
import type { MemoryEntry } from "@/types";
import { Badge } from "@/components/ui/Badge";
import { PageLoader } from "@/components/ui/LoadingSpinner";
import { categoryColor, categoryIcon, formatRelativeTime } from "@/lib/utils";

const PROFILE_FIELDS = [
  { key: "skillLevel", label: "Skill Level", placeholder: "Beginner / Intermediate / Advanced" },
  { key: "preferredEditingSoftware", label: "Editing Software", placeholder: "CapCut, Premiere Pro, DaVinci..." },
  { key: "preferredFilmingDevice", label: "Filming Device", placeholder: "iPhone, Samsung, Sony A6400..." },
  { key: "preferredPlatforms", label: "Platforms", placeholder: "Instagram, TikTok, YouTube..." },
  { key: "preferredReelDuration", label: "Preferred Duration", placeholder: "20-30 seconds, 60 seconds..." },
];

export default function MemoryPage() {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [profile, setProfile] = useState<{ summary: string; hasProfile: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [profileForm, setProfileForm] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [customMemory, setCustomMemory] = useState("");
  const [savingCustom, setSavingCustom] = useState(false);
  const [hindsightAvailable, setHindsightAvailable] = useState(true);
  const [activeTab, setActiveTab] = useState<"memories" | "profile">("memories");

  const fetchData = async () => {
    setLoading(true);
    try {
      const [memRes, profileRes] = await Promise.all([
        fetch("/api/memory", { headers: { "x-user-id": "reelcraft-user-default" } }),
        fetch("/api/memory?mode=profile", { headers: { "x-user-id": "reelcraft-user-default" } }),
      ]);
      const memData = await memRes.json();
      const profileData = await profileRes.json();

      if (memData.success) setMemories(memData.data.memories ?? []);
      if (profileData.success) setProfile(profileData.data);
    } catch {
      setHindsightAvailable(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const saveProfile = async () => {
    setSaving(true);
    try {
      const profile: Record<string, string | string[]> = {};
      for (const [k, v] of Object.entries(profileForm)) {
        if (v.trim()) {
          if (k === "preferredPlatforms") {
            profile[k] = v.split(",").map((s) => s.trim()).filter(Boolean);
          } else {
            profile[k] = v.trim();
          }
        }
      }
      await fetch("/api/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-user-id": "reelcraft-user-default" },
        body: JSON.stringify({ type: "profile", profile }),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
      fetchData();
    } catch { /* ignore */ } finally {
      setSaving(false);
    }
  };

  const saveCustomMemory = async () => {
    if (!customMemory.trim()) return;
    setSavingCustom(true);
    try {
      await fetch("/api/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-user-id": "reelcraft-user-default" },
        body: JSON.stringify({ type: "memory", content: customMemory.trim(), category: "profile" }),
      });
      setCustomMemory("");
      fetchData();
    } catch { /* ignore */ } finally {
      setSavingCustom(false);
    }
  };

  const byCategory = memories.reduce<Record<string, MemoryEntry[]>>((acc, m) => {
    const cat = m.category ?? "profile";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(m);
    return acc;
  }, {});

  if (loading) return <PageLoader message="Loading your creative memory…" />;

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-brand-500/20 flex items-center justify-center">
              <Brain className="w-5 h-5 text-brand-400" />
            </div>
            <h1 className="text-3xl font-display font-bold text-white">Creative Memory</h1>
          </div>
          <p className="text-gray-400 max-w-lg">
            ReelCraft AI uses Hindsight to remember your preferences and adapt every guide to your unique workflow.
          </p>
        </div>
        <button onClick={fetchData} className="btn-ghost gap-1.5">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {!hindsightAvailable && (
        <div className="glass rounded-xl p-4 border border-yellow-500/30 bg-yellow-500/5 flex items-start gap-3 mb-8">
          <AlertCircle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-yellow-300">Hindsight not configured</p>
            <p className="text-xs text-gray-400 mt-1">
              Add your <code className="text-yellow-400">HINDSIGHT_API_KEY</code> to <code className="text-yellow-400">.env.local</code> to enable persistent memory.
              Get your key at <a href="https://hindsight.vectorize.io" className="text-brand-400 underline" target="_blank" rel="noreferrer">hindsight.vectorize.io</a>
            </p>
          </div>
        </div>
      )}

      {/* Profile summary card */}
      {profile?.hasProfile && profile.summary && (
        <div className="glass rounded-2xl p-6 mb-8 border border-brand-500/20">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-brand-400" />
            <h2 className="text-sm font-semibold text-brand-300">MY CREATIVE PROFILE</h2>
          </div>
          <p className="text-sm text-gray-300 leading-relaxed">{profile.summary}</p>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {[
          { label: "Total Memories", value: memories.length, color: "text-brand-400" },
          { label: "Profile", value: byCategory.profile?.length ?? 0, color: "text-purple-400" },
          { label: "Creative", value: byCategory.creative?.length ?? 0, color: "text-accent-400" },
          { label: "Workflow", value: byCategory.workflow?.length ?? 0, color: "text-blue-400" },
        ].map((s) => (
          <div key={s.label} className="glass rounded-xl p-4 text-center">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/[0.08] mb-6">
        {(["memories", "profile"] as const).map((t) => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-3 text-sm font-medium capitalize transition-all border-b-2 -mb-px ${
              activeTab === t ? "text-white border-brand-500" : "text-gray-500 border-transparent hover:text-gray-300"
            }`}>
            {t === "memories" ? `All Memories (${memories.length})` : "Set Profile"}
          </button>
        ))}
      </div>

      {/* Memories tab */}
      {activeTab === "memories" && (
        <div className="space-y-6">
          {memories.length === 0 ? (
            <div className="glass rounded-2xl p-12 text-center">
              <Brain className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <h3 className="text-white font-semibold mb-2">No memories yet</h3>
              <p className="text-gray-400 text-sm mb-6">
                Start a conversation in the Chat, analyze a Reel, or fill in your profile below to build your creative memory.
              </p>
              <button onClick={() => setActiveTab("profile")} className="btn-primary">
                <Plus className="w-4 h-4" />
                Set Up Your Profile
              </button>
            </div>
          ) : (
            <>
              {/* Quick add */}
              <div className="glass rounded-xl p-4">
                <p className="text-sm font-medium text-gray-300 mb-3">Add a memory manually</p>
                <div className="flex gap-2">
                  <input
                    className="input flex-1"
                    placeholder="e.g. I prefer warm color grades, I film in my kitchen, I like transitions that match the beat..."
                    value={customMemory}
                    onChange={(e) => setCustomMemory(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && saveCustomMemory()}
                  />
                  <button onClick={saveCustomMemory} disabled={savingCustom || !customMemory.trim()}
                    className="btn-primary px-4">
                    {savingCustom ? <div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" /> : <Plus className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Memories by category */}
              {Object.entries(byCategory).map(([cat, items]) => (
                <div key={cat}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-lg">{categoryIcon(cat)}</span>
                    <h3 className="text-sm font-semibold text-white capitalize">{cat}</h3>
                    <Badge variant="default" className="text-xs">{items.length}</Badge>
                  </div>
                  <div className="space-y-2">
                    {items.map((m, i) => (
                      <div key={i} className="glass rounded-xl px-4 py-3 flex items-start gap-3">
                        <span className={`text-xs badge mt-0.5 flex-shrink-0 ${categoryColor(cat)}`}>
                          {cat}
                        </span>
                        <p className="text-sm text-gray-300 flex-1">{m.text}</p>
                        {m.whyItMatters && (
                          <p className="text-xs text-gray-600 italic hidden md:block max-w-xs text-right flex-shrink-0">
                            {m.whyItMatters}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Profile tab */}
      {activeTab === "profile" && (
        <div className="glass rounded-2xl p-6">
          <h2 className="text-lg font-bold text-white mb-2">Set Your Creative Profile</h2>
          <p className="text-sm text-gray-400 mb-6">
            This information is stored in Hindsight and used to personalize every Reel guide for your workflow.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            {PROFILE_FIELDS.map(({ key, label, placeholder }) => (
              <div key={key}>
                <label className="block text-sm font-medium text-gray-300 mb-1.5">{label}</label>
                <input
                  className="input"
                  placeholder={placeholder}
                  value={profileForm[key] ?? ""}
                  onChange={(e) => setProfileForm((f) => ({ ...f, [key]: e.target.value }))}
                />
              </div>
            ))}
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-1.5">Preferred content style</label>
            <input
              className="input"
              placeholder="e.g. Fast-paced food Reels, cinematic travel, minimal lifestyle..."
              value={profileForm.editingStyle ?? ""}
              onChange={(e) => setProfileForm((f) => ({ ...f, editingStyle: e.target.value }))}
            />
          </div>

          <button onClick={saveProfile} disabled={saving}
            className={`flex items-center gap-2 py-3 px-6 rounded-xl font-semibold text-white transition-all ${saving ? "opacity-50 cursor-not-allowed bg-white/5" : "btn-primary"}`}>
            {saving ? (
              <><div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />Saving to Hindsight…</>
            ) : saved ? (
              <><CheckCircle className="w-4 h-4 text-green-400" />Saved to Memory!</>
            ) : (
              <><Brain className="w-4 h-4" />Save to Hindsight Memory</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
