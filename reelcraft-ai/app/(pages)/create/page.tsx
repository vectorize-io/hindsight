"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Wand2, Brain, ArrowRight, Copy, Check, AlertCircle } from "lucide-react";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { MemoryBanner } from "@/components/ui/MemoryBanner";
import { Badge } from "@/components/ui/Badge";

const PLATFORMS = ["Instagram Reels", "TikTok", "YouTube Shorts", "Facebook Reels"];
const SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"];
const DURATIONS = ["10-15 seconds", "20-30 seconds", "30-60 seconds", "60-90 seconds"];

function CreatePageInner() {
  const searchParams = useSearchParams();
  const refId = searchParams.get("ref") ?? "";

  const [form, setForm] = useState({
    topic: "",
    promotingOrShowing: "",
    equipment: "",
    phoneOrCamera: "",
    editingSoftware: "",
    skillLevel: "Beginner",
    platform: "Instagram Reels",
    desiredDuration: "20-30 seconds",
    style: "",
    differences: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [memories, setMemories] = useState<string[]>([]);
  const [memoryUsed, setMemoryUsed] = useState(false);

  // Pre-fill from memory
  useEffect(() => {
    fetch("/api/memory?mode=profile", { headers: { "x-user-id": "reelcraft-user-default" } })
      .then((r) => r.json())
      .then((d) => {
        if (d.success && d.data.hasProfile) {
          const summary: string = d.data.summary ?? "";
          const memories = d.data.memories ?? [];
          setMemories(memories.map((m: { text: string }) => m.text));

          // Extract quick values from memories
          for (const m of memories) {
            const t = (m.text ?? "").toLowerCase();
            if (t.includes("capcut")) setForm((f) => ({ ...f, editingSoftware: f.editingSoftware || "CapCut" }));
            if (t.includes("premiere")) setForm((f) => ({ ...f, editingSoftware: f.editingSoftware || "Adobe Premiere Pro" }));
            if (t.includes("iphone")) setForm((f) => ({ ...f, phoneOrCamera: f.phoneOrCamera || "iPhone" }));
            if (t.includes("android") || t.includes("samsung")) setForm((f) => ({ ...f, phoneOrCamera: f.phoneOrCamera || "Android phone" }));
            if (t.includes("beginner")) setForm((f) => ({ ...f, skillLevel: f.skillLevel === "Beginner" ? "Beginner" : f.skillLevel }));
            if (t.includes("intermediate")) setForm((f) => ({ ...f, skillLevel: "Intermediate" }));
            if (t.includes("advanced")) setForm((f) => ({ ...f, skillLevel: "Advanced" }));
          }

          if (summary) setMemoryUsed(true);
        }
      })
      .catch(() => {});
  }, []);

  const update = (k: keyof typeof form, v: string) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async () => {
    if (!form.topic.trim()) { setError("Please enter your topic."); return; }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/create-version", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-user-id": "reelcraft-user-default" },
        body: JSON.stringify({ referenceProjectId: refId, ...form }),
      });
      const data = await res.json();
      if (!data.success) { setError(data.error ?? "Failed to generate"); return; }
      setResult(data.data.customGuide);
      setMemoryUsed(data.data.memoryUsed);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const copyResult = () => {
    if (result) { navigator.clipboard.writeText(result); setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-500/30 bg-accent-500/10 text-accent-300 text-xs font-semibold mb-4">
          <Wand2 className="w-3 h-3" />
          Create My Version
        </div>
        <h1 className="text-4xl font-display font-bold text-white mb-3">
          Create Your Original Reel
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto">
          {refId ? "Using the production structure from your analyzed Reel as inspiration — all content will be completely original." : "Describe your Reel idea and get a complete production plan."}
        </p>
      </div>

      {memoryUsed && memories.length > 0 && <MemoryBanner memories={memories} />}

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Topic */}
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            What is your topic? <span className="text-red-400">*</span>
          </label>
          <input className="input" placeholder="e.g. My homemade pasta recipe, My travel vlog from Goa, My fitness transformation..." value={form.topic} onChange={(e) => update("topic", e.target.value)} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">What are you promoting or showing?</label>
          <input className="input" placeholder="e.g. My restaurant, My YouTube channel, My fitness program..." value={form.promotingOrShowing} onChange={(e) => update("promotingOrShowing", e.target.value)} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Equipment available</label>
          <input className="input" placeholder="e.g. iPhone 14, tripod, ring light..." value={form.equipment} onChange={(e) => update("equipment", e.target.value)} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Phone or camera</label>
          <input className="input" placeholder="e.g. iPhone 15 Pro, Samsung S24, Sony A6400..." value={form.phoneOrCamera} onChange={(e) => update("phoneOrCamera", e.target.value)} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Editing software</label>
          <input className="input" placeholder="e.g. CapCut, Premiere Pro, DaVinci Resolve..." value={form.editingSoftware} onChange={(e) => update("editingSoftware", e.target.value)} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Skill level</label>
          <div className="flex gap-2">
            {SKILL_LEVELS.map((s) => (
              <button key={s} onClick={() => update("skillLevel", s)}
                className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${form.skillLevel === s ? "text-white" : "glass text-gray-400 hover:text-gray-200"}`}
                style={form.skillLevel === s ? { background: "linear-gradient(135deg, #c026d3, #ea580c)" } : {}}>
                {s}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Target platform</label>
          <div className="grid grid-cols-2 gap-2">
            {PLATFORMS.map((p) => (
              <button key={p} onClick={() => update("platform", p)}
                className={`py-2.5 rounded-xl text-sm font-medium transition-all ${form.platform === p ? "text-white" : "glass text-gray-400 hover:text-gray-200"}`}
                style={form.platform === p ? { background: "linear-gradient(135deg, #c026d3, #ea580c)" } : {}}>
                {p}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Desired duration</label>
          <div className="grid grid-cols-2 gap-2">
            {DURATIONS.map((d) => (
              <button key={d} onClick={() => update("desiredDuration", d)}
                className={`py-2.5 rounded-xl text-sm font-medium transition-all ${form.desiredDuration === d ? "text-white" : "glass text-gray-400 hover:text-gray-200"}`}
                style={form.desiredDuration === d ? { background: "linear-gradient(135deg, #c026d3, #ea580c)" } : {}}>
                {d}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Style you want</label>
          <input className="input" placeholder="e.g. Fast-paced, cinematic, minimal, ASMR-heavy..." value={form.style} onChange={(e) => update("style", e.target.value)} />
        </div>

        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-300 mb-2">What should be different from the reference?</label>
          <textarea className="input resize-none h-20" placeholder="e.g. Make it more humorous, add B-roll, shorter hook, different music vibe..." value={form.differences} onChange={(e) => update("differences", e.target.value)} />
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm mb-6">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      <button onClick={handleSubmit} disabled={loading || !form.topic.trim()}
        className={`w-full flex items-center justify-center gap-2 py-4 rounded-xl font-semibold text-white transition-all ${loading || !form.topic.trim() ? "opacity-40 cursor-not-allowed bg-white/5" : "btn-primary"}`}>
        {loading ? (
          <><div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />Generating your personalized guide…</>
        ) : (
          <><Wand2 className="w-5 h-5" />Generate My Reel Plan<ArrowRight className="w-4 h-4" /></>
        )}
      </button>

      {result && (
        <div className="mt-10">
          {memoryUsed && (
            <div className="flex items-center gap-2 mb-4 text-sm text-brand-300">
              <Brain className="w-4 h-4" />
              <span>This plan was personalized using your saved creative preferences.</span>
              <Badge variant="brand">Memory Used</Badge>
            </div>
          )}

          <div className="glass rounded-2xl p-6 relative">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-white">Your Original Reel Plan</h2>
              <button onClick={copyResult} className="btn-ghost text-xs gap-1.5">
                {copied ? <><Check className="w-3.5 h-3.5 text-green-400" />Copied!</> : <><Copy className="w-3.5 h-3.5" />Copy</>}
              </button>
            </div>
            <MarkdownRenderer content={result} />
          </div>
        </div>
      )}
    </div>
  );
}

export default function CreatePage() {
  return (
    <Suspense fallback={<div className="max-w-4xl mx-auto px-4 py-20 text-center text-gray-400">Loading…</div>}>
      <CreatePageInner />
    </Suspense>
  );
}
